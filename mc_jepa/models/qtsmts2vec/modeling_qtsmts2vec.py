"""TS2Vec baseline for the series JEPA (configuration_qtsmts2vec.py).

Per (stock, decision time) window, with the encoder E of ``qtsmfjepa`` (bidirectional, no CLS token) and one patch
as one TS2Vec timestamp, following the official ``ts2vec.py`` / ``models/losses.py``:

    two crops [eleft, right) and [left, eright) of the P context patches, shifted by a random offset per window,
    overlapping on [left, right) (crop_l >= crop_min_patches patches)
    r1 = E(mask(crop 1))[-crop_l:],  r2 = E(mask(crop 2))[:crop_l]       binomial timestamp mask, per view
    loss = mean over levels of  alpha * instance(r1, r2) + (1 - alpha) * temporal(r1, r2),
           then r <- maxpool_2(r) over time, down to one timestamp

Instance negatives are the other windows of the micro-batch at the same timestamp; temporal negatives are the other
timestamps of the same window; similarities are raw dot products. Position embeddings count from each crop's start,
since window-absolute positions would let the temporal term be solved from position alone. The module mirrors
``QTSMFJepaModel``'s interface (``encode``, ``forward(context, future, symbol_mask)``, ``loss`` and ``metrics``
outputs), so ``panel_jepa_trainer`` and ``tools.eval_jepa_probe`` run it unchanged.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from mc_jepa.models.qtsmts2vec.configuration_qtsmts2vec import QTSMTs2VecConfig


class QTSMTs2VecMLPBlock(nn.Module):
    """Pre-LN residual MLP block."""

    def __init__(self, config: QTSMTs2VecConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.fc2(self.act(self.fc1(self.norm(x)))))


class QTSMTs2VecEncoderLayer(GradientCheckpointingLayer):
    """Pre-LN bidirectional transformer layer."""

    def __init__(self, config: QTSMTs2VecConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlp = QTSMTs2VecMLPBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S, T, D = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(S, T, 3, self.num_heads, D // self.num_heads).unbind(2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        x = x + self.dropout(self.proj(attn.transpose(1, 2).reshape(S, T, D)))
        return self.mlp(x)


class QTSMTs2VecStockEncoder(nn.Module):
    """E: [S, T, patch_len * F] patches (optionally masked) -> [S, T, D] normed patch states."""

    def __init__(self, config: QTSMTs2VecConfig):
        super().__init__()
        self.patch_len = config.patch_len
        self.max_patches = config.max_patches
        self.input_clip = config.input_clip
        self.patch_embed = nn.Linear(config.patch_len * config.in_features, config.hidden_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.max_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([QTSMTs2VecEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """[S, L, F] bars -> [S, L / patch_len, patch_len * F] clipped patches."""
        S, L, F_ = x.shape
        assert L % self.patch_len == 0, f"sequence length {L} is not a multiple of patch_len {self.patch_len}"
        num_patches = L // self.patch_len
        assert num_patches <= self.max_patches, f"{num_patches} patches exceed max_patches {self.max_patches}"
        clip = self.input_clip
        x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)
        return x.reshape(S, num_patches, self.patch_len * F_)

    def forward(self, patches: torch.Tensor, keep: torch.Tensor | None = None) -> torch.Tensor:
        """keep [S, T]: patch embeddings to keep (TS2Vec timestamp masking zeroes the rest, before positions)."""
        h = self.patch_embed(patches.to(self.patch_embed.weight.dtype))
        if keep is not None:
            h = h * keep[..., None].to(h.dtype)
        h = self.dropout(h + self.position_embeddings[:, : h.shape[1]])
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


@dataclass
class QTSMTs2VecOutput(ModelOutput):
    """
    loss: hierarchical contrastive loss (only when computed, see ``forward``).
    stock_states: [B, N, D] readout states of the uncropped, unmasked window (zero at padding); ``None`` with a loss.
    market_state, peer_states: always ``None``; kept so callers of ``QTSMFJepaModel`` work unchanged.
    metrics: detached scalar losses and collapse diagnostics.
    """

    loss: torch.FloatTensor | None = None
    stock_states: torch.FloatTensor | None = None
    market_state: torch.FloatTensor | None = None
    peer_states: torch.FloatTensor | None = None
    metrics: dict | None = None


def _spectrum(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Effective rank (exp entropy of the covariance spectrum) and mean per-dimension std of rows x [M, D]."""
    x = x.detach().float().reshape(-1, x.shape[-1])
    if x.shape[0] < 2:
        nan = x.new_tensor(float("nan"))
        return nan, nan
    x = x - x.mean(dim=0)
    with torch.autocast(x.device.type, enabled=False):  # called inside the bf16 forward; eigh has no bf16 kernel
        cov = x.T @ x / (x.shape[0] - 1)
        eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    p = eig / eig.sum().clamp_min(1e-12)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()), cov.diagonal().sqrt().mean()


def instance_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """z1, z2 [B, T, D]: at each timestamp, the other view of the same window against all other windows."""
    B = z1.shape[0]
    if B == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0).transpose(0, 1)  # [T, 2B, D]
    return _contrast(z, B)


def temporal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """z1, z2 [B, T, D]: in each window, the other view at the same timestamp against all other timestamps."""
    T = z1.shape[1]
    if T == 1:
        return z1.new_tensor(0.0)
    return _contrast(torch.cat([z1, z2], dim=1), T)  # [B, 2T, D]


def _contrast(z: torch.Tensor, n: int) -> torch.Tensor:
    """z [G, 2n, D], rows i and n + i positive; InfoNCE over the other 2n - 1 rows (official TS2Vec layout)."""
    sim = z @ z.transpose(1, 2)  # [G, 2n, 2n]
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1] + torch.triu(sim, diagonal=1)[:, :, 1:]  # drop the diagonal
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(n, device=z.device)
    return (logits[:, i, n + i - 1].mean() + logits[:, n + i, i].mean()) / 2


def hierarchical_contrastive_loss(
    z1: torch.Tensor, z2: torch.Tensor, alpha: float = 0.5, temporal_unit: int = 0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official ``hierarchical_contrastive_loss``; also returns the level-averaged instance and temporal terms."""
    instance = temporal = z1.new_tensor(0.0)
    d = 0
    while z1.shape[1] > 1:
        instance = instance + instance_contrastive_loss(z1, z2)
        if d >= temporal_unit:
            temporal = temporal + temporal_contrastive_loss(z1, z2)
        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
    if z1.shape[1] == 1:
        instance = instance + instance_contrastive_loss(z1, z2)
        d += 1
    instance, temporal = instance / d, temporal / d
    return alpha * instance + (1 - alpha) * temporal, instance, temporal


class QTSMTs2VecPreTrainedModel(PreTrainedModel):
    config: QTSMTs2VecConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["QTSMTs2VecEncoderLayer"]

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()
        elif isinstance(module, QTSMTs2VecStockEncoder):
            module.position_embeddings.data.normal_(mean=0.0, std=std)


class QTSMTs2VecModel(QTSMTs2VecPreTrainedModel):
    def __init__(self, config: QTSMTs2VecConfig):
        super().__init__(config)
        self.encoder = QTSMTs2VecStockEncoder(config)
        self.peer_pooler = None  # no panel context; tools.eval_jepa_probe checks this attribute
        self.post_init()

    def _readout(self, h: torch.Tensor, readout: str | None = None) -> torch.Tensor:
        """h [S, P, D] -> [S, D]."""
        readout = readout or self.config.readout
        if readout == "max":
            return h.amax(dim=1)
        return h[:, -1] if readout == "last" else h.mean(dim=1)

    def encode(self, context: torch.Tensor, symbol_mask: torch.Tensor, readout: str | None = None, **kwargs):
        """Frozen features for transfer: (stock_states [B, N, D], None, None), like ``QTSMFJepaModel.encode``."""
        mask = symbol_mask.bool()
        states = self._readout(self.encoder(self.encoder.patchify(context[mask])), readout)
        z = states.new_zeros(*mask.shape, states.shape[-1])
        z[mask] = states
        return z, None, None

    def _crops(self, P: int, S: int, generator: torch.Generator | None):
        """Official TS2Vec cropping: (start, length) of both views [S] / int, and the overlap length crop_l."""

        def randint(low, high, size=()):
            return (
                torch.randint(low, high, size, generator=generator)
                if size
                else int(torch.randint(low, high, (1,), generator=generator))
            )

        crop_l = randint(min(self.config.crop_min_patches, P), P + 1)
        crop_left = randint(0, P - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = randint(0, crop_left + 1)
        crop_eright = randint(crop_right, P + 1)
        offset = randint(-crop_eleft, P - crop_eright + 1, (S,))
        return (offset + crop_eleft, crop_right - crop_eleft), (offset + crop_left, crop_eright - crop_left), crop_l

    def forward(
        self,
        context: torch.Tensor,
        future: torch.Tensor | None = None,
        symbol_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        compute_loss: bool | None = None,
        **kwargs,
    ) -> QTSMTs2VecOutput:
        """
        Args:
            context: [B, N, lookback, F] (N = 1 for ``series_collate_fn``).
            future: unused by the loss; with ``compute_loss`` left at ``None`` it turns the loss on, as for
                ``QTSMFJepaModel``. Without a loss only the states are returned.
            symbol_mask: [B, N] real stocks; all ``True`` if omitted.

        Crops and masks come from the global RNG in training and from a fixed seed in eval, so the validation loss
        is comparable across steps.
        """
        config = self.config
        B, N = context.shape[:2]
        mask = torch.ones(B, N, dtype=torch.bool, device=context.device) if symbol_mask is None else symbol_mask.bool()
        compute_loss = future is not None if compute_loss is None else compute_loss
        if not compute_loss:
            return QTSMTs2VecOutput(stock_states=self.encode(context, mask)[0])

        patches = self.encoder.patchify(context[mask])  # [S, P, patch]
        S, P, _ = patches.shape
        generator = None if self.training else torch.Generator().manual_seed(0)
        rows = torch.arange(S, device=patches.device)[:, None]
        *crops, crop_l = self._crops(P, S, generator)
        views = []
        for start, length in crops:
            idx = start.to(patches.device)[:, None] + torch.arange(length, device=patches.device)
            keep = torch.rand(S, length, generator=generator).to(patches.device) >= config.mask_prob
            views.append(self.encoder(patches[rows, idx], keep))
        # the overlap [left, right) is the tail of view 1 and the head of view 2
        z1, z2 = views[0][:, -crop_l:].float(), views[1][:, :crop_l].float()
        loss, instance, temporal = hierarchical_contrastive_loss(z1, z2, config.alpha, config.temporal_unit)

        metrics = {
            "loss/instance": instance.detach(),
            "loss/temporal": temporal.detach(),
            "crop/overlap": z1.new_tensor(float(crop_l)),
        }
        if return_diagnostics:
            with torch.no_grad():
                h = self.encoder(patches)
            states = {"max_state": h.amax(dim=1), "mean_state": h.mean(dim=1), "last_state": h[:, -1]}
            for name, rows_ in states.items():
                erank, std = _spectrum(rows_)
                metrics[f"erank/{name}"] = erank
                metrics[f"std/{name}"] = std
        return QTSMTs2VecOutput(loss=loss, metrics=metrics)


AutoConfig.register(QTSMTs2VecConfig.model_type, QTSMTs2VecConfig)
AutoModel.register(QTSMTs2VecConfig, QTSMTs2VecModel)

__all__ = ["QTSMTs2VecModel", "QTSMTs2VecPreTrainedModel", "QTSMTs2VecOutput"]
