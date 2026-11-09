"""Autoregressive next-patch baseline for the series JEPA (configuration_qtsmar.py).

Per (stock, decision time) window, with the encoder E of ``qtsmfjepa`` made causal and without a CLS token, trained
like a decoder-only LM with teacher forcing (one forward pass scores every prefix):

    x_1 .. x_P = context patches,  x_{P+1} .. = future patches (``future_target``)
    h_p = E(x_1 .. x_p),           W h_p ~ (x_{p+1}, .., x_{p+K})      raw bars, K = pred_patches

Bars are z-scored by the context statistics in the dataset (so early patches see a little of the later context's
mean / std; the future targets do not). The module mirrors ``QTSMFJepaModel``'s interface (``encode``,
``forward(context, future, symbol_mask)``, ``loss`` and ``metrics`` outputs), so ``panel_jepa_trainer`` and
``tools.eval_jepa_probe`` run it unchanged.
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

from mc_jepa.models.qtsmar.configuration_qtsmar import QTSMArConfig


class QTSMArMLPBlock(nn.Module):
    """Pre-LN residual MLP block."""

    def __init__(self, config: QTSMArConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.fc2(self.act(self.fc1(self.norm(x)))))


class QTSMArDecoderLayer(GradientCheckpointingLayer):
    """Pre-LN causal transformer layer."""

    def __init__(self, config: QTSMArConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlp = QTSMArMLPBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S, T, D = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(S, T, 3, self.num_heads, D // self.num_heads).unbind(2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=self.attn_dropout if self.training else 0.0,
            is_causal=True,
        )
        x = x + self.dropout(self.proj(attn.transpose(1, 2).reshape(S, T, D)))
        return self.mlp(x)


class QTSMArStockEncoder(nn.Module):
    """E: [S, L, F] bars -> [S, L / patch_len, D] normed causal patch states."""

    def __init__(self, config: QTSMArConfig):
        super().__init__()
        self.patch_len = config.patch_len
        self.max_patches = config.max_patches
        self.input_clip = config.input_clip
        self.patch_embed = nn.Linear(config.patch_len * config.in_features, config.hidden_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.max_patches, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([QTSMArDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        S, L, F_ = x.shape
        assert L % self.patch_len == 0, f"sequence length {L} is not a multiple of patch_len {self.patch_len}"
        num_patches = L // self.patch_len
        assert num_patches <= self.max_patches, f"{num_patches} patches exceed max_patches {self.max_patches}"

        clip = self.input_clip
        x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)
        h = self.patch_embed(x.reshape(S, num_patches, self.patch_len * F_).to(self.patch_embed.weight.dtype))
        h = self.dropout(h + self.position_embeddings[:, :num_patches])
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


@dataclass
class QTSMArOutput(ModelOutput):
    """
    loss: next-patch loss (only when computed, see ``forward``).
    stock_states: [B, N, D] readout states (zero at padding).
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


class QTSMArPreTrainedModel(PreTrainedModel):
    config: QTSMArConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["QTSMArDecoderLayer"]

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
        elif isinstance(module, QTSMArStockEncoder):
            module.position_embeddings.data.normal_(mean=0.0, std=std)


class QTSMArModel(QTSMArPreTrainedModel):
    def __init__(self, config: QTSMArConfig):
        super().__init__(config)
        self.encoder = QTSMArStockEncoder(config)
        self.peer_pooler = None  # no panel context; tools.eval_jepa_probe checks this attribute
        self.head = nn.Linear(config.hidden_size, config.pred_patches * config.patch_len * config.in_features)
        self.post_init()

    def _readout(self, h: torch.Tensor, readout: str | None = None) -> torch.Tensor:
        """h [S, P, D] -> [S, D]."""
        readout = readout or self.config.readout
        return h[:, -1] if readout == "last" else h.mean(dim=1)

    def encode(self, context: torch.Tensor, symbol_mask: torch.Tensor, readout: str | None = None, **kwargs):
        """Frozen features for transfer: (stock_states [B, N, D], None, None), like ``QTSMFJepaModel.encode``."""
        mask = symbol_mask.bool()
        states = self._readout(self.encoder(context[mask]), readout)
        z = states.new_zeros(*mask.shape, states.shape[-1])
        z[mask] = states
        return z, None, None

    def _elementwise_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        clip = self.config.target_clip
        target = torch.nan_to_num(target.float(), nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)
        if self.config.loss == "huber":
            return F.huber_loss(pred.float(), target, reduction="none", delta=self.config.huber_delta)
        return (pred.float() - target).square()

    def forward(
        self,
        context: torch.Tensor,
        future: torch.Tensor | None = None,
        symbol_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        compute_loss: bool | None = None,
        **kwargs,
    ) -> QTSMArOutput:
        """
        Args:
            context: [B, N, lookback, F] (N = 1 for ``series_collate_fn``).
            future: [B, N, lookahead, F]; ``future_target`` needs it.
            symbol_mask: [B, N] real stocks; all ``True`` if omitted.
            compute_loss: defaults to ``future is not None``, as for ``QTSMFJepaModel``; without a loss only the
                states are returned.

        Shapes: S real stocks over the batch, P = lookback / patch_len patches, K = pred_patches, D hidden size.
        """
        config = self.config
        B, N = context.shape[:2]
        mask = torch.ones(B, N, dtype=torch.bool, device=context.device) if symbol_mask is None else symbol_mask.bool()
        compute_loss = future is not None if compute_loss is None else compute_loss
        if not compute_loss:
            return QTSMArOutput(stock_states=self.encode(context, mask)[0])

        x = context[mask]  # [S, L, F]
        S, L, F_ = x.shape
        P, K, patch = L // config.patch_len, config.pred_patches, config.patch_len * F_
        h = self.encoder(x)  # [S, P, D]
        pred = self.head(h).view(S, P, K, patch)

        # targets: state p predicts patches p + 1 .. p + K of the context, continued into the future
        seq = x.reshape(S, P, patch)
        if config.future_target:
            assert future is not None and future.shape[2] >= K * config.patch_len, (
                f"future_target needs at least pred_patches * patch_len = {K * config.patch_len} future bars"
            )
            seq = torch.cat([seq, future[mask][:, : K * config.patch_len].reshape(S, K, patch)], dim=1)
        idx = torch.arange(1, P + 1, device=x.device)[:, None] + torch.arange(K, device=x.device)  # [P, K]
        valid = idx < seq.shape[1]  # [P, K]; all True with future_target
        target = seq[:, idx.clamp_max(seq.shape[1] - 1)]  # [S, P, K, patch]
        per_step = self._elementwise_loss(pred, target).mean(dim=-1)  # [S, P, K]
        weights = valid.float()[None].expand_as(per_step)
        loss = (per_step * weights).sum() / weights.sum().clamp_min(1.0)

        metrics = {"loss/recon": loss.detach()}
        if config.future_target:  # the decision-time state forecasting the first future patch
            metrics["loss/last"] = per_step[:, -1, 0].mean().detach()
        if K > 1:
            for k in range(K):
                w = weights[..., k]
                metrics[f"loss/step{k + 1}"] = ((per_step[..., k] * w).sum() / w.sum().clamp_min(1.0)).detach()
        if return_diagnostics:
            for name, rows in {"stock_state": h[:, -1], "mean_state": h.mean(dim=1), "pred": pred[:, -1]}.items():
                erank, std = _spectrum(rows)
                metrics[f"erank/{name}"] = erank
                metrics[f"std/{name}"] = std

        z = h.new_zeros(B, N, h.shape[-1])
        z[mask] = self._readout(h)
        return QTSMArOutput(loss=loss, stock_states=z, metrics=metrics)


AutoConfig.register(QTSMArConfig.model_type, QTSMArConfig)
AutoModel.register(QTSMArConfig, QTSMArModel)

__all__ = ["QTSMArModel", "QTSMArPreTrainedModel", "QTSMArOutput"]
