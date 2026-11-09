"""Reconstruction baselines for the series JEPA (configuration_qtsmmae.py).

Per (stock, decision time) window, with the same encoder E as ``qtsmfjepa``:

    future:  z = E(context)_cls,               P(z, h) ~ future[h - 1 : h - 1 + Delta]     raw bars, per horizon
    masked:  tokens = E(context, patch_mask),   W tokens_p ~ context patch p                for masked patches p

Bars are z-scored by the context statistics in the dataset. The masked loss then reads as ~(1 - R^2) against
predicting zero; future blocks drift away from the context, so their variance (and the future loss at init) exceeds 1,
growing with the horizon.
The module mirrors ``QTSMFJepaModel``'s interface (``encode``, ``forward(context, future, symbol_mask)``, ``loss`` and
``metrics`` outputs), so ``panel_jepa_trainer`` and ``tools.eval_jepa_probe`` run it unchanged.
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

from mc_jepa.models.qtsmmae.configuration_qtsmmae import QTSMMaeConfig


class QTSMMaeMLPBlock(nn.Module):
    """Pre-LN residual MLP block."""

    def __init__(self, config: QTSMMaeConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.fc2(self.act(self.fc1(self.norm(x)))))


class QTSMMaeEncoderLayer(GradientCheckpointingLayer):
    """Pre-LN bidirectional transformer layer."""

    def __init__(self, config: QTSMMaeConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlp = QTSMMaeMLPBlock(config)

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


class QTSMMaeStockEncoder(nn.Module):
    """E: [S, L, F] bars -> [S, 1 + L / patch_len, D] normed CLS + patch states.

    With ``patch_mask`` [S, P], masked patch embeddings are replaced by the learned mask token before the position
    embeddings are added (SimMIM). The mask token exists only for the ``masked`` objective, since DDP rejects
    parameters without gradients.
    """

    def __init__(self, config: QTSMMaeConfig):
        super().__init__()
        self.patch_len = config.patch_len
        self.max_patches = config.max_patches
        self.input_clip = config.input_clip
        self.patch_embed = nn.Linear(config.patch_len * config.in_features, config.hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.mask_token = None
        if config.objective == "masked":
            self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.max_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([QTSMMaeEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor, patch_mask: torch.Tensor | None = None) -> torch.Tensor:
        S, L, F_ = x.shape
        assert L % self.patch_len == 0, f"sequence length {L} is not a multiple of patch_len {self.patch_len}"
        num_patches = L // self.patch_len
        assert num_patches <= self.max_patches, f"{num_patches} patches exceed max_patches {self.max_patches}"

        clip = self.input_clip
        x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)
        tokens = self.patch_embed(x.reshape(S, num_patches, self.patch_len * F_).to(self.patch_embed.weight.dtype))
        if patch_mask is not None:
            tokens = torch.where(patch_mask[..., None], self.mask_token.to(tokens.dtype), tokens)
        h = torch.cat([self.cls_token.expand(S, -1, -1), tokens], dim=1)
        h = self.dropout(h + self.position_embeddings[:, : num_patches + 1])
        for layer in self.layers:
            h = layer(h)
        return self.norm(h)


class QTSMMaePredictor(nn.Module):
    """Residual MLP over the concatenation of its inputs, projected to ``out_features``."""

    def __init__(self, config: QTSMMaeConfig, num_inputs: int, out_features: int):
        super().__init__()
        self.inp = nn.Linear(num_inputs * config.hidden_size, config.hidden_size)
        self.blocks = nn.ModuleList([QTSMMaeMLPBlock(config) for _ in range(config.predictor_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out = nn.Linear(config.hidden_size, out_features)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x = self.inp(torch.cat(inputs, dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))


@dataclass
class QTSMMaeOutput(ModelOutput):
    """
    loss: reconstruction loss (only when computed, see ``forward``).
    stock_states: [B, N, D] readout states (zero at padding); for ``masked`` training passes, of the masked input.
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


class QTSMMaePreTrainedModel(PreTrainedModel):
    config: QTSMMaeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["QTSMMaeEncoderLayer"]

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
        elif isinstance(module, QTSMMaeStockEncoder):
            module.cls_token.data.normal_(mean=0.0, std=std)
            if module.mask_token is not None:
                module.mask_token.data.normal_(mean=0.0, std=std)
            module.position_embeddings.data.normal_(mean=0.0, std=std)


class QTSMMaeModel(QTSMMaePreTrainedModel):
    def __init__(self, config: QTSMMaeConfig):
        super().__init__(config)
        D, F_ = config.hidden_size, config.in_features
        self.encoder = QTSMMaeStockEncoder(config)
        self.peer_pooler = None  # no panel context; tools.eval_jepa_probe checks this attribute
        if config.objective == "future":
            self.horizon_embeddings = nn.Embedding(len(config.horizons), D)
            self.future_predictor = QTSMMaePredictor(config, num_inputs=2, out_features=config.target_block_len * F_)
        else:
            self.recon_head = nn.Linear(D, config.patch_len * F_)
        self.post_init()

    def _readout(self, h: torch.Tensor, readout: str | None = None) -> torch.Tensor:
        """h [S, 1 + P, D] -> [S, D]."""
        readout = readout or self.config.readout
        return h[:, 0] if readout == "cls" else h[:, 1:].mean(dim=1)

    def encode(self, context: torch.Tensor, symbol_mask: torch.Tensor, readout: str | None = None, **kwargs):
        """Frozen features for transfer: (stock_states [B, N, D], None, None), like ``QTSMFJepaModel.encode``."""
        mask = symbol_mask.bool()
        states = self._readout(self.encoder(context[mask]), readout)
        z = states.new_zeros(*mask.shape, states.shape[-1])
        z[mask] = states
        return z, None, None

    def _patch_mask(self, S: int, num_patches: int, device: torch.device) -> torch.Tensor:
        """[S, P] boolean, exactly ``round(mask_ratio * P / span)`` spans of ``mask_span`` patches masked per row."""
        span = self.config.mask_span
        assert num_patches % span == 0, f"{num_patches} patches are not a multiple of mask_span {span}"
        num_spans = num_patches // span
        k = min(max(1, round(self.config.mask_ratio * num_spans)), num_spans - 1)
        ids = torch.rand(S, num_spans, device=device).argsort(dim=1)[:, :k]
        mask = torch.zeros(S, num_spans, dtype=torch.bool, device=device).scatter_(1, ids, True)
        return mask.repeat_interleave(span, dim=1)

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
    ) -> QTSMMaeOutput:
        """
        Args:
            context: [B, N, lookback, F] (N = 1 for ``series_collate_fn``).
            future: [B, N, lookahead, F]; the ``future`` objective needs it.
            symbol_mask: [B, N] real stocks; all ``True`` if omitted.
            compute_loss: defaults to ``future is not None``, as for ``QTSMFJepaModel``; without a loss only the
                (unmasked) states are returned.

        Shapes: S real stocks over the batch, P = lookback / patch_len patches, H horizons, D hidden size.
        """
        config = self.config
        B, N = context.shape[:2]
        mask = torch.ones(B, N, dtype=torch.bool, device=context.device) if symbol_mask is None else symbol_mask.bool()
        compute_loss = future is not None if compute_loss is None else compute_loss
        if not compute_loss:
            return QTSMMaeOutput(stock_states=self.encode(context, mask)[0])

        x = context[mask]  # [S, L, F]
        S, L, F_ = x.shape
        num_patches = L // config.patch_len
        metrics = {}
        if config.objective == "future":
            assert future is not None and future.shape[2] >= config.lookahead, (
                f"the future objective needs lookahead >= {config.lookahead} future bars"
            )
            h = self.encoder(x)  # [S, 1 + P, D]
            z = h[:, 0]  # [S, D]; the CLS state, as in the JEPA
            fut = future[mask]  # [S, lookahead, F]
            delta = config.target_block_len
            target = torch.stack([fut[:, k - 1 : k - 1 + delta] for k in config.horizons], dim=1)  # [S, H, Delta, F]
            H = target.shape[1]
            horizon = self.horizon_embeddings.weight  # [H, D]
            pred = self.future_predictor(z[:, None].expand(S, H, -1), horizon[None].expand(S, -1, -1))
            per_horizon = self._elementwise_loss(pred, target.reshape(S, H, delta * F_)).mean(dim=(0, 2))  # [H]
            loss = per_horizon.mean()  # horizons weighted equally
            metrics.update(
                {f"loss/future_h{k}": v.detach() for k, v in zip(config.horizons, per_horizon, strict=False)}
            )
            pred_rows = pred.reshape(-1, pred.shape[-1])
        else:
            patch_mask = self._patch_mask(S, num_patches, x.device)  # [S, P]
            h = self.encoder(x, patch_mask)  # [S, 1 + P, D]
            pred = self.recon_head(h[:, 1:])  # [S, P, patch_len * F]
            target = x.reshape(S, num_patches, config.patch_len * F_)
            per_patch = self._elementwise_loss(pred, target).mean(dim=-1)  # [S, P]
            weights = patch_mask.float()
            loss = (per_patch * weights).sum() / weights.sum().clamp_min(1.0)
            metrics["loss/visible"] = ((per_patch * (1 - weights)).sum() / (1 - weights).sum().clamp_min(1.0)).detach()
            pred_rows = pred[patch_mask]

        metrics["loss/recon"] = loss.detach()
        if return_diagnostics:
            for name, rows in {"stock_state": h[:, 0], "mean_state": h[:, 1:].mean(dim=1), "pred": pred_rows}.items():
                erank, std = _spectrum(rows)
                metrics[f"erank/{name}"] = erank
                metrics[f"std/{name}"] = std

        z = h.new_zeros(B, N, h.shape[-1])
        z[mask] = self._readout(h)
        return QTSMMaeOutput(loss=loss, stock_states=z, metrics=metrics)


AutoConfig.register(QTSMMaeConfig.model_type, QTSMMaeConfig)
AutoModel.register(QTSMMaeConfig, QTSMMaeModel)

__all__ = ["QTSMMaeModel", "QTSMMaePreTrainedModel", "QTSMMaeOutput"]
