"""CPC baseline for the series JEPA (configuration_qtsmcpc.py).

Per (stock, decision time) window i and horizon h, with the encoder E and predictor P of ``qtsmfjepa``:

    z_i      = E(context_i)                                   CLS state of the context
    p_{i,h}  = P(z_i, e_h)                                    horizon-conditioned prediction
    u_{i,h}  = E'(future_i[h - 1 : h - 1 + Delta])            E' = E with gradient (target="online", CPC)
                                                              or a stop-gradient EMA copy (target="ema")
    loss     = InfoNCE of cos(p_{i,h}, u_{i,h}) / tau against the other targets u_{j,h'} of the micro-batch

This is the JEPA objective with the EMA-target MSE replaced by a contrastive one: the context, the future blocks
(z-scored by the context's stats, ``target_window="block"``, ``target_norm="context"``), the encoder, the readout and
the predictor are the JEPA's, so ``target="online"`` against ``qtsmfjepa`` isolates what the EMA target without
negatives buys, and ``target="ema"`` (an EMA target with negatives) separates the two changes. With
``negatives="all"`` the candidates include window i's other horizons, CPC's same-sequence negatives. The module mirrors
``QTSMFJepaModel``'s interface (``encode``, ``forward(context, future, symbol_mask)``, ``update_target_encoder``,
``loss`` and ``metrics`` outputs), so ``panel_jepa_trainer`` and ``tools.eval_jepa_probe`` run it unchanged; module
names match the JEPA's (``encoder.*``, ``target_encoder.*``), so ``load_encoders`` works across the two.
"""

import copy
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from mc_jepa.models.qtsmcpc.configuration_qtsmcpc import QTSMCpcConfig


class QTSMCpcMLPBlock(nn.Module):
    """Pre-LN residual MLP block."""

    def __init__(self, config: QTSMCpcConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.fc2(self.act(self.fc1(self.norm(x)))))


class QTSMCpcEncoderLayer(GradientCheckpointingLayer):
    """Pre-LN bidirectional transformer layer."""

    def __init__(self, config: QTSMCpcConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlp = QTSMCpcMLPBlock(config)

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


class QTSMCpcStockEncoder(nn.Module):
    """E: [S, L, F] bars -> [S, D] CLS (or mean patch) state, the JEPA's encoder without scale or daily tokens."""

    def __init__(self, config: QTSMCpcConfig):
        super().__init__()
        self.patch_len = config.patch_len
        self.max_patches = config.max_patches
        self.input_clip = config.input_clip
        self.patch_embed = nn.Linear(config.patch_len * config.in_features, config.hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.max_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([QTSMCpcEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x: torch.Tensor, readout: str = "cls") -> torch.Tensor:
        """``readout``: ``"cls"`` returns the CLS state, ``"mean"`` the mean patch state."""
        S, L, F_ = x.shape
        assert L % self.patch_len == 0, f"sequence length {L} is not a multiple of patch_len {self.patch_len}"
        num_patches = L // self.patch_len
        assert num_patches <= self.max_patches, f"{num_patches} patches exceed max_patches {self.max_patches}"

        clip = self.input_clip
        x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)
        tokens = self.patch_embed(x.reshape(S, num_patches, self.patch_len * F_).to(self.patch_embed.weight.dtype))
        h = torch.cat([self.cls_token.expand(S, -1, -1), tokens], dim=1)
        h = self.dropout(h + self.position_embeddings[:, : num_patches + 1])
        for layer in self.layers:
            h = layer(h)
        if readout == "mean":
            return self.norm(h[:, 1:]).mean(dim=1)
        return self.norm(h[:, 0])


class QTSMCpcPredictor(nn.Module):
    """Residual MLP over the concatenation of its inputs."""

    def __init__(self, config: QTSMCpcConfig, num_inputs: int):
        super().__init__()
        self.inp = nn.Linear(num_inputs * config.hidden_size, config.hidden_size)
        self.blocks = nn.ModuleList([QTSMCpcMLPBlock(config) for _ in range(config.predictor_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x = self.inp(torch.cat(inputs, dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.out(self.norm(x))


@dataclass
class QTSMCpcOutput(ModelOutput):
    """
    loss: InfoNCE loss, averaged over horizons (only when ``future`` is given).
    stock_states: [B, N, D] context states z_i (zero at padding).
    market_state, peer_states: always ``None``; kept so callers of ``QTSMFJepaModel`` work unchanged.
    metrics: detached scalar losses, retrieval accuracies and collapse diagnostics.
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


def info_nce(
    pred: torch.Tensor, target: torch.Tensor, temperature: float, negatives: str = "all"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """InfoNCE of predictions pred [S, H, D] against their targets [S, H, D] on cosine similarity over temperature.

    ``negatives="all"``: each (i, h) against all S * H targets; ``"horizon"``: against the S targets of horizon h.
    Returns the loss [H], the top-1 accuracy [H] and, with ``"all"``, the share of predictions whose top candidate is
    another horizon of their own window (NaN with ``"horizon"``).
    """
    S, H, D = pred.shape
    q, k = F.normalize(pred.float(), dim=-1), F.normalize(target.float(), dim=-1)
    if negatives == "all":
        logits = q.reshape(S * H, D) @ k.reshape(S * H, D).T / temperature  # [SH, SH]
        labels = torch.arange(S * H, device=pred.device)
        loss = F.cross_entropy(logits, labels, reduction="none").view(S, H).mean(dim=0)
        top = logits.argmax(dim=-1)
        acc = (top == labels).float().view(S, H).mean(dim=0)
        same_window = ((top // H == labels // H) & (top != labels)).float().mean()
    else:
        logits = torch.einsum("shd,thd->hst", q, k) / temperature  # [H, S, S]
        labels = torch.arange(S, device=pred.device).expand(H, S)
        loss = F.cross_entropy(logits.reshape(H * S, S), labels.reshape(-1), reduction="none").view(H, S).mean(dim=1)
        acc = (logits.argmax(dim=-1) == labels).float().mean(dim=1)
        same_window = pred.new_tensor(float("nan"))
    return loss, acc.detach(), same_window.detach()


class QTSMCpcPreTrainedModel(PreTrainedModel):
    config: QTSMCpcConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["QTSMCpcEncoderLayer"]

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
        elif isinstance(module, QTSMCpcStockEncoder):
            module.cls_token.data.normal_(mean=0.0, std=std)
            module.position_embeddings.data.normal_(mean=0.0, std=std)


class QTSMCpcModel(QTSMCpcPreTrainedModel):
    def __init__(self, config: QTSMCpcConfig):
        super().__init__(config)
        self.encoder = QTSMCpcStockEncoder(config)
        self.target_encoder = None  # the online encoder with target="online"; the trainer then skips the EMA step
        if config.target == "ema":
            self.target_encoder = copy.deepcopy(self.encoder)
            self.target_encoder.requires_grad_(False)
        self.horizon_embeddings = nn.Embedding(len(config.horizons), config.hidden_size)
        self.stock_predictor = QTSMCpcPredictor(config, num_inputs=2)
        self.peer_pooler = None  # no panel context; tools.eval_jepa_probe checks this attribute
        self.post_init()
        if self.target_encoder is not None:
            self.target_encoder.load_state_dict(self.encoder.state_dict())

    @torch.no_grad()
    def update_target_encoder(self, momentum: float):
        """EMA step of the target encoder towards the online encoder (``target="ema"``; a no-op otherwise)."""
        if self.target_encoder is None:
            return
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=True):
            target.lerp_(online.detach().to(target.dtype), 1.0 - momentum)

    def encode(self, context: torch.Tensor, symbol_mask: torch.Tensor, readout: str | None = None, **kwargs):
        """Frozen features for transfer: (stock_states [B, N, D], None, None), like ``QTSMFJepaModel.encode``."""
        mask = symbol_mask.bool()
        states = self.encoder(context[mask], readout or self.config.readout)
        z = states.new_zeros(*mask.shape, states.shape[-1])
        z[mask] = states
        return z, None, None

    def _future_blocks(self, future: torch.Tensor) -> torch.Tensor:
        """future [S, lookahead, F] -> [S, H, Delta, F]; future[:, k] is bar t + 1 + k."""
        delta = self.config.target_block_len
        assert future.shape[1] >= self.config.lookahead, (
            f"future has {future.shape[1]} bars, horizons {self.config.horizons} with block length {delta} "
            f"need lookahead >= {self.config.lookahead}"
        )
        return torch.stack([future[:, h - 1 : h - 1 + delta] for h in self.config.horizons], dim=1)

    def forward(
        self,
        context: torch.Tensor,
        future: torch.Tensor | None = None,
        symbol_mask: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        **kwargs,
    ) -> QTSMCpcOutput:
        """
        Args:
            context: [B, N, lookback, F] (N = 1 for ``series_collate_fn``).
            future: [B, N, lookahead, F]; without it only the states are returned.
            symbol_mask: [B, N] real stocks; all ``True`` if omitted. Every real stock of the micro-batch is one
                window; negatives are drawn across panels and stocks alike.
        """
        config = self.config
        B, N = context.shape[:2]
        mask = torch.ones(B, N, dtype=torch.bool, device=context.device) if symbol_mask is None else symbol_mask.bool()
        z_all = self.encode(context, mask)[0]  # [B, N, D]
        if future is None:
            return QTSMCpcOutput(stock_states=z_all)

        z = z_all[mask]  # [S, D]
        S, H = z.shape[0], len(config.horizons)
        blocks = self._future_blocks(future[mask])  # [S, H, Delta, F]
        flat = blocks.reshape(S * H, *blocks.shape[2:])
        if self.target_encoder is None:
            target = self.encoder(flat, config.readout)
        else:
            with torch.no_grad():
                self.target_encoder.eval()
                target = self.target_encoder(flat, config.readout)
        target = target.view(S, H, -1)  # [S, H, D]
        horizon = self.horizon_embeddings.weight[None].expand(S, -1, -1)  # [S, H, D]
        pred = self.stock_predictor(z[:, None].expand(S, H, -1), horizon)  # [S, H, D]

        with torch.autocast(context.device.type, enabled=False):
            loss_h, acc, same_window = info_nce(pred, target, config.temperature, config.negatives)
        loss = loss_h.mean()
        candidates = S * H if config.negatives == "all" else S
        metrics = {
            "loss/infonce": loss.detach(),
            "cpc/acc": acc.mean(),
            "cpc/chance": loss.new_tensor(1.0 / candidates),
            "cpc/pos_cos": F.cosine_similarity(pred.float(), target.float(), dim=-1).mean().detach(),
        }
        metrics.update({f"loss/infonce_h{h}": v.detach() for h, v in zip(config.horizons, loss_h, strict=True)})
        metrics.update({f"cpc/acc_h{h}": v for h, v in zip(config.horizons, acc, strict=True)})
        if config.negatives == "all":
            metrics["cpc/same_window_err"] = same_window
        if return_diagnostics:
            for name, x in {"stock_state": z, "target": target, "pred_stock": pred}.items():
                erank, std = _spectrum(x)
                metrics[f"erank/{name}"] = erank
                metrics[f"std/{name}"] = std
        return QTSMCpcOutput(loss=loss, stock_states=z_all, metrics=metrics)


AutoConfig.register(QTSMCpcConfig.model_type, QTSMCpcConfig)
AutoModel.register(QTSMCpcConfig, QTSMCpcModel)

__all__ = ["QTSMCpcModel", "QTSMCpcPreTrainedModel", "QTSMCpcOutput"]
