"""Factorized panel JEPA.

Per decision time t, with stocks i in the eligible panel U_t:

    z_i      = E(context_i)                                  BERT-like encoder over the stock's own history
    m, m_-i  = G({z_j : j in U_t}), G({z_j : j != i})        permutation-invariant peer pooler
    u+_{i,h} = sg(E_ema(future_i[h - 1 : h - 1 + Delta]))    EMA target encoder on a future block
    c+_h     = mean_i u+_{i,h},   r+_{i,h} = u+_{i,h} - c+_h

    P_market(m, h) ~ c+_h,        P_stock(z_i, m_-i, h) ~ r+_{i,h}

Market-conditioned variant (``target_mode="single"``, ``market_condition="loo_mean"``): no residual target; the stock
head predicts u+_{i,h} both without and with the realized leave-one-out market target c+_{-i,h} (stop-gradient):

    P_stock(z_i, 0, h) ~ u+_{i,h},    P_stock(z_i, W c+_{-i,h}, h) ~ u+_{i,h}

With ``condition_predictor="factor"`` the two passes are a(z_i, h) and a(z_i, h) + B(z_i, h) g(c+_{-i,h}), an explicit
factor model with stock-specific loadings B; ``factor_stop_base`` makes the second pass sg(a) + B g(c), and
``factor_loadings="linear"`` makes B linear in z_i. With ``condition_predictor="separate"`` they are two predictors that
share only z_i.

With u+_i = a_i + B_i f+ + e_i, the unconditional optimum a_i + B_i E[f+ | x_i] barely depends on the exposure B_i
when the market direction f+ is unpredictable; the conditional optimum a_i + B_i f+ does, so z_i has to encode it.

Learned hindsight latent (``market_condition="peer_attn"``): the condition is attention over stock i's peers, queries
from z_i, keys from the peers' EMA context states (their past), values their targets u+_{j,h} (their future), self
excluded; ``loo_mean`` is its uniform-attention special case. With ``condition_query="global"`` the queries are learned
vectors shared by every stock, so each head is a learned peer portfolio that z_i cannot steer.

Same-time contrastive loss (``contrastive_weight``): P_stock(z_i, 0, h) has to pick u+_{i,h} among the panel's targets
at the same decision time (InfoNCE on panel-centered targets), so the shared market move cannot help and only
relative, stock-specific information scores.

Lead-lag loss (``peer_predict_weight``): P_peer(z_k, z_i, h) ~ u+_{i,h} - sg(P_stock(z_i, 0, h)) for a random peer k of
the panel, so z_k has to carry what its history says about other stocks' futures beyond their own histories.

SIGReg (``sigreg_weight``): the Epps-Pulley distance of the context states z_i, on random 1-D projections, to an
isotropic Gaussian N(0, I) (LeJEPA), an explicit anti-collapse term next to the EMA target.

Panel mixer (``panel_mixer_layers > 0``): the stock head reads z~_i = M({z_j : j in U_t})_i, transformer layers of
stock-stock attention over the panel's context states, instead of z_i, so peers' histories can inform stock i's
future (relative moves, lead-lag). With ``freeze_encoder`` only M and the heads train on top of a fixed encoder.

Each target is standardized by its running mean and total variance before the MSE, so predicting zero scores ~1,
the loss reads as ~(1 - R^2), and neither the common nor the residual target dominates by scale.
"""

import copy
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.activations import ACT2FN
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput

from mc_jepa.models.qtsmfjepa.configuration_qtsmfjepa import QTSMFJepaConfig


def _no_autocast(device: torch.device):
    return torch.autocast(device_type=device.type, enabled=False) if device.type in ("cuda", "cpu") else nullcontext()


class QTSMFJepaMLPBlock(nn.Module):
    """Pre-LN residual MLP block."""

    def __init__(self, config: QTSMFJepaConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.act = ACT2FN[config.hidden_act]
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.fc2(self.act(self.fc1(self.norm(x)))))


class QTSMFJepaEncoderLayer(GradientCheckpointingLayer):
    """Pre-LN bidirectional transformer layer."""

    def __init__(self, config: QTSMFJepaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.attn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = config.attention_probs_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.mlp = QTSMFJepaMLPBlock(config)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        """x [S, T, D]; ``key_mask`` [S, T] marks the tokens every token may attend to (all if ``None``)."""
        S, T, D = x.shape
        q, k, v = self.qkv(self.attn_norm(x)).view(S, T, 3, self.num_heads, D // self.num_heads).unbind(2)
        attn = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=None if key_mask is None else key_mask[:, None, None, :],
            dropout_p=self.attn_dropout if self.training else 0.0,
        )
        x = x + self.dropout(self.proj(attn.transpose(1, 2).reshape(S, T, D)))
        return self.mlp(x)


def scale_features(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Scale token input from the raw per-feature mean and std [..., F] of a window: [..., 2F] (asinh mean, log std)."""
    mean, std = mean.float(), std.float()
    return torch.cat([torch.asinh(mean), std.clamp_min(1e-8).log()], dim=-1).nan_to_num(0.0)


class QTSMFJepaStockEncoder(nn.Module):
    """E: [S, L, F] bars -> [S, D] CLS (or mean patch) state. Bars are patched into L / patch_len tokens.

    With ``config.scale_token`` set, an optional scale [S, 2F] (``scale_features``) becomes one more token after the
    CLS token; it has no position embedding, its MLP output tells it apart. With ``config.daily_lookback``, optional
    daily bars [S, daily_lookback, daily_features] become daily_lookback / daily_patch_len more tokens, with their
    own position embeddings, between those and the intraday patches (the target encoder is called without them).
    """

    def __init__(self, config: QTSMFJepaConfig):
        super().__init__()
        self.patch_len = config.patch_len
        self.max_patches = config.max_patches
        self.input_clip = config.input_clip
        self.patch_embed = nn.Linear(config.patch_len * config.in_features, config.hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, config.max_patches + 1, config.hidden_size))
        self.scale_embed = None
        if config.scale_token:
            self.scale_embed = nn.Sequential(
                nn.Linear(2 * config.in_features, config.hidden_size),
                ACT2FN[config.hidden_act],
                nn.Linear(config.hidden_size, config.hidden_size),
            )
        self.daily_embed = self.daily_position_embeddings = None
        if config.daily_lookback > 0:
            self.daily_patch_len = config.daily_patch_len
            self.daily_embed = nn.Linear(config.daily_patch_len * config.daily_features, config.hidden_size)
            num_daily = config.daily_lookback // config.daily_patch_len
            self.daily_position_embeddings = nn.Parameter(torch.zeros(1, num_daily, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layers = nn.ModuleList([QTSMFJepaEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        readout: str = "cls",
        daily: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``readout``: ``"cls"`` returns the CLS state, ``"mean"`` the mean intraday patch state."""
        S, L, F_ = x.shape
        assert L % self.patch_len == 0, f"sequence length {L} is not a multiple of patch_len {self.patch_len}"
        num_patches = L // self.patch_len
        assert num_patches <= self.max_patches, f"{num_patches} patches exceed max_patches {self.max_patches}"

        clip = self.input_clip
        x = torch.nan_to_num(x, nan=0.0, posinf=clip, neginf=-clip).clamp(-clip, clip)
        tokens = self.patch_embed(x.reshape(S, num_patches, self.patch_len * F_).to(self.patch_embed.weight.dtype))
        h = torch.cat([self.cls_token.expand(S, -1, -1), tokens], dim=1)
        h = h + self.position_embeddings[:, : num_patches + 1]
        if scale is not None:
            assert self.scale_embed is not None, "a scale input needs config.scale_token"
            scale_token = self.scale_embed(scale.to(self.patch_embed.weight.dtype))[:, None]  # [S, 1, D]
            h = torch.cat([h[:, :1], scale_token, h[:, 1:]], dim=1)
        if daily is not None:
            assert self.daily_embed is not None, "a daily input needs config.daily_lookback"
            num_daily = daily.shape[1] // self.daily_patch_len
            daily = daily.reshape(S, num_daily, -1).to(self.daily_embed.weight.dtype)
            daily_tokens = self.daily_embed(daily) + self.daily_position_embeddings[:, :num_daily]
            k = h.shape[1] - num_patches  # CLS (and scale) tokens
            h = torch.cat([h[:, :k], daily_tokens, h[:, k:]], dim=1)
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h)
        if readout == "mean":
            return self.norm(h[:, -num_patches:]).mean(dim=1)
        return self.norm(h[:, 0])


class QTSMFJepaPeerPooler(nn.Module):
    """G: per-stock MLP, then multi-query attention pooling over the panel.

    There is deliberately no stock-stock attention: with pooling as the only interaction, the leave-one-out summary
    m_-i is exact and costs O(N) instead of N separate passes.
    """

    def __init__(self, config: QTSMFJepaConfig):
        super().__init__()
        D, K = config.hidden_size, config.num_market_tokens
        self.num_heads = config.num_attention_heads
        self.num_queries = K
        self.phi = nn.ModuleList([QTSMFJepaMLPBlock(config) for _ in range(config.panel_mlp_layers)])
        self.phi_norm = nn.LayerNorm(D, eps=config.layer_norm_eps)
        self.queries = nn.Parameter(torch.zeros(K, D))
        self.key = nn.Linear(D, D)
        self.value = nn.Linear(D, D)
        self.out = nn.Linear(K * D, D)
        self.out_norm = nn.LayerNorm(D, eps=config.layer_norm_eps)

    def forward(
        self, z: torch.Tensor, mask: torch.Tensor, leave_one_out: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """z [B, N, D], mask [B, N] -> market [B, D], peers [B, N, D] (zero for stocks without peers)."""
        B, N, D = z.shape
        K, H = self.num_queries, self.num_heads
        h = z
        for block in self.phi:
            h = block(h)
        h = self.phi_norm(h)
        k = self.key(h).view(B, N, H, D // H)
        v = self.value(h).view(B, N, H, D // H)

        with _no_autocast(z.device):
            k, v = k.float(), v.float()
            q = self.queries.float().view(K, H, D // H)
            scores = torch.einsum("khd,bnhd->bkhn", q, k) / math.sqrt(D // H)
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))
            shift = scores.amax(dim=-1, keepdim=True).nan_to_num(0.0, neginf=0.0)
            e = torch.exp(scores - shift)  # [B, K, H, N], 0 at padding
            den = e.sum(dim=-1)  # [B, K, H]
            num = torch.einsum("bkhn,bnhd->bkhd", e, v)
            market = (num / den[..., None].clamp_min(1e-30)).reshape(B, K * D)

            peers = None
            if leave_one_out:
                # float64: subtracting stock i's own term cancels badly when it carries most of the weight
                e64 = e.double()
                num_loo = num.double()[:, :, :, None] - e64[..., None] * v.double().transpose(1, 2)[:, None]
                den_loo = den.double()[..., None] - e64  # [B, K, H, N]
                peers = (num_loo / den_loo.clamp_min(1e-30)[..., None]).float()  # [B, K, H, N, D / H]
                peers = peers.permute(0, 3, 1, 2, 4).reshape(B, N, K * D)

        market = self.out_norm(self.out(market.to(z.dtype)))
        if peers is not None:
            has_peers = mask & (mask.sum(dim=1, keepdim=True) > 1)
            peers = self.out_norm(self.out(peers.to(z.dtype))) * has_peers[..., None]
        return market, peers


class QTSMFJepaPanelMixer(nn.Module):
    """M: transformer layers over the stocks of each panel, z [B, N, D] -> mixed states [B, N, D].

    Unlike the pooler, every stock attends to every other stock of its panel, so a mixed state can carry relative
    information such as a stock's move against its peers or lead-lag. Its inputs are context states only, so mixing
    never sees any stock's future.
    """

    def __init__(self, config: QTSMFJepaConfig):
        super().__init__()
        self.layers = nn.ModuleList([QTSMFJepaEncoderLayer(config) for _ in range(config.panel_mixer_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """z [B, N, D], mask [B, N] -> [B, N, D], zero at padding."""
        key_mask = mask | ~mask.any(dim=1, keepdim=True)  # an empty panel attends to its padding, not to nothing
        h = z
        for layer in self.layers:
            h = layer(h, key_mask)
        return self.norm(h) * mask[..., None]


class QTSMFJepaPredictor(nn.Module):
    """Residual MLP over the concatenation of its inputs; ``out_act`` applies ``hidden_act`` to the output, for a
    predictor that feeds another linear layer."""

    def __init__(self, config: QTSMFJepaConfig, num_inputs: int, out_act: bool = False):
        super().__init__()
        self.inp = nn.Linear(num_inputs * config.hidden_size, config.hidden_size)
        self.blocks = nn.ModuleList([QTSMFJepaMLPBlock(config) for _ in range(config.predictor_layers)])
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.out = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_act = ACT2FN[config.hidden_act] if out_act else nn.Identity()

    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        x = self.inp(torch.cat(inputs, dim=-1))
        for block in self.blocks:
            x = block(x)
        return self.out_act(self.out(self.norm(x)))


@dataclass
class QTSMFJepaOutput(ModelOutput):
    """
    loss: weighted sum of the normalized market and stock losses (only when ``future`` is given).
    stock_states: [B, N, D] context states z_i (zero at padding).
    mixed_states: [B, N, D] states after the panel mixer (``None`` without one).
    market_state: [B, D] panel summary m_t.
    peer_states: [B, N, D] leave-one-out summaries m_-i (``None`` if the model has no peer context).
    metrics: detached scalar losses and collapse diagnostics.
    """

    loss: torch.FloatTensor | None = None
    stock_states: torch.FloatTensor | None = None
    mixed_states: torch.FloatTensor | None = None
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
    cov = x.T @ x / (x.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(0)
    p = eig / eig.sum().clamp_min(1e-12)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-12))).sum()), cov.diagonal().sqrt().mean()


def sigreg(x: torch.Tensor, num_slices: int = 256, knots: int = 17, t_max: float = 3.0) -> torch.Tensor:
    """Sketched Isotropic Gaussian Regularization (LeJEPA, arXiv:2511.08544) of rows x [M, D]: the Epps-Pulley
    statistic of x projected on ``num_slices`` random unit directions against N(0, 1), averaged over directions.

    Each 1-D statistic is the integral of |phi_M(t) - exp(-t^2 / 2)|^2 exp(-t^2 / 2) over t, with phi_M the empirical
    characteristic function of the projection; the integrand is even, so it is twice the trapezoid rule on ``knots``
    points of [0, t_max]. It is 0 iff every projection is standard normal, i.e. x ~ N(0, I) (Cramer-Wold). Unlike the
    paper it is not multiplied by M, so its weight does not depend on the batch size.
    """
    x = x.float()
    directions = F.normalize(torch.randn(x.shape[-1], num_slices, device=x.device), dim=0)  # [D, slices]
    t = torch.linspace(0, t_max, knots, device=x.device)
    gauss = torch.exp(-0.5 * t.square())  # characteristic function of N(0, 1), also the weight
    xt = (x @ directions)[..., None] * t  # [M, slices, knots]
    err = (xt.cos().mean(dim=0) - gauss).square() + xt.sin().mean(dim=0).square()  # |phi_M(t) - phi(t)|^2
    return 2 * torch.trapezoid(err * gauss, t, dim=-1).mean()


class QTSMFJepaPreTrainedModel(PreTrainedModel):
    config: QTSMFJepaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["QTSMFJepaEncoderLayer"]

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
        elif isinstance(module, QTSMFJepaStockEncoder):
            module.cls_token.data.normal_(mean=0.0, std=std)
            module.position_embeddings.data.normal_(mean=0.0, std=std)
            if module.daily_position_embeddings is not None:
                module.daily_position_embeddings.data.normal_(mean=0.0, std=std)
        elif isinstance(module, QTSMFJepaPeerPooler):
            module.queries.data.normal_(mean=0.0, std=std)


class QTSMFJepaModel(QTSMFJepaPreTrainedModel):
    def __init__(self, config: QTSMFJepaConfig):
        super().__init__(config)
        D, H = config.hidden_size, len(config.horizons)
        self.encoder = QTSMFJepaStockEncoder(config)
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_encoder.requires_grad_(False)

        self.has_market_head = config.target_mode != "single"
        needs_pooler = config.use_peer_context or self.has_market_head
        self.peer_pooler = QTSMFJepaPeerPooler(config) if needs_pooler else None
        self.panel_mixer = QTSMFJepaPanelMixer(config) if config.panel_mixer_layers > 0 else None
        self.horizon_embeddings = nn.Embedding(H, D)
        self.condition_proj = None  # market condition bottleneck, with config.market_condition
        if config.market_condition != "none":
            assert not config.use_peer_context, "a market condition replaces the peer context"
            cond_in = D * config.condition_heads if config.market_condition == "peer_attn" else D
            if config.condition_predictor == "factor":  # g(c): the condition's factors
                self.condition_proj = nn.Linear(cond_in, config.condition_dim, bias=False)
            else:  # low-rank W c, nonlinear with config.stacked_linear_act
                act = [ACT2FN[config.hidden_act]] if config.stacked_linear_act else []
                self.condition_proj = nn.Sequential(
                    nn.Linear(cond_in, config.condition_dim), *act, nn.Linear(config.condition_dim, D)
                )
        self.condition_ramp = 1.0  # multiplier of config.condition_weight, set each step by the trainer's schedule
        self.factor_loadings = self.loadings_out = None  # B(z_i, h), with condition_predictor="factor"
        if self.condition_proj is not None and config.condition_predictor == "factor":
            if config.factor_loadings == "linear":  # one map W_h z_i + b_h per horizon
                self.loadings_out = nn.ModuleList(nn.Linear(D, D * config.condition_dim) for _ in range(H))
            else:
                self.factor_loadings = QTSMFJepaPredictor(
                    config, num_inputs=2 + int(config.use_peer_context), out_act=config.stacked_linear_act
                )
                self.loadings_out = nn.Linear(D, D * config.condition_dim)
        self.condition_query = self.condition_key = self.condition_sink = None  # peer attention of "peer_attn"
        if config.market_condition == "peer_attn":
            assert D % config.condition_heads == 0, "hidden_size must be a multiple of condition_heads"
            if config.condition_query == "global":  # one query per head; zero = loo_mean (plus the sink) at step 0
                self.condition_query = nn.Parameter(torch.zeros(config.condition_heads, D // config.condition_heads))
            else:
                self.condition_query = nn.Linear(D, D)
            self.condition_key = nn.Linear(D, D)
            self.condition_sink = nn.Parameter(torch.zeros(config.condition_heads))  # per-head sink logit
        shared_condition = self.condition_proj is not None and config.condition_predictor == "shared"
        num_stock_inputs = 2 + int(config.use_peer_context) + int(shared_condition)
        self.stock_predictor = QTSMFJepaPredictor(config, num_inputs=num_stock_inputs)
        self.cond_stock_predictor = None  # P_c(z_i, W c, h), with condition_predictor="separate"
        if self.condition_proj is not None and config.condition_predictor == "separate":
            self.cond_stock_predictor = QTSMFJepaPredictor(config, num_inputs=num_stock_inputs + 1)
        self.market_predictor = QTSMFJepaPredictor(config, num_inputs=2) if self.has_market_head else None
        # lead-lag head P_peer(z_partner, z_i, h), with config.peer_predict_weight
        self.peer_predictor = QTSMFJepaPredictor(config, num_inputs=3) if config.peer_predict_weight > 0 else None
        self.scale_predictor = self.scale_out = None  # supervised future-scale head, with config.scale_aux_weight
        if config.scale_aux_weight > 0:
            self.scale_predictor = QTSMFJepaPredictor(config, num_inputs=2, out_act=config.stacked_linear_act)
            self.scale_out = nn.Linear(D, 2 * config.in_features)

        # running first/second moments of each loss target, per horizon, to normalize the losses
        for name in ("market", "stock"):
            self.register_buffer(f"{name}_target_mean", torch.zeros(H, D, dtype=torch.float32))
            self.register_buffer(f"{name}_target_sq", torch.ones(H, D, dtype=torch.float32))
        if self.scale_predictor is not None:
            self.register_buffer("scale_target_mean", torch.zeros(H, 2 * config.in_features, dtype=torch.float32))
            self.register_buffer("scale_target_sq", torch.ones(H, 2 * config.in_features, dtype=torch.float32))
        self.register_buffer("target_stats_initialized", torch.zeros((), dtype=torch.bool))

        self.post_init()
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        if config.freeze_encoder:
            self.encoder.requires_grad_(False)

    @torch.no_grad()
    def update_target_encoder(self, momentum: float):
        """EMA step of the target encoder towards the online encoder; call after each optimizer step. A no-op with
        ``config.freeze_encoder``, which keeps the target encoder it was given."""
        if self.config.freeze_encoder:
            return
        for target, online in zip(self.target_encoder.parameters(), self.encoder.parameters(), strict=False):
            target.lerp_(online.detach().to(target.dtype), 1.0 - momentum)

    def _encode_panel(
        self,
        encoder: nn.Module,
        x: torch.Tensor,
        mask: torch.Tensor,
        scale: torch.Tensor | None = None,
        readout: str | None = None,
        daily: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode only the real stocks: x [B, N, *extra, T, F], scale [B, N, *extra, 2F] -> [B, N, *extra, D];
        daily [B, N, days, daily_features] only for contexts (no extra axes). ``readout`` defaults to
        ``config.readout``."""
        readout = readout or self.config.readout
        flat = x[mask]  # [S, *extra, T, F]
        extra = flat.shape[1:-2]
        flat_scale = None if scale is None else scale[mask].reshape(-1, scale.shape[-1])
        flat_daily = None if daily is None else daily[mask]
        states = encoder(flat.reshape(-1, *flat.shape[-2:]), flat_scale, readout, flat_daily)
        states = states.reshape(flat.shape[0], *extra, -1)
        out = states.new_zeros(*mask.shape, *states.shape[1:])
        out[mask] = states
        return out

    def _pool(self, z: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.peer_pooler is None:
            return None, None
        return self.peer_pooler(z, mask, leave_one_out=self.config.use_peer_context)

    def mix(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Panel-mixed states [B, N, D] of context states z [B, N, D]; z itself without a panel mixer."""
        return z if self.panel_mixer is None else self.panel_mixer(z, mask)

    def encode(
        self,
        context: torch.Tensor,
        symbol_mask: torch.Tensor,
        use_target_encoder: bool = False,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        readout: str | None = None,
        daily: torch.Tensor | None = None,
    ):
        """Frozen features for transfer: (stock_states [B, N, D], market_state [B, D], peer_states [B, N, D]).

        ``mean`` / ``std`` [B, N, F] are the dataset's per-window normalization statistics, required with
        ``config.scale_token`` (either encoder then embeds the context's scale token). ``readout`` overrides
        ``config.readout`` (the pooler is trained on states of ``config.readout``). Panel-mixed states
        come from ``mix``. ``daily`` [B, N, days, daily_features] is required with ``config.daily_lookback``.
        """
        encoder = self.target_encoder if use_target_encoder else self.encoder
        scale = None
        if self.config.scale_token:
            assert mean is not None and std is not None, "config.scale_token needs mean and std"
            scale = scale_features(mean, std)
        assert (daily is not None) == (self.config.daily_lookback > 0), "daily bars iff config.daily_lookback"
        with torch.no_grad() if self.config.freeze_encoder else nullcontext():
            z = self._encode_panel(encoder, context, symbol_mask, scale, readout, daily)
        market, peers = self._pool(z, symbol_mask)
        return z, market, peers

    @staticmethod
    def _block_scale(blocks: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Absolute scale of each future block [B, N, H, Delta, F], normalized by the context's mean / std
        [B, N, F]: ``scale_features`` [B, N, H, 2F] of the block's raw mean and std."""
        blocks = blocks.float()
        mean, std = mean.float()[:, :, None], std.float()[:, :, None]
        return scale_features(mean + std * blocks.mean(dim=-2), std * blocks.std(dim=-2, unbiased=False))

    @staticmethod
    def _self_normalize(blocks: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Blocks [B, N, H, Delta, F] z-scored by the context's std [B, N, F] -> z-scored by each block's own raw
        mean and std, matching the dataset's ``(x - mean) / (std + 1e-5)`` on raw values."""
        blocks, std = blocks.float(), std.float()[:, :, None, None]
        centered = blocks - blocks.mean(dim=-2, keepdim=True)
        return centered * std / (std * blocks.std(dim=-2, unbiased=False, keepdim=True) + 1e-5)

    def _future_blocks(self, future: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """future [B, N, lookahead, F] -> [B, N, H, Delta, F]; future[:, :, k] is bar t + 1 + k.

        With ``config.target_window="end"``, block h is bars t+h-Delta+1 .. t+h of context [B, N, lookback, F]
        followed by future (both z-scored by the context's stats).
        """
        delta = self.config.target_block_len
        assert future.shape[2] >= self.config.lookahead, (
            f"future has {future.shape[2]} bars, horizons {self.config.horizons} with block length {delta} "
            f"need lookahead >= {self.config.lookahead}"
        )
        if self.config.target_window == "end":
            lookback = context.shape[2]
            assert delta <= lookback + min(self.config.horizons), "end windows must start inside the context"
            seq = torch.cat([context, future.to(context.dtype)], dim=2)  # bar t+k at index lookback - 1 + k
            return torch.stack([seq[:, :, lookback + h - delta : lookback + h] for h in self.config.horizons], dim=2)
        return torch.stack([future[:, :, h - 1 : h - 1 + delta] for h in self.config.horizons], dim=2)

    @torch.no_grad()
    def _target_stats(self, name: str, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and variance [H, D] of rows target [M, H, D], from running moments (updated in training)."""
        stats = torch.stack([target.sum(dim=0), target.square().sum(dim=0)])  # [2, H, D]
        count = target.new_tensor(float(target.shape[0]))
        if self.training and dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            packed = torch.cat([stats.flatten(), count[None]])
            dist.all_reduce(packed)
            stats, count = packed[:-1].view_as(stats), packed[-1]
        mean, sq = stats / count.clamp_min(1.0)
        running_mean, running_sq = getattr(self, f"{name}_target_mean"), getattr(self, f"{name}_target_sq")
        if self.training:
            if self.target_stats_initialized:
                running_mean.lerp_(mean, 1.0 - self.config.loss_stats_momentum)
                running_sq.lerp_(sq, 1.0 - self.config.loss_stats_momentum)
            else:
                running_mean.copy_(mean)
                running_sq.copy_(sq)
        elif not self.target_stats_initialized:
            running_mean, running_sq = mean, sq
        return running_mean, (running_sq - running_mean.square()).clamp_min(0)

    def _normalized_mse(
        self, name: str, pred: torch.Tensor, target: torch.Tensor, stats: tuple | None = None
    ) -> torch.Tensor:
        """Per-horizon MSE of pred against the standardized target, for rows [M, H, D] -> [H].

        Predictions live in the standardized space: target minus its running mean, over its total std / sqrt(D).
        ``stats`` reuses the (mean, var) of an earlier ``_target_stats`` call instead of updating them again.
        """
        mean, var = self._target_stats(name, target) if stats is None else stats
        scale = (var.sum(dim=-1).clamp_min(1e-6) / target.shape[-1]).sqrt()[:, None]
        return (pred - (target - mean) / scale).square().mean(dim=-1).mean(dim=0)

    def _standardized_mse(self, name: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Like ``_normalized_mse``, but each target dimension over its own std (for targets in mixed units)."""
        mean, var = self._target_stats(name, target)
        return (pred - (target - mean) / var.clamp_min(1e-6).sqrt()).square().mean(dim=-1).mean(dim=0)

    def _context_subset(self, mask: torch.Tensor) -> torch.Tensor:
        """At most ``config.context_symbols`` random real stocks per panel [B, N] (all with 0; the first ones in
        eval, so validation is deterministic)."""
        k = self.config.context_symbols
        if k <= 0:
            return mask
        if self.training:
            score = torch.rand(mask.shape, device=mask.device)
        else:
            score = torch.linspace(0, 1, mask.shape[1], device=mask.device).expand(mask.shape)
        score = score.masked_fill(~mask, 2.0)
        rank = score.argsort(dim=1).argsort(dim=1)
        return mask & (rank < k)

    def _market_condition(
        self,
        u: torch.Tensor,
        u_valid: torch.Tensor,
        common: torch.Tensor,
        counts: torch.Tensor,
        batch_idx,
        source: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Leave-one-out mean target c+_{-i} of each scored stock [S, H, D], standardized by the running moments of
        the panel mean c+ (zero for stocks without peers, the same input as the unconditional pass). With ``source``
        [B] (``config.condition_placebo``), panel b gets the full mean c+ of panel ``source[b]`` instead."""
        n = counts[batch_idx]  # [S, 1, 1]
        loo = (u.sum(dim=1)[batch_idx] - u_valid) / (n - 1).clamp_min(1)
        if source is not None:  # placebo: the full mean c+ of another panel, which never contains stock i
            loo = common[source][batch_idx]
        mean, var = self._target_stats("market", common)
        scale = (var.sum(dim=-1).clamp_min(1e-6) / u.shape[-1]).sqrt()[:, None]
        return ((loo - mean) / scale) * (n > 1)

    @staticmethod
    def _placebo_source(timestamp: torch.Tensor | None, B: int, device: torch.device) -> tuple[torch.Tensor, float]:
        """Panel whose market target each panel gets under ``config.condition_placebo`` [B]: a random derangement of
        the micro-batch, redrawn up to 8 times for panels that drew their own date (timestamp YYYYMMDDHHMMSS). Also
        returns the share of panels left on their own date."""
        perm = torch.randperm(B, device=device)
        source = torch.empty_like(perm)
        source[perm] = perm.roll(1)  # a cycle through all panels: nobody gets its own target
        if timestamp is None or B < 2:
            return source, 0.0
        date = timestamp.to(device) // 1_000_000
        for _ in range(8):
            same = date[source] == date
            if not same.any():
                break
            source[same] = torch.randint(B, (int(same.sum()),), device=device)
        return source, (date[source] == date).float().mean().item()

    def _peer_attention(
        self, rows: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor, batch_idx, pos
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``peer_attn`` condition: scored states rows [S, D] attend over their panel's other stocks, keys [B, N, D]
        (EMA context states), values [B, N, H, D] (standardized targets), plus a sink slot per head with a learned
        logit and a zero value that absorbs the weight no peer deserves. With ``condition_query="global"`` the
        queries are the learned per-head vectors and rows only sets S. Returns [S, H, heads * D] (zero without
        peers), the mean entropy of the peer weights renormalized without the sink over log(peers) (1 = uniform,
        i.e. ``loo_mean``), and the mean sink weight."""
        S, D = rows.shape
        Hh = self.config.condition_heads
        if self.config.condition_query == "global":
            q = self.condition_query.to(rows.dtype)[None].expand(S, -1, -1)  # [S, Hh, dk]
        else:
            q = self.condition_query(rows).view(S, Hh, D // Hh)
        k = self.condition_key(keys.to(rows.dtype)).view(*keys.shape[:2], Hh, D // Hh)[batch_idx]  # [S, N, Hh, dk]
        with _no_autocast(rows.device):
            scores = torch.einsum("sad,snad->san", q.float(), k.float()) / math.sqrt(D // Hh)  # [S, Hh, N]
            allowed = mask[batch_idx].clone()  # [S, N]
            allowed[torch.arange(S, device=rows.device), pos] = False  # never its own future
            scores = scores.masked_fill(~allowed[:, None], float("-inf"))
            sink = self.condition_sink.float()[None, :, None].expand(S, -1, 1)  # [S, Hh, 1]
            weights = torch.cat([scores, sink], dim=-1).softmax(dim=-1)  # [S, Hh, N + 1]
            sink_weight = weights[..., -1]  # [S, Hh]
            weights = weights[..., :-1]
            out = torch.einsum("san,snhd->shad", weights, values.float()[batch_idx])  # [S, H, Hh, D]
            n_peers = allowed.sum(dim=1)  # [S]
            p = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            entropy = -(p * p.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=1)  # [S]
            has = n_peers > 1
            rel = (entropy[has] / n_peers[has].float().log()).mean() if has.any() else entropy.new_tensor(float("nan"))
            sink_mass = sink_weight[has].mean() if has.any() else entropy.new_tensor(float("nan"))
        return out.reshape(S, out.shape[1], Hh * D), rel.detach(), sink_mass.detach()

    def _contrastive_loss(
        self, pred, u, mask, stats, batch_idx, pos, scale: torch.Tensor | None
    ) -> tuple[torch.Tensor, dict]:
        """Same-time InfoNCE [H]: pred [S, H, D] (standardized space) against every target of its panel u
        [B, N, H, D], keys centered on the panel mean; ``scale`` [B, N, 2F] context scale for nearest-scale
        candidates (``contrastive_neighbors``)."""
        config = self.config
        mean, var = stats
        std_scale = (var.sum(dim=-1).clamp_min(1e-6) / u.shape[-1]).sqrt()[:, None]  # [H, 1]
        tgt = (u - mean) / std_scale * mask[..., None, None]  # [B, N, H, D]
        counts = mask.sum(dim=1).clamp_min(1)[:, None, None, None]
        keys = F.normalize(tgt - tgt.sum(dim=1, keepdim=True) / counts, dim=-1)
        logits = torch.einsum("shd,snhd->shn", F.normalize(pred, dim=-1), keys[batch_idx])
        logits = logits / config.contrastive_temperature  # [S, H, N]
        allowed = mask[batch_idx].clone()  # [S, N]
        K = config.contrastive_neighbors
        if K > 0 and scale is not None and K + 1 < mask.shape[1]:
            sf = scale.float()
            rows = sf[mask]
            sf = (sf - rows.mean(dim=0)) / rows.std(dim=0).nan_to_num(1.0).clamp_min(1e-6)  # per-feature standardized
            own = sf[batch_idx, pos]  # [S, 2F]
            dist2 = (sf[batch_idx] - own[:, None]).square().sum(dim=-1).masked_fill(~allowed, float("inf"))
            dist2[torch.arange(len(pos), device=pos.device), pos] = -1.0  # the positive is always a candidate
            keep = dist2.topk(K + 1, dim=1, largest=False).indices
            near = torch.zeros_like(allowed).scatter_(1, keep, True)
            allowed = allowed & near
        logits = logits.masked_fill(~allowed[:, None], float("-inf"))
        n_cand = allowed.sum(dim=1)  # [S]
        valid = (n_cand > 1).float()
        H = logits.shape[1]
        ce = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), pos.repeat_interleave(H), reduction="none")
        ce = ce.view(-1, H)  # [S, H]
        loss = (ce * valid[:, None]).sum(dim=0) / valid.sum().clamp_min(1.0)
        acc = (logits.argmax(dim=-1) == pos[:, None]).float()
        denom = valid.sum().clamp_min(1.0)
        metrics = {
            "contrastive/acc": ((acc * valid[:, None]).sum() / (denom * H)).detach(),
            "contrastive/chance": ((valid / n_cand.clamp_min(1)).sum() / denom).detach(),
        }
        return loss, metrics

    def _peer_loss(self, rows, pred, u_valid, stats, horizon, enc_mask, batch_idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Lead-lag loss [H] and explained share [H]: P_peer(z_k, z_i, h) predicts stock i's standardized target
        minus its own detached prediction pred [S, H, D], for one random partner k of the same panel (the next
        scored stock in eval)."""
        S, H = pred.shape[:2]
        mean, var = stats
        std_scale = (var.sum(dim=-1).clamp_min(1e-6) / u_valid.shape[-1]).sqrt()[:, None]
        target = (u_valid - mean) / std_scale - pred.detach()  # [S, H, D]
        n_b = enc_mask.sum(dim=1)  # [B]
        start = (n_b.cumsum(0) - n_b)[batch_idx]  # [S]; scored rows are grouped by panel
        n = n_b[batch_idx]
        r = torch.arange(S, device=pred.device) - start
        if self.training:
            shift = (torch.rand(S, device=pred.device) * (n - 1).clamp_min(1)).floor().long() + 1
        else:
            shift = torch.ones_like(r)
        partner = start + (r + shift) % n.clamp_min(1)
        valid = (n > 1).float()
        expand = lambda x: x[:, None].expand(S, H, -1)  # noqa: E731
        peer_pred = self.peer_predictor(expand(rows[partner]), expand(rows), horizon[None].expand(S, -1, -1))
        with _no_autocast(pred.device):
            err = (peer_pred.float() - target).square().mean(dim=-1)  # [S, H]
            denom = valid.sum().clamp_min(1.0)
            loss = (err * valid[:, None]).sum(dim=0) / denom
            base = (target.square().mean(dim=-1) * valid[:, None]).sum(dim=0) / denom
        return loss, (1 - loss / base.clamp_min(1e-6)).detach()

    def forward(
        self,
        context: torch.Tensor,
        future: torch.Tensor | None = None,
        symbol_mask: torch.Tensor | None = None,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        daily: torch.Tensor | None = None,
        timestamp: torch.Tensor | None = None,
        return_diagnostics: bool = False,
        **kwargs,
    ) -> QTSMFJepaOutput:
        """
        Args:
            context: [B, N, lookback, F] from ``CrossSectionPanelDataset.collate_fn``.
            future: [B, N, lookahead, F]; without it only the states are returned.
            symbol_mask: [B, N] real stocks; all ``True`` if omitted.
            mean, std: [B, N, F] per-window normalization statistics of the context; required with
                ``config.scale_token``.
            daily: [B, N, daily_lookback, daily_features] daily bars before the decision date, for the online
                encoder only; required with ``config.daily_lookback``.
            timestamp: [B] decision times (YYYYMMDDHHMMSS); ``config.condition_placebo`` avoids same-date sources.

        Shapes:
            B: panels (decision times) in the batch; in series mode each panel is one (stock, decision time) window.
            N: stocks per panel, padded to the largest panel (1 in series mode).
            S: real stocks over the batch, ``symbol_mask.sum()``; padded slots are dropped.
            H: number of horizons, ``len(config.horizons)``.
            D: ``config.hidden_size``.
            F: features per bar, ``config.in_features``.
        """
        config = self.config
        B, N = context.shape[:2]
        mask = torch.ones(B, N, dtype=torch.bool, device=context.device) if symbol_mask is None else symbol_mask.bool()

        # with config.context_symbols, only a subset is encoded and scored; all stocks enter the market condition
        enc_mask = mask if future is None else self._context_subset(mask)
        z, market, peers = self.encode(context, enc_mask, mean=mean, std=std, daily=daily)  # [B,N,D], [B,D], [B,N,D]
        mixed = self.mix(z, enc_mask)  # [B, N, D]; the state the stock predictor reads
        output = QTSMFJepaOutput(
            stock_states=z,
            mixed_states=mixed if self.panel_mixer is not None else None,
            market_state=market,
            peer_states=peers,
        )
        if future is None:
            return output

        with torch.no_grad():
            self.target_encoder.eval()
            blocks = self._future_blocks(future, context)  # [B, N, H, Delta, F]
            block_scale = None
            future_scale = None  # [S, H, 2F] regression target of the scale head
            if self.scale_predictor is not None:
                assert mean is not None and std is not None, "config.scale_aux_weight needs mean and std"
                future_scale = self._block_scale(blocks, mean, std)[mask]
            if config.scale_token and config.target_scale == "context":
                block_scale = scale_features(mean, std)[:, :, None].expand(-1, -1, blocks.shape[2], -1)
            elif config.scale_token:
                block_scale = self._block_scale(blocks, mean, std)  # [B, N, H, 2F]
            target_input = blocks
            if config.target_norm == "self":
                assert std is not None, "config.target_norm='self' needs the context std"
                target_input = self._self_normalize(blocks, std)
            u = self._encode_panel(self.target_encoder, target_input, mask, block_scale).float()  # [B, N, H, D]
            counts = mask.sum(dim=1).clamp_min(1)[:, None, None]  # [B, 1, 1]
            common = u.sum(dim=1) / counts  # [B, H, D]; padded rows of u are zero
            batch_idx = enc_mask.nonzero(as_tuple=True)[0]  # [S]; batch row of each scored stock
            u_valid = u[enc_mask]  # [S, H, D]
            residual = u_valid - common[batch_idx]  # [S, H, D]
            enc_pos = enc_mask.nonzero(as_tuple=True)[1]  # [S]; panel slot of each scored stock
            stock_stats = self._target_stats("stock", u_valid) if config.target_mode == "single" else None
            condition = None  # [S, H, D] standardized c+_{-i}, with config.market_condition
            placebo_same_date = None  # share of panels whose placebo target is from their own date
            if config.market_condition == "loo_mean":
                source = None
                if config.condition_placebo:
                    source, placebo_same_date = self._placebo_source(timestamp, B, u.device)
                condition = self._market_condition(u, u_valid, common, counts, batch_idx, source)
            peer_keys = peer_values = None  # [B, N, D], [B, N, H, D] of the "peer_attn" condition
            if config.market_condition == "peer_attn":
                ctx_scale = scale_features(mean, std) if config.scale_token else None
                peer_keys = self._encode_panel(self.target_encoder, context, mask, ctx_scale, daily=daily)
                s_mean, s_var = stock_stats
                s_scale = (s_var.sum(dim=-1).clamp_min(1e-6) / u.shape[-1]).sqrt()[:, None]
                peer_values = (u - s_mean) / s_scale * mask[..., None, None]

        H = len(config.horizons)
        horizon = self.horizon_embeddings.weight  # [H, D]
        S = u_valid.shape[0]
        stock_inputs = [mixed[enc_mask][:, None].expand(S, H, -1)]  # [S, H, D]
        if config.use_peer_context:
            stock_inputs.append(peers[enc_mask][:, None].expand(S, H, -1))  # [S, H, D]
        stock_pred_cond = None  # [S, H, D] prediction given the market condition
        attn_entropy = attn_sink = None
        if peer_keys is not None:
            condition, attn_entropy, attn_sink = self._peer_attention(
                stock_inputs[0][:, 0], peer_keys, peer_values, mask, batch_idx, enc_pos
            )
        is_factor = condition is not None and config.condition_predictor == "factor"
        if is_factor:
            factors = self.condition_proj(condition.to(stock_inputs[0].dtype))  # [S, H, K]
            if self.factor_loadings is None:  # linear loadings, from z_i alone
                loadings = torch.stack([out(stock_inputs[0][:, 0]) for out in self.loadings_out], dim=1)
            else:
                loadings = self.loadings_out(self.factor_loadings(*stock_inputs, horizon[None].expand(S, -1, -1)))
            exposure = (loadings.view(S, H, -1, factors.shape[-1]) @ factors[..., None]).squeeze(-1)  # [S, H, D]
        elif condition is not None and self.cond_stock_predictor is not None:
            projected = self.condition_proj(condition.to(stock_inputs[0].dtype))
            stock_pred_cond = self.cond_stock_predictor(*stock_inputs, projected, horizon[None].expand(S, -1, -1))
        elif condition is not None:
            projected = self.condition_proj(condition.to(stock_inputs[0].dtype))
            stock_pred_cond = self.stock_predictor(*stock_inputs, projected, horizon[None].expand(S, -1, -1))
            stock_inputs.append(torch.zeros_like(projected))  # the unconditional pass gets a zero condition
        stock_pred = self.stock_predictor(*stock_inputs, horizon[None].expand(S, -1, -1))  # [S, H, D]
        if is_factor:  # a(z_i, h) + B(z_i, h) g(c); the unconditional pass is a
            stock_pred_cond = (stock_pred.detach() if config.factor_stop_base else stock_pred) + exposure
        market_pred = None  # [B, H, D] with a market head
        if self.has_market_head:
            market_pred = self.market_predictor(market[:, None].expand(B, H, -1), horizon[None].expand(B, -1, -1))
        scale_pred = None  # [S, H, 2F] with config.scale_aux_weight
        if self.scale_predictor is not None:
            scale_pred = self.scale_out(self.scale_predictor(stock_inputs[0], horizon[None].expand(S, -1, -1)))

        with _no_autocast(context.device):
            stock_pred = stock_pred.float()
            market_pred = market_pred.float() if market_pred is not None else None
            if config.target_mode == "factorized":
                market_loss = self._normalized_mse("market", market_pred, common)  # [H]
                stock_loss = self._normalized_mse("stock", stock_pred, residual)  # [H]
            elif config.target_mode == "unfactorized":
                market_loss = None
                stock_loss = self._normalized_mse("stock", stock_pred + market_pred[batch_idx], u_valid)  # [H]
            else:
                market_loss = None
                stock_loss = self._normalized_mse("stock", stock_pred, u_valid, stock_stats)  # [H]
            cond_loss = uncond_loss = None  # [H], with a market condition
            if stock_pred_cond is not None:
                uncond_loss = stock_loss
                cond_loss = self._normalized_mse("stock", stock_pred_cond.float(), u_valid, stock_stats)
                w = config.condition_weight * self.condition_ramp
                stock_loss = (1 - w) * uncond_loss + w * cond_loss
            contrastive_loss = contrastive_metrics = None  # [H], with config.contrastive_weight
            if config.contrastive_weight > 0:
                ctx_scale = scale_features(mean, std) if mean is not None and std is not None else None
                contrastive_loss, contrastive_metrics = self._contrastive_loss(
                    stock_pred, u, mask, stock_stats, batch_idx, enc_pos, ctx_scale
                )
            peer_loss = peer_explained = None  # [H], with config.peer_predict_weight
            if self.peer_predictor is not None:
                peer_loss, peer_explained = self._peer_loss(
                    stock_inputs[0][:, 0], stock_pred, u_valid, stock_stats, horizon, enc_mask, batch_idx
                )
            scale_loss = None  # [H]
            if scale_pred is not None:
                scale_loss = self._standardized_mse("scale", scale_pred.float(), future_scale)
            if self.training:
                self.target_stats_initialized.fill_(True)

            loss = config.stock_loss_weight * stock_loss.mean()  # stock_loss [H] -> scalar, horizons weighted equally
            metrics = {"loss/stock": stock_loss.mean().detach()}
            metrics.update({f"loss/stock_h{h}": v.detach() for h, v in zip(config.horizons, stock_loss, strict=False)})
            if cond_loss is not None:
                metrics["loss/stock_uncond"] = uncond_loss.mean().detach()
                metrics["loss/stock_cond"] = cond_loss.mean().detach()
                # share of the unconditional error the realized market explains, per horizon
                explained = (1 - cond_loss / uncond_loss.clamp_min(1e-6)).detach()
                metrics.update({f"cond/explained_h{h}": v for h, v in zip(config.horizons, explained, strict=False)})
            if placebo_same_date is not None:
                metrics["cond/placebo_same_date"] = torch.tensor(placebo_same_date)
            if attn_entropy is not None:
                metrics["cond/attn_entropy"] = attn_entropy
                metrics["cond/attn_sink"] = attn_sink
            if contrastive_loss is not None:
                loss = loss + config.contrastive_weight * contrastive_loss.mean()
                metrics["loss/contrastive"] = contrastive_loss.mean().detach()
                metrics.update(contrastive_metrics)
            if config.sigreg_weight > 0:
                sigreg_loss = sigreg(z[enc_mask], config.sigreg_slices)
                loss = loss + config.sigreg_weight * sigreg_loss
                metrics["loss/sigreg"] = sigreg_loss.detach()
            if peer_loss is not None:
                loss = loss + config.peer_predict_weight * peer_loss.mean()
                metrics["loss/peer"] = peer_loss.mean().detach()
                metrics.update(
                    {f"peer/explained_h{h}": v for h, v in zip(config.horizons, peer_explained, strict=False)}
                )
            if market_loss is not None:
                loss = loss + config.market_loss_weight * market_loss.mean()
                metrics["loss/market"] = market_loss.mean().detach()
                metrics.update(
                    {f"loss/market_h{h}": v.detach() for h, v in zip(config.horizons, market_loss, strict=False)}
                )
            if scale_loss is not None:
                loss = loss + config.scale_aux_weight * scale_loss.mean()
                metrics["loss/scale_aux"] = scale_loss.mean().detach()

            if return_diagnostics:
                mixed_rows = mixed[enc_mask] if self.panel_mixer is not None else None
                metrics.update(
                    self._diagnostics(
                        z[enc_mask], market, u_valid, common, residual, stock_pred, market_pred, mixed=mixed_rows
                    )
                )

        output.loss = loss
        output.metrics = metrics
        return output

    @torch.no_grad()
    def _diagnostics(
        self, z, market, target, common, residual, stock_pred, market_pred, mixed=None
    ) -> dict[str, torch.Tensor]:
        """Collapse monitors: effective rank and std of states, targets and predictions, plus target variances.

        Market-level rows number B * H per micro-batch, so their effective rank is bounded by that; compare them
        across runs with the same batch size.
        """
        tensors = {
            "stock_state": z,
            "mixed_state": mixed,
            "market_state": market,
            "target": target,
            "target_common": common,
            "target_residual": residual,
            "pred_stock": stock_pred,
            "pred_market": market_pred,
        }
        metrics = {}
        for name, x in tensors.items():
            if x is None:
                continue
            erank, std = _spectrum(x)
            metrics[f"erank/{name}"] = erank
            metrics[f"std/{name}"] = std
        metrics["var/target_common"] = common.var(dim=0, unbiased=False).sum(dim=-1).mean()
        metrics["var/target_residual"] = residual.square().sum(dim=-1).mean()
        return metrics


AutoConfig.register(QTSMFJepaConfig.model_type, QTSMFJepaConfig)
AutoModel.register(QTSMFJepaConfig, QTSMFJepaModel)

__all__ = ["QTSMFJepaModel", "QTSMFJepaPreTrainedModel", "QTSMFJepaOutput"]
