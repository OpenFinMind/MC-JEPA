"""Factorized panel JEPA configuration."""

from transformers.configuration_utils import PretrainedConfig

TARGET_MODES = ("factorized", "unfactorized", "single")
MARKET_CONDITIONS = ("none", "loo_mean", "peer_attn")


class QTSMFJepaConfig(PretrainedConfig):
    r"""
    Configuration for [`QTSMFJepaModel`]: a BERT-like stock encoder, a permutation-invariant peer pooler, an EMA
    target encoder, and horizon-conditioned market / stock predictors trained on ``CrossSectionPanelDataset`` panels.

    Args:
        in_features (`int`): number of input features ``F`` per bar.
        patch_len (`int`): bars per token. ``lookback`` and ``target_block_len`` must be multiples of it.
        max_patches (`int`): size of the learned position table (excluding the CLS token).
        input_clip (`float`): inputs are z-scored by the dataset, then clamped to ``[-input_clip, input_clip]``.
        readout (`str`): the state a stock encoder returns, for the online context state z_i, the EMA targets u+ and
            the frozen probe features alike: ``"cls"`` the CLS token, ``"mean"`` the mean over its patch states (the
            CLS token is then read only by the patches' attention).
        hidden_size, num_hidden_layers, num_attention_heads, intermediate_size, hidden_act, hidden_dropout_prob,
        attention_probs_dropout_prob, layer_norm_eps, initializer_range: BERT-style stock encoder ``E``.
        num_market_tokens (`int`): learned pooling queries in the peer pooler ``G``.
        panel_mlp_layers (`int`): residual MLP blocks applied to each stock state before pooling.
        predictor_layers (`int`): residual MLP blocks in each predictor.
        horizons (`list[int]`): target block ``h`` covers bars ``t+h .. t+h+target_block_len-1`` (``h >= 1``).
        target_block_len (`int`): bars ``Delta`` per future target block.
        target_mode (`str`):
            - ``"factorized"``: market head predicts the cross-sectional mean target ``c+``, stock head predicts the
              residual ``r+ = u+ - c+`` (the proposed method).
            - ``"unfactorized"``: same two heads, their sum predicts ``u+`` (the decisive control).
            - ``"single"``: stock head alone predicts ``u+`` (plain JEPA).
        use_peer_context (`bool`): feed the leave-one-out peer summary ``m_{-i}`` to the stock head; ``False``
            gives the asset-only control.
        market_loss_weight, stock_loss_weight (`float`): weights of the two normalized losses.
        loss_stats_momentum (`float`): EMA momentum of the running target moments that normalize each loss.
        ema_momentum, ema_momentum_end (`float`): target encoder EMA momentum, linearly annealed by the trainer.
        scale_token (`bool`): give both encoders the per-window scale that the dataset's z-score removes,
            as one extra token embedded from ``asinh(mean)`` and ``log(std)`` of each feature: the online encoder
            gets the context's scale, the target encoder each future block's own absolute scale, so the predictor
            has to forecast it. Both always get the token, so
            the EMA target encoder never runs on an input type the online encoder was not trained on.
        target_scale (`str`): the scale token of the target encoder with ``scale_token``: ``"block"`` each future
            block's own scale (the predictor forecasts it), ``"context"`` the context's scale (the control that
            sees scale without having to forecast it).
        scale_aux_weight (`float`): weight of a supervised head that regresses each future block's
            ``scale_features`` [H, 2F] from the online state (per-feature standardized MSE); 0 disables it. With
            ``target_scale="context"`` it is the control that forecasts scale outside the latent target.
        target_window (`str`): which bars target ``h`` covers: ``"block"`` the ``target_block_len`` bars starting at
            t+h; ``"end"`` the ``target_block_len`` bars ending at t+h, which may reach back into the context (with
            ``target_block_len`` = lookback and ``target_norm="self"``, the input the online encoder sees at t+h).
        target_norm (`str`): how each future block is normalized for the target encoder: ``"context"`` by the
            context's mean / std, as the dataset returns it; ``"self"`` by the block's own mean / std, exactly as the
            dataset normalizes a context window, so a block as long as the lookback is the input the online encoder
            would see at that later time (its scale then comes only from the scale token; needs mean / std).
        market_condition (`str`): what the stock head is told about the realized future market
            (market-conditioned JEPA):
            - ``"none"``: nothing.
            - ``"loo_mean"``: the leave-one-out panel mean of the EMA targets, c+_{-i,h} = mean_{j != i} u+_{j,h},
              standardized and projected through a ``condition_dim`` bottleneck. Stop-gradient; used only in
              pretraining, so frozen states never need the future.
            - ``"peer_attn"``: a learned hindsight latent. Stock i attends over its panel's other stocks with
              ``condition_heads`` heads, queries from z_i, keys from the peers' context states under the EMA encoder
              (their past), values their standardized EMA targets u+_{j,h} (their realized future); the heads go
              through the ``condition_dim`` bottleneck. Uniform attention is ``loo_mean``; learned weights pick the
              peers (sector, style) whose outcomes explain stock i's, so z_i has to encode which peers those are.
              Each head has one more softmax slot, a sink with a learned logit and a zero value, that absorbs the
              weight no peer deserves, so a head can shrink its average toward zero.
        condition_weight (`float`): with a market condition, the stock loss is ``(1 - w) * L(P(z_i, 0, h), u+) +
            w * L(P(z_i, c+_{-i}, h), u+)``: the same predictor scored without (zero condition) and with it.
            0 is plain JEPA, 1 the purely conditional objective.
        condition_predictor (`str`): how the stock head reads the market condition:
            - ``"shared"``: one predictor P(z_i, W c, h), with a zero condition for the unconditional pass.
            - ``"separate"``: two predictors, P_u(z_i, h) for the unconditional pass and P_c(z_i, W c, h) for the
              conditional one, so the passes share only z_i.
            - ``"factor"``: an explicit factor model, P(z_i, c, h) = a(z_i, h) + B(z_i, h) g(c), with g a linear map
              (no bias) of the condition to ``condition_dim`` factors and B(z_i, h) [D, condition_dim] the stock's
              loadings from its own head. The unconditional pass is a(z_i, h) alone, so the condition reaches z_i
              only through the loadings and a(z_i, h) is the same function in both passes.
        factor_stop_base (`bool`): with ``condition_predictor="factor"``, the conditional pass is sg(a(z_i, h)) +
            B(z_i, h) g(c), so a(z_i, h) is trained only by the unconditional loss (the single objective) and the
            conditional loss reaches z_i only through the loadings.
        factor_loadings (`str`): with ``condition_predictor="factor"``, the form of the loadings B(z_i, h):
            - ``"mlp"``: a predictor head over (z_i, h), then a linear map to [D, condition_dim].
            - ``"linear"``: B(z_i, h) = reshape(W_h z_i + b_h), one affine map per horizon, so the exposure is linear
              in z_i, the form a frozen linear probe reads.
        condition_placebo (`bool`): with ``market_condition="loo_mean"``, a placebo control: each panel gets the
            standardized market target c+ of another panel of the micro-batch, from another date when one exists,
            instead of its own c+_{-i}. The condition keeps its distribution but carries no information on the
            panel's own future, so a gain that survives it is regularization, not exposure.
        condition_ramp_start, condition_ramp_steps (`int`): schedule of ``condition_weight`` over training steps: 0
            before step ``condition_ramp_start``, then linear up to ``condition_weight`` over
            ``condition_ramp_steps`` (set by the trainer). Both 0 keep the weight constant.
        condition_heads (`int`): attention heads of the ``peer_attn`` condition (one weighted peer average each).
        condition_query (`str`): the queries of the ``peer_attn`` condition:
            - ``"state"``: projected from z_i, so each stock picks its own peers.
            - ``"global"``: one learned query per head, shared by every stock (zero-initialized, so training starts
              from ``loo_mean`` plus the sink). The peer weights then depend only on the peers' past states, so each
              head is a learned, characteristic-weighted market portfolio of the peers' futures, the same for every
              stock of the panel up to its own exclusion, and z_i cannot steer the condition toward the peers that
              best reveal its own future.
        condition_dim (`int`): bottleneck width of the market condition (roughly the number of latent factors).
        stacked_linear_act (`bool`): put ``hidden_act`` between back-to-back linear layers, which otherwise collapse
            into one linear map: the two layers of the ``condition_dim`` bottleneck (``condition_predictor`` shared /
            separate), and the predictor output before ``loadings_out`` (``factor_loadings="mlp"``) and ``scale_out``.
        sigreg_weight (`float`): weight of SIGReg (LeJEPA, arXiv:2511.08544) on the online context states z_i: the
            Epps-Pulley statistic of z_i on ``sigreg_slices`` random unit directions against N(0, 1), which pushes
            the states towards an isotropic Gaussian N(0, I). 0 disables it.
        sigreg_slices (`int`): random directions per SIGReg evaluation (redrawn every forward pass).
        context_symbols (`int`): encode at most this many random stocks per panel with the online encoder and
            score only them; every stock of the panel still enters the market condition through the cheap target
            encoder. 0 encodes every stock.
        contrastive_weight (`float`): weight of a same-time contrastive loss on the unconditional stock prediction: it
            has to pick stock i's own standardized target u+_{i,h} among the targets of every stock of its panel
            (InfoNCE, cosine similarity). Keys are centered on the panel mean, so the shared market move cancels and
            only stock-specific (relative) information scores. 0 disables it.
        contrastive_temperature (`float`): softmax temperature of the contrastive loss.
        contrastive_neighbors (`int`): with ``scale_token``, the candidates are stock i and the K stocks of its panel
            nearest in context scale (``scale_features``), so a target cannot be told apart by its stock's
            persistent volatility or activity level alone. 0 keeps every stock of the panel.
        peer_predict_weight (`float`): weight of a lead-lag loss: a peer head reads a random partner's state and
            stock i's state and predicts what stock i's own prediction misses, u+_{i,h} - sg(P(z_i, h))
            (standardized), so z_partner has to carry information about other stocks' futures. 0 disables it.
        panel_mixer_layers (`int`): transformer layers of stock-stock attention over each panel's encoded context
            states (``QTSMFJepaPanelMixer``); the stock predictor then reads the mixed state instead of z_i, so the
            mixer learns what peers' histories tell about stock i's future. 0 disables it (the stock's own state).
        freeze_encoder (`bool`): keep the online and target encoders fixed (no gradient, no EMA update), e.g. a
            series encoder loaded with the trainer's ``init_encoder_from``; only the mixer and heads train.
        daily_lookback (`int`): completed trading days before the decision date given to the online encoder as a
            second, longer context (``mc_jepa/datasets/daily.py``; the dataset needs ``data.daily_dir``). 0
            disables it. The target encoder never sees daily bars, so the targets do not change.
        daily_patch_len (`int`): days per daily token (``daily_lookback / daily_patch_len`` tokens).
        daily_features (`int`): features per day, ``DAILY_FEATURES``.
        daily_feature_set (`str`): which daily features the dataset builds, ``features-1`` or ``features-2``
            (``mc_jepa/datasets/daily.py``); stored here so a checkpoint's probes rebuild the same input.
    """

    model_type = "qtsmfjepa"

    def __init__(
        self,
        in_features=61,
        patch_len=10,
        max_patches=64,
        input_clip=10.0,
        readout="cls",
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        num_market_tokens=4,
        panel_mlp_layers=2,
        predictor_layers=2,
        horizons=(1, 60, 240),
        target_block_len=30,
        target_mode="factorized",
        use_peer_context=True,
        market_loss_weight=1.0,
        stock_loss_weight=1.0,
        loss_stats_momentum=0.99,
        ema_momentum=0.996,
        ema_momentum_end=1.0,
        scale_token=False,
        target_scale="block",
        scale_aux_weight=0.0,
        target_window="block",
        target_norm="context",
        market_condition="none",
        condition_weight=1.0,
        condition_dim=32,
        stacked_linear_act=False,
        condition_predictor="shared",
        factor_stop_base=False,
        factor_loadings="mlp",
        condition_heads=4,
        condition_query="state",
        condition_ramp_start=0,
        condition_ramp_steps=0,
        condition_placebo=False,
        contrastive_weight=0.0,
        contrastive_temperature=0.1,
        contrastive_neighbors=0,
        peer_predict_weight=0.0,
        sigreg_weight=0.0,
        sigreg_slices=256,
        context_symbols=0,
        panel_mixer_layers=0,
        freeze_encoder=False,
        daily_lookback=0,
        daily_patch_len=5,
        daily_features=6,
        daily_feature_set="features-1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert target_mode in TARGET_MODES, f"target_mode must be one of {TARGET_MODES}, got {target_mode}"
        assert hidden_size % num_attention_heads == 0
        assert target_block_len % patch_len == 0, "target_block_len must be a multiple of patch_len"
        assert all(h >= 1 for h in horizons), "horizons count bars after the decision time and start at 1"

        self.in_features = in_features
        self.patch_len = patch_len
        self.max_patches = max_patches
        self.input_clip = input_clip
        assert readout in ("cls", "mean"), f"unknown readout {readout}"
        self.readout = readout
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.num_market_tokens = num_market_tokens
        self.panel_mlp_layers = panel_mlp_layers
        self.predictor_layers = predictor_layers
        self.horizons = list(horizons)
        self.target_block_len = target_block_len
        self.target_mode = target_mode
        self.use_peer_context = use_peer_context
        self.market_loss_weight = market_loss_weight
        self.stock_loss_weight = stock_loss_weight
        self.loss_stats_momentum = loss_stats_momentum
        self.ema_momentum = ema_momentum
        self.ema_momentum_end = ema_momentum_end
        self.scale_token = bool(scale_token)
        assert target_scale in ("block", "context"), f"unknown target_scale {target_scale}"
        self.target_scale = target_scale
        self.scale_aux_weight = float(scale_aux_weight)
        assert target_window in ("block", "end"), f"unknown target_window {target_window}"
        self.target_window = target_window
        assert target_norm in ("context", "self"), f"unknown target_norm {target_norm}"
        self.target_norm = target_norm
        assert market_condition in MARKET_CONDITIONS, f"market_condition must be one of {MARKET_CONDITIONS}"
        assert market_condition == "none" or target_mode == "single", "a market condition needs target_mode=single"
        self.market_condition = market_condition
        self.condition_weight = float(condition_weight)
        self.condition_dim = condition_dim
        self.stacked_linear_act = bool(stacked_linear_act)
        assert condition_predictor in ("shared", "separate", "factor"), (
            f"unknown condition_predictor {condition_predictor}"
        )
        self.condition_predictor = condition_predictor
        assert factor_loadings in ("mlp", "linear"), f"unknown factor_loadings {factor_loadings}"
        assert condition_predictor == "factor" or (not factor_stop_base and factor_loadings == "mlp"), (
            "factor_stop_base and factor_loadings need condition_predictor=factor"
        )
        self.factor_stop_base = bool(factor_stop_base)
        self.factor_loadings = factor_loadings
        self.condition_heads = condition_heads
        assert condition_query in ("state", "global"), f"unknown condition_query {condition_query}"
        self.condition_query = condition_query
        self.condition_ramp_start = int(condition_ramp_start)
        self.condition_ramp_steps = int(condition_ramp_steps)
        assert not condition_placebo or market_condition == "loo_mean", "condition_placebo needs loo_mean"
        self.condition_placebo = bool(condition_placebo)
        self.contrastive_weight = float(contrastive_weight)
        self.contrastive_temperature = float(contrastive_temperature)
        self.contrastive_neighbors = contrastive_neighbors
        self.peer_predict_weight = float(peer_predict_weight)
        assert target_mode == "single" or (contrastive_weight == 0 and peer_predict_weight == 0), (
            "the contrastive and peer losses need target_mode=single"
        )
        self.sigreg_weight = float(sigreg_weight)
        self.sigreg_slices = sigreg_slices
        assert not (freeze_encoder and sigreg_weight > 0), "SIGReg regularizes the encoder, which freeze_encoder fixes"
        self.context_symbols = context_symbols
        self.panel_mixer_layers = panel_mixer_layers
        self.freeze_encoder = bool(freeze_encoder)
        assert daily_lookback % daily_patch_len == 0, "daily_lookback must be a multiple of daily_patch_len"
        self.daily_lookback = daily_lookback
        self.daily_patch_len = daily_patch_len
        self.daily_features = daily_features
        assert daily_feature_set in ("features-1", "features-2"), f"unknown daily_feature_set {daily_feature_set}"
        self.daily_feature_set = daily_feature_set

    @property
    def lookahead(self) -> int:
        """Future bars ``CrossSectionPanelDataset`` must return to cover every target block."""
        if self.target_window == "end":
            return max(self.horizons)
        return max(self.horizons) + self.target_block_len - 1


__all__ = ["QTSMFJepaConfig"]
