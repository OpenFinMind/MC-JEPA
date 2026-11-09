"""CPC (contrastive predictive coding) baseline for the series JEPA (``qtsmfjepa``, ``target_mode: single``)."""

from transformers.configuration_utils import PretrainedConfig

TARGETS = ("online", "ema")
NEGATIVES = ("all", "horizon")


class QTSMCpcConfig(PretrainedConfig):
    r"""
    Configuration for [`QTSMCpcModel`]: the stock encoder and stock predictor of [`QTSMFJepaModel`] with the JEPA's
    future-block targets, trained with InfoNCE (CPC, van den Oord et al. 2018) instead of the MSE against an EMA
    target: each prediction has to pick its own future-block latent among the other targets of the micro-batch.

    Args:
        in_features, patch_len, max_patches, input_clip, readout, hidden_size, num_hidden_layers, num_attention_heads,
        intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, layer_norm_eps,
        initializer_range: stock encoder, identical to ``QTSMFJepaConfig`` (CLS token, bidirectional attention).
        predictor_layers (`int`): residual MLP blocks of the horizon-conditioned predictor, as in the JEPA.
        horizons (`list[int]`), target_block_len (`int`): target block ``h`` covers bars ``t+h .. t+h+Delta-1``, as in
            the JEPA (``target_window="block"``, ``target_norm="context"``).
        target (`str`): the encoder of the future blocks:
            - ``"online"``: the online encoder itself, with gradient (CPC).
            - ``"ema"``: a stop-gradient EMA copy (MoCo-style), the arm between CPC and the JEPA: the JEPA's target
              with negatives.
        ema_momentum, ema_momentum_end (`float`): EMA momentum of ``target="ema"``, linearly annealed by the trainer.
        negatives (`str`): the candidates of each prediction (window i, horizon h), besides its own target:
            - ``"all"``: every other (window, horizon) target of the micro-batch, including window i's other horizons
              (CPC's same-sequence negatives).
            - ``"horizon"``: the other windows' targets at the same horizon h.
        temperature (`float`): softmax temperature of the cosine similarities.
    """

    model_type = "qtsmcpc"

    def __init__(
        self,
        in_features=5,
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
        predictor_layers=2,
        horizons=(1, 60, 240),
        target_block_len=30,
        target="online",
        ema_momentum=0.996,
        ema_momentum_end=1.0,
        negatives="all",
        temperature=0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert readout in ("cls", "mean"), f"unknown readout {readout}"
        assert target in TARGETS, f"target must be one of {TARGETS}, got {target}"
        assert negatives in NEGATIVES, f"negatives must be one of {NEGATIVES}, got {negatives}"
        assert hidden_size % num_attention_heads == 0
        assert target_block_len % patch_len == 0, "target_block_len must be a multiple of patch_len"
        assert all(h >= 1 for h in horizons), "horizons count bars after the decision time and start at 1"
        assert temperature > 0

        self.in_features = in_features
        self.patch_len = patch_len
        self.max_patches = max_patches
        self.input_clip = input_clip
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
        self.predictor_layers = predictor_layers
        self.horizons = list(horizons)
        self.target_block_len = target_block_len
        self.target = target
        self.ema_momentum = ema_momentum
        self.ema_momentum_end = ema_momentum_end
        self.negatives = negatives
        self.temperature = float(temperature)

    @property
    def lookahead(self) -> int:
        """Future bars the dataset must return: up to the end of the last target block."""
        return max(self.horizons) + self.target_block_len - 1


__all__ = ["QTSMCpcConfig"]
