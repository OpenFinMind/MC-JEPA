"""TS2Vec (hierarchical contrastive) baseline for the series JEPA (``qtsmfjepa``, ``target_mode: single``)."""

from transformers.configuration_utils import PretrainedConfig

READOUTS = ("max", "mean", "last")


class QTSMTs2VecConfig(PretrainedConfig):
    r"""
    Configuration for [`QTSMTs2VecModel`]: the stock encoder of [`QTSMFJepaModel`] without a CLS token, trained with
    the TS2Vec objective (Yue et al., AAAI 2022) on patches: two overlapping random crops of each window, timestamp
    masking, and the hierarchical (instance + temporal) contrastive loss on the overlap.

    Args:
        in_features, patch_len, max_patches, input_clip, hidden_size, num_hidden_layers, num_attention_heads,
        intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, layer_norm_eps,
        initializer_range: stock encoder, identical to ``QTSMFJepaConfig`` (bidirectional attention). Position
            embeddings count from the start of each crop, not of the window.
        crop_min_patches (`int`): shortest overlap of the two crops, in patches (official ``2 ** (temporal_unit + 1)``).
        mask_prob (`float`): probability of zeroing a patch embedding in each view (official binomial mask, 0.5).
        alpha (`float`): weight of the instance term; ``1 - alpha`` weights the temporal term.
        temporal_unit (`int`): hierarchy levels below which the temporal term is skipped (official default 0).
        horizons (`list[int]`), target_block_len (`int`): unused by the loss; kept so ``lookahead`` matches the
            JEPA's and the dataset samples exactly the same windows.
        readout (`str`): frozen feature returned by ``encode``: the ``"max"`` over patch states (TS2Vec's
            ``encoding_window='full_series'``), their ``"mean"``, or the ``"last"`` patch state.
    """

    model_type = "qtsmts2vec"

    def __init__(
        self,
        in_features=5,
        patch_len=10,
        max_patches=64,
        input_clip=10.0,
        hidden_size=256,
        num_hidden_layers=6,
        num_attention_heads=8,
        intermediate_size=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        crop_min_patches=2,
        mask_prob=0.5,
        alpha=0.5,
        temporal_unit=0,
        horizons=(1, 60, 240),
        target_block_len=30,
        readout="max",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert readout in READOUTS, f"readout must be one of {READOUTS}, got {readout}"
        assert hidden_size % num_attention_heads == 0
        assert 1 <= crop_min_patches <= max_patches
        assert 0.0 <= mask_prob < 1.0 and 0.0 <= alpha <= 1.0
        assert all(h >= 1 for h in horizons), "horizons count bars after the decision time and start at 1"

        self.in_features = in_features
        self.patch_len = patch_len
        self.max_patches = max_patches
        self.input_clip = input_clip
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.crop_min_patches = crop_min_patches
        self.mask_prob = mask_prob
        self.alpha = alpha
        self.temporal_unit = temporal_unit
        self.horizons = list(horizons)
        self.target_block_len = target_block_len
        self.readout = readout

    @property
    def lookahead(self) -> int:
        """Future bars the dataset must return; the same as the JEPA's, so both sample the same windows."""
        return max(self.horizons) + self.target_block_len - 1


__all__ = ["QTSMTs2VecConfig"]
