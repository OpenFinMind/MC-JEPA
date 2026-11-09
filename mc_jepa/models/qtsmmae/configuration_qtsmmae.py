"""Reconstruction baselines for the series JEPA (``qtsmfjepa``, ``target_mode: single``)."""

from transformers.configuration_utils import PretrainedConfig

OBJECTIVES = ("future", "masked")
READOUTS = ("cls", "mean")
LOSSES = ("mse", "huber")


class QTSMMaeConfig(PretrainedConfig):
    r"""
    Configuration for [`QTSMMaeModel`]: the same BERT-like stock encoder as [`QTSMFJepaModel`], trained by
    reconstructing bars instead of predicting EMA-encoder latents.

    Args:
        in_features, patch_len, max_patches, input_clip, hidden_size, num_hidden_layers, num_attention_heads,
        intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, layer_norm_eps,
        initializer_range: stock encoder, identical to ``QTSMFJepaConfig``.
        objective (`str`):
            - ``"future"``: from the CLS state and a horizon embedding, a residual MLP predicts the raw future block
              (bars ``t+h .. t+h+target_block_len-1``, z-scored by the context statistics). It is the series JEPA
              with the target moved from latent space to observation space; everything else is shared.
            - ``"masked"``: PatchTST / SimMIM masked-patch modeling. Masked patch embeddings are replaced by a
              learned mask token and a linear head reconstructs the masked patches of the context.
        predictor_layers (`int`): residual MLP blocks in the future predictor.
        horizons (`list[int]`), target_block_len (`int`): future blocks, as in ``QTSMFJepaConfig``. The masked
            objective ignores them for its loss, but the dataset still returns ``lookahead`` future bars, so both
            objectives train on exactly the same windows as the JEPA.
        mask_ratio (`float`): share of context patches masked (``"masked"`` only).
        mask_span (`int`): masked patches come in contiguous spans of this many patches; 1 is i.i.d. masking.
        target_clip (`float`): targets are clamped to ``[-target_clip, target_clip]`` like the inputs.
        loss (`str`): ``"mse"`` or ``"huber"`` (with ``huber_delta``) on the clamped targets.
        readout (`str`): frozen feature returned by ``encode``: the ``"cls"`` state or the ``"mean"`` of the patch
            states. The masked loss never trains the CLS state directly, so probe both.
    """

    model_type = "qtsmmae"

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
        objective="future",
        predictor_layers=2,
        horizons=(1, 60, 240),
        target_block_len=30,
        mask_ratio=0.4,
        mask_span=1,
        target_clip=10.0,
        loss="mse",
        huber_delta=1.0,
        readout="cls",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert objective in OBJECTIVES, f"objective must be one of {OBJECTIVES}, got {objective}"
        assert readout in READOUTS, f"readout must be one of {READOUTS}, got {readout}"
        assert loss in LOSSES, f"loss must be one of {LOSSES}, got {loss}"
        assert hidden_size % num_attention_heads == 0
        assert 0.0 < mask_ratio < 1.0 and mask_span >= 1
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
        self.objective = objective
        self.predictor_layers = predictor_layers
        self.horizons = list(horizons)
        self.target_block_len = target_block_len
        self.mask_ratio = mask_ratio
        self.mask_span = mask_span
        self.target_clip = target_clip
        self.loss = loss
        self.huber_delta = huber_delta
        self.readout = readout

    @property
    def lookahead(self) -> int:
        """Future bars the dataset must return; the same as the JEPA's, so both sample the same windows."""
        return max(self.horizons) + self.target_block_len - 1


__all__ = ["QTSMMaeConfig"]
