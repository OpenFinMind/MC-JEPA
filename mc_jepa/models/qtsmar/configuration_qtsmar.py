"""Autoregressive next-patch baseline for the series JEPA (``qtsmfjepa``, ``target_mode: single``)."""

from transformers.configuration_utils import PretrainedConfig

READOUTS = ("last", "mean")
LOSSES = ("mse", "huber")


class QTSMArConfig(PretrainedConfig):
    r"""
    Configuration for [`QTSMArModel`]: the stock encoder of [`QTSMFJepaModel`] made causal (no CLS token) and trained
    like a decoder-only LM on patches: every patch state predicts the raw patch(es) that follow it.

    Args:
        in_features, patch_len, max_patches, input_clip, hidden_size, num_hidden_layers, num_attention_heads,
        intermediate_size, hidden_act, hidden_dropout_prob, attention_probs_dropout_prob, layer_norm_eps,
        initializer_range: stock encoder, identical to ``QTSMFJepaConfig`` except for the causal attention mask.
        pred_patches (`int`): patches each state predicts, ``p + 1 .. p + pred_patches`` (1 is plain next-patch).
        future_target (`bool`): extend the sequence with the first ``pred_patches`` future patches, so the last
            states (the ``last`` readout at the decision time) also get targets. Without it, states whose targets
            run past the context are left out of the loss.
        horizons (`list[int]`), target_block_len (`int`): unused by the loss; kept so ``lookahead`` matches the
            JEPA's and the dataset samples exactly the same windows.
        target_clip (`float`): targets are clamped to ``[-target_clip, target_clip]`` like the inputs.
        loss (`str`): ``"mse"`` or ``"huber"`` (with ``huber_delta``) on the clamped targets.
        readout (`str`): frozen feature returned by ``encode``: the ``"last"`` patch state (the only one that sees
            the whole context) or the ``"mean"`` of the patch states.
    """

    model_type = "qtsmar"

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
        pred_patches=1,
        future_target=False,
        horizons=(1, 60, 240),
        target_block_len=30,
        target_clip=10.0,
        loss="mse",
        huber_delta=1.0,
        readout="last",
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert readout in READOUTS, f"readout must be one of {READOUTS}, got {readout}"
        assert loss in LOSSES, f"loss must be one of {LOSSES}, got {loss}"
        assert hidden_size % num_attention_heads == 0
        assert pred_patches >= 1
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
        self.pred_patches = pred_patches
        self.future_target = future_target
        self.horizons = list(horizons)
        self.target_block_len = target_block_len
        self.target_clip = target_clip
        self.loss = loss
        self.huber_delta = huber_delta
        self.readout = readout
        assert not future_target or self.lookahead >= pred_patches * patch_len, (
            f"future_target needs lookahead >= pred_patches * patch_len = {pred_patches * patch_len}"
        )

    @property
    def lookahead(self) -> int:
        """Future bars the dataset must return; the same as the JEPA's, so both sample the same windows."""
        return max(self.horizons) + self.target_block_len - 1


__all__ = ["QTSMArConfig"]
