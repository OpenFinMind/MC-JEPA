"""Self-supervised trainer for panel JEPA models (``qtsmfjepa``) and their baselines (``qtsmmae``, ``qtsmar``,
``qtsmts2vec``, ``qtsmcpc``).

``data.format`` picks the samples:

- ``panel``: ``CrossSectionPanelDataset`` (v4), one sample is one decision time with every eligible stock.
- ``series``: ``StockSeriesDataset`` (v3), one sample is one (stock, decision time) window, fed to the model as a
  one-stock panel. There are no peers, so the model must use ``target_mode: single`` without peer context.

A micro-batch holds ``micro_batch_size_per_gpu`` samples; gradients accumulate to ``train_batch_size`` samples per
optimizer step, after which the EMA target encoder is updated. DDP only: the model is small, and FSDP1 flat
parameters cannot mix the frozen target encoder with trainable weights.
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"

import logging
import random
import time
from contextlib import nullcontext

import hydra
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from safetensors.torch import load_file
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoConfig, AutoModel

import mc_jepa.models  # noqa: F401
from mc_jepa.datasets.v3.series_dataset import StockSeriesDataset
from mc_jepa.datasets.v4.panel_dataset import CrossSectionPanelDataset
from mc_jepa.utils.logger import print_rank_0
from mc_jepa.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from mc_jepa.utils.tracking import Tracking

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("QUANT_TSM_LOGGING_LEVEL", "WARN"))

# model-section keys that configure the trainer, not the architecture
TRAINER_MODEL_KEYS = ("name", "enable_gradient_checkpointing")
NO_DECAY_NAMES = ("cls_token", "mask_token", "position_embeddings", "queries", "horizon_embeddings")


def series_collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Stack ``StockSeriesDataset`` samples as one-stock panels: context [B, 1, lookback, F], symbol_mask [B, 1]."""
    collated = torch.utils.data.default_collate(batch)
    out = {"timestamp": collated.pop("timestamp"), "symbol_ids": collated.pop("symbol_id")[:, None]}
    out.update({k: v[:, None] for k, v in collated.items()})
    out["symbol_mask"] = torch.ones(len(batch), 1, dtype=torch.bool)
    return out


def _load_symbols(symbols) -> list[str] | None:
    """``null``, a list, or a text file with one symbol per line."""
    if symbols is None or not isinstance(symbols, str):
        return None if symbols is None else list(symbols)
    with open(os.path.expanduser(symbols)) as f:
        return [line.strip() for line in f if line.strip()]


def load_encoders(model, path: str) -> None:
    """Copy the online and target encoders of a saved checkpoint (e.g. a series run) into ``model``; the other
    modules keep their fresh initialization. The encoder architectures must match."""
    prefixes = ("encoder.", "target_encoder.")
    state = load_file(os.path.join(os.path.expanduser(path), "model.safetensors"))
    state = {k: v for k, v in state.items() if k.startswith(prefixes)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [k for k in missing if k.startswith(prefixes)]
    assert not missing and not unexpected, f"encoder mismatch: missing {missing}, unexpected {unexpected}"


def load_model_init(model, path: str) -> None:
    """Initialize ``model`` from every weight of a saved checkpoint, e.g. to finetune a panel single run with a
    market condition. A predictor input layer that gained inputs (the condition, which sits before the last,
    horizon input) keeps the saved columns and gets zero columns for the new inputs, so the model starts out
    computing what the checkpoint did; modules the checkpoint lacks keep their fresh initialization."""
    state = load_file(os.path.join(os.path.expanduser(path), "model.safetensors"))
    own = model.state_dict()
    D = model.config.hidden_size
    for k, v in state.items():
        assert k in own, f"unexpected key {k} in {path}"
        if v.shape == own[k].shape:
            continue
        (rows, cols), (saved_rows, saved_cols) = own[k].shape, v.shape
        widens = k.endswith(".inp.weight") and rows == saved_rows and cols > saved_cols and (cols - saved_cols) % D == 0
        assert widens, f"shape mismatch for {k}: {tuple(v.shape)} saved, {tuple(own[k].shape)} in the model"
        widened = v.new_zeros(own[k].shape)
        widened[:, : v.shape[1] - D] = v[:, :-D]
        widened[:, -D:] = v[:, -D:]
        state[k] = widened
        print_rank_0(f"{k}: {tuple(v.shape)} -> {tuple(widened.shape)}, new inputs zero")
    missing, _ = model.load_state_dict(state, strict=False)
    print_rank_0(f"Fresh (not in {path}): {missing}")


class PanelJepaTrainer:
    def __init__(self, config, local_rank: int):
        self.config = config
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = self.device.type == "cuda" and config.trainer.get("bf16", True)

        self._build_model()
        self._build_dataloader()
        self._build_optimizer()

    def _build_model(self):
        config = self.config
        model_kwargs = OmegaConf.to_container(config.model, resolve=True)
        name = model_kwargs["name"]
        model_kwargs = {k: v for k, v in model_kwargs.items() if k not in TRAINER_MODEL_KEYS}
        num_features = len(config.data.features)
        if model_kwargs.get("in_features") is None:
            model_kwargs["in_features"] = num_features
        assert model_kwargs["in_features"] == num_features, "model.in_features must equal len(data.features)"
        self.model_config = AutoConfig.for_model(name, **model_kwargs)
        self.data_format = config.data.get("format", "panel")
        assert self.data_format in ("panel", "series"), f"Unknown data.format: {self.data_format}"
        if self.data_format == "series" and name == "qtsmfjepa":
            assert self.model_config.target_mode == "single" and not self.model_config.use_peer_context, (
                "series samples have no peers: use model.target_mode=single and model.use_peer_context=false"
            )

        if config.trainer.model_path is None:
            print_rank_0("Initializing model from scratch")
            model = AutoModel.from_config(self.model_config, dtype=torch.float32)
        else:
            print_rank_0(f"Continue from {config.trainer.model_path}")
            model = AutoModel.from_pretrained(config.trainer.model_path, config=self.model_config, dtype=torch.float32)
        if config.trainer.get("init_encoder_from") is not None:
            print_rank_0(f"Encoders from {config.trainer.init_encoder_from}")
            load_encoders(model, config.trainer.init_encoder_from)
        if config.trainer.get("init_model_from") is not None:
            print_rank_0(f"All weights from {config.trainer.init_model_from}")
            load_model_init(model, config.trainer.init_model_from)
        if config.model.get("enable_gradient_checkpointing", False):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        print_rank_0(model)
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print_rank_0(f"Model has {num_trainable} trainable / {sum(p.numel() for p in model.parameters())} parameters")

        self.model = model.to(self.device)
        self.ddp_model = DistributedDataParallel(
            self.model,
            device_ids=[self.device.index] if self.device.type == "cuda" else None,
            find_unused_parameters=False,
        )

    def _build_dataset(self, split, deterministic: bool) -> CrossSectionPanelDataset | StockSeriesDataset:
        data = self.config.data
        # a split may override data.times_of_day, e.g. to keep validation fixed while ablating training times
        times_of_day = split.get("times_of_day", data.times_of_day)
        times_of_day = None if times_of_day is None else list(times_of_day)
        daily_lookback = getattr(self.model_config, "daily_lookback", 0)  # 0 for models without daily bars
        daily = {
            "daily_dir": data.get("daily_dir"),
            "daily_lookback": daily_lookback,
            "daily_feature_set": getattr(self.model_config, "daily_feature_set", "features-1"),
        }
        if self.data_format == "series":
            return StockSeriesDataset(
                data_prefix=os.path.expanduser(split.prefix),
                features=list(data.features),
                lookback=data.lookback,
                lookahead=self.model_config.lookahead,
                times_of_day=times_of_day,
                start_timestamp=split.start_timestamp,
                end_timestamp=split.end_timestamp,
                symbols=_load_symbols(split.symbols),
                num_samples=split.get("num_samples"),
                seed=data.seed,
                normalize=True,
                warmup=data.mmap_warmup,
                **daily,
            )
        return CrossSectionPanelDataset(
            data_prefix=os.path.expanduser(split.prefix),
            features=list(data.features),
            lookback=data.lookback,
            lookahead=self.model_config.lookahead,
            times_of_day=times_of_day,
            start_timestamp=split.start_timestamp,
            end_timestamp=split.end_timestamp,
            symbols=_load_symbols(split.symbols),
            min_symbols=data.min_symbols,
            span_ratio=data.get("span_ratio", 1.0),
            max_symbols=split.get("max_symbols"),
            deterministic=deterministic,
            seed=data.seed,
            normalize=True,
            warmup=data.mmap_warmup,
            **daily,
        )

    def _collate_fn(self):
        return series_collate_fn if self.data_format == "series" else CrossSectionPanelDataset.collate_fn

    def _build_dataloader(self):
        data = self.config.data
        micro = data.micro_batch_size_per_gpu
        assert data.train_batch_size % (self.world_size * micro) == 0, (
            f"train_batch_size {data.train_batch_size} must be divisible by world size {self.world_size} x "
            f"micro_batch_size_per_gpu {micro}"
        )
        self.grad_accum = data.train_batch_size // (self.world_size * micro)
        assert data.lookback % self.model_config.patch_len == 0, "data.lookback must be a multiple of patch_len"

        self.train_dataset = self._build_dataset(data.train, deterministic=False)
        print_rank_0(f"Train samples: {len(self.train_dataset)}, lookahead {self.model_config.lookahead}")
        self.train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True,
            seed=data.seed,
            drop_last=True,
        )
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=micro,
            sampler=self.train_sampler,
            num_workers=data.num_workers,
            pin_memory=self.device.type == "cuda",
            drop_last=True,
            persistent_workers=data.num_workers > 0,
            collate_fn=self._collate_fn(),
        )
        assert len(self.train_dataloader) >= self.grad_accum, "not enough train samples for one optimizer step"

        self.val_dataloader = None
        if data.get("val") is not None and data.val.get("prefix"):
            self.val_dataset = self._build_dataset(data.val, deterministic=True)
            print_rank_0(f"Val samples: {len(self.val_dataset)}")
            self.val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=micro,
                sampler=DistributedSampler(
                    self.val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False
                ),
                num_workers=data.num_workers,
                pin_memory=self.device.type == "cuda",
                collate_fn=self._collate_fn(),
            )

    def _build_optimizer(self):
        optim_config = self.config.optim
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (no_decay if param.ndim < 2 or any(k in name for k in NO_DECAY_NAMES) else decay).append(param)
        groups = [
            {"params": decay, "weight_decay": optim_config.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        assert optim_config.type == "adamw", f"Unknown optimizer type: {optim_config.type}"
        self.optimizer = optim.AdamW(groups, lr=optim_config.lr, betas=tuple(optim_config.betas))

        self.total_training_steps = self.config.data.total_training_steps
        if optim_config.get("warmup_steps_ratio") is not None:
            num_warmup_steps = int(self.total_training_steps * optim_config.warmup_steps_ratio)
        else:
            num_warmup_steps = optim_config.warmup_steps
        scheduler = optim_config.get("lr_scheduler", "cosine")
        if scheduler == "cosine":
            schedule_fn = get_cosine_schedule_with_warmup
        elif scheduler == "wsd":
            schedule_fn = get_wsd_schedule_with_warmup
        else:
            raise ValueError(f"Unknown lr scheduler: {scheduler}")
        self.lr_scheduler = schedule_fn(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.total_training_steps,
            min_lr_ratio=optim_config.get("min_lr_ratio", 0.0),
        )

    def ema_momentum(self, step: int) -> float | None:
        """``None`` for models without an EMA target encoder (``qtsmmae``, ``qtsmcpc`` with ``target: online``) or
        with frozen encoders."""
        if getattr(self.model, "target_encoder", None) is None or getattr(self.model_config, "freeze_encoder", False):
            return None
        start, end = self.model_config.ema_momentum, self.model_config.ema_momentum_end
        return start + (end - start) * min(step / self.total_training_steps, 1.0)

    def condition_ramp(self, step: int) -> float | None:
        """Multiplier of ``condition_weight`` at ``step`` (``model.condition_ramp_start`` / ``_steps``); ``None``
        without a schedule."""
        start = getattr(self.model_config, "condition_ramp_start", 0)
        steps = getattr(self.model_config, "condition_ramp_steps", 0)
        if start == 0 and steps == 0:
            return None
        return min(max((step - start) / steps, 0.0), 1.0) if steps > 0 else float(step >= start)

    def _forward(self, batch: dict[str, torch.Tensor], diagnostics: bool):
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if self.use_bf16 else nullcontext()
        with autocast:
            return self.ddp_model(
                context=batch["context"],
                future=batch["future"],
                symbol_mask=batch["symbol_mask"],
                mean=batch.get("mean"),  # per-window normalization statistics, for config.scale_token
                std=batch.get("std"),
                daily=batch.get("daily"),  # daily bars before the decision date, for model.daily_lookback
                timestamp=batch.get("timestamp"),  # decision times, for model.condition_placebo
                return_diagnostics=diagnostics,
            )

    def _all_reduce_mean(self, metrics: dict[str, torch.Tensor]) -> dict[str, float]:
        keys = sorted(metrics)
        if not keys:
            return {}
        values = torch.stack([metrics[k].float().to(self.device) for k in keys])
        dist.all_reduce(values)
        return dict(zip(keys, (values / self.world_size).tolist(), strict=False))

    def training_step(self, micro_batches: list[dict], step: int) -> dict[str, float]:
        start_time = time.time()
        self.ddp_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        diagnostics = step % self.config.trainer.diag_freq == 0
        ramp = self.condition_ramp(step)
        if ramp is not None:
            self.model.condition_ramp = ramp

        sums: dict[str, torch.Tensor] = {}
        counts: dict[str, int] = {}
        for i, batch in enumerate(micro_batches):
            sync = nullcontext() if i == len(micro_batches) - 1 else self.ddp_model.no_sync()
            with sync:
                output = self._forward(batch, diagnostics=diagnostics and i == 0)
                (output.loss / len(micro_batches)).backward()
            for k, v in {"loss": output.loss.detach(), **output.metrics}.items():
                sums[k] = sums.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1

        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.optim.clip_grad)
        momentum = self.ema_momentum(step)
        if torch.isfinite(grad_norm):
            self.optimizer.step()
            if momentum is not None:
                self.model.update_target_encoder(momentum)
        else:
            print(f"WARN: in rank {self.rank}, grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad(set_to_none=True)
        self.lr_scheduler.step()

        metrics = self._all_reduce_mean({k: sums[k] / counts[k] for k in sums} | {"grad_norm": grad_norm})
        metrics = {f"train/{k}": v for k, v in metrics.items()}
        if momentum is not None:
            metrics["train/ema_momentum"] = momentum
        if ramp is not None:
            metrics["train/condition_weight"] = self.model_config.condition_weight * ramp
        metrics.update(
            {
                "train/lr": self.lr_scheduler.get_last_lr()[0],
                "train/symbols_per_panel": float(sum(b["symbol_mask"].sum() for b in micro_batches))
                / sum(len(b["symbol_mask"]) for b in micro_batches),
                "train/step_time(s)": time.time() - start_time,
            }
        )
        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.ddp_model.eval()
        sums: dict[str, torch.Tensor] = {}
        num_batches = 0
        for batch in self.val_dataloader:
            output = self._forward(batch, diagnostics=True)
            for k, v in {"loss": output.loss, **output.metrics}.items():
                sums[k] = sums.get(k, 0.0) + v
            num_batches += 1
        metrics = self._all_reduce_mean({k: v / max(num_batches, 1) for k, v in sums.items()})
        return {f"val/{k}": v for k, v in metrics.items()}

    def save_checkpoint(self, step: int):
        path = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")
        if self.rank == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path)
        dist.barrier()

    def fit(self):
        config = self.config
        tracking = None
        if self.rank == 0:
            tracking = Tracking(
                project_name=config.trainer.project_name,
                experiment_name=config.trainer.experiment_name,
                default_backend=config.trainer.logger,
                config=OmegaConf.to_container(config, resolve=True),
            )

        global_step, epoch = 0, 0
        last_val_metrics = None
        progress = tqdm(total=self.total_training_steps, desc="Train", disable=self.rank != 0)
        while global_step < self.total_training_steps:
            self.train_sampler.set_epoch(epoch)
            micro_batches = []
            for batch in self.train_dataloader:
                micro_batches.append(batch)
                if len(micro_batches) < self.grad_accum:
                    continue
                global_step += 1
                metrics = self.training_step(micro_batches, global_step)
                micro_batches = []
                progress.update(1)

                is_last_step = global_step >= self.total_training_steps
                if self.val_dataloader is not None and (is_last_step or global_step % config.trainer.test_freq == 0):
                    last_val_metrics = self.validate()
                    metrics.update(last_val_metrics)
                if tracking is not None:
                    tracking.log(data=metrics, step=global_step)
                if is_last_step or global_step % config.trainer.save_freq == 0:
                    self.save_checkpoint(global_step)
                if is_last_step:
                    break
            epoch += 1
        progress.close()
        print_rank_0(f"Final validation metrics: {last_val_metrics}")


def run(config):
    # a node with a broken driver lists GPUs in nvidia-smi but fails CUDA init; training would silently fall to CPU
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available (check torch.cuda.is_available() on this node)")
    dist.init_process_group("nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print_rank_0(OmegaConf.to_yaml(config))
    trainer = PanelJepaTrainer(config, local_rank=local_rank)
    trainer.fit()
    dist.destroy_process_group()


@hydra.main(config_path="../configs/panel", config_name="qtsmfjepa-panel-ohlcv-10m", version_base=None)
def main(config):
    from contextlib import redirect_stdout

    if config.get("timestamp") is None:
        config.timestamp = int(time.time())
    config.trainer.experiment_name = f"{config.timestamp}"
    prefix = "" if config.trainer.model_path is None else "continue-"
    if config.trainer.default_local_dir is None:
        config.trainer.default_local_dir = (
            f"./experiments/{prefix}{config.trainer.project_name}-{config.trainer.experiment_name}"
        )
    os.makedirs(config.trainer.default_local_dir, exist_ok=True)
    os.environ["TENSORBOARD_DIR"] = f"{config.trainer.default_local_dir}/tensorboard"
    print(f"Experiment dir: {config.trainer.default_local_dir}")
    if int(os.environ.get("RANK", 0)) == 0:
        OmegaConf.save(config, f"{config.trainer.default_local_dir}/config.yaml")

    random.seed(config.trainer.seed)
    np.random.seed(config.trainer.seed)
    torch.manual_seed(config.trainer.seed)

    with open(f"{config.trainer.default_local_dir}/train.log", "a") as fout, redirect_stdout(fout):
        run(config)


if __name__ == "__main__":
    main()
