from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from typing import Any, cast

import hydra
import torch
import torch.nn.functional as F
from anydataset.types import AudioItem, AudioView, Modality, Role
from anytrain.framework.flow_matching import ContinuousFlowRuntime, ODESampler
from anytrain.lightning import DebugCallback
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, DistributedSampler, Subset

from speech_to_speech.callback import DistributedContract
from speech_to_speech.callback.logging import CodecOracleLogger, event, stage
from speech_to_speech.loss.flow_matching import AcousticFlowLoss
from speech_to_speech.loss.types import LossItem
from speech_to_speech.model.acoustic import AcousticFlowDecoder
from zhuyin.datasets.wmt19_tts import wmt19_tts_codec


class LongCatOracle(pl.LightningModule):
    def __init__(
        self,
        codebook: Tensor,
        feature_dim: int,
        *,
        initialization: str,
        seed: int,
        dequantize: Callable[[Tensor], Tensor],
        flow: ContinuousFlowRuntime,
        learning_rate: float,
        weight_decay: float,
        target_mean: Tensor,
        target_std: Tensor,
    ) -> None:
        super().__init__()
        weight = embedding_weight(codebook, initialization, seed=seed)
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)
        self.decoder = AcousticFlowDecoder(weight.size(-1), feature_dim)
        self.flow = flow
        self.objective = AcousticFlowLoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.target_mean = nn.Buffer(target_mean)
        self.target_std = nn.Buffer(target_std)
        self.dequantize = dequantize
        self._current: LossItem | None = None
        self._logged_dequantize = False

    def condition(self, semantic_codes: Tensor) -> Tensor:
        return self.embedding(semantic_codes)

    def target(
        self,
        acoustic_codes: Tensor,
        *,
        normalize: bool,
        log_first: bool = True,
    ) -> Tensor:
        if log_first and not self._logged_dequantize:
            with stage(
                "train.first_dequantize",
                code_shape=list(acoustic_codes.shape),
            ):
                target = self.dequantize(acoustic_codes).float()
            self._logged_dequantize = True
        else:
            target = self.dequantize(acoustic_codes).float()
        if not normalize:
            return target
        return (target - self.target_mean) / self.target_std

    def training_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> dict[str, Any]:
        del batch_idx
        codes = batch["codes"]
        mask = batch["mask"]
        safe_codes = codes.masked_fill(~mask[..., None], 0)
        semantic_codes = safe_codes[..., 0]
        acoustic_codes = safe_codes[..., 1:]
        condition = self.condition(semantic_codes)
        target = self.target(acoustic_codes, normalize=True)
        target = target.masked_fill(~mask[..., None], 0)
        item = self.objective(self.decoder, condition, target, mask, self.flow)
        self._current = item
        loss = item.loss.mean()
        self.log(
            "train/flow_loss", loss, on_step=True, prog_bar=True, sync_dist=True
        )
        self.log("train/batch_size", float(codes.size(0)), on_step=True, sync_dist=True)
        self.log(
            "train/valid_frames", mask.sum().float(), on_step=True, sync_dist=True
        )
        return {"loss": loss, "flow_matching": item}

    def on_after_backward(self) -> None:
        self._current = None

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [*self.embedding.parameters(), *self.decoder.parameters()],
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @torch.no_grad()
    def sample(self, semantic_codes: Tensor, *, seed: int) -> Tensor:
        condition = self.condition(semantic_codes)
        generator = torch.Generator(device=condition.device).manual_seed(seed)
        noise = torch.randn(
            (*condition.shape[:2], self.decoder.latent_dim),
            device=condition.device,
            dtype=condition.dtype,
            generator=generator,
        )
        normalized = self.flow.sample(
            self.decoder,
            noise,
            condition=condition,
        ).final
        return normalized * self.target_std + self.target_mean


class UnifiedTokenOracle(pl.LightningModule):
    def __init__(
        self,
        codebook: Tensor,
        max_length: int,
        *,
        initialization: str,
        seed: int,
        layers: int,
        heads: int,
        feedforward_dim: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        weight = embedding_weight(codebook, initialization, seed=seed)
        special = random_weight(
            weight.new_empty((1, weight.size(-1))),
            weight,
            seed=seed + 1,
        )
        self.vocab_size = weight.size(0)
        self.bos_id = self.vocab_size
        self.embedding = nn.Embedding.from_pretrained(
            torch.cat((weight, special), dim=0),
            freeze=False,
        )
        self.position = nn.Embedding(max_length, weight.size(-1))
        layer = nn.TransformerEncoderLayer(
            d_model=weight.size(-1),
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = nn.Linear(weight.size(-1), self.vocab_size, bias=False)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(self, codes: Tensor, frame_mask: Tensor | None = None) -> Tensor:
        if frame_mask is None:
            frame_mask = torch.ones_like(codes, dtype=torch.bool)
        safe_codes = codes.masked_fill(~frame_mask, 0)
        inputs = torch.cat(
            (
                codes.new_full((codes.size(0), 1), self.bos_id),
                safe_codes[:, :-1],
            ),
            dim=1,
        )
        positions = torch.arange(codes.size(1), device=codes.device)
        hidden = self.embedding(inputs) + self.position(positions)[None]
        causal_mask = torch.ones(
            (codes.size(1), codes.size(1)),
            dtype=torch.bool,
            device=codes.device,
        ).triu(diagonal=1)
        return self.head(
            self.backbone(
                hidden,
                mask=causal_mask,
                src_key_padding_mask=~frame_mask,
                is_causal=True,
            )
        )

    def training_step(self, batch: Mapping[str, Tensor], batch_idx: int) -> Tensor:
        del batch_idx
        codes = batch["codes"]
        frame_mask = batch["mask"]
        logits = self(codes, frame_mask)
        labels = codes.masked_fill(~frame_mask, -100)
        loss = F.cross_entropy(logits.transpose(1, 2), labels)
        accuracy = (
            logits.argmax(dim=-1).eq(codes).masked_select(frame_mask).float().mean()
        )
        self.log(
            "train/token_loss", loss, on_step=True, prog_bar=True, sync_dist=True
        )
        self.log(
            "train/token_accuracy",
            accuracy,
            on_step=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("train/batch_size", float(codes.size(0)), on_step=True, sync_dist=True)
        self.log(
            "train/valid_frames",
            frame_mask.sum().float(),
            on_step=True,
            sync_dist=True,
        )
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    @torch.no_grad()
    def predict(self, codes: Tensor) -> Tensor:
        return self(codes).argmax(dim=-1)


class PreparedCodeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: DictConfig,
        codec: DictConfig,
        *,
        output_dir: Path,
        seed: int,
    ) -> None:
        super().__init__()
        self.data = data
        self.codec = codec
        self.output_dir = output_dir
        self.seed = seed
        self.dataset: Any | None = None
        self.sampler: DistributedSampler[Any] | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self.dataset is not None:
            return
        dataset = wmt19_tts_codec(
            codec=str(self.codec.name),
            root=path(self.data.root),
            split=str(self.data.split),
        )
        sample_limit = self.data.sample_limit
        if sample_limit is not None:
            dataset = Subset(dataset, range(min(int(sample_limit), len(dataset))))
        self.dataset = dataset

    def train_dataloader(self):
        if self.dataset is None:
            raise RuntimeError("PreparedCodeDataModule.setup() must run first.")
        sampler = None
        if self.trainer.world_size > 1:
            sampler = DistributedSampler(
                self.dataset,
                num_replicas=self.trainer.world_size,
                rank=self.trainer.global_rank,
                shuffle=True,
                seed=self.seed,
                drop_last=False,
            )
        self.sampler = sampler
        loader = DataLoader(
            self.dataset,
            batch_size=int(self.data.batch_size),
            sampler=sampler,
            shuffle=sampler is None,
            num_workers=int(self.data.num_workers),
            pin_memory=bool(self.data.pin_memory),
            persistent_workers=(
                bool(self.data.persistent_workers) and int(self.data.num_workers) > 0
            ),
            collate_fn=partial(code_collate, codec=self.codec, data=self.data),
        )
        if not bool(self.data.lba.enabled):
            return loader
        from lba import LBA

        return LBA(
            loader,
            len_fn=partial(code_length, codec=self.codec, data=self.data),
            max_padded_length=round(
                float(self.data.lba.max_batch_seconds)
                * float(self.codec.frame_rate)
            ),
            max_padding_ratio=float(self.data.lba.max_padding_ratio),
            prefetch_batches=int(self.data.lba.prefetch_batches),
            planner_mode=str(self.data.lba.planner_mode),
            drop_last_flush=bool(self.data.lba.drop_last_flush),
            log_dir=self.output_dir / "lba",
        )


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    run(config)


def run(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    pl.seed_everything(int(config.train.seed), workers=True)
    device = process_device()
    event(
        "run",
        "start",
        codec=str(config.codec.name),
        objective=str(config.codec.objective),
        initialization=str(config.init.name),
    )
    output_dir = Path(str(config.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    codes = load_codes(config.data, config.codec)
    codec = build_codec(config.codec, device=device)
    if str(config.codec.objective) == "flow":
        module, metadata = build_flow(config, codec, codes)
    elif str(config.codec.objective) == "token":
        module, metadata = build_token(config, codec, codes)
    else:
        raise ValueError(f"unsupported codec oracle objective: {config.codec.objective}")

    callback = CodecOracleLogger(
        objective=str(config.codec.objective),
        codec=codec,
        codes=codes,
        output_dir=output_dir,
        sample_rate=int(codec.sample_rate),
        seed=int(config.train.seed),
        sample_every_n_steps=int(config.logging.sample_every_n_steps),
        histogram_every_n_steps=int(config.logging.histogram_every_n_steps),
        save_audio=bool(config.logging.save_audio),
        metadata=metadata,
    )
    callbacks: list[Callback] = [
        callback,
        DistributedContract(int(config.trainer.expected_world_size)),
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="step-{step}",
            save_last=True,
            save_top_k=0,
        ),
    ]
    if bool(config.logging.nonfinite_check):
        callbacks.append(DebugCallback())
    with stage("logger.build", logger=str(config.logging.logger)):
        logger = build_logger(config.logging, output_dir)
        logger.log_hyperparams(
            cast(dict[str, Any], OmegaConf.to_container(config, resolve=True))
        )
    trainer = pl.Trainer(
        accelerator=str(config.trainer.accelerator),
        devices=config.trainer.devices,
        precision=str(config.trainer.precision),
        max_steps=int(config.train.max_steps),
        max_epochs=int(config.trainer.max_epochs),
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        enable_checkpointing=bool(config.trainer.enable_checkpointing),
        gradient_clip_val=float(config.trainer.gradient_clip_val),
        default_root_dir=str(output_dir),
        logger=logger,
        callbacks=callbacks,
        strategy=str(config.trainer.strategy),
        use_distributed_sampler=bool(config.trainer.use_distributed_sampler),
    )
    with stage("trainer.fit", objective=str(config.codec.objective)):
        if bool(config.data.lba.enabled):
            trainer.fit(
                module,
                datamodule=PreparedCodeDataModule(
                    config.data,
                    config.codec,
                    output_dir=output_dir,
                    seed=int(config.train.seed),
                ),
            )
        else:
            trainer.fit(
                module,
                train_dataloaders=codes_loader(
                    codes,
                    objective=str(config.codec.objective),
                ),
            )


def load_codes(data: DictConfig, codec: DictConfig) -> Tensor:
    name = str(codec.name)
    with stage(
        "dataset.load",
        codec=name,
        split=str(data.split),
        sample_index=int(data.sample_index),
    ):
        dataset = wmt19_tts_codec(
            codec=name,
            root=path(data.root),
            split=str(data.split),
        )
        codes = prepared_codes(dataset[int(data.sample_index)], codec=codec, data=data)
    event(
        "dataset.sample",
        "ready",
        codec=name,
        code_shape=list(codes.shape),
        code_min=int(codes.min()),
        code_max=int(codes.max()),
    )
    return codes


def prepared_codes(
    sample: Mapping[Any, Any],
    *,
    codec: DictConfig,
    data: DictConfig,
) -> Tensor:
    item = sample[(Role.TARGET, Modality.AUDIO)]
    if not isinstance(item, AudioItem):
        raise TypeError("WMT19 target audio must be an AudioItem.")
    codes = item.views[AudioView(str(codec.view))]
    if not isinstance(codes, Tensor) or codes.dim() != 2:
        raise ValueError("prepared codec codes must have shape [frame, codebook].")
    frames = min(
        codes.size(0),
        round(float(data.max_seconds) * float(codec.frame_rate)),
    )
    codes = codes[:frames].long().contiguous()
    if codes.size(0) == 0:
        raise ValueError("selected prepared codec sequence is empty.")
    return codes


def code_length(
    sample: Mapping[Any, Any],
    *,
    codec: DictConfig,
    data: DictConfig,
) -> int:
    return prepared_codes(sample, codec=codec, data=data).size(0)


def code_collate(
    samples: list[Mapping[Any, Any]],
    *,
    codec: DictConfig,
    data: DictConfig,
) -> dict[str, Tensor]:
    values = [prepared_codes(sample, codec=codec, data=data) for sample in samples]
    codes = pad_sequence(values, batch_first=True, padding_value=-1)
    mask = (codes >= 0).all(dim=-1)
    if str(codec.objective) == "token":
        codes = codes[..., 0]
    return {"codes": codes, "mask": mask}


def build_codec(config: DictConfig, *, device: torch.device) -> Any:
    name = str(config.name)
    with stage("codec.load", codec=name):
        if name == "longcat":
            from anytrain.codec.longcat import LongCat

            codec = LongCat.from_pretrained(
                cache_dir=path(config.cache_dir),
                decoder=str(config.decoder),
                device=device,
                local_files_only=bool(config.local_files_only),
            )
        elif name == "unicodec":
            from anytrain.codec.unicodec import UniCodec

            codec = UniCodec.from_pretrained(
                cache_dir=path(config.cache_dir),
                device=device,
                domain=str(config.domain),
                bandwidth_id=int(config.bandwidth_id),
                local_files_only=bool(config.local_files_only),
            )
        else:
            raise ValueError(f"unsupported codec oracle codec: {name}")
    return codec


@torch.no_grad()
def build_flow(
    config: DictConfig,
    codec: Any,
    codes: Tensor,
) -> tuple[LongCatOracle, dict[str, Any]]:
    if codes.size(-1) < 2:
        raise ValueError("flow oracle requires semantic and acoustic codebooks.")
    semantic_codes = codes[:, 0]
    acoustic_codes = codes[:, 1:]
    with stage(
        "codec.dequantize_probe",
        codec=str(config.codec.name),
        code_shape=list(acoustic_codes.shape),
    ):
        target = codec.acoustic_codes_to_features(
            acoustic_codes.unsqueeze(0).to(codec.device)
        ).float()
    normalized, mean, std = normalize_target(
        target,
        enabled=bool(config.train.normalize_features),
    )
    del normalized
    codebook = codec.semantic_codebook.detach().float()
    flow = ContinuousFlowRuntime(
        sampler=ODESampler(
            method=str(config.train.flow.method),
            nfe=int(config.train.flow.nfe),
            num_steps=int(config.train.flow.num_steps),
            return_intermediates=False,
        )
    )
    module = LongCatOracle(
        codebook.cpu(),
        target.size(-1),
        initialization=str(config.init.name),
        seed=int(config.train.seed),
        dequantize=codec.acoustic_codes_to_features,
        flow=flow,
        learning_rate=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
        target_mean=mean.cpu(),
        target_std=std.cpu(),
    )
    metadata = common_metadata(config, codes, codebook) | {
        "semantic_frames": int(semantic_codes.size(0)),
        "feature_dim": int(target.size(-1)),
        "feature_mean": float(target.mean()),
        "feature_std": float(target.std(correction=0)),
    }
    return module, metadata


@torch.no_grad()
def build_token(
    config: DictConfig,
    codec: Any,
    codes: Tensor,
) -> tuple[UnifiedTokenOracle, dict[str, Any]]:
    if codes.size(-1) != 1:
        raise ValueError("unified token oracle requires exactly one codebook.")
    vocab_size = int(codec.codebook_sizes[0])
    ids = torch.arange(vocab_size, device=codec.device).view(1, vocab_size, 1)
    with stage("codec.codebook_extract", codec=str(config.codec.name), rows=vocab_size):
        codebook = codec.codes_to_features(ids)[0].detach().float()
    module = UnifiedTokenOracle(
        codebook.cpu(),
        round(float(config.data.max_seconds) * float(config.codec.frame_rate)),
        initialization=str(config.init.name),
        seed=int(config.train.seed),
        layers=int(config.token.layers),
        heads=int(config.token.heads),
        feedforward_dim=int(config.token.feedforward_dim),
        dropout=float(config.token.dropout),
        learning_rate=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
    )
    return module, common_metadata(config, codes, codebook)


def codes_loader(codes: Tensor, *, objective: str) -> DataLoader:
    if objective == "flow":
        value = codes
    elif objective == "token":
        value = codes[:, 0]
    else:
        raise ValueError(f"unsupported codec oracle objective: {objective}")
    sample = {
        "codes": value,
        "mask": torch.ones(value.size(0), dtype=torch.bool),
    }
    return DataLoader([sample], batch_size=1, num_workers=0)


def embedding_weight(codebook: Tensor, initialization: str, *, seed: int) -> Tensor:
    if codebook.dim() != 2 or not torch.is_floating_point(codebook):
        raise ValueError("codec codebook must have shape [vocab, dim] and floating dtype.")
    if initialization == "codec":
        return codebook.clone()
    if initialization == "random":
        return random_weight(torch.empty_like(codebook), codebook, seed=seed)
    raise ValueError(f"unsupported audio embedding initialization: {initialization}")


def random_weight(output: Tensor, reference: Tensor, *, seed: int) -> Tensor:
    generator = torch.Generator(device=output.device).manual_seed(seed)
    output.normal_(
        mean=float(reference.mean()),
        std=float(reference.std(correction=0)),
        generator=generator,
    )
    return output


def normalize_target(target: Tensor, *, enabled: bool) -> tuple[Tensor, Tensor, Tensor]:
    if not enabled:
        return target, target.new_zeros((1, 1, target.size(-1))), target.new_ones(
            (1, 1, target.size(-1))
        )
    mean = target.mean(dim=(0, 1), keepdim=True)
    std = target.std(dim=(0, 1), correction=0, keepdim=True).clamp_min(1e-5)
    return (target - mean) / std, mean, std


def common_metadata(
    config: DictConfig,
    codes: Tensor,
    codebook: Tensor,
) -> dict[str, Any]:
    return {
        "codec": str(config.codec.name),
        "objective": str(config.codec.objective),
        "initialization": str(config.init.name),
        "code_shape": list(codes.shape),
        "codebook_shape": list(codebook.shape),
        "codebook_mean": float(codebook.mean()),
        "codebook_std": float(codebook.std(correction=0)),
        "frame_rate": float(config.codec.frame_rate),
        "max_seconds": float(config.data.max_seconds),
    }


def build_logger(config: DictConfig, output_dir: Path):
    name = str(config.logger)
    if name == "tensorboard":
        return TensorBoardLogger(save_dir=str(output_dir), name="tensorboard")
    if name == "csv":
        return CSVLogger(save_dir=str(output_dir), name="csv")
    raise ValueError("logging.logger must be tensorboard or csv.")


def process_device() -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    return device


def path(value: Any) -> Path | None:
    return None if value is None else Path(str(value)).expanduser()

if __name__ == "__main__":
    main()
