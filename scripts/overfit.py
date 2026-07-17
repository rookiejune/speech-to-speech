from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Union, cast

import hydra
import torch
from anydataset.types import Sample as RawSample
from lightning import pytorch as pl
from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from speech_to_speech.callback import StageConfig, StageSwitcher, WorldSizeContract
from speech_to_speech.callback.logging import (
    FlowMatchingLogger,
    GradLogger,
    GradNormLogger,
    OutputsLogger,
    SampleLogger,
    TextRetentionLogger,
)
from speech_to_speech.datamodule import Collator, DataRuntime, ModelBatch
from speech_to_speech.generation.batch import requests_from_batch
from speech_to_speech.loss import (
    FlowObjective,
    Outputs,
    RVQObjective,
    TokenObjective,
    WavLMTeacher,
    loss_items,
)
from speech_to_speech.model import (
    AcousticType,
    DecoderConfig,
    FlowRepaConfig,
    TokenModel,
    SpeechToSpeechFlowModel,
    SpeechToSpeechRVQModel,
)
from speech_to_speech.pl_module.protocol import (
    FlowCompositionModel,
    RVQCompositionModel,
)
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.pl_module import SpeechToSpeech
from speech_to_speech.reporting import window_summary
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import init_runtime
from speech_to_speech.runtime.types import Codec
from speech_to_speech.task import Task
if __package__:
    from ._acoustic_evaluation import evaluate
else:
    from _acoustic_evaluation import evaluate


class LossSummary(Callback):
    def __init__(self) -> None:
        super().__init__()
        self.values: dict[str, list[float]] = {}

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        if not isinstance(outputs, Mapping):
            return
        typed_outputs = cast(Outputs, outputs)
        self._append("loss", typed_outputs["loss"])
        for name, item in loss_items(typed_outputs):
            self._append(name, item.loss)

    def report(
        self,
        window: int = 20,
    ) -> dict[str, dict[str, float | int | None]]:
        report = {}
        for name, values in self.values.items():
            report[name] = window_summary(values, window)
        return report

    def _append(self, name: str, value: Tensor) -> None:
        self.values.setdefault(name, []).append(float(value.detach().float().mean()))


class AcousticEvaluation(Callback):
    def __init__(
        self,
        model: SpeechToSpeechFlowModel | SpeechToSpeechRVQModel,
        batch: ModelBatch,
        codec: Codec,
        output_dir: Path,
        *,
        every_n_steps: int,
        seeds: Sequence[int],
    ) -> None:
        super().__init__()
        self.model = model
        self.batch = batch
        self.codec = codec
        self.path = output_dir / "evaluation.json"
        self.every_n_steps = every_n_steps
        self.seeds = tuple(seeds)
        self.values: dict[int, dict[str, float]] = {}

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module
        if trainer.is_global_zero:
            self.evaluate(trainer, 0)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, outputs, batch, batch_idx
        if (
            trainer.is_global_zero
            and trainer.global_step % self.every_n_steps == 0
        ):
            self.evaluate(trainer, trainer.global_step)

    def on_train_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        del pl_module
        if trainer.is_global_zero:
            self.evaluate(trainer, trainer.global_step)

    def evaluate(self, trainer: pl.Trainer, step: int) -> None:
        if step in self.values:
            return
        metrics = evaluate(self.model, self.batch, self.codec, seeds=self.seeds)
        self.values[step] = metrics
        if trainer.logger is not None:
            trainer.logger.log_metrics(
                {f"evaluation/{name}": value for name, value in metrics.items()},
                step=step,
            )
        self.path.write_text(
            json.dumps(
                {str(key): value for key, value in self.values.items()},
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


class FixedDataModule(LightningDataModule):
    def __init__(
        self,
        codec: str,
        runtime: DataRuntime,
        task_weights: Mapping[Task, float],
        sample_index: int,
        *,
        root: Path | None = None,
        split: str = "train",
    ) -> None:
        super().__init__()
        self.codec = codec
        self.runtime = runtime
        self.collator = Collator(runtime, task_weights)
        self.sample_index = sample_index
        self.root = root
        self.split = split
        self._dataset: Dataset[RawSample] | None = None
        self._training: Subset[RawSample] | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._dataset is not None:
            return
        if self.codec != self.runtime.codec_name:
            raise ValueError(
                "fixed datamodule and runtime must use the same codec: "
                f"{self.codec!r} != {self.runtime.codec_name!r}."
            )
        from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

        self._dataset = cast(
            Dataset[RawSample],
            cast(
                object,
                wmt19_tts_codec(
                    codec=self.codec,
                    root=self.root,
                    split=self.split,
                ),
            ),
        )
        self._training = Subset(self._dataset, [self.sample_index])

    def set_task_weights(self, task_weights: Mapping[Task, float]) -> None:
        self.collator.set_task_weights(task_weights)

    def train_samples(self, indices: Sequence[int]) -> list[RawSample]:
        if self._dataset is None:
            raise RuntimeError("FixedDataModule.setup() must run before reading samples.")
        return [self._dataset[index] for index in indices]

    def train_dataloader(self) -> Iterable[ModelBatch]:
        if self._training is None:
            raise RuntimeError("FixedDataModule.setup() must run before training.")
        return DataLoader(
            self._training,
            batch_size=1,
            num_workers=0,
            collate_fn=self.collator,
        )


@hydra.main(version_base=None, config_path="../configs", config_name="overfit")
def main(config: DictConfig) -> None:
    run(config)


def run(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    output_dir = Path(str(config.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(int(config.train.seed), workers=True)
    rt = init_runtime(runtime_config(config))
    layout = rt.layout
    codec = rt.codec
    backbone = rt.backbone
    flow_matching = rt.flow_matching
    task = Task(str(config.task))
    acoustic_type = AcousticType(str(config.acoustic.type))
    datamodule = FixedDataModule(
        str(config.codec.name),
        rt,
        {task: 1.0},
        int(config.data.sample_index),
        root=(
            None
            if config.data.root is None
            else Path(str(config.data.root)).expanduser()
        ),
        split=str(config.data.split),
    )

    repa_teacher = None
    repa_weight = None
    if acoustic_type is AcousticType.FLOW:
        repa_weight = config.acoustic.repa.weight
    if repa_weight is not None:
        repa_teacher = WavLMTeacher(
            codec,
            checkpoint=str(config.acoustic.repa.teacher_checkpoint),
            layer=int(config.acoustic.repa.teacher_layer),
            device=backbone.get_input_embeddings().weight.device,
        )
    torch.manual_seed(int(config.train.seed))
    hidden_dim = config.acoustic.decoder.hidden_dim
    decoder = DecoderConfig(
        hidden_dim=None if hidden_dim is None else int(hidden_dim),
        layers=int(config.acoustic.decoder.layers),
        heads=int(config.acoustic.decoder.heads),
        ffn_ratio=int(config.acoustic.decoder.ffn_ratio),
    )
    repa = (
        None
        if repa_teacher is None
        else FlowRepaConfig(
            feature_dim=repa_teacher.feature_dim,
            student_layer=(
                None
                if config.acoustic.repa.student_layer is None
                else int(config.acoustic.repa.student_layer)
            ),
        )
    )
    uses_acoustic_decoder = bool(codec.acoustic_codebook_sizes)
    module_config = ModuleConfig(
        learning_rate=float(config.optimizer.learning_rate),
        weight_decay=float(config.optimizer.weight_decay),
    )
    evaluation: AcousticEvaluation | None = None
    if not uses_acoustic_decoder:
        token_model = TokenModel(runtime=rt)
        objective = TokenObjective(layout)
        module = SpeechToSpeech(
            module_config,
            model=token_model,
            objective=objective,
        )
        model = token_model
    elif acoustic_type is AcousticType.FLOW:
        flow_model = SpeechToSpeechFlowModel(
            runtime=rt,
            decoder=decoder,
            repa=repa,
        )
        objective = FlowObjective(
            layout,
            flow_matching,
            repa=(
                None
                if repa_weight is None or repa_teacher is None
                else {
                    "weight": float(repa_weight),
                    "teacher": repa_teacher,
                }
            ),
        )
        module = SpeechToSpeech[FlowCompositionModel](
            module_config,
            model=flow_model,
            objective=objective,
        )
        model = flow_model
    else:
        rvq_model = SpeechToSpeechRVQModel(
            runtime=rt,
            decoder=decoder,
        )
        objective = RVQObjective(layout)
        module = SpeechToSpeech[RVQCompositionModel](
            module_config,
            model=rvq_model,
            objective=objective,
        )
        model = rvq_model
    if uses_acoustic_decoder:
        datamodule.setup("fit")
        batch = next(iter(datamodule.train_dataloader()))
        evaluation = AcousticEvaluation(
            cast(
                Union[SpeechToSpeechFlowModel, SpeechToSpeechRVQModel],
                model,
            ),
            batch,
            codec,
            output_dir,
            every_n_steps=max(1, int(config.train.max_steps) // 5),
            seeds=range(4),
        )
    summary = LossSummary()
    if acoustic_type is AcousticType.FLOW:
        loss_pair = (
            ("flow_matching", "repa")
            if repa_weight is not None
            else ("token", "flow_matching")
        )
    else:
        loss_pair = ("token", "rvq")
    callbacks = cast(list[Callback], [
        OutputsLogger(),
        GradNormLogger(),
        TextRetentionLogger(
            {
                "zh_en": {
                    "instruction": "Translate into English: 昨晚的暴雨导致三趟列车晚点。",
                    "reference": "Last night's heavy rain delayed three trains.",
                },
            },
            every_n_steps=1,
            max_new_tokens=8,
        ),
        StageSwitcher(
            StageConfig(
                task_weights_by_stage=[{task: 1.0}],
                epoch_milestones=[],
            )
        ),
        summary,
    ])
    if bool(config.callbacks.sample.enabled):
        callbacks.insert(
            2,
            SampleLogger(
                [int(config.data.sample_index)],
                every_n_steps=int(config.callbacks.sample.every_n_steps),
            ),
        )
    if evaluation is not None:
        callbacks.append(evaluation)
    if uses_acoustic_decoder:
        callbacks.insert(
            1,
            GradLogger(
                loss_pair,
                "model.acoustic_flow.decoder.input.weight"
                if acoustic_type is AcousticType.FLOW
                else "model.acoustic_decoder.decoder.layers.0.self_attn.q_proj.weight",
                every_n_steps=1,
            ),
        )
    if uses_acoustic_decoder and acoustic_type is AcousticType.FLOW:
        callbacks.insert(1, FlowMatchingLogger(flow_matching, every_n_steps=1))
    trainer = build_trainer(config, output_dir, callbacks)
    trainer.fit(module, datamodule=datamodule)

    if not trainer.is_global_zero:
        return

    if uses_acoustic_decoder:
        if evaluation is None:
            raise RuntimeError("acoustic evaluation is unavailable.")
        generation = evaluate_generation(module, evaluation.batch, codec)
        (output_dir / "generation.json").write_text(
            json.dumps(generation, indent=2, sort_keys=True) + "\n"
        )

    acoustic_decoder_parameters = (
        sum(parameter.numel() for parameter in model.acoustic_flow.decoder.parameters())
        if isinstance(model, SpeechToSpeechFlowModel)
        else sum(parameter.numel() for parameter in model.acoustic_decoder.parameters())
        if isinstance(model, SpeechToSpeechRVQModel)
        else 0
    )
    result = {
        "task": task.value,
        "sample_index": int(config.data.sample_index),
        "max_steps": int(config.train.max_steps),
        "parameters": {
            "total": sum(parameter.numel() for parameter in model.parameters()),
            "trainable": sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            ),
            "acoustic_decoder": acoustic_decoder_parameters,
        },
        "metrics": summary.report(),
    }
    result_path = output_dir / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


def build_trainer(
    config: DictConfig,
    output_dir: Path,
    callbacks: list[Callback],
) -> pl.Trainer:
    callbacks = [
        WorldSizeContract(int(config.trainer.expected_world_size)),
        *callbacks,
    ]
    return pl.Trainer(
        accelerator=str(config.trainer.accelerator),
        devices=config.trainer.devices,
        precision=cast(Any, str(config.trainer.precision)),
        max_steps=int(config.train.max_steps),
        max_epochs=int(config.trainer.max_epochs),
        default_root_dir=str(output_dir),
        logger=build_logger(config.logging, output_dir),
        callbacks=callbacks,
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        enable_checkpointing=bool(config.trainer.enable_checkpointing),
        gradient_clip_val=float(config.trainer.gradient_clip_val),
        strategy=str(config.trainer.strategy),
        use_distributed_sampler=bool(config.trainer.use_distributed_sampler),
    )


@torch.no_grad()
def evaluate_generation(
    module: SpeechToSpeech,
    batch: ModelBatch,
    codec: Codec,
) -> dict[str, Any]:
    device = next(module.parameters()).device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = module.generate(
        requests_from_batch(batch),
        max_new_tokens=64,
        do_sample=False,
    )[0]
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    audio = result["audio"]
    if audio is None:
        raise RuntimeError("acoustic task generation did not return audio.")
    features = audio["features"]
    waveform = audio["waveform"]
    if features is None:
        raise RuntimeError("LongCat generation did not return acoustic features.")
    if not bool(torch.isfinite(features).all() and torch.isfinite(waveform).all()):
        raise RuntimeError("generation returned non-finite acoustic output.")
    duration = waveform.numel() / codec.sample_rate
    return {
        "token_ids": result["response_ids"].detach().cpu().tolist(),
        "feature_shape": list(features.shape),
        "waveform_shape": list(waveform.shape),
        "duration_seconds": duration,
        "elapsed_seconds": elapsed,
        "rtf": elapsed / duration,
        "finite": True,
    }


def runtime_config(config: DictConfig) -> RuntimeConfig:
    audio_tokenizer = config.runtime.audio_tokenizer
    device = torch.device(str(config.runtime.device))
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    return RuntimeConfig(
        codec=str(config.codec.name),
        backbone=str(config.runtime.backbone),
        audio_tokenizer=(
            None if audio_tokenizer is None else str(audio_tokenizer)
        ),
        device=str(device),
        dtype=str(config.runtime.dtype),
        attn_implementation=str(config.runtime.attn_implementation),
        flow_method=str(config.flow.method),
        flow_nfe=int(config.flow.nfe),
        flow_num_steps=int(config.flow.num_steps),
    )


def build_logger(config: DictConfig, output_dir: Path):
    name = str(config.name)
    if name == "tensorboard":
        return TensorBoardLogger(save_dir=str(output_dir), name="tensorboard")
    if name == "csv":
        return CSVLogger(save_dir=str(output_dir), name="csv")
    raise ValueError("logging.name must be tensorboard or csv.")


if __name__ == "__main__":
    main()
