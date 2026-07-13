from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import hydra
from anydataset.types import Sample as RawSample
from lightning import pytorch as pl
from lightning.pytorch import LightningDataModule
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from speech_to_speech.callback import StageConfig, StageSwitcher
from speech_to_speech.callback.logging import (
    FlowMatchingLogger,
    GradLogger,
    GradNormLogger,
    OutputsLogger,
    SampleLogger,
    TextRetentionLogger,
)
from speech_to_speech.datamodule import Collator, ModelBatch, Task
from speech_to_speech.loss import Loss, Outputs, RVQLoss, WavLMTeacher, loss_items
from speech_to_speech.model import Config as ModelConfig
from speech_to_speech.model import SpeechToSpeechFlowModel, SpeechToSpeechRVQModel
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.pl_module import SpeechToSpeech
from speech_to_speech.reporting import window_summary
from speech_to_speech.runtime import Config as RuntimeConfig
from speech_to_speech.runtime import init_runtime


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


class FixedDataModule(LightningDataModule):
    def __init__(
        self,
        codec: str,
        strategy: Mapping[Task, float],
        sample_index: int,
    ) -> None:
        super().__init__()
        self.codec = codec
        self.collator = Collator(strategy)
        self.sample_index = sample_index
        self._dataset: Dataset[RawSample] | None = None
        self._training: Subset[RawSample] | None = None

    def setup(self, stage: str | None = None) -> None:
        del stage
        if self._dataset is not None:
            return
        from zhuyin.datasets.wmt19_tts import wmt19_tts_codec

        self._dataset = cast(
            Dataset[RawSample],
            cast(object, wmt19_tts_codec(codec=self.codec, split="train")),
        )
        self._training = Subset(self._dataset, [self.sample_index])

    def set_strategy(self, strategy: Mapping[Task, float]) -> None:
        self.collator.set_strategy(strategy)

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


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    run(config)


def run(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    output_dir = Path(str(config.output_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(int(config.train.seed), workers=True)
    rt = init_runtime(
        RuntimeConfig(
            codec=str(config.codec.name),
            backbone=str(config.runtime.backbone),
            audio_tokenizer=str(config.runtime.audio_tokenizer),
            device=str(config.runtime.device),
            dtype=str(config.runtime.dtype),
            attn_implementation=str(config.runtime.attn_implementation),
        )
    )
    task = Task(str(config.task))
    datamodule = FixedDataModule(
        str(config.codec.name),
        {task: 1.0},
        int(config.data.sample_index),
    )

    repa_teacher = None
    if config.objective == "rvq" and config.repa.weight is not None:
        raise ValueError("REPA is only defined for the flow objective.")
    if config.repa.weight is not None:
        repa_teacher = WavLMTeacher(
            rt.codec,
            checkpoint=str(config.repa.teacher),
            layer=int(config.repa.layer),
            device=rt.backbone.get_input_embeddings().weight.device,
        )
    model_config = (
        ModelConfig()
        if repa_teacher is None
        else ModelConfig(acoustic_repa_dim=repa_teacher.feature_dim)
    )
    model = (
        SpeechToSpeechFlowModel(model_config, runtime_snapshot=rt)
        if config.objective == "flow"
        else SpeechToSpeechRVQModel(model_config, runtime_snapshot=rt)
    )
    loss = (
        Loss(
            rt.layout,
            rt.flow_matching,
            repa=(
                None
                if config.repa.weight is None or repa_teacher is None
                else {"weight": float(config.repa.weight), "teacher": repa_teacher}
            ),
        )
        if config.objective == "flow"
        else RVQLoss(rt.layout)
    )
    module = SpeechToSpeech(
        ModuleConfig(
            learning_rate=float(config.optimizer.learning_rate),
            weight_decay=float(config.optimizer.weight_decay),
        ),
        model=model,
        loss=loss,
    )
    summary = LossSummary()
    loss_pair = (
        ("flow_matching", "repa")
        if config.repa.weight is not None
        else ("semantic", "flow_matching")
        if config.objective == "flow"
        else ("semantic", "causal_lm")
    )
    callbacks = cast(list[Callback], [
        OutputsLogger(),
        GradLogger(
            loss_pair,
            "model.acoustic_decoder.input.weight"
            if config.objective == "flow"
            else "model.acoustic_decoder.decoder.layers.0.self_attn.q_proj.weight",
            every_n_steps=1,
        ),
        GradNormLogger(),
        SampleLogger([int(config.data.sample_index)], every_n_steps=1),
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
        StageSwitcher(StageConfig(strategies=[{task: 1.0}], milestones=[])),
        summary,
    ])
    if config.objective == "flow":
        callbacks.insert(1, FlowMatchingLogger(rt.flow_matching, every_n_steps=1))
    trainer = pl.Trainer(
        accelerator=str(config.trainer.accelerator),
        devices=config.trainer.devices,
        precision=cast(Any, str(config.trainer.precision)),
        max_steps=int(config.train.max_steps),
        default_root_dir=str(output_dir),
        logger=TensorBoardLogger(save_dir=str(output_dir), name="tensorboard"),
        callbacks=callbacks,
        log_every_n_steps=int(config.trainer.log_every_n_steps),
        enable_checkpointing=bool(config.trainer.enable_checkpointing),
    )
    trainer.fit(module, datamodule=datamodule)

    result = {
        "task": task.value,
        "sample_index": int(config.data.sample_index),
        "max_steps": int(config.train.max_steps),
        "metrics": summary.report(),
    }
    result_path = output_dir / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
if __name__ == "__main__":
    main()
