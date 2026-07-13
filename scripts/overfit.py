from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor

from speech_to_speech.callback import StageConfig, StageSwitcher
from speech_to_speech.callback.logging import (
    FlowMatchingLogger,
    GradLogger,
    GradNormLogger,
    OutputsLogger,
    SampleLogger,
    TextRetentionLogger,
)
from speech_to_speech.datamodule import Config as DataConfig
from speech_to_speech.datamodule import DataModule, Task
from speech_to_speech.loss import Loss, Outputs, WavLMTeacher, loss_items
from speech_to_speech.model import Config as ModelConfig
from speech_to_speech.model import SpeechToSpeechFlowModel
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


def main(argv: Sequence[str] | None = None) -> None:
    args = parser().parse_args(argv)
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    pl.seed_everything(args.seed, workers=True)
    rt = init_runtime(
        RuntimeConfig(
            codec=args.codec,
            backbone=args.backbone,
            audio_tokenizer=args.audio_tokenizer,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn_implementation,
        )
    )
    task = Task(args.task)
    datamodule = DataModule(
        DataConfig(codec=args.codec, dataloader={"batch_size": 1, "num_workers": 0}),
        {task: 1.0},
    )

    repa_teacher = None
    if args.repa_weight is not None:
        repa_teacher = WavLMTeacher(
            rt.codec,
            checkpoint=args.repa_teacher,
            layer=args.repa_layer,
            device=rt.backbone.get_input_embeddings().weight.device,
        )
    model_config = (
        ModelConfig()
        if repa_teacher is None
        else ModelConfig(acoustic_repa_dim=repa_teacher.feature_dim)
    )
    model = SpeechToSpeechFlowModel(model_config, runtime_snapshot=rt)
    module = SpeechToSpeech(
        ModuleConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
        model=model,
        loss=Loss(
            rt.layout,
            rt.flow_matching,
            repa_weight=args.repa_weight,
            repa_teacher=repa_teacher,
        ),
    )
    summary = LossSummary()
    callbacks: list[Callback] = [
        OutputsLogger(),
        FlowMatchingLogger(rt.flow_matching, every_n_steps=1),
        GradLogger(
            ("flow_matching", "repa"),
            "model.acoustic_decoder.input.weight",
            every_n_steps=1,
        ),
        GradNormLogger(),
        SampleLogger([args.sample_index], every_n_steps=1),
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
    ]
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=args.max_steps,
        default_root_dir=str(output_dir),
        logger=TensorBoardLogger(save_dir=str(output_dir), name="tensorboard"),
        callbacks=callbacks,
        log_every_n_steps=1,
        enable_checkpointing=False,
    )
    trainer.fit(module, datamodule=datamodule)

    result = {
        "task": task.value,
        "sample_index": args.sample_index,
        "max_steps": args.max_steps,
        "metrics": summary.report(),
    }
    result_path = output_dir / "metrics.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=[task.value for task in Task])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--audio-tokenizer", required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--repa-weight", type=float)
    parser.add_argument("--repa-teacher", default="microsoft/wavlm-base")
    parser.add_argument("--repa-layer", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--codec", default="longcat")
    parser.add_argument("--backbone", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser


if __name__ == "__main__":
    main()
