from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import TensorBoardLogger
from torch import Tensor
from torch.utils.data import DataLoader

from speech_to_speech.callback.logging import FlowMatchingLogger, OutputsLogger
from speech_to_speech.datamodule.collator import Collator
from speech_to_speech.datamodule.types import Task
from speech_to_speech.loss import Loss, LossItem
from speech_to_speech.model.acoustic import SpeechToSpeechFlowModel
from speech_to_speech.pl_module import Config as ModuleConfig
from speech_to_speech.pl_module import SpeechToSpeech
from speech_to_speech.runtime import init_runtime
from speech_to_speech.runtime.singleton import Config as RuntimeConfig
from zhuyin.datasets.wmt19_tts import wmt19_tts_codec


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
        total = outputs.get("loss")
        if isinstance(total, Tensor):
            self._append("loss", total)
        for name in ("semantic", "flow_matching"):
            item = outputs.get(name)
            if isinstance(item, LossItem):
                self._append(name, item.loss)

    def report(self, window: int = 20) -> dict[str, dict[str, float | int]]:
        report = {}
        for name, values in self.values.items():
            size = min(window, len(values))
            first = sum(values[:size]) / size
            last = sum(values[-size:]) / size
            report[name] = {
                "steps": len(values),
                "window": size,
                "first_mean": first,
                "last_mean": last,
                "last_to_first": last / first,
            }
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
    dataset = wmt19_tts_codec(codec=args.codec, split=args.split)
    sample = dataset[args.sample_index]
    loader = DataLoader(
        [sample],
        batch_size=1,
        num_workers=0,
        collate_fn=Collator({task: 1.0}),
    )

    model = SpeechToSpeechFlowModel(runtime_snapshot=rt)
    module = SpeechToSpeech(
        ModuleConfig(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        ),
        model=model,
        loss=Loss(rt.layout, rt.flow_matching),
    )
    summary = LossSummary()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        max_steps=args.max_steps,
        default_root_dir=str(output_dir),
        logger=TensorBoardLogger(save_dir=str(output_dir), name="tensorboard"),
        callbacks=[OutputsLogger(), FlowMatchingLogger(every_n_steps=10), summary],
        log_every_n_steps=1,
        enable_checkpointing=False,
    )
    trainer.fit(module, train_dataloaders=loader)

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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--codec", default="longcat")
    parser.add_argument("--backbone", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser


if __name__ == "__main__":
    main()
