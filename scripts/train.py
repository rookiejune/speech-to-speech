from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from speech_to_speech.datamodule import SpeechToSpeechDataModule
from speech_to_speech.datamodule.example import speech_pair_from_sample
from speech_to_speech.model.DiT.model import DiT
from speech_to_speech.model.orchestrator import Orchestrator, dit_config
from speech_to_speech.pl_module import SpeechToSpeechModule
from speech_to_speech.runtime import longcat_codec, prepare_longcat_tokenizer, qwen3_tokenizer
from speech_to_speech.smoke import _dataset, load_config


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.max_steps is not None:
        config = replace(config, train=replace(config.train, max_steps=args.max_steps))

    seed_everything(config.train.seed, workers=True)
    tokenizer = qwen3_tokenizer(config.model)
    bpe = prepare_longcat_tokenizer(
        (speech_pair_from_sample(sample) for sample in _dataset(config.data)),
        datasets=config.data.datasets,
        config=config.bpe,
    )
    acoustic_training = config.train.acoustic_loss_weight > 0.0
    model = Orchestrator(
        dit=DiT(dit_config()) if acoustic_training else None,
        model_config=config.model,
        bpe_config=config.bpe,
        tokenizer=tokenizer,
        bpe_vocab_size=bpe.vocab_size,
    )
    module = SpeechToSpeechModule(
        model,
        config.train,
        bpe=bpe if acoustic_training else None,
        acoustic_feature_extractor=longcat_codec() if acoustic_training else None,
    )
    datamodule = SpeechToSpeechDataModule(
        config.data,
        config.tasks,
        model.embed_tokens,
        tokenizer=tokenizer,
        bpe_tokenizer=bpe,
        bpe=config.bpe,
    )

    root = Path(args.default_root_dir)
    logger = TensorBoardLogger(
        save_dir=str(root / "tensorboard"),
        name=args.name,
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(root / "checkpoints" / args.name),
        filename="{step:08d}",
        monitor="train/loss",
        mode="min",
        save_top_k=args.save_top_k,
        save_last=True,
        every_n_train_steps=args.checkpoint_every_n_steps,
    )
    trainer = Trainer(
        default_root_dir=str(root),
        max_steps=config.train.max_steps,
        accelerator=args.accelerator or accelerator(config.train.device),
        devices=args.devices,
        strategy=args.strategy,
        precision=config.train.precision,
        logger=logger,
        callbacks=[checkpoint],
        enable_checkpointing=True,
        enable_model_summary=not args.no_model_summary,
        enable_progress_bar=not args.no_progress_bar,
        log_every_n_steps=args.log_every_n_steps,
    )
    trainer.fit(module, datamodule=datamodule, ckpt_path=args.ckpt_path)
    print(f"training finished: global_step={trainer.global_step}")


def accelerator(device: str) -> str:
    if device in {"cuda", "gpu"}:
        return "cuda"
    return device


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run speech-to-speech training.")
    parser.add_argument("config", help="Path to a speech-to-speech YAML config.")
    parser.add_argument("--name", default="wmt19-quality-longrun")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--default-root-dir", default="outputs/train")
    parser.add_argument("--accelerator")
    parser.add_argument("--devices", type=devices, default=1)
    parser.add_argument("--strategy", default="auto")
    parser.add_argument("--ckpt-path")
    parser.add_argument("--log-every-n-steps", type=positive_int, default=10)
    parser.add_argument("--checkpoint-every-n-steps", type=positive_int, default=500)
    parser.add_argument("--save-top-k", type=int, default=2)
    parser.add_argument("--no-model-summary", action="store_true")
    parser.add_argument("--no-progress-bar", action="store_true")
    return parser.parse_args(argv)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive.")
    return parsed


def devices(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


if __name__ == "__main__":
    main()
