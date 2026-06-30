from __future__ import annotations

import os
from collections.abc import Iterable
from functools import partial
from pathlib import Path

import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from speech_to_speech.config import DatasetFactoryConfig, SpeechToSpeechConfig
from speech_to_speech.dataset import dataset_metadata, training_dataset
from speech_to_speech.datamodule import SpeechToSpeechDataModule
from speech_to_speech.datamodule.example import speech_pair_from_sample
from speech_to_speech.model.orchestrator import Orchestrator
from speech_to_speech.pl_module import (
    SpeechToSpeechModule,
    TaskGenerationLogger,
    TaskSampleLogger,
)
from speech_to_speech.runtime import longcat_codec, prepare_longcat_tokenizer, qwen3_tokenizer
from speech_to_speech.smoke import (
    _accelerator,
    _mapping,
    _speech_to_speech_config,
    _validate_acoustic_training_model,
)
from speech_to_speech.types import SpeechPair


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _bind_local_cuda_device()
    config = _speech_to_speech_config(
        _mapping(OmegaConf.to_container(cfg, resolve=True), "config")
    )
    if config.trainer.ckpt_path is not None:
        allow_trusted_checkpoint_globals()
    _validate_acoustic_training_model(config.model, config.train)

    seed_everything(config.train.seed, workers=True)
    tokenizer = qwen3_tokenizer(config.model)
    dataset_factory = config.datamodule.dataset_factory
    bpe = prepare_longcat_tokenizer(
        partial(_speech_pairs, dataset_factory),
        datasets=dataset_metadata(dataset_factory),
        config=config.bpe,
    )
    model = Orchestrator(
        model_config=config.model,
        bpe_config=config.bpe,
        tokenizer=tokenizer,
        bpe_vocab_size=bpe.vocab_size,
    )
    acoustic_training = config.train.acoustic_loss_weight > 0.0
    module = SpeechToSpeechModule(
        model,
        config.train,
        bpe=bpe if acoustic_training else None,
        acoustic_feature_extractor=longcat_codec() if acoustic_training else None,
    )
    datamodule = SpeechToSpeechDataModule(
        config.datamodule,
        config.tasks,
        model.embed_tokens,
        tokenizer=tokenizer,
        bpe_tokenizer=bpe,
        bpe=config.bpe,
    )

    root = _trainer_root(config.trainer.default_root_dir)
    logger = TensorBoardLogger(
        save_dir=str(root / "tensorboard"),
        name=config.trainer.name,
    )
    callbacks = _callbacks(config, root=root, tokenizer=tokenizer)
    trainer_kwargs = {
        "default_root_dir": str(root),
        "max_steps": config.train.max_steps,
        "accelerator": config.trainer.accelerator or _accelerator(config.train),
        "devices": config.trainer.devices,
        "strategy": config.trainer.strategy,
        "precision": config.train.precision,
        "logger": logger,
        "callbacks": callbacks,
        "enable_checkpointing": config.trainer.callbacks.checkpoint.enabled,
        "enable_model_summary": config.trainer.enable_model_summary,
        "enable_progress_bar": config.trainer.enable_progress_bar,
    }
    if config.trainer.log_every_n_steps is not None:
        trainer_kwargs["log_every_n_steps"] = config.trainer.log_every_n_steps
    trainer = Trainer(**trainer_kwargs)
    trainer.fit(module, datamodule=datamodule, ckpt_path=config.trainer.ckpt_path)
    print(f"training finished: global_step={trainer.global_step}")


def _bind_local_cuda_device() -> None:
    rank = os.environ.get("LOCAL_RANK")
    if rank is None or not torch.cuda.is_available():
        return
    torch.cuda.set_device(int(rank))


def _trainer_root(default_root_dir: str | Path | None) -> Path:
    if default_root_dir is None:
        from zhuyin.env import train_dir

        return train_dir("speech-to-speech")
    if isinstance(default_root_dir, str) and not default_root_dir:
        raise ValueError("trainer.default_root_dir must not be empty.")
    return Path(default_root_dir)


def _callbacks(
    config: SpeechToSpeechConfig,
    *,
    root: Path,
    tokenizer: object,
) -> list[Callback]:
    trainer = config.trainer
    callbacks = []
    checkpoint = trainer.callbacks.checkpoint
    if checkpoint.enabled:
        callbacks.append(
            ModelCheckpoint(
                dirpath=str(root / "checkpoints" / trainer.name),
                filename=checkpoint.filename,
                monitor=checkpoint.monitor,
                mode=checkpoint.mode,
                save_top_k=checkpoint.save_top_k,
                save_last=checkpoint.save_last,
                every_n_train_steps=checkpoint.every_n_steps,
            )
        )
    learning_rate_monitor = trainer.callbacks.learning_rate_monitor
    if learning_rate_monitor.enabled:
        callbacks.append(
            LearningRateMonitor(logging_interval=learning_rate_monitor.logging_interval)
        )
    sample = trainer.callbacks.sample
    if sample.enabled:
        callbacks.append(
            TaskSampleLogger(
                datamodule=config.datamodule,
                tasks=config.tasks,
                bpe=config.bpe,
                every_n_steps=sample.every_n_steps,
                samples_per_task=sample.samples_per_task,
                max_audio_samples=sample.max_audio_samples,
            )
        )
    generation = trainer.callbacks.generation
    if generation.enabled and generation.every_n_steps is not None:
        callbacks.append(
            TaskGenerationLogger(
                datamodule=config.datamodule,
                bpe=config.bpe,
                tokenizer=tokenizer,
                every_n_steps=generation.every_n_steps,
                sample_index=generation.sample_index,
                flow_steps=generation.flow_steps,
                chunk_size=generation.chunk_size,
                left_context_chunks=generation.left_context_chunks,
                guidance_scale=generation.guidance_scale,
                acoustic_sampler=generation.acoustic_sampler,
                preview_tokens=generation.preview_tokens,
                max_audio_samples=generation.max_audio_samples,
            )
        )
    return callbacks


def _speech_pairs(dataset_factory: DatasetFactoryConfig) -> Iterable[SpeechPair]:
    for sample in training_dataset(dataset_factory):
        yield speech_pair_from_sample(sample)


def allow_trusted_checkpoint_globals() -> None:
    from anytrain.optim.config import MuonAdjustLRFn
    from torch.serialization import add_safe_globals

    add_safe_globals([MuonAdjustLRFn])


if __name__ == "__main__":
    main()
