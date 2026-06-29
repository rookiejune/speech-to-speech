from __future__ import annotations

import os
from collections.abc import Iterable
from functools import partial
from pathlib import Path

import hydra
import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from speech_to_speech.config import DatasetFactoryConfig
from speech_to_speech.dataset import dataset_metadata, training_dataset
from speech_to_speech.datamodule import SpeechToSpeechDataModule
from speech_to_speech.datamodule.example import speech_pair_from_sample
from speech_to_speech.model.DiT.model import DiT
from speech_to_speech.model.orchestrator import Orchestrator, dit_config
from speech_to_speech.pl_module import (
    SpeechToSpeechModule,
    TaskGenerationLogger,
    TaskSampleLogger,
)
from speech_to_speech.runtime import longcat_codec, prepare_longcat_tokenizer, qwen3_tokenizer
from speech_to_speech.smoke import _accelerator, _mapping, _speech_to_speech_config
from speech_to_speech.types import SpeechPair


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    _bind_local_cuda_device()
    config = _speech_to_speech_config(
        _mapping(OmegaConf.to_container(cfg, resolve=True), "config")
    )
    if config.trainer.ckpt_path is not None:
        allow_trusted_checkpoint_globals()

    seed_everything(config.train.seed, workers=True)
    tokenizer = qwen3_tokenizer(config.model)
    dataset_factory = config.datamodule.dataset_factory
    bpe = prepare_longcat_tokenizer(
        partial(_speech_pairs, dataset_factory),
        datasets=dataset_metadata(dataset_factory),
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
        config.datamodule,
        config.tasks,
        model.embed_tokens,
        tokenizer=tokenizer,
        bpe_tokenizer=bpe,
        bpe=config.bpe,
    )

    root = Path(config.trainer.default_root_dir)
    logger = TensorBoardLogger(
        save_dir=str(root / "tensorboard"),
        name=config.trainer.name,
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(root / "checkpoints" / config.trainer.name),
        filename="{step:08d}",
        monitor="loss",
        mode="min",
        save_top_k=config.trainer.save_top_k,
        save_last=True,
        every_n_train_steps=config.trainer.checkpoint_every_n_steps,
    )
    callbacks = [checkpoint]
    if config.trainer.sample_log_every_n_steps is not None:
        callbacks.append(
            TaskSampleLogger(
                datamodule=config.datamodule,
                tasks=config.tasks,
                bpe=config.bpe,
                every_n_steps=config.trainer.sample_log_every_n_steps,
                samples_per_task=config.trainer.samples_per_task,
                max_audio_samples=config.trainer.sample_log_max_audio_samples,
            )
        )
    if config.trainer.generation_log_every_n_steps is not None:
        callbacks.append(
            TaskGenerationLogger(
                datamodule=config.datamodule,
                bpe=config.bpe,
                tokenizer=tokenizer,
                every_n_steps=config.trainer.generation_log_every_n_steps,
                sample_index=config.trainer.generation_sample_index,
                flow_steps=config.trainer.generation_flow_steps,
                chunk_size=config.trainer.generation_chunk_size,
                guidance_scale=config.trainer.generation_guidance_scale,
                acoustic_sampler=config.trainer.generation_acoustic_sampler,
                preview_tokens=config.trainer.generation_preview_tokens,
                max_audio_samples=config.trainer.generation_log_max_audio_samples,
            )
        )
    trainer_kwargs = {
        "default_root_dir": str(root),
        "max_steps": config.train.max_steps,
        "accelerator": config.trainer.accelerator or _accelerator(config.train),
        "devices": config.trainer.devices,
        "strategy": config.trainer.strategy,
        "precision": config.train.precision,
        "logger": logger,
        "callbacks": callbacks,
        "enable_checkpointing": True,
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


def _speech_pairs(dataset_factory: DatasetFactoryConfig) -> Iterable[SpeechPair]:
    for sample in training_dataset(dataset_factory):
        yield speech_pair_from_sample(sample)


def allow_trusted_checkpoint_globals() -> None:
    from anytrain.optim.config import MuonAdjustLRFn
    from torch.serialization import add_safe_globals

    add_safe_globals([MuonAdjustLRFn])


if __name__ == "__main__":
    main()
