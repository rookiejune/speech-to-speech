from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import torch
from anydataset import AudioItem, AudioView, Modality, Role
from anydataset.store import DatasetWriter

from speech_to_speech.config import ModelConfig
from speech_to_speech import smoke
from helpers import MockTokenizer


class SmokeRunnerTest(unittest.TestCase):
    def test_load_config_reads_qwen3_smoke_yaml(self) -> None:
        config = smoke.load_config("configs/qwen3_smoke.yaml")

        self.assertEqual(
            config.data.datasets,
            ("store://~/repos/anydataset/storage/fleurs-full-longcat",),
        )
        self.assertEqual(config.tasks.enabled, ("autoregression",))
        self.assertEqual(config.bpe.vocab_size, 100000)
        self.assertEqual(config.train.max_steps, 100)

    def test_load_config_reads_wmt19_mixed_smoke_yaml(self) -> None:
        config = smoke.load_config("configs/wmt19_tts_longcat_mixed_smoke.yaml")

        self.assertEqual(
            config.data.datasets,
            ("store:///mnt/pami202/zhuyin/datasets/wmt19-tts-longcat/longcat-delta:train",),
        )
        self.assertEqual(config.data.batch_size, 4)
        self.assertEqual(config.tasks.enabled, ("autoregression", "translation"))
        self.assertTrue(config.model.load_in_4bit)
        self.assertTrue(config.model.lora.enabled)

    def test_run_smoke_wires_real_dataset_to_trainer(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = _write_store(root / "store")
            config_path = root / "smoke.yaml"
            config_path.write_text(
                f"""
data:
  datasets:
    - store://{store}:train
  cache_root: {root / "cache"}
  batch_size: 1
  num_workers: 0
  pin_memory: false
  drop_last: false
bpe:
  cache_dir_env: BPE_CACHE_DIR
  codec_name: longcat
  vocab_size: 16
  max_piece_frames: 4
tasks:
  enabled:
    - autoregression
model:
  model_name_or_path: Qwen/Qwen3-0.6B
  trust_remote_code: false
  train_text_embedding: false
  train_audio_embedding: true
  train_audio_special_tokens: true
  train_backbone: false
  train_dit: false
  lora:
    enabled: false
    rank: 16
    alpha: 32
    dropout: 0.05
    targets:
      - q_proj
train:
  max_steps: 4
  learning_rate: 0.0001
  optimizer_preset: pretrain
  optimizer: adamw
  weight_decay:
  schedule: constant
  warmup_steps: 0
  stable_steps:
  decay_steps:
  min_lr_ratio: 0.1
  seed: 0
  device: cpu
  precision: "32-true"
""",
                encoding="utf-8",
            )
            with patch.object(smoke, "qwen3_tokenizer", return_value=MockTokenizer()) as qwen:
                with patch.object(smoke, "prepare_longcat_tokenizer", return_value=_bpe()):
                    with patch.object(smoke, "Orchestrator", FakeOrchestrator):
                        with patch.object(smoke, "SpeechToSpeechModule", FakeModule):
                            with patch.object(smoke, "Trainer", FakeTrainer):
                                with patch.object(smoke, "seed_everything") as seed:
                                    trainer = smoke.run_smoke(
                                        config_path,
                                        max_steps=1,
                                        default_root_dir=root / "outputs",
                                        enable_progress_bar=False,
                                    )

            self.assertIsInstance(trainer, FakeTrainer)
            seed.assert_called_once_with(0, workers=True)
            qwen.assert_called_once()
            self.assertIsInstance(qwen.call_args.args[0], ModelConfig)
            self.assertEqual(FakeTrainer.last_kwargs["max_steps"], 1)
            self.assertEqual(FakeTrainer.last_kwargs["accelerator"], "cpu")
            self.assertFalse(FakeTrainer.last_kwargs["enable_progress_bar"])
            self.assertEqual(FakeTrainer.last_fit_module.train.max_steps, 1)
            batch = next(iter(FakeTrainer.last_fit_datamodule.train_dataloader()))
            self.assertEqual(batch.input_ids.size(0), 1)


class FakeOrchestrator:
    def __init__(
        self,
        *,
        model_config: ModelConfig,
        bpe_config: object,
        tokenizer: object,
        bpe_vocab_size: int,
    ) -> None:
        del bpe_config, tokenizer
        self.model_config = model_config
        self.embed_tokens = _embedding(bpe_vocab_size)


class FakeModule:
    def __init__(self, model: FakeOrchestrator, train: object) -> None:
        self.model = model
        self.train = train


class FakeTrainer:
    last_kwargs: dict[str, object] = {}
    last_fit_module: FakeModule
    last_fit_datamodule: object

    def __init__(self, **kwargs: object) -> None:
        self.global_step = 0
        FakeTrainer.last_kwargs = kwargs

    def fit(self, module: FakeModule, *, datamodule: object) -> None:
        FakeTrainer.last_fit_module = module
        FakeTrainer.last_fit_datamodule = datamodule
        self.global_step = 1


class FakeBPE:
    vocab_size = 16

    def encode_units(self, units: list[int]) -> list[int]:
        return [int(unit) for unit in units]


def _bpe() -> FakeBPE:
    return FakeBPE()


def _embedding(vocab_size: int):
    from anytrain.idspace import IdSpace, IdSpaceEmbedding, Modality as IdModality, ModalityBlock

    space = IdSpace(
        {
            "<|endoftext|>": 0,
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "user": 3,
            "assistant": 4,
            "\n": 5,
            "<think>": 6,
            "</think>": 7,
            "boa": 16,
            "eoa": 17,
        },
        [
            ModalityBlock(IdModality.TEXT, 0, 16),
            ModalityBlock(IdModality.AUDIO, 18, vocab_size),
        ],
    )
    return IdSpaceEmbedding(space, 4)


def _write_store(path: Path) -> Path:
    DatasetWriter(path, dataset_id="toy-s2s", split="train").write(
        [
            {
                (Role.SOURCE, Modality.AUDIO): AudioItem(
                    views={
                        AudioView.LONGCAT: {
                            "semantic_codes": torch.tensor([0, 1, 2]),
                            "acoustic_codes": torch.zeros((4, 3), dtype=torch.long),
                        }
                    }
                ),
                (Role.TARGET, Modality.AUDIO): AudioItem(
                    views={
                        AudioView.LONGCAT: {
                            "semantic_codes": torch.tensor([2, 1]),
                            "acoustic_codes": torch.zeros((4, 2), dtype=torch.long),
                        }
                    }
                ),
            }
        ]
    )
    return path


if __name__ == "__main__":
    unittest.main()
