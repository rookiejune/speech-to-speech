from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import torch

from speech_to_speech.config import DataModuleConfig
from speech_to_speech.datamodule.batch_builder import CausalLMBatchBuilder
from speech_to_speech.pl_module.sample_logger import (
    TaskSampleLogger,
    _canary_pair,
    _decode_side,
    _generation_specs,
    _should_log_generation_step,
)
from speech_to_speech.types import LongCatPair, LongCatSide
from helpers import (
    MockFrameBPE,
    MockTokenizer,
    isolated_anydataset_home,
    patched_wmt19_longcat,
    toy_embedding,
    write_toy_longcat_store,
)


class FakeCodec:
    def __init__(self) -> None:
        self.semantic_shape: tuple[int, ...] | None = None
        self.acoustic_shape: tuple[int, ...] | None = None

    def decode(self, semantic_ids: torch.Tensor, acoustic_ids: torch.Tensor) -> torch.Tensor:
        self.semantic_shape = tuple(semantic_ids.shape)
        self.acoustic_shape = tuple(acoustic_ids.shape)
        return torch.ones(1, 8)


class TaskSampleLoggerTest(unittest.TestCase):
    def test_task_sample_logger_accepts_lightning_runtime_attributes(self) -> None:
        logger = TaskSampleLogger()
        marker = object()

        logger.log = marker

        self.assertIs(logger.log, marker)

    def test_task_sample_logger_accepts_start_only_schedule(self) -> None:
        logger = TaskSampleLogger(every_n_steps=0)

        self.assertEqual(logger.every_n_steps, 0)

    def test_decode_side_rejects_bpe_acoustic_length_mismatch(self) -> None:
        side = LongCatSide(
            semantic_ids=torch.tensor([1, 2]),
            acoustic_ids=torch.zeros((4, 2), dtype=torch.long),
        )

        with self.assertRaisesRegex(
            ValueError,
            "BPE-expanded semantic length must match LongCat acoustic length",
        ):
            _decode_side(
                FakeCodec(),
                MockFrameBPE(expanded=[1, 2, 3]),
                side,
                max_audio_samples=None,
            )

    def test_decode_side_passes_expanded_semantic_and_acoustic_to_codec(self) -> None:
        codec = FakeCodec()
        side = LongCatSide(
            semantic_ids=torch.tensor([1, 2]),
            acoustic_ids=torch.zeros((4, 3), dtype=torch.long),
        )

        audio = _decode_side(codec, MockFrameBPE(expanded=[1, 2, 2]), side, max_audio_samples=4)

        self.assertEqual(tuple(audio.shape), (1, 4))
        self.assertEqual(codec.semantic_shape, (1, 3))
        self.assertEqual(codec.acoustic_shape, (1, 4, 3))

    def test_generation_schedule_logs_first_update_then_interval(self) -> None:
        self.assertFalse(_should_log_generation_step(0, 5, None))
        self.assertTrue(_should_log_generation_step(1, 5, None))
        self.assertFalse(_should_log_generation_step(1, 5, 1))
        self.assertFalse(_should_log_generation_step(5, 5, 1))
        self.assertTrue(_should_log_generation_step(6, 5, 1))

    def test_canary_pair_reads_fixed_dataset_index(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            store = write_toy_longcat_store(
                root / "store",
                pairs=(([0], [1]), ([2, 3], [4])),
            )

            with isolated_anydataset_home(root):
                with patched_wmt19_longcat(store):
                    pair = _canary_pair(DataModuleConfig(), 1)

        self.assertEqual(pair.source.semantic_ids.tolist(), [2, 3])
        self.assertEqual(pair.target.semantic_ids.tolist(), [4])

    def test_generation_specs_use_one_pair_for_all_directions(self) -> None:
        pair = LongCatPair(
            source=LongCatSide(
                semantic_ids=torch.tensor([0, 1]),
                acoustic_ids=torch.zeros((4, 2), dtype=torch.long),
            ),
            target=LongCatSide(
                semantic_ids=torch.tensor([2]),
                acoustic_ids=torch.zeros((4, 1), dtype=torch.long),
            ),
        )
        builder = CausalLMBatchBuilder(
            toy_embedding(audio_vocab_size=8),
            tokenizer=MockTokenizer(),
        )

        specs = _generation_specs(builder, MockFrameBPE(vocab_size=8), pair)

        self.assertEqual(
            [spec.name for spec in specs],
            ["source_ar", "target_ar", "source_to_target", "target_to_source"],
        )
        self.assertIsNone(specs[0].prefix)
        self.assertIs(specs[0].reference, pair.source)
        self.assertIsNone(specs[1].prefix)
        self.assertIs(specs[1].reference, pair.target)
        self.assertIs(specs[2].prefix, pair.source)
        self.assertIs(specs[2].reference, pair.target)
        self.assertIs(specs[3].prefix, pair.target)
        self.assertIs(specs[3].reference, pair.source)

    def test_generation_specs_use_full_teacher_forcing_ar_target(self) -> None:
        pair = LongCatPair(
            source=LongCatSide(
                semantic_ids=torch.tensor([0, 1, 2]),
                acoustic_ids=torch.zeros((4, 3), dtype=torch.long),
            ),
            target=LongCatSide(
                semantic_ids=torch.tensor([3, 4, 5]),
                acoustic_ids=torch.zeros((4, 3), dtype=torch.long),
            ),
        )
        builder = CausalLMBatchBuilder(
            toy_embedding(audio_vocab_size=8),
            tokenizer=MockTokenizer(),
        )

        specs = _generation_specs(builder, MockFrameBPE(vocab_size=8), pair)

        self.assertEqual(int(specs[0].batch.labels.ne(-100).sum().item()), 5)
        self.assertEqual(int(specs[1].batch.labels.ne(-100).sum().item()), 5)
        self.assertEqual(specs[0].batch.target_audio.semantic_ids.tolist(), [[0, 1, 2]])
        self.assertEqual(specs[1].batch.target_audio.semantic_ids.tolist(), [[3, 4, 5]])


if __name__ == "__main__":
    unittest.main()
