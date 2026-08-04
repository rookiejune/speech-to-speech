from __future__ import annotations

import unittest
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from anydataset.dataset import MapStyleABC
from anydataset.dataset.speaker import SpeakerAudioGrid
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Lang,
    Modality,
    Role,
    Sample,
    TextItem,
    TextMeta,
    TextView,
)
from anytrain.codec import AcousticLayout
from anytrain.module.idspace import Layout

from speech_to_speech.datamodule.dataset.speech import (
    DatasetConfig,
    DatasetName,
    SpeakerGridCellsDataset,
    load_dataset,
)
from speech_to_speech.datamodule.config import DataLoaderConfig, SpeechConfig
from speech_to_speech.datamodule.build.single import SingleCollator
from speech_to_speech.datamodule.types import ModelBatch
from speech_to_speech.audio_stream import AudioStream
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.runtime.audio_tokenizer import BiCodecAudioTokenizer
from speech_to_speech.task import Task


class SpeakerGridDatasetTest(unittest.TestCase):
    def test_all_speakers_preserve_flat_cell_order(self):
        grid = _grid()
        dataset = SpeakerGridCellsDataset(grid)

        self.assertEqual(len(dataset), 4)
        self.assertIs(dataset[0], grid.cells[0])
        self.assertIs(dataset[-1], grid.cells[3])
        self.assertEqual(
            [dataset.global_index(index) for index in range(4)],
            [0, 1, 2, 3],
        )

    def test_selected_speaker_maps_one_cell_per_text_row(self):
        grid = _grid()
        dataset = SpeakerGridCellsDataset(grid, speaker="bob")

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.global_index(0), 1)
        self.assertEqual(dataset.global_index(1), 3)
        self.assertIs(dataset[0], grid.cells[1])
        self.assertIs(dataset[1], grid.cells[3])

    def test_selected_speaker_shards_rows_after_filtering(self):
        cells = _Cells(_cells())
        dataset = SpeakerGridCellsDataset(
            SpeakerAudioGrid(cells, ("alice", "bob")),
            speaker="bob",
        )

        rank_0 = list(
            dataset._shuffle(
                shuffle=False,
                seed=0,
                epoch=0,
                num_replicas=2,
                rank=0,
            )
        )
        rank_1 = list(
            dataset._shuffle(
                shuffle=False,
                seed=0,
                epoch=0,
                num_replicas=2,
                rank=1,
            )
        )

        self.assertEqual(rank_0, [(0,)])
        self.assertEqual(rank_1, [(1,)])
        self.assertEqual(cells.shuffle_requests, [(1, 0), (1, 0)])

    def test_unknown_speaker_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not present"):
            SpeakerGridCellsDataset(_grid(), speaker="unknown")

    def test_reference_context_pairs_same_speaker_and_wraps_rows(self):
        dataset = SpeakerGridCellsDataset(_grid(), with_audio_context=True)

        first = dataset[0]
        last = dataset[2]

        self.assertEqual(
            first[(Role.DEFAULT, Modality.TEXT)].views[TextView.SPEAKERS],
            first.audio_context[(Role.DEFAULT, Modality.TEXT)].views[
                TextView.SPEAKERS
            ],
        )
        self.assertEqual(
            first.audio_context[(Role.DEFAULT, Modality.TEXT)].meta[
                TextMeta.SOURCE_INDEX
            ],
            1,
        )
        self.assertEqual(
            last.audio_context[(Role.DEFAULT, Modality.TEXT)].meta[
                TextMeta.SOURCE_INDEX
            ],
            0,
        )

    def test_reference_context_requires_two_rows(self):
        one_row = SpeakerAudioGrid(_Cells(_cells()[:2]), ("alice", "bob"))

        with self.assertRaisesRegex(ValueError, "at least two"):
            SpeakerGridCellsDataset(one_row, with_audio_context=True)

    def test_loader_selects_runtime_codec_and_speaker(self):
        load = Mock()
        load.return_value = SimpleNamespace(load=lambda: _grid())
        config = DatasetConfig(
            name=DatasetName.QWEN_TTS_SPEAKER,
            root="/tmp/qwen-bicodec",
            split="train",
            speaker="bob",
        )

        with _qwen_tts_loader(load):
            dataset = load_dataset(
                config,
                _runtime(AudioSequenceLayout.FLATTENED),
            )

        self.assertIsInstance(dataset, SpeakerGridCellsDataset)
        self.assertEqual(len(dataset), 2)
        load.assert_called_once_with(
            codec="bicodec",
            root=Path("/tmp/qwen-bicodec"),
            split="train",
        )

    def test_loader_enables_reference_context_for_reference_route(self):
        load = Mock()
        load.return_value = SimpleNamespace(load=lambda: _grid())
        with _qwen_tts_loader(load):
            dataset = load_dataset(
                DatasetConfig(name=DatasetName.QWEN_TTS_SPEAKER),
                SimpleNamespace(
                    codec_name="bicodec",
                    audio_view=AudioView.BICODEC,
                    audio_sequence_layout=AudioSequenceLayout.SEMANTIC,
                    audio_tokenizer=BiCodecAudioTokenizer(
                        semantic_codebook_size=16,
                        acoustic_codebook_sizes=(5, 7),
                        acoustic_unit_length=3,
                    ),
                ),
            )

        self.assertIsInstance(dataset, SpeakerGridCellsDataset)
        self.assertTrue(dataset.with_audio_context)

    def test_loader_rejects_unsupported_codec(self):
        config = DatasetConfig(name=DatasetName.QWEN_TTS_SPEAKER)

        with self.assertRaisesRegex(ValueError, "bicodec and longcat"):
            load_dataset(
                config,
                SimpleNamespace(codec_name="unicodec"),
            )

    def test_speaker_grid_requires_single_data_shape(self):
        with self.assertRaisesRegex(ValueError, "datamodule shape single"):
            SpeechConfig(
                codec="bicodec",
                dataloader=DataLoaderConfig(batch_size=1, num_workers=0),
                dataset=DatasetConfig(name=DatasetName.QWEN_TTS_SPEAKER),
            )


class BiCodecSpeakerCellTest(unittest.TestCase):
    def test_reference_route_keeps_target_semantic_out_of_prompt(self):
        runtime = _runtime(AudioSequenceLayout.SEMANTIC)
        sample = SpeakerGridCellsDataset(
            _grid(),
            with_audio_context=True,
        )[0]

        batch = SingleCollator(runtime, {Task.TTS: 1.0})([sample])

        self.assertIsInstance(batch, ModelBatch)
        if batch.audio_contexts is None or batch.audio_contexts[0] is None:
            self.fail("reference audio context was dropped from the model batch")
        torch.testing.assert_close(
            batch.audio_contexts[0].semantic,
            torch.tensor([[2], [3]]),
        )
        row = batch.input_ids[0]
        boa_positions = (row == runtime.boa_token_id).nonzero(as_tuple=False).flatten()
        self.assertGreaterEqual(boa_positions.numel(), 2)
        prompt_boa = int(boa_positions[0].item())
        prompt_eoa = int(
            (row[prompt_boa + 1 :] == runtime.eoa_token_id)
            .nonzero(as_tuple=False)[0]
            .item()
        ) + prompt_boa + 1
        local_prompt = row[prompt_boa + 1 : prompt_eoa] - 10
        decoded = runtime.audio_tokenizer.decode_streams(
            local_prompt,
            (AudioStream.ACOUSTIC,),
        )
        self.assertIsNone(decoded.semantic)
        torch.testing.assert_close(
            decoded.acoustic,
            torch.tensor([[0, 1], [2, 3], [4, 5]]),
        )

    def test_semantic_only_and_full_sequence_build_tts_batches(self):
        for audio_sequence_layout in (
            AudioSequenceLayout.SEMANTIC,
            AudioSequenceLayout.FLATTENED,
        ):
            with self.subTest(audio_sequence_layout=audio_sequence_layout.value):
                runtime = _runtime(audio_sequence_layout)
                cell = SpeakerGridCellsDataset(
                    _grid(),
                    with_audio_context=(
                        runtime.audio_sequence_layout is AudioSequenceLayout.SEMANTIC
                    ),
                )[0]
                batch = SingleCollator(runtime, {Task.TTS: 1.0})([cell])

                self.assertIsInstance(batch, ModelBatch)
                self.assertEqual(batch.tasks, [Task.TTS])
                self.assertIsNone(batch.acoustic_target)
                prompt_length = int(batch.generation_prompt_lengths[0].item())
                response = batch.input_ids[0, prompt_length:]
                semantic_marker = runtime.layout.to_global(
                    "audio",
                    torch.tensor(runtime.audio_tokenizer.semantic_token_id),
                )
                acoustic_marker = runtime.layout.to_global(
                    "audio",
                    torch.tensor(runtime.audio_tokenizer.acoustic_token_id),
                )
                if runtime.audio_sequence_layout is AudioSequenceLayout.FLATTENED:
                    self.assertIsNone(batch.audio_contexts[0])
                    self.assertEqual(int(response[0]), int(acoustic_marker))
                    acoustic_index = (response == acoustic_marker).nonzero(
                        as_tuple=False,
                    )[0].item()
                    semantic_index = (response == semantic_marker).nonzero(
                        as_tuple=False,
                    )[0].item()
                    self.assertLess(acoustic_index, semantic_index)
                else:
                    self.assertIsNotNone(batch.audio_contexts[0])
                    self.assertEqual(int(response[0]), int(semantic_marker))
                    boa_positions = (
                        batch.input_ids[0] == runtime.boa_token_id
                    ).nonzero(as_tuple=False)
                    self.assertGreaterEqual(boa_positions.numel(), 2)


class _Cells(MapStyleABC):
    def __init__(self, samples: Sequence[Sample]) -> None:
        self.samples = tuple(samples)
        self.shuffle_requests: list[tuple[int, int]] = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]

    def _shuffle(
        self,
        *,
        shuffle: bool,
        seed: int,
        epoch: int,
        num_replicas: int,
        rank: int,
    ) -> Iterator[Sequence[int]]:
        del shuffle, seed, epoch
        self.shuffle_requests.append((num_replicas, rank))
        yield (0, 1)
        yield (2, 3)


class _TextTokenizer:
    special_tokens_map: dict[str, str] = {}

    def __len__(self) -> int:
        return 10

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [1, 2]

    def apply_chat_template(self, conversation, **kwargs) -> str:
        del kwargs
        return f"<user>{conversation[0]['content']}</user><assistant>"


def _grid() -> SpeakerAudioGrid:
    return SpeakerAudioGrid(_Cells(_cells()), ("alice", "bob"))


def _cells() -> tuple[Sample, ...]:
    samples: list[Sample] = []
    for row in range(2):
        for speaker_index, speaker in enumerate(("alice", "bob")):
            offset = row * 2 + speaker_index
            samples.append(
                {
                    (Role.DEFAULT, Modality.TEXT): TextItem(
                        views={
                            TextView.TEXT: f"utterance {row}",
                            TextView.SPEAKERS: speaker,
                        },
                        meta={
                            TextMeta.LANG: Lang.EN,
                            TextMeta.SOURCE_INDEX: row,
                        },
                    ),
                    (Role.DEFAULT, Modality.AUDIO): AudioItem(
                        views={
                            AudioView.BICODEC: {
                                "semantic": torch.tensor(
                                    [[offset], [offset + 1]],
                                    dtype=torch.long,
                                ),
                                "acoustic": torch.tensor(
                                    [[0, 1], [2, 3], [4, 5]],
                                    dtype=torch.long,
                                ),
                            }
                        },
                        meta={
                            AudioMeta.DURATION: 0.04,
                            AudioMeta.SPEAKER_ID: speaker,
                        },
                    ),
                }
            )
    return tuple(samples)


def _runtime(audio_sequence_layout: AudioSequenceLayout):
    tokenizer = BiCodecAudioTokenizer(
        semantic_codebook_size=16,
        acoustic_codebook_sizes=(5, 7),
        acoustic_unit_length=3,
    )
    return SimpleNamespace(
        codec_name="bicodec",
        audio_view=AudioView.BICODEC,
        codec_frame_rate=50.0,
        audio_sequence_layout=audio_sequence_layout,
        semantic_codec_artifact=(
            "/tmp/bicodec-semantic"
            if audio_sequence_layout is AudioSequenceLayout.SEMANTIC
            else None
        ),
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
        acoustic_unit_length=3,
        text_tokenizer=_TextTokenizer(),
        audio_tokenizer=tokenizer,
        layout=Layout(text=(0, 10), audio=(10, 10 + tokenizer.vocab_size)),
        pad_token_id=0,
        eos_token_id=1,
        boa_token_id=10 + tokenizer.vocab_size,
        eoa_token_id=11 + tokenizer.vocab_size,
    )


def _qwen_tts_loader(load: Mock):
    zhuyin = ModuleType("zhuyin")
    datasets = ModuleType("zhuyin.datasets")
    wmt19 = ModuleType("zhuyin.datasets.wmt19")
    qwen_tts = ModuleType("zhuyin.datasets.wmt19.qwen_tts")
    qwen_tts.speaker_grid = load
    return patch.dict(
        sys.modules,
        {
            "zhuyin": zhuyin,
            "zhuyin.datasets": datasets,
            "zhuyin.datasets.wmt19": wmt19,
            "zhuyin.datasets.wmt19.qwen_tts": qwen_tts,
        },
    )


if __name__ == "__main__":
    unittest.main()
