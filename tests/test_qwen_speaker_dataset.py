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
from anydataset.dataset.speaker import SpeakerAudioGrid, SpeakerAudioRow
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
from anytrain.module.idspace import Layout

from speech_to_speech.datamodule.dataset.speech import (
    DatasetConfig,
    DatasetName,
    SpeakerGridCellsDataset,
    load_dataset,
)
from speech_to_speech.datamodule.config import DataLoaderConfig, SpeechConfig
from speech_to_speech.datamodule.single import SingleCollator
from speech_to_speech.datamodule.module import _sample_audio_frame_cost
from speech_to_speech.datamodule.batch import ModelBatch
from speech_to_speech.datamodule.sample import AudioContextCostRow
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.runtime.audio_schema import AudioTokenSpec
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

    def test_cost_row_delegates_after_speaker_index_mapping(self):
        cells = _CostCells(_cells())
        dataset = SpeakerGridCellsDataset(
            SpeakerAudioGrid(cells, ("alice", "bob")),
            speaker="bob",
        )

        row = dataset.cost_row(-1)

        self.assertEqual(row, ("cost", 3))
        self.assertEqual(cells.cost_requests, [3])
        self.assertEqual(cells.item_requests, [])

    def test_selected_speaker_maps_every_text_role_row(self):
        grid = _multi_role_grid()
        dataset = SpeakerGridCellsDataset(grid, speaker="bob")

        self.assertEqual(grid.shape, (2, 2, 2))
        self.assertEqual(len(dataset), 4)
        self.assertEqual(
            [dataset.global_index(index) for index in range(len(dataset))],
            [1, 3, 5, 7],
        )

    def test_context_cost_sums_two_metadata_rows_without_payload_reads(self):
        cells = _CostCells(
            _cells(),
            cost_rows=tuple(
                _cost_row(duration)
                for duration in (0.02, 0.04, 0.06, 0.08)
            ),
        )
        dataset = SpeakerGridCellsDataset(
            SpeakerAudioGrid(cells, ("alice", "bob")),
            speaker="bob",
            with_audio_context=True,
        )

        row = dataset.cost_row(0)

        self.assertIsInstance(row, AudioContextCostRow)
        self.assertEqual(cells.cost_requests, [1, 3])
        self.assertEqual(cells.item_requests, [])
        self.assertEqual(_sample_audio_frame_cost(row, frame_rate=50.0), 6)

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

    def test_reference_context_uses_text_rows_not_source_axis(self):
        cells = _CostCells(_multi_role_cells(source_count=1))
        dataset = SpeakerGridCellsDataset(
            _multi_role_grid(source_count=1, cells=cells),
            speaker="bob",
            with_audio_context=True,
        )

        self.assertEqual(len(dataset), 2)
        _ = dataset[0]
        _ = dataset[1]
        self.assertEqual(cells.item_requests, [1, 3, 3, 1])

    def test_reference_context_requires_two_rows(self):
        one_row = SpeakerAudioGrid(_Cells(_cells()[:2]), ("alice", "bob"))

        with self.assertRaisesRegex(ValueError, "at least two"):
            SpeakerGridCellsDataset(one_row, with_audio_context=True)

    def test_loader_selects_runtime_codec_and_speaker(self):
        load = Mock()
        view = load.return_value
        selected = view.filter.return_value
        selected.load.return_value = _grid()
        config = DatasetConfig(
            name=DatasetName.QWEN_TTS_SPEAKER,
            root="/tmp/qwen-bicodec",
            split="train",
            filter="grid_quality_v1",
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
        view.filter.assert_called_once_with("grid_quality_v1")
        selected.load.assert_called_once_with()

    def test_loader_does_not_infer_reference_context_from_runtime(self):
        load = Mock()
        view = load.return_value
        selected = view.filter.return_value
        selected.load.return_value = _grid()
        with _qwen_tts_loader(load):
            dataset = load_dataset(
                DatasetConfig(name=DatasetName.QWEN_TTS_SPEAKER, filter=None),
                SimpleNamespace(
                    codec_name="bicodec",
                    audio_view=AudioView.BICODEC,
                    audio_sequence_layout=AudioSequenceLayout.FLATTENED,
                    audio_tokenizer=BiCodecAudioTokenizer(
                        semantic_codebook_size=16,
                        global_codebook_sizes=(5, 7),
                        global_unit_length=3,
                    ),
                ),
            )

        self.assertIsInstance(dataset, SpeakerGridCellsDataset)
        self.assertFalse(dataset.with_audio_context)
        view.filter.assert_called_once_with(None)

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
    def test_explicit_reference_serializes_global_without_batch_side_channel(self):
        runtime = _runtime(AudioSequenceLayout.FLATTENED)
        sample = SpeakerGridCellsDataset(
            _grid(),
            with_audio_context=True,
        )[0]

        batch = SingleCollator(runtime, {Task.TTS: 1.0})([sample])

        self.assertIsInstance(batch, ModelBatch)
        row = batch.input_ids[0]
        boa_positions = (row == runtime.boa_token_id).nonzero(as_tuple=False).flatten()
        self.assertGreaterEqual(boa_positions.numel(), 2)
        prompt_boa = int(boa_positions[0].item())
        prompt_eoa = int(
            (row[prompt_boa + 1 :] == runtime.eoa_token_id)
            .nonzero(as_tuple=False)[0]
            .item()
        ) + prompt_boa + 1
        audio_start, _ = runtime.layout.blocks["audio"]
        local_prompt = row[prompt_boa + 2 : prompt_eoa] - audio_start
        decoded = runtime.audio_tokenizer.decode_streams(
            local_prompt,
        )
        self.assertIsNone(decoded.semantic_codes)
        torch.testing.assert_close(
            decoded.global_codes,
            torch.tensor([[0, 1], [2, 3], [4, 5]]),
        )

    def test_input_presence_selects_tts_response_opening_marker(self):
        for with_audio_context in (False, True):
            with self.subTest(with_audio_context=with_audio_context):
                runtime = _runtime(AudioSequenceLayout.FLATTENED)
                cell = SpeakerGridCellsDataset(
                    _grid(),
                    with_audio_context=with_audio_context,
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
                global_marker = runtime.layout.to_global(
                    "audio",
                    torch.tensor(runtime.audio_tokenizer.global_token_id),
                )
                if not with_audio_context:
                    self.assertEqual(int(response[2]), int(global_marker))
                    global_index = (response == global_marker).nonzero(
                        as_tuple=False,
                    )[0].item()
                    semantic_index = (response == semantic_marker).nonzero(
                        as_tuple=False,
                    )[0].item()
                    self.assertLess(global_index, semantic_index)
                else:
                    self.assertEqual(int(response[2]), int(semantic_marker))
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


class _CostCells(_Cells):
    def __init__(
        self,
        samples: Sequence[Sample],
        *,
        cost_rows: Sequence[object] | None = None,
    ) -> None:
        super().__init__(samples)
        self.cost_rows = cost_rows
        self.cost_requests: list[int] = []
        self.item_requests: list[int] = []

    def __getitem__(self, index: int) -> Sample:
        self.item_requests.append(index)
        return super().__getitem__(index)

    def cost_row(self, index: int) -> object:
        self.cost_requests.append(index)
        if self.cost_rows is not None:
            return self.cost_rows[index]
        return "cost", index


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


def _multi_role_grid(
    *,
    source_count: int = 2,
    cells: _Cells | None = None,
) -> SpeakerAudioGrid:
    return SpeakerAudioGrid(
        _Cells(_multi_role_cells(source_count)) if cells is None else cells,
        ("alice", "bob"),
        row_specs=tuple(
            SpeakerAudioRow(source_index=source, role=role)
            for source in range(source_count)
            for role in (Role.SOURCE, Role.TARGET)
        ),
    )


def _multi_role_cells(source_count: int) -> tuple[Sample, ...]:
    samples: list[Sample] = []
    for source in range(source_count):
        for role in (Role.SOURCE, Role.TARGET):
            for speaker_index, speaker in enumerate(("alice", "bob")):
                offset = source * 4 + int(role is Role.TARGET) * 2 + speaker_index
                samples.append(
                    {
                        (Role.DEFAULT, Modality.TEXT): TextItem(
                            views={
                                TextView.TEXT: f"source {source} {role.value}",
                                TextView.SPEAKERS: speaker,
                            },
                            meta={
                                TextMeta.LANG: Lang.EN,
                                TextMeta.SOURCE_INDEX: source,
                            },
                        ),
                        (Role.DEFAULT, Modality.AUDIO): AudioItem(
                            views={
                                AudioView.BICODEC: {
                                    "semantic": torch.tensor([[offset]]),
                                    "global": torch.tensor([[0, 1]]),
                                }
                            },
                            meta={
                                AudioMeta.DURATION: 0.02 * (offset + 1),
                                AudioMeta.SPEAKER_ID: speaker,
                            },
                        ),
                    }
                )
    return tuple(samples)


def _cost_row(duration: float) -> SimpleNamespace:
    return SimpleNamespace(
        items=(
            (
                (Role.DEFAULT, Modality.AUDIO),
                {AudioMeta.DURATION.value: duration},
            ),
        ),
    )


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
                                "global": torch.tensor(
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
        global_codebook_sizes=(5, 7),
        global_unit_length=3,
    )
    audio_start = 10
    boa_token_id = audio_start + tokenizer.vocab_size
    spec = AudioTokenSpec.create(
        codec_name="bicodec",
        sequence_layout=audio_sequence_layout.value,
        tokenizer=tokenizer,
    )
    return SimpleNamespace(
        codec_name="bicodec",
        audio_view=AudioView.BICODEC,
        codec_frame_rate=50.0,
        audio_sequence_layout=audio_sequence_layout,
        acoustic_generator_artifact=(
            "/tmp/bicodec-semantic"
            if audio_sequence_layout is AudioSequenceLayout.SEMANTIC
            else None
        ),
        global_unit_length=3,
        text_tokenizer=_TextTokenizer(),
        audio_tokenizer=tokenizer,
        input_audio_tokenizer=tokenizer,
        input_audio_token_spec=spec,
        audio_token_spec=spec,
        output_audio_token_spec=spec,
        layout=Layout(text=(0, audio_start), audio=(audio_start, boa_token_id + 4)),
        pad_token_id=0,
        eos_token_id=1,
        boa_token_id=boa_token_id,
        eoa_token_id=boa_token_id + 1,
        mask_token_id=boa_token_id + 2,
        audio_schema_token_id=boa_token_id + 3,
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
