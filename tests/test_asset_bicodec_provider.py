from __future__ import annotations

import unittest
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast

import torch
from anydataset.dataset import MapStyleABC
from anydataset.dataset.collate import collate_fn
from anydataset.store import ViewMaterializer
from anydataset.types import (
    AudioItem,
    AudioReq,
    AudioView,
    Lang,
    Modality,
    Role,
    Sample,
    TextItem,
    TextMeta,
    TextReq,
    TextView,
)
from anytrain.codec import (
    SemanticGlobalCodec,
    SemanticGlobalCodes,
)

from speech_to_speech.datamodule._asset_provider import BiCodecProvider
from speech_to_speech.datamodule.parse import speech_from_codes
from speech_to_speech.datamodule.contract import DataRuntime
from speech_to_speech.datamodule.sample import Language
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.runtime.audio_tokenizer import BiCodecAudioTokenizer


class _FakeBiCodec:
    name = "bicodec"
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_frame_rate = 50.0
    semantic_codebook_sizes = (64,)
    global_codebook_sizes = (8, 9)
    global_unit_length = 2
    global_feature_dim = 4
    calls: list[tuple[tuple[int, ...], int]]

    def __init__(self) -> None:
        self.calls = []

    def tokenize(self, audio: torch.Tensor, sample_rate: int) -> SemanticGlobalCodes:
        self.calls.append((tuple(audio.shape), sample_rate))
        batch, _, frames = audio.shape
        semantic = (
            torch.arange(frames, dtype=torch.long)
            .view(1, frames, 1)
            .expand(batch, -1, -1)
            .contiguous()
        )
        slots = torch.arange(self.global_unit_length, dtype=torch.long)
        global_codes = torch.stack(
            (
                slots % self.global_codebook_sizes[0],
                (slots + 1) % self.global_codebook_sizes[1],
            ),
            dim=-1,
        ).expand(batch, -1, -1)
        return SemanticGlobalCodes(semantic=semantic, global_codes=global_codes)


@dataclass(frozen=True)
class _ProviderFactory:
    def __call__(self, device: str) -> BiCodecProvider:
        if device != "cpu":
            raise AssertionError(device)
        return BiCodecProvider(cast(SemanticGlobalCodec, _FakeBiCodec()))


class _PairDataset(MapStyleABC):
    def __init__(self) -> None:
        self.samples = (
            _sample(source_frames=4, target_frames=3, with_text=True),
            _sample(source_frames=2, target_frames=4, with_text=True),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]


@dataclass(frozen=True)
class _DatasetFactory:
    def __call__(self) -> _PairDataset:
        return _PairDataset()


class BiCodecProviderTest(unittest.TestCase):
    def test_batches_variable_lengths_and_multiple_roles(self) -> None:
        codec = _FakeBiCodec()
        provider = BiCodecProvider(cast(SemanticGlobalCodec, codec))
        source = (Role.SOURCE, Modality.AUDIO)
        target = (Role.TARGET, Modality.AUDIO)
        batch = collate_fn(
            {
                source: AudioReq(views=frozenset({AudioView.WAVEFORM})),
                target: AudioReq(views=frozenset({AudioView.WAVEFORM})),
            }
        )(
            [
                _sample(source_frames=4, target_frames=3),
                _sample(source_frames=2, target_frames=4),
            ]
        )

        outputs = provider.call_batch(batch)

        self.assertIsInstance(outputs, dict)
        source_outputs = outputs[source]
        target_outputs = outputs[target]
        self.assertEqual(
            [tuple(value["semantic"].shape) for value in source_outputs],
            [(4, 1), (2, 1)],
        )
        self.assertEqual(
            [tuple(value["semantic"].shape) for value in target_outputs],
            [(3, 1), (4, 1)],
        )
        for value in (*source_outputs, *target_outputs):
            self.assertEqual(set(value), {"semantic", "global"})
            self.assertEqual(tuple(value["global"].shape), (2, 2))
            self.assertEqual(value["semantic"].device.type, "cpu")
            self.assertTrue(value["semantic"].is_contiguous())
            self.assertEqual(value["global"].device.type, "cpu")
            self.assertTrue(value["global"].is_contiguous())
        self.assertEqual(
            codec.calls,
            [
                ((1, 1, 4), 16_000),
                ((1, 1, 2), 16_000),
                ((1, 1, 3), 16_000),
                ((1, 1, 4), 16_000),
            ],
        )

    def test_rejects_non_positive_global_unit_length(self) -> None:
        codec = _FakeBiCodec()
        codec.global_unit_length = 0

        with self.assertRaisesRegex(ValueError, "requires global units"):
            BiCodecProvider(cast(SemanticGlobalCodec, codec))

    def test_store_round_trips_through_workspace_loader(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            ViewMaterializer(
                root / "bicodec",
                split="train",
                batch_size=2,
                commit_samples=2,
                write_workers=0,
                keep_schema={
                    (Role.SOURCE, Modality.TEXT): TextReq(
                        views=frozenset({TextView.TEXT}),
                        meta=frozenset({TextMeta.LANG}),
                    ),
                    (Role.TARGET, Modality.TEXT): TextReq(
                        views=frozenset({TextView.TEXT}),
                        meta=frozenset({TextMeta.LANG}),
                    ),
                },
            ).write(
                dataset_factory=_DatasetFactory(),
                provider_factory=_ProviderFactory(),
                devices="cpu",
            )

            from zhuyin.datasets.wmt19 import moss_tts

            dataset = (
                moss_tts.codec(
                    moss_tts.Codec.BICODEC,
                    root=root,
                    split="train",
                )
                .filter(None)
                .load()
            )
            sample = dataset[0]

            for role, semantic_frames in ((Role.SOURCE, 4), (Role.TARGET, 3)):
                audio = sample[(role, Modality.AUDIO)]
                self.assertIsInstance(audio, AudioItem)
                value = audio.views[AudioView.BICODEC]
                self.assertEqual(set(value), {"semantic", "global"})
                self.assertEqual(tuple(value["semantic"].shape), (semantic_frames, 1))
                self.assertEqual(tuple(value["global"].shape), (2, 2))
                speech = speech_from_codes(
                    value,
                    text_token_ids=torch.tensor([1], dtype=torch.long),
                    language=Language.ZH,
                    duration_seconds=semantic_frames / 50.0,
                    runtime=_runtime(),
                )
                self.assertIsNone(speech.acoustic_codes)
                torch.testing.assert_close(speech.global_codes, value["global"])


def _sample(
    *,
    source_frames: int,
    target_frames: int,
    with_text: bool = False,
) -> Sample:
    sample: Sample = {
        (Role.SOURCE, Modality.AUDIO): AudioItem(
            views={
                AudioView.WAVEFORM: (
                    torch.arange(source_frames, dtype=torch.float32).unsqueeze(0),
                    16_000,
                )
            }
        ),
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={
                AudioView.WAVEFORM: (
                    torch.arange(target_frames, dtype=torch.float32).unsqueeze(0),
                    16_000,
                )
            }
        ),
    }
    if with_text:
        sample[(Role.SOURCE, Modality.TEXT)] = TextItem(
            views={TextView.TEXT: "source"},
            meta={TextMeta.LANG: Lang.ZH},
        )
        sample[(Role.TARGET, Modality.TEXT)] = TextItem(
            views={TextView.TEXT: "target"},
            meta={TextMeta.LANG: Lang.EN},
        )
    return sample


def _runtime() -> DataRuntime:
    return cast(
        DataRuntime,
        SimpleNamespace(
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
            audio_view=AudioView.BICODEC,
            audio_tokenizer=BiCodecAudioTokenizer(
                semantic_codebook_size=64,
                global_codebook_sizes=(8, 9),
                global_unit_length=2,
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
