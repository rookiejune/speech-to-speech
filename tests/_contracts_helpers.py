from __future__ import annotations

# ruff: noqa: F401

import json
import pickle
import sys
import unittest
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import torch
from anydataset import AnyDataset, Source, Spec
from anydataset.store import DatasetWriter
from anydataset.types import (
    AudioItem,
    AudioMeta,
    AudioView,
    Lang,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextView,
)
from hydra import compose, initialize_config_dir
from anytrain.codec import AcousticLayout, SemanticAcousticCodes
from anytrain.module.idspace import Layout
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf
from torch import nn
from anydataset.dataset import MapStyleABC

from speech_to_speech.callback import OnDeviceCodecMaterializer, build_parameter_policy
from speech_to_speech.datamodule.config import (
    DataLoaderConfig,
    DataLoaderCostsConfig,
    SpeechConfig,
)
from speech_to_speech.datamodule._helper.task import TaskWeights, allocate_tasks
from speech_to_speech.datamodule.collate.collator import Collator, TextCollator
from speech_to_speech.datamodule.dataset.speech import (
    DatasetConfig,
    DatasetName,
    SplitManifestDataset,
    ToyDataset,
    load_dataset,
)
from speech_to_speech.datamodule.collate.joint import LoaderSchedule, ScheduledDataLoader
from speech_to_speech.datamodule.module import DataModule, LoaderSpec
from speech_to_speech.datamodule.diagnostic import SampleSplit
from speech_to_speech.datamodule.build.single import SingleCollator
from speech_to_speech.datamodule.dataset.text import (
    TextConfig,
    TextDatasetConfig,
    TextDatasetName,
    load_text_dataset,
)
from speech_to_speech.datamodule.types import DataShape
from speech_to_speech.datamodule.parse.parser import (
    _parse_audio_item,
    parse_sample,
    parse_text_sample,
)
from speech_to_speech.datamodule.build.sample import build_sample
from speech_to_speech.datamodule.build.single import build_single_sample, parse_single_sample
from speech_to_speech.datamodule.protocol import DataRuntimeSnapshot
from speech_to_speech.datamodule.types import (
    Language,
    ModelBatch,
    ModelSample,
    RawSpeech,
    RawSpeechBatch,
)
from speech_to_speech.model import ToyConfig
from speech_to_speech.model.acoustic import AcousticType
from speech_to_speech.runtime import (
    AudioSequenceLayout,
    Config,
    Runtime,
)
from speech_to_speech.runtime.runtime import audio_tokenizer, dtype
from speech_to_speech.runtime.audio_tokenizer import (
    BiCodecAudioTokenizer,
    FlattenedAudioTokenizer,
    NativeAudioTokenizer,
    TorchCodecBPE,
)
from speech_to_speech.parameter_policy import (
    PARAMETER_POLICY_SPECS,
    ParameterGroup,
    ParameterPolicyName,
    apply_parameter_policy,
    default_parameter_policy_config,
)
from speech_to_speech.task import Task
from scripts._overfit_config import overfit as parse_overfit
from scripts._config_common import TokenModelConfig
from scripts._entry import runtime_config
from scripts.create_split_manifest import build_manifest
from scripts.overfit import (
    _prepare_generation_module,
    build_trainer,
    run,
)


class _Tokenizer:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def encode(self, text: str, *, add_special_tokens: bool = False):
        self.encoded = (text, add_special_tokens)
        return [1, 2]


class _ChatTokenizer(_Tokenizer):
    def apply_chat_template(self, conversation, **kwargs) -> str:
        del kwargs
        return f"<user>{conversation[0]['content']}</user><assistant>"


class _StageBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_hidden_layers=3)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(nn.Linear(1, 1) for _ in range(3))
        self.model.norm = nn.LayerNorm(1)


class _StageAcousticDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.embed_tokens = nn.Embedding(1, 1)
        self.codebook_embeddings = nn.ModuleList(nn.Embedding(1, 1) for _ in range(2))
        self.embedding_projections = nn.ModuleList(nn.Linear(1, 1) for _ in range(2))
        self.head = nn.Linear(1, 1)


class _TokenEmbedding(nn.Module):
    def __init__(self, *, rows: int = 1, dim: int = 1) -> None:
        super().__init__()
        self.embeddings = nn.ModuleDict({"audio": nn.Embedding(rows, dim)})
        self.adapters = nn.ModuleDict({"audio": nn.Linear(dim, dim)})


class _StageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _StageBackbone()
        self.token_embedding = _TokenEmbedding()
        self.audio_output_adapter = nn.Linear(1, 1)
        self.acoustic_decoder = _StageAcousticDecoder()




def _sample(task: Task) -> ModelSample:
    return ModelSample.from_sequence(
        torch.tensor([1, 2]),
        torch.tensor([-100, 2]),
        task=task,
        prediction=task.prediction_modality,
    )


def _target_sample(
    codes: torch.Tensor,
    *,
    semantic_codes: torch.Tensor | None = None,
) -> ModelSample:
    frames = codes.size(0)
    return ModelSample.from_sequence(
        torch.tensor([1, 4]),
        torch.tensor([-100, 4]),
        acoustic_target={
            "semantic_codes": (
                torch.ones((frames, 1), dtype=torch.long)
                if semantic_codes is None
                else semantic_codes
            ),
            "codes": codes,
            "token_positions": torch.ones(frames, dtype=torch.long),
        },
        task=Task.TTS,
        prediction=Task.TTS.prediction_modality,
    )


def _assert_store_local_batches(
    test: unittest.TestCase,
    sampler: object,
) -> None:
    batches = [tuple(sorted(batch)) for batch in cast(Any, sampler)]
    test.assertEqual(set(batches), {(0, 1), (2, 3)})
    for batch in batches:
        test.assertLessEqual(len(batch), 2)


@contextmanager
def _store_dataset(samples, *, split: str = "train"):
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        with patch.dict("os.environ", {"ANYDATASET_HOME": str(root / "cache")}):
            output = root / "dataset"
            DatasetWriter(
                output,
                dataset_id="toy-speech",
                split=split,
                max_shard_samples=2,
            ).write(samples)
            yield AnyDataset(Spec(source=Source.STORE, path=str(output), split=split))


def _speech_datamodule(runtime, config: SpeechConfig, task_weights) -> DataModule:
    return DataModule(
        runtime,
        {"train": LoaderSpec.speech(config, task_weights)},
    )


def _store_train_loader(
    test: unittest.TestCase,
    datamodule: DataModule,
    dataset: AnyDataset,
):
    with patch(
        "speech_to_speech.datamodule.module.load_dataset",
        return_value=dataset,
    ) as load:
        datamodule.setup()
        loader = cast(Any, datamodule.train_dataloader())

    load.assert_called_once()
    test.assertIs(loader.dataset, dataset)
    return loader


def _assert_store_sampler(
    test: unittest.TestCase,
    loader,
    dataset: AnyDataset,
    *,
    max_batch_memory: int = 2,
    max_batch_samples: int = 2,
):
    sampler = loader.batch_sampler
    test.assertIs(sampler.dataset, dataset)
    test.assertEqual(sampler.max_batch_memory, max_batch_memory)
    test.assertEqual(sampler.max_batch_samples, max_batch_samples)
    test.assertTrue(sampler.shuffle)
    _assert_store_local_batches(test, sampler)
    return sampler


def _load_wmt19_dataset(config: DatasetConfig):
    dataset = [Mock()]
    view = Mock()
    filtered = Mock()
    filtered.load.return_value = dataset
    view.filter.return_value = filtered
    moss_tts = SimpleNamespace(codec=Mock(return_value=view))
    wmt19 = ModuleType("zhuyin.datasets.wmt19")
    wmt19.moss_tts = moss_tts

    with patch.dict(
        sys.modules,
        {
            "zhuyin": ModuleType("zhuyin"),
            "zhuyin.datasets": ModuleType("zhuyin.datasets"),
            "zhuyin.datasets.wmt19": wmt19,
        },
    ):
        loaded = load_dataset(config, _data_runtime())

    return loaded, dataset, moss_tts, filtered, view


def _compose(*overrides: str, config_name: str = "overfit") -> DictConfig:
    root = Path(__file__).parents[1]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(root / "configs"),
    ):
        return compose(config_name=config_name, overrides=list(overrides))


def _data_runtime():
    return SimpleNamespace(
        codec_name="longcat",
        audio_view=AudioView.LONGCAT,
        codec_frame_rate=50.0,
        audio_sequence_layout=AudioSequenceLayout.SEMANTIC,
        semantic_codec_artifact=None,
        acoustic_layout=AcousticLayout.FRAME_ALIGNED,
        acoustic_unit_length=None,
        text_tokenizer=_Tokenizer(10),
        audio_tokenizer=NativeAudioTokenizer(vocab_size=8),
        layout=Layout(text=(0, 10), audio=(10, 20)),
        pad_token_id=0,
        eos_token_id=1,
        boa_token_id=18,
        eoa_token_id=19,
        mask_token_id=20,
        codec=_longcat_codec(),
    )


def _bicodec_data_runtime():
    tokenizer = BiCodecAudioTokenizer(
        semantic_vocab_size=8,
        acoustic_codebook_sizes=(3,),
        acoustic_unit_length=2,
    )
    audio_start = 10
    boa_token_id = audio_start + tokenizer.vocab_size
    return SimpleNamespace(
        codec_name="bicodec",
        audio_view=AudioView.BICODEC,
        codec_frame_rate=50.0,
        audio_sequence_layout=AudioSequenceLayout.FLATTENED,
        semantic_codec_artifact=None,
        acoustic_layout=AcousticLayout.FIXED_LENGTH,
        acoustic_unit_length=2,
        text_tokenizer=_ChatTokenizer(10),
        audio_tokenizer=tokenizer,
        layout=Layout(text=(0, 10), audio=(audio_start, boa_token_id + 3)),
        pad_token_id=0,
        eos_token_id=1,
        boa_token_id=boa_token_id,
        eoa_token_id=boa_token_id + 1,
        mask_token_id=boa_token_id + 2,
        codec=_StructuredEncodingCodec(),
    )


def _loader(batch_size: int = 1) -> DataLoaderConfig:
    return DataLoaderConfig(batch_size=batch_size, num_workers=0)


def _longcat_codec():
    return SimpleNamespace(
        sample_rate=16_000,
        semantic_feature_dim=4,
        semantic_codebook=torch.zeros(5, 4),
        codebook_sizes=(5, 3),
        acoustic_feature_dim=4,
        acoustic_codebook_sizes=(3,),
        acoustic_codes_to_features=Mock(),
        decode_features=Mock(),
        frame_rate=50.0,
    )


class _EncodingCodec:
    sample_rate = 16_000
    frame_rate = 50.0
    codebook_sizes = (5, 3)

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.input_dtypes: list[torch.dtype] = []
        self.autocast_enabled: list[bool] = []

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self.calls.append((tuple(audio.shape), sample_rate))
        self.input_dtypes.append(audio.dtype)
        self.autocast_enabled.append(torch.is_autocast_enabled(audio.device.type))
        return audio.new_tensor([[[0, 2], [1, 3]]], dtype=torch.long)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return codes.new_zeros((codes.size(0), 1, codes.size(1)), dtype=torch.float32)


class _StructuredEncodingCodec:
    sample_rate = 16_000
    frame_rate = 50.0
    semantic_codebook = torch.zeros(8, 4)
    semantic_codebook_sizes = (8,)
    acoustic_codebook_sizes = (3,)
    acoustic_layout = AcousticLayout.FIXED_LENGTH
    acoustic_unit_length = 2
    acoustic_feature_dim = 4

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int]] = []
        self.input_dtypes: list[torch.dtype] = []
        self.autocast_enabled: list[bool] = []

    def tokenize(
        self,
        audio: torch.Tensor,
        sample_rate: int,
    ) -> SemanticAcousticCodes:
        self.calls.append((tuple(audio.shape), sample_rate))
        self.input_dtypes.append(audio.dtype)
        self.autocast_enabled.append(torch.is_autocast_enabled(audio.device.type))
        return SemanticAcousticCodes(
            semantic=torch.tensor([[[1], [2]]], device=audio.device),
            acoustic=torch.tensor([[[0], [1]]], device=audio.device),
        )

    def detokenize(self, codes: object) -> torch.Tensor:
        del codes
        return torch.zeros(1, 1, 1)

    def acoustic_codes_to_features(self, acoustic_codes: torch.Tensor) -> torch.Tensor:
        return acoustic_codes.to(dtype=torch.float32)

    def decode_features(
        self,
        semantic_codes: torch.Tensor,
        acoustic_features: torch.Tensor,
    ) -> torch.Tensor:
        del semantic_codes
        return acoustic_features.new_zeros((1, 1, 1))


def _raw_sample(index: int = 0):
    def audio(offset: int) -> AudioItem:
        return AudioItem(
            views={
                AudioView.LONGCAT: torch.tensor(
                    [[offset, offset + 2], [offset + 1, offset + 3]]
                )
            },
            meta={AudioMeta.DURATION: 0.04},
        )

    return {
        (Role.SOURCE, Modality.AUDIO): audio(index),
        (Role.SOURCE, Modality.TEXT): TextItem(
            views={TextView.TEXT: "source text"},
            meta={TextMeta.LANG: Lang.ZH},
        ),
        (Role.TARGET, Modality.AUDIO): audio(index + 4),
        (Role.TARGET, Modality.TEXT): TextItem(
            views={TextView.TEXT: "target text"},
            meta={TextMeta.LANG: Lang.EN},
        ),
    }


def _raw_single_sample(index: int = 0):
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={
                AudioView.LONGCAT: torch.tensor(
                    [[index, index + 2], [index + 1, index + 3]]
                )
            },
            meta={AudioMeta.DURATION: 0.04},
        ),
        (Role.DEFAULT, Modality.TEXT): TextItem(
            views={TextView.TEXT: "single text"},
            meta={TextMeta.LANG: Lang.EN},
        ),
    }


def _raw_single_waveform_sample():
    return {
        (Role.DEFAULT, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (torch.ones(1, 4), 4)},
            meta={},
        ),
        (Role.DEFAULT, Modality.TEXT): TextItem(
            views={TextView.TEXT: "single text"},
            meta={TextMeta.LANG: Lang.EN},
        ),
    }


def _raw_pair_waveform_sample():
    return {
        (Role.SOURCE, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (torch.ones(1, 4), 4)},
            meta={},
        ),
        (Role.SOURCE, Modality.TEXT): TextItem(
            views={TextView.TEXT: "source text"},
            meta={TextMeta.LANG: Lang.ZH},
        ),
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.WAVEFORM: (torch.ones(1, 6), 4)},
            meta={},
        ),
        (Role.TARGET, Modality.TEXT): TextItem(
            views={TextView.TEXT: "target text"},
            meta={TextMeta.LANG: Lang.EN},
        ),
    }


def _raw_sample_without_duration(index: int = 0):
    sample = _raw_sample(index)
    for role in (Role.SOURCE, Role.TARGET):
        sample[(role, Modality.AUDIO)].meta.pop(AudioMeta.DURATION)
    return sample


def _raw_text_sample():
    return {
        (Role.SOURCE, Modality.TEXT): TextItem(
            views={TextView.TEXT: "source text"},
            meta={TextMeta.LANG: Lang.ZH},
        ),
        (Role.TARGET, Modality.TEXT): TextItem(
            views={TextView.TEXT: "target text"},
            meta={TextMeta.LANG: Lang.EN},
        ),
    }



__all__ = [name for name in globals() if not name.startswith("__")]
