from __future__ import annotations

import json
import pickle
import sys
import unittest
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
from speech_to_speech.model import Config as ModelConfig, ToyConfig
from speech_to_speech.model.acoustic import AcousticType
from speech_to_speech.runtime import (
    AudioRepresentation,
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
from speech_to_speech.stage import (
    PARAMETER_POLICY_SPECS,
    ParameterGroup,
    ParameterPolicyName,
    apply_parameter_policy,
    default_parameter_policy_config,
)
from speech_to_speech.task import Task
from scripts._overfit_config import overfit as parse_overfit
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


class ContractTest(unittest.TestCase):
    def test_dataloader_config_validates_loader_values(self):
        invalid = (
            ({"batch_size": 0, "num_workers": 0}, ValueError, "positive"),
            ({"batch_size": True, "num_workers": 0}, TypeError, "integer"),
            ({"batch_size": 1, "num_workers": -1}, ValueError, "non-negative"),
            (
                {"batch_size": 1, "num_workers": 0, "pin_memory": 1},
                TypeError,
                "boolean",
            ),
        )
        for kwargs, error, message in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(error, message):
                DataLoaderConfig(**kwargs)
        costs_invalid = (
            ({"enabled": True}, ValueError, "max_batch_frames"),
            (
                {"enabled": True, "max_batch_frames": 1, "planning_window": 0},
                ValueError,
                "planning_window",
            ),
            ({"enabled": 1}, TypeError, "boolean"),
        )
        for kwargs, error, message in costs_invalid:
            with self.subTest(costs=kwargs), self.assertRaisesRegex(error, message):
                DataLoaderCostsConfig(**kwargs)
        with self.assertRaisesRegex(TypeError, "DataLoaderCostsConfig"):
            DataLoaderConfig(
                batch_size=1,
                num_workers=0,
                costs=cast(DataLoaderCostsConfig, {}),
            )
        structured = OmegaConf.structured(DataLoaderConfig)
        merged = OmegaConf.merge(
            structured,
            {"batch_size": 8, "num_workers": 4, "pin_memory": True},
        )
        loader = OmegaConf.to_object(merged)
        self.assertIsInstance(loader, DataLoaderConfig)
        self.assertIsInstance(loader.costs, DataLoaderCostsConfig)
        self.assertFalse(loader.costs.enabled)
        with self.assertRaisesRegex(ValueError, "text loaders"):
            TextConfig(
                dataloader=DataLoaderConfig(
                    batch_size=1,
                    num_workers=0,
                    costs=DataLoaderCostsConfig(
                        enabled=True,
                        max_batch_frames=8,
                    ),
                )
            )

    def test_datamodule_configs_reject_mapping_loader_config(self):
        loader = cast(DataLoaderConfig, {"batch_size": 1, "num_workers": 0})

        with self.assertRaisesRegex(TypeError, "DataLoaderConfig"):
            SpeechConfig(codec="longcat", dataloader=loader)
        with self.assertRaisesRegex(TypeError, "DataLoaderConfig"):
            TextConfig(dataloader=loader)

    def test_runtime_loads_semantic_codec_artifact_with_existing_backend(self):
        backend = SimpleNamespace(name="longcat")
        support = SimpleNamespace()
        semantic_runtime = SimpleNamespace(sample_rate=24_000, frame_rate=50.0, decode=Mock())
        runtime = Runtime(
            Config(
                codec="longcat",
                semantic_codec_artifact="/tmp/semantic-codec",
            )
        )

        with (
            patch(
                "speech_to_speech.runtime.runtime.load_codec",
                return_value=backend,
            ),
            patch(
                "semantic_acoustic_codec.runtime.load_artifact",
                return_value=support,
            ) as load,
            patch(
                "semantic_acoustic_codec.runtime.SemanticCodecRuntime",
                return_value=semantic_runtime,
            ) as bind,
        ):
            loaded = runtime.semantic_codec

        self.assertIs(loaded, semantic_runtime)
        load.assert_called_once_with(
            Path("/tmp/semantic-codec"),
            device=None,
        )
        bind.assert_called_once_with(support, backend)

    def test_runtime_rejects_semantic_codec_without_artifact(self):
        runtime = Runtime(Config(codec="longcat"))

        with self.assertRaisesRegex(RuntimeError, "semantic_codec_artifact"):
            _ = runtime.semantic_codec

    def test_semantic_codec_artifact_disables_acoustic_side_channel(self):
        runtime = Runtime(
            Config(
                codec="longcat",
                semantic_codec_artifact="/tmp/semantic-codec",
            )
        )

        with patch("speech_to_speech.runtime.runtime.load_codec") as load:
            self.assertFalse(runtime.acoustic_side_channel)

        load.assert_not_called()

    def test_worker_runtime_snapshot_excludes_model_and_codec(self):
        runtime = SimpleNamespace(
            codec_name="longcat",
            audio_view=AudioView.LONGCAT,
            codec_frame_rate=50.0,
            audio_representation=AudioRepresentation.DECOUPLED,
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
            codec=object(),
            backbone=object(),
        )

        snapshot = pickle.loads(pickle.dumps(DataRuntimeSnapshot.from_runtime(runtime)))

        self.assertFalse(hasattr(snapshot, "codec"))
        self.assertFalse(hasattr(snapshot, "backbone"))
        self.assertEqual(snapshot.codec_frame_rate, 50.0)
        self.assertEqual(snapshot.layout.blocks, runtime.layout.blocks)
        self.assertIs(snapshot.layout, snapshot.layout)

    def test_trainer_presets_have_one_composable_schema(self):
        configs = [
            _compose(),
            _compose("trainer=ddp"),
            _compose(
                "experiment=unicodec_ddp_smoke",
            ),
        ]
        expected = {
            "accelerator",
            "devices",
            "strategy",
            "use_distributed_sampler",
            "precision",
            "max_epochs",
            "log_every_n_steps",
            "enable_checkpointing",
            "gradient_clip_val",
        }

        for config in configs:
            self.assertEqual(set(config.trainer), expected)
            self.assertEqual(config.trainer.devices, "auto")

        self.assertEqual(
            configs[1].trainer.strategy,
            "ddp_find_unused_parameters_true",
        )
        self.assertEqual(
            configs[2].trainer.strategy,
            "ddp_find_unused_parameters_true",
        )
        self.assertEqual(configs[2].trainer.precision, "bf16-mixed")
        self.assertTrue(configs[1].trainer.use_distributed_sampler)
        self.assertFalse(configs[2].trainer.use_distributed_sampler)

    @patch("scripts.overfit.pl.Trainer")
    @patch("scripts.overfit.build_logger")
    def test_overfit_trainer_consumes_the_unicodec_ddp_contract(
        self,
        logger,
        trainer,
    ):
        callbacks = [Callback()]
        with TemporaryDirectory() as output_root:
            config = parse_overfit(
                _compose(
                    "experiment=unicodec_ddp_smoke",
                    f"repo_output_root={output_root}",
                )
            )
            output_dir = Path(self.id())

            built = build_trainer(config, output_dir, callbacks)

        self.assertIs(built, trainer.return_value)
        kwargs = trainer.call_args.kwargs
        self.assertEqual(kwargs["devices"], "auto")
        self.assertEqual(kwargs["strategy"], "ddp_find_unused_parameters_true")
        self.assertEqual(kwargs["max_epochs"], -1)
        self.assertEqual(kwargs["precision"], "bf16-mixed")
        self.assertFalse(kwargs["use_distributed_sampler"])
        self.assertEqual(kwargs["gradient_clip_val"], 1.0)
        self.assertTrue(kwargs["enable_checkpointing"])
        self.assertEqual(kwargs["num_sanity_val_steps"], 0)
        self.assertIs(kwargs["logger"], logger.return_value)
        self.assertEqual(kwargs["callbacks"], callbacks)

    def test_public_configs_support_omegaconf_structured(self):
        runtime_config = OmegaConf.structured(Config)
        model_config = OmegaConf.structured(ModelConfig)

        self.assertIsNone(runtime_config.audio_tokenizer)
        self.assertIsNone(runtime_config.device)
        self.assertEqual(model_config.semantic_audio_adapter, "linear")
        self.assertEqual(model_config.audio_input_adapter.type, "mlp")
        self.assertEqual(model_config.audio_output_adapter.type, "none")

    def test_acoustic_presets_expose_only_supported_options(self):
        flow = _compose()
        rvq = _compose("model/acoustic=rvq")
        none = _compose("model/acoustic=none")

        self.assertEqual(flow.acoustic.type, "flow")
        self.assertEqual(flow.acoustic.repa.teacher_layer, 9)
        self.assertIn("student_layer", flow.acoustic.repa)
        self.assertNotIn("normalize_features", flow.acoustic)
        self.assertEqual(flow.runtime.codec, "longcat")
        self.assertEqual(flow.model.semantic_audio_adapter, "linear")
        self.assertEqual(rvq.acoustic.type, "rvq")
        self.assertNotIn("repa", rvq.acoustic)
        self.assertEqual(none.acoustic.type, "none")
        self.assertEqual(none.acoustic.name, "token")
        self.assertNotIn("decoder", none.acoustic)

    def test_overfit_acoustic_branch_constructs_evaluation_on_py39(self):
        class EvaluationReached(Exception):
            pass

        runtime = SimpleNamespace(
            layout=Mock(),
            codec=SimpleNamespace(acoustic_codebook_sizes=(1024,), frame_rate=50.0),
            acoustic_side_channel=True,
            backbone=Mock(),
            flow_matching=Mock(),
        )
        datamodule = Mock()
        datamodule.train_dataloader.return_value = [Mock()]
        model = Mock()
        with TemporaryDirectory() as output_dir:
            config = parse_overfit(
                _compose(
                    "runtime=longcat_native",
                    f"repo_output_root={output_dir}",
                    "output_subdir=contract-test",
                    "train.max_steps=1",
                    "model.acoustic.decoder.layers=1",
                    "model.acoustic.decoder.heads=1",
                    "model.acoustic.decoder.ffn_ratio=1",
                )
            )
            with (
                patch("scripts.overfit.pl.seed_everything"),
                patch("scripts.overfit.runtime_config", return_value=Mock()),
                patch("scripts.overfit.Runtime", return_value=runtime),
                patch("scripts.overfit.DataModule", return_value=datamodule),
                patch(
                    "scripts.overfit.build",
                    return_value=(AcousticType.FLOW, Mock(), model),
                ),
                patch(
                    "scripts.overfit.AcousticEvaluation", side_effect=EvaluationReached
                ),
            ):
                with self.assertRaises(EvaluationReached):
                    run(config)

    @patch("scripts.overfit.torch.cuda.set_device")
    def test_post_fit_generation_uses_runtime_device(self, set_device):
        module = Mock()
        module.parameters.return_value = iter(
            [SimpleNamespace(device=torch.device("cuda", 0))]
        )

        device = _prepare_generation_module(module, torch.device("cuda", 0))

        self.assertEqual(device, torch.device("cuda", 0))
        set_device.assert_called_once_with(torch.device("cuda", 0))
        module.to.assert_called_once_with(torch.device("cuda", 0))

    def test_task_is_the_modality_source_of_truth(self):
        self.assertIs(Task.S2ST.source_modality, Modality.AUDIO)
        self.assertIs(Task.S2ST.target_modality, Modality.AUDIO)
        self.assertTrue(Task.S2ST.uses_source_role)
        self.assertIs(Task.MT.source_modality, Modality.TEXT)
        self.assertIs(Task.MT.target_modality, Modality.TEXT)
        self.assertTrue(Task.MT.uses_source_role)
        self.assertIsNone(Task.AUDIO_AR.source_modality)
        self.assertIs(Task.ASR.target_modality, Modality.TEXT)
        self.assertFalse(Task.TTS.uses_source_role)
        self.assertIsNone(Task.PARALLEL_AR.target_modality)
        self.assertIsNone(Task.INTERLEAVED_AR.target_modality)
        self.assertTrue(Task.PARALLEL_AR.prediction_modality.supervises_text)
        self.assertTrue(Task.PARALLEL_AR.prediction_modality.supervises_audio)

    def test_runtime_separates_audio_id_capabilities(self):
        rt = Runtime(Config())
        rt.__dict__["text_tokenizer"] = _Tokenizer(10)
        rt.__dict__["audio_tokenizer"] = SimpleNamespace(vocab_size=3)

        self.assertEqual(rt.audio_head_range, (10, 16))
        self.assertEqual(rt.codec_audio_range, (10, 13))
        self.assertEqual(rt.audio_generation_allowed_ids, (10, 11, 12, 14))
        self.assertEqual(rt.mask_token_id, 15)
        self.assertNotIn(rt.boa_token_id, rt.audio_generation_allowed_ids)
        self.assertNotIn(rt.mask_token_id, rt.audio_generation_allowed_ids)
        self.assertTrue(rt.is_codec_audio_id(12))
        self.assertFalse(rt.is_codec_audio_id(rt.eoa_token_id))

    def test_unified_codec_uses_semantic_codes_without_acoustic_side_channel(self):
        item = AudioItem(views={AudioView.UNICODEC: torch.tensor([[1], [2], [3]])})
        semantic, acoustic = _parse_audio_item(item, AudioView.UNICODEC)

        self.assertTrue(torch.equal(semantic, torch.tensor([[1], [2], [3]])))
        self.assertIsNone(acoustic)

    def test_stable_codec_uses_full_codes_without_acoustic_side_channel(self):
        codes = torch.tensor([[1], [2], [3]])
        item = AudioItem(views={AudioView.STABLE: codes})

        semantic, acoustic = _parse_audio_item(item, AudioView.STABLE)

        self.assertIs(semantic, codes)
        self.assertIsNone(acoustic)

    def test_bicodec_runtime_rejects_fixed_length_structured_codec(self):
        with self.assertRaisesRegex(ValueError, "fixed-length structured codec"):
            Config(codec="bicodec")

    def test_parser_rejects_non_codec_audio_views(self):
        item = AudioItem(
            views={AudioView.WAVEFORM: torch.zeros(2, 2)},
            meta={},
        )

        with self.assertRaisesRegex(ValueError, "unsupported codec audio view"):
            _parse_audio_item(item, AudioView.WAVEFORM)

    @patch("speech_to_speech.runtime.backbone.hf.AutoModelForCausalLM.from_pretrained")
    def test_backbone_loading_forwards_runtime_configuration(self, from_pretrained):
        backbone = Mock()
        moved = Mock()
        backbone.to.return_value = moved
        from_pretrained.return_value = backbone
        rt = Runtime(
            Config(
                backbone="fake/backbone",
                device="cuda",
                dtype="bfloat16",
                attn_implementation="flash_attention_2",
            )
        )

        loaded = rt.backbone

        from_pretrained.assert_called_once_with(
            "fake/backbone",
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        backbone.to.assert_called_once_with("cuda")
        self.assertIs(loaded, moved)

    def test_runtime_dtype_is_explicit(self):
        self.assertIs(dtype("float16"), torch.float16)
        with self.assertRaisesRegex(ValueError, "unknown torch dtype"):
            dtype("not_a_dtype")

    def test_overfit_runtime_config_preserves_native_audio_tokenizer(self):
        with TemporaryDirectory() as output_root:
            config = parse_overfit(
                _compose(
                    "runtime=unicodec",
                    "model/acoustic=none",
                    "audio_sequence_layout=flattened",
                    "runtime.backbone=fake/backbone",
                    f"repo_output_root={output_root}",
                )
            )

        with patch.dict("os.environ", {"LOCAL_RANK": "1"}):
            result = runtime_config(config.runtime)

        self.assertEqual(result.codec, "unicodec")
        self.assertIsNone(result.audio_tokenizer)
        self.assertEqual(result.device, "cuda:1")

    @patch("anytrain.framework.flow_matching.ContinuousFlowRuntime")
    @patch("anytrain.framework.flow_matching.ODESampler")
    def test_runtime_forwards_flow_configuration(self, sampler, flow_runtime):
        configured_sampler = Mock()
        sampler.return_value = configured_sampler
        rt = Runtime(Config(flow_method="euler", flow_nfe=7, flow_num_steps=6))

        loaded = rt.flow_matching

        sampler.assert_called_once_with(
            method="euler",
            nfe=7,
            num_steps=6,
            return_intermediates=False,
        )
        flow_runtime.assert_called_once_with(sampler=configured_sampler)
        self.assertIs(loaded, flow_runtime.return_value)

    def test_audio_tokenizer_loads_an_explicit_artifact_path(self):
        tokenizer = SimpleNamespace()
        wrapped = SimpleNamespace()
        codec_bpe = Mock(return_value=tokenizer)
        module = ModuleType("zhuyin.tokenizers.codec_bpe")
        module.codec_bpe = codec_bpe
        modules = {
            "zhuyin": ModuleType("zhuyin"),
            "zhuyin.tokenizers": ModuleType("zhuyin.tokenizers"),
            "zhuyin.tokenizers.codec_bpe": module,
        }
        with (
            patch.dict(sys.modules, modules),
            patch.object(TorchCodecBPE, "wrap", return_value=wrapped) as wrap,
        ):
            loaded = audio_tokenizer("~/bpe/longcat/vocab_100k")

        codec_bpe.assert_called_once_with(Path("~/bpe/longcat/vocab_100k").expanduser())
        wrap.assert_called_once_with(tokenizer)
        self.assertIs(loaded, wrapped)

    def test_raw_text_is_encoded_at_the_datamodule_boundary(self):
        tokenizer = _Tokenizer(10)
        runtime = SimpleNamespace(
            audio_view=AudioView.LONGCAT,
            codec_frame_rate=50.0,
            audio_representation=AudioRepresentation.DECOUPLED,
            semantic_codec_artifact=None,
            acoustic_layout=AcousticLayout.FRAME_ALIGNED,
            acoustic_unit_length=None,
            text_tokenizer=tokenizer,
            audio_tokenizer=NativeAudioTokenizer(vocab_size=8),
        )
        raw = _raw_sample()

        pair = parse_sample(raw, runtime)

        self.assertTrue(torch.equal(pair.source.text_token_ids, torch.tensor([1, 2])))
        self.assertIs(pair.source.language, Language.ZH)
        self.assertIs(pair.target.language, Language.EN)
        self.assertEqual(pair.source.acoustic_codes.shape, (2, 1))
        self.assertTrue(
            torch.equal(pair.source.acoustic_codes, torch.tensor([[2], [3]]))
        )
        self.assertEqual(tokenizer.encoded, ("target text", False))

    def test_parse_sample_infers_duration_from_codec_frames_when_metadata_missing(self):
        runtime = _data_runtime()

        pair = parse_sample(_raw_sample_without_duration(), runtime)

        self.assertEqual(pair.source.duration_seconds, 0.04)
        self.assertEqual(pair.target.duration_seconds, 0.04)

    def test_build_sample_uses_inferred_audio_seconds_for_audio_tasks(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        pair = parse_sample(_raw_sample_without_duration(), runtime)

        tts = build_sample(pair, Task.TTS, runtime)
        s2st = build_sample(pair, Task.S2ST, runtime)

        self.assertEqual(tts.audio_seconds, 0.04)
        self.assertEqual(s2st.audio_seconds, 0.08)

    def test_single_collator_builds_tts_from_default_utterance(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        utterance = parse_single_sample(_raw_single_sample(), runtime)
        sample = build_single_sample(utterance, Task.TTS, runtime)

        batch = SingleCollator(runtime, {Task.TTS: 1.0})([_raw_single_sample()])

        self.assertEqual(sample.task, Task.TTS)
        self.assertEqual(batch.tasks, [Task.TTS])
        self.assertIsNotNone(batch.acoustic_target)
        supervised = batch.token_labels[batch.token_labels.ne(-100)]
        self.assertTrue(torch.equal(supervised, torch.tensor([10, 11, 19])))
        self.assertAlmostEqual(float(batch.audio_seconds[0].item()), 0.04)

    def test_single_collator_builds_asr_from_the_same_utterance_shape(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)

        batch = SingleCollator(runtime, {Task.ASR: 1.0})([_raw_single_sample()])

        self.assertEqual(batch.tasks, [Task.ASR])
        self.assertIsNone(batch.acoustic_target)
        supervised = batch.token_labels[batch.token_labels.ne(-100)]
        self.assertTrue(torch.equal(supervised, torch.tensor([1, 2, 1])))

    def test_single_text_task_does_not_require_or_encode_audio(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        runtime.codec = _EncodingCodec()

        batch = SingleCollator(
            runtime,
            {Task.TEXT_AR: 1.0},
            encode_missing_codes=True,
        )([_raw_single_waveform_sample()])

        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(batch.tasks, [Task.TEXT_AR])
        self.assertEqual(runtime.codec.calls, [])

    def test_single_collator_emits_raw_batch_only_for_explicit_waveform_fallback(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        raw = _raw_single_waveform_sample()

        with self.assertRaisesRegex(ValueError, "missing .* codec"):
            SingleCollator(runtime, {Task.TTS: 1.0})([raw])

        batch = SingleCollator(
            runtime,
            {Task.TTS: 1.0},
            encode_missing_codes=True,
        )([raw])

        self.assertIsInstance(batch, RawSpeechBatch)
        self.assertEqual(batch.tasks, [Task.TTS])
        target = batch.samples[0].target
        self.assertIsInstance(target, RawSpeech)
        self.assertEqual(target.sample_rate, 4)
        self.assertEqual(target.duration_seconds, 1.0)

    def test_on_device_codec_materializer_converts_raw_single_batch(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        runtime.codec = _EncodingCodec()
        raw = SingleCollator(
            runtime,
            {Task.TTS: 1.0},
            encode_missing_codes=True,
        )([_raw_single_waveform_sample()])

        with torch.autocast("cpu", dtype=torch.bfloat16):
            batch = OnDeviceCodecMaterializer(runtime)(
                raw,
                device=torch.device("cpu"),
            )

        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(batch.tasks, [Task.TTS])
        self.assertIsNotNone(batch.acoustic_target)
        self.assertEqual(runtime.codec.calls, [((1, 1, 4), 4)])
        self.assertEqual(runtime.codec.input_dtypes, [torch.float32])
        self.assertEqual(runtime.codec.autocast_enabled, [False])

    def test_bicodec_online_tokenize_stays_fp32_outside_autocast(self):
        runtime = _bicodec_data_runtime()
        raw = SingleCollator(
            runtime,
            {Task.TTS: 1.0},
            encode_missing_codes=True,
        )([_raw_single_waveform_sample()])

        with torch.autocast("cpu", dtype=torch.bfloat16):
            batch = OnDeviceCodecMaterializer(runtime)(
                raw,
                device=torch.device("cpu"),
            )

        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(batch.tasks, [Task.TTS])
        self.assertEqual(runtime.codec.calls, [((1, 1, 4), 4)])
        self.assertEqual(runtime.codec.input_dtypes, [torch.float32])
        self.assertEqual(runtime.codec.autocast_enabled, [False])

    def test_longcat_route_uses_frame_encode_when_codec_has_both_capabilities(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        runtime.codec = _EncodingCodec()
        raw = SingleCollator(
            runtime,
            {Task.TTS: 1.0},
            encode_missing_codes=True,
        )([_raw_single_waveform_sample()])

        with patch(
            "speech_to_speech.callback.codec.supports_structured",
            return_value=True,
        ), patch(
            "speech_to_speech.callback.codec.structured_codec",
        ) as structured:
            batch = OnDeviceCodecMaterializer(runtime)(raw, device=torch.device("cpu"))

        self.assertIsInstance(batch, ModelBatch)
        structured.assert_not_called()
        self.assertEqual(runtime.codec.calls, [((1, 1, 4), 4)])

    def test_pair_waveform_fallback_encodes_both_s2st_roles(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        runtime.codec = _EncodingCodec()
        sample = _raw_pair_waveform_sample()

        with self.assertRaisesRegex(ValueError, "missing .* codec"):
            Collator(runtime, {Task.S2ST: 1.0})([sample])

        raw = Collator(
            runtime,
            {Task.S2ST: 1.0},
            encode_missing_codes=True,
        )([sample])
        batch = OnDeviceCodecMaterializer(runtime)(raw, device=torch.device("cpu"))

        self.assertIsInstance(raw, RawSpeechBatch)
        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(batch.tasks, [Task.S2ST])
        self.assertEqual(
            runtime.codec.calls,
            [((1, 1, 4), 4), ((1, 1, 6), 4)],
        )

    def test_pair_waveform_fallback_encodes_only_task_audio_roles(self):
        for task, expected_shape in (
            (Task.S2TT, (1, 1, 4)),
            (Task.TTS, (1, 1, 6)),
        ):
            with self.subTest(task=task):
                runtime = _data_runtime()
                runtime.text_tokenizer = _ChatTokenizer(10)
                runtime.codec = _EncodingCodec()
                raw = Collator(
                    runtime,
                    {task: 1.0},
                    encode_missing_codes=True,
                )([_raw_pair_waveform_sample()])

                batch = OnDeviceCodecMaterializer(runtime)(
                    raw,
                    device=torch.device("cpu"),
                )

                self.assertIsInstance(batch, ModelBatch)
                self.assertEqual(batch.tasks, [task])
                self.assertEqual(runtime.codec.calls, [(expected_shape, 4)])

    def test_pair_waveform_fallback_can_mix_prepared_and_raw_samples(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        runtime.codec = _EncodingCodec()
        raw = Collator(
            runtime,
            {Task.S2ST: 1.0},
            encode_missing_codes=True,
        )([_raw_sample(), _raw_pair_waveform_sample()])

        batch = OnDeviceCodecMaterializer(runtime)(raw, device=torch.device("cpu"))

        self.assertIsInstance(raw, RawSpeechBatch)
        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(batch.tasks, [Task.S2ST, Task.S2ST])
        self.assertEqual(len(runtime.codec.calls), 2)

    def test_datamodule_can_select_single_shape_without_changing_pair_default(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(),
            shape=DataShape.SINGLE,
            encode_missing_codes=True,
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )

        with patch(
            "speech_to_speech.datamodule.module.load_dataset",
            return_value=[_raw_single_waveform_sample()],
        ):
            datamodule.setup()
            batch = next(iter(datamodule.train_dataloader()))

        self.assertIsInstance(batch, RawSpeechBatch)
        self.assertEqual(
            datamodule.loader_specs["train"].speech_config.shape,
            DataShape.SINGLE,
        )

    def test_datamodule_wires_waveform_fallback_for_pair_shape(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(),
            shape=DataShape.PAIR,
            encode_missing_codes=True,
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.S2ST: 1.0})},
        )

        with patch(
            "speech_to_speech.datamodule.module.load_dataset",
            return_value=[_raw_pair_waveform_sample()],
        ):
            datamodule.setup()
            batch = next(iter(datamodule.train_dataloader()))

        self.assertIsInstance(batch, RawSpeechBatch)
        self.assertEqual(batch.tasks, [Task.S2ST])

    def test_full_codec_sequence_flattens_complete_codes_without_acoustic_target(self):
        tokenizer = FlattenedAudioTokenizer(
            codebook_sizes=(8, 10),
            codec_name="longcat",
        )
        audio_start = 10
        runtime = SimpleNamespace(
            audio_view=AudioView.LONGCAT,
            codec_frame_rate=50.0,
            audio_representation=AudioRepresentation.FULL_CODEC_SEQUENCE,
            audio_sequence_layout=AudioSequenceLayout.FLATTENED,
            semantic_codec_artifact=None,
            acoustic_layout=AcousticLayout.FRAME_ALIGNED,
            acoustic_unit_length=None,
            text_tokenizer=_ChatTokenizer(10),
            audio_tokenizer=tokenizer,
            layout=Layout(
                text=(0, audio_start),
                audio=(audio_start, audio_start + tokenizer.vocab_size + 3),
            ),
            pad_token_id=0,
            eos_token_id=1,
            boa_token_id=audio_start + tokenizer.vocab_size,
            eoa_token_id=audio_start + tokenizer.vocab_size + 1,
            mask_token_id=audio_start + tokenizer.vocab_size + 2,
        )
        raw = _raw_sample()

        pair = parse_sample(raw, runtime)
        sample = build_sample(pair, Task.S2ST, runtime)

        source_codes = raw[(Role.SOURCE, Modality.AUDIO)].views[AudioView.LONGCAT]
        target_codes = raw[(Role.TARGET, Modality.AUDIO)].views[AudioView.LONGCAT]
        self.assertTrue(torch.equal(pair.source.semantic_codes, source_codes))
        self.assertIsNone(pair.source.acoustic_codes)
        self.assertTrue(torch.equal(pair.target.semantic_codes, target_codes))
        self.assertIsNone(pair.target.acoustic_codes)
        self.assertTrue(torch.equal(pair.target.audio_token_ids, tokenizer.encode(target_codes)))
        self.assertEqual(int(pair.target.audio_token_spans.sum().item()), target_codes.size(0))
        self.assertIsNone(sample.acoustic_target)

        supervised = sample.token_labels[sample.token_labels.ne(-100)]
        expected = torch.cat(
            [
                pair.target.audio_token_ids + audio_start,
                torch.tensor([runtime.eoa_token_id]),
            ]
        )
        self.assertTrue(torch.equal(supervised, expected))

    def test_text_parser_ignores_audio_fields(self):
        tokenizer = _Tokenizer(10)
        runtime = SimpleNamespace(text_tokenizer=tokenizer)

        pair = parse_text_sample(_raw_text_sample(), runtime)

        self.assertTrue(torch.equal(pair.source.text_token_ids, torch.tensor([1, 2])))
        self.assertIs(pair.source.language, Language.ZH)
        self.assertIs(pair.target.language, Language.EN)
        self.assertEqual(tokenizer.encoded, ("target text", False))

    def test_text_collator_builds_mt_batches_without_audio_runtime(self):
        runtime = SimpleNamespace(
            text_tokenizer=_ChatTokenizer(32),
            layout=Layout(text=(0, 32), audio=(32, 36)),
            pad_token_id=0,
            eos_token_id=31,
        )

        batch = TextCollator(runtime, {Task.MT: 1.0})([_raw_text_sample()])

        self.assertEqual(batch.tasks, [Task.MT])
        self.assertIsNone(batch.acoustic_target)
        self.assertTrue(batch.token_labels.ne(-100).any())
        labels = batch.token_labels[batch.token_labels.ne(-100)]
        self.assertTrue((labels >= 0).all())
        self.assertTrue((labels < 32).all())

    def test_text_collator_rejects_audio_tasks(self):
        runtime = SimpleNamespace(
            text_tokenizer=_ChatTokenizer(32),
            layout=Layout(text=(0, 32), audio=(32, 36)),
            pad_token_id=0,
            eos_token_id=31,
        )

        with self.assertRaisesRegex(ValueError, "text-only"):
            TextCollator(runtime, {Task.TTS: 1.0})

    @patch("anydataset.presets.WMT19")
    def test_text_dataset_config_loads_anydataset_wmt19(self, wmt19):
        config = TextDatasetConfig(
            name=TextDatasetName.WMT19,
            split="validation",
            source_lang="de",
            target_lang="en",
        )

        loaded = load_text_dataset(config)

        self.assertIs(loaded, wmt19.return_value)
        wmt19.assert_called_once_with(
            split="validation",
            source_lang="de",
            target_lang="en",
        )

    def test_text_datamodule_reads_toy_text_without_codec_runtime(self):
        runtime = SimpleNamespace(
            text_tokenizer=_ChatTokenizer(32),
            layout=Layout(text=(0, 32), audio=(32, 36)),
            pad_token_id=0,
            eos_token_id=31,
        )
        datamodule = DataModule(
            runtime,
            {
                "mt": LoaderSpec.text(
                    TextConfig(
                        dataloader=_loader(2),
                        dataset=TextDatasetConfig(
                            name=TextDatasetName.TOY,
                            toy_samples=2,
                        ),
                    ),
                    {Task.MT: 1.0},
                )
            },
        )

        datamodule.setup()
        batch = next(iter(datamodule.train_dataloader()))

        self.assertEqual(batch.input_ids.size(0), 2)
        self.assertEqual(batch.tasks, [Task.MT, Task.MT])
        self.assertIsNone(batch.acoustic_target)

    def test_text_validation_dataloader_limits_samples(self):
        runtime = SimpleNamespace(
            text_tokenizer=_ChatTokenizer(32),
            layout=Layout(text=(0, 32), audio=(32, 36)),
            pad_token_id=0,
            eos_token_id=31,
        )
        text_config = TextConfig(
            dataloader=_loader(4),
            dataset=TextDatasetConfig(
                name=TextDatasetName.TOY,
                toy_samples=5,
            ),
        )
        datamodule = DataModule(
            runtime,
            {"mt": LoaderSpec.text(text_config, {Task.MT: 1.0})},
            validation=LoaderSpec.text(
                text_config,
                {Task.MT: 1.0},
                max_samples=2,
            ),
        )

        datamodule.setup()
        batches = list(datamodule.val_dataloader())

        self.assertEqual(sum(batch.input_ids.size(0) for batch in batches), 2)
        self.assertTrue(
            all(task is Task.MT for batch in batches for task in batch.tasks)
        )

    def test_scheduled_dataloader_rotates_homogeneous_loaders_by_weight(self):
        speech = ModelBatch.from_samples([_sample(Task.TTS)], pad_token_id=99)
        mt = ModelBatch.from_samples([_sample(Task.MT)], pad_token_id=99)
        loader = ScheduledDataLoader(
            {"speech": [speech], "mt": [mt]},
            LoaderSchedule({"speech": 2.0, "mt": 1.0}),
        )

        iterator = iter(loader)
        tasks = [next(iterator).tasks[0] for _ in range(6)]

        self.assertEqual(
            tasks,
            [Task.TTS, Task.MT, Task.TTS, Task.TTS, Task.MT, Task.TTS],
        )

    def test_scheduled_dataloader_interleaves_one_accumulation_window(self):
        speech = ModelBatch.from_samples([_sample(Task.TTS)], pad_token_id=99)
        mt = ModelBatch.from_samples([_sample(Task.MT)], pad_token_id=99)
        with self.assertRaisesRegex(ValueError, "too small"):
            LoaderSchedule(
                {"speech": 9.0, "mt": 1.0},
                accumulate_grad_batches=8,
            )
        loader = ScheduledDataLoader(
            {"speech": [speech], "mt": [mt]},
            LoaderSchedule(
                {"speech": 2.0, "mt": 1.0},
                accumulate_grad_batches=3,
            ),
        )

        batches = list(islice(loader, 3))

        self.assertTrue(all(isinstance(batch, ModelBatch) for batch in batches))
        self.assertEqual(
            [batch.tasks[0] for batch in batches],
            [Task.TTS, Task.MT, Task.TTS],
        )

    def test_datamodule_sets_up_loaders_and_returns_scheduled_loader(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(32)
        speech = LoaderSpec.speech(
            SpeechConfig(
                codec="longcat",
                dataloader=_loader(),
                dataset=DatasetConfig(
                    name=DatasetName.TOY,
                    toy_samples=1,
                    toy_frames=2,
                ),
            ),
            {Task.TTS: 1.0},
        )
        mt = LoaderSpec.text(
            TextConfig(
                dataloader=_loader(),
                dataset=TextDatasetConfig(
                    name=TextDatasetName.TOY,
                    toy_samples=1,
                ),
            ),
            {Task.MT: 1.0},
        )
        datamodule = DataModule(
            runtime,
            {"speech": speech, "mt": mt},
            LoaderSchedule(
                {"speech": 1.0, "mt": 1.0},
                accumulate_grad_batches=2,
            ),
        )

        datamodule.setup("fit")
        loader = datamodule.train_dataloader()
        iterator = iter(loader)

        self.assertEqual(datamodule.schedule.accumulate_grad_batches, 2)
        batches = [next(iterator), next(iterator)]
        self.assertEqual([batch.tasks[0] for batch in batches], [Task.TTS, Task.MT])

    def test_datamodule_validates_loader_names(self):
        runtime = _data_runtime()
        speech = LoaderSpec.speech(
            SpeechConfig(
                codec="longcat",
                dataloader=_loader(),
                dataset=DatasetConfig(name=DatasetName.TOY),
            ),
            {Task.TTS: 1.0},
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            DataModule(
                runtime,
                {"speech": speech},
                LoaderSchedule({"speech": 1.0, "mt": 1.0}),
            )
        with self.assertRaisesRegex(ValueError, "finite positive"):
            LoaderSchedule({"speech": 0.0, "mt": 0.0})

    @patch("speech_to_speech.datamodule.module.load_dataset")
    def test_datamodule_setup_loads_dataset_once(self, load_dataset):
        load_dataset.return_value = []
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(),
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )

        datamodule.setup()
        datamodule.setup()

        load_dataset.assert_called_once_with(config.dataset, runtime)

    @patch("speech_to_speech.datamodule.module.load_dataset")
    def test_datamodule_keeps_standard_loader_for_non_store_dataset(
        self,
        load_dataset,
    ):
        load_dataset.return_value = [_raw_sample(), _raw_sample()]
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(2),
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )

        datamodule.setup()
        loader = cast(Any, datamodule.train_dataloader())

        self.assertIs(loader.dataset, load_dataset.return_value)
        self.assertEqual(loader.batch_size, 2)

    @patch("speech_to_speech.datamodule.module.load_dataset")
    def test_datamodule_rejects_enabled_costs_for_non_mapstyle_dataset(
        self,
        load_dataset,
    ):
        load_dataset.return_value = [_raw_sample(), _raw_sample()]
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=DataLoaderConfig(
                batch_size=2,
                num_workers=0,
                costs=DataLoaderCostsConfig(
                    enabled=True,
                    max_batch_frames=8,
                ),
            ),
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )
        datamodule.setup()
        with self.assertRaisesRegex(ValueError, "non-MapStyle"):
            datamodule.train_dataloader()

    @patch("speech_to_speech.datamodule.module.load_dataset")
    def test_datamodule_rejects_enabled_costs_for_fixed_sample_loader(
        self,
        load_dataset,
    ):
        load_dataset.return_value = [_raw_sample(), _raw_sample()]
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=DataLoaderConfig(
                batch_size=1,
                num_workers=0,
                costs=DataLoaderCostsConfig(
                    enabled=True,
                    max_batch_frames=8,
                ),
            ),
        )
        datamodule = DataModule(
            runtime,
            {
                "train": LoaderSpec.speech(
                    config,
                    {Task.TTS: 1.0},
                    sample_index=0,
                )
            },
        )
        datamodule.setup()
        with self.assertRaisesRegex(ValueError, "fixed-sample"):
            datamodule.train_dataloader()

    def test_toy_dataset_uses_codec_shapes_and_value_ranges(self):
        cases = (
            (
                "longcat",
                SimpleNamespace(
                    sample_rate=16_000,
                    semantic_feature_dim=4,
                    semantic_codebook=torch.zeros(5, 4),
                    codebook_sizes=(5, 3, 7),
                    acoustic_feature_dim=4,
                    acoustic_codebook_sizes=(3, 7),
                    acoustic_codes_to_features=Mock(),
                    decode_features=Mock(),
                    frame_rate=50.0,
                ),
                AudioView.LONGCAT,
                (5, 3, 7),
            ),
            (
                "unicodec",
                SimpleNamespace(
                    semantic_feature_dim=4,
                    semantic_codebook=torch.zeros(11, 4),
                    codebook_sizes=(11,),
                    frame_rate=50.0,
                ),
                AudioView.UNICODEC,
                (11,),
            ),
        )

        for codec_name, codec, view, sizes in cases:
            with self.subTest(codec=codec_name):
                dataset = ToyDataset(codec_name, codec, samples=2, frames=3)
                first = dataset[0]
                again = dataset[0]
                self.assertEqual(len(dataset), 2)
                for role in (Role.SOURCE, Role.TARGET):
                    item = first[(role, Modality.AUDIO)]
                    codes = item.views[view]
                    self.assertEqual(tuple(codes.shape), (3, len(sizes)))
                    for codebook, size in enumerate(sizes):
                        self.assertTrue((codes[:, codebook] >= 0).all())
                        self.assertTrue((codes[:, codebook] < size).all())
                    self.assertTrue(
                        torch.equal(codes, again[(role, Modality.AUDIO)].views[view])
                    )

    def test_datamodule_loads_toy_data_without_prepared_dataset(self):
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(),
            dataset=DatasetConfig(
                name=DatasetName.TOY,
                toy_samples=2,
                toy_frames=3,
            ),
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )

        datamodule.setup()

        self.assertEqual(
            len(
                datamodule.diagnostic_samples(
                    [0, 1],
                    split=SampleSplit.TRAIN,
                    loader_name="train",
                )
            ),
            2,
        )
        loader = cast(Any, datamodule.train_dataloader())
        self.assertEqual(loader.batch_sampler.max_batch_samples, 1)

    def test_datamodule_shards_child_loader_indices_across_ranks(self):
        runtime = _data_runtime()
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(2),
            dataset=DatasetConfig(
                name=DatasetName.TOY,
                toy_samples=6,
                toy_frames=3,
            ),
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )
        datamodule.setup()
        sampler = cast(Any, datamodule.train_dataloader()).batch_sampler

        rank_batches = []
        for rank in range(2):
            with patch(
                "anydataset.dataset.batching.rank",
                return_value=(2, rank),
            ):
                rank_batches.append(list(sampler))

        rank_indices = [
            {index for batch in batches for index in batch}
            for batches in rank_batches
        ]
        self.assertTrue(rank_indices[0].isdisjoint(rank_indices[1]))
        self.assertEqual(rank_indices[0] | rank_indices[1], set(range(6)))
        self.assertEqual(len(rank_batches[0]), len(rank_batches[1]))

    def test_datamodule_uses_anydataset_batches_for_store_backed_data(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.dict("os.environ", {"ANYDATASET_HOME": str(root / "cache")}):
                output = root / "dataset"
                DatasetWriter(
                    output,
                    dataset_id="toy-speech",
                    split="train",
                    max_shard_samples=2,
                ).write([_raw_sample(index) for index in range(4)])
                dataset = AnyDataset(
                    Spec(source=Source.STORE, path=str(output), split="train")
                )
                runtime = _data_runtime()
                config = SpeechConfig(
                    codec="longcat",
                    dataloader=_loader(2),
                )
                datamodule = DataModule(
                    runtime,
                    {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
                )

                with patch(
                    "speech_to_speech.datamodule.module.load_dataset",
                    return_value=dataset,
                ) as load:
                    datamodule.setup()
                    loader = cast(Any, datamodule.train_dataloader())

                load.assert_called_once()
                self.assertIs(loader.dataset, dataset)
                sampler = loader.batch_sampler
                self.assertIs(sampler.dataset, dataset)
                self.assertEqual(sampler.max_batch_memory, 2)
                self.assertEqual(sampler.max_batch_samples, 2)
                self.assertTrue(sampler.shuffle)
                _assert_store_local_batches(self, sampler)
                self.assertEqual(
                    len(
                        datamodule.diagnostic_samples(
                            [0, 1],
                            split=SampleSplit.TRAIN,
                            loader_name="train",
                        )
                    ),
                    2,
                )

    def test_datamodule_smoke_uses_audio_frame_costs_for_store_backed_data(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.dict("os.environ", {"ANYDATASET_HOME": str(root / "cache")}):
                output = root / "dataset"
                DatasetWriter(
                    output,
                    dataset_id="toy-speech",
                    split="train",
                    max_shard_samples=2,
                ).write([_raw_sample(index) for index in range(4)])
                dataset = AnyDataset(
                    Spec(source=Source.STORE, path=str(output), split="train")
                )
                runtime = _data_runtime()
                config = SpeechConfig(
                    codec="longcat",
                    dataloader=DataLoaderConfig(
                        batch_size=2,
                        num_workers=0,
                        costs=DataLoaderCostsConfig(
                            enabled=True,
                            max_batch_frames=8,
                            planning_window=4,
                        ),
                    ),
                )
                datamodule = DataModule(
                    runtime,
                    {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
                )

                with patch(
                    "speech_to_speech.datamodule.module.load_dataset",
                    return_value=dataset,
                ):
                    datamodule.setup()
                    loader = cast(Any, datamodule.train_dataloader())

                sampler = loader.batch_sampler
                self.assertIsNotNone(sampler.costs)
                self.assertEqual(sampler.costs[0], 4)
                self.assertEqual(sampler.max_batch_memory, 8)
                self.assertEqual(sampler.max_batch_samples, 2)
                self.assertEqual(sampler.planning_window, 4)
                _assert_store_local_batches(self, sampler)

    def test_datamodule_uses_store_backed_data_without_duration(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.dict("os.environ", {"ANYDATASET_HOME": str(root / "cache")}):
                output = root / "dataset"
                DatasetWriter(
                    output,
                    dataset_id="toy-speech",
                    split="train",
                    max_shard_samples=2,
                ).write([_raw_sample_without_duration(index) for index in range(4)])
                dataset = AnyDataset(
                    Spec(source=Source.STORE, path=str(output), split="train")
                )
                runtime = _data_runtime()
                runtime.text_tokenizer = _ChatTokenizer(10)
                config = SpeechConfig(
                    codec="longcat",
                    dataloader=_loader(2),
                )
                datamodule = DataModule(
                    runtime,
                    {"train": LoaderSpec.speech(config, {Task.S2ST: 1.0})},
                )

                with patch(
                    "speech_to_speech.datamodule.module.load_dataset",
                    return_value=dataset,
                ) as load:
                    datamodule.setup()
                    loader = cast(Any, datamodule.train_dataloader())

                load.assert_called_once()
                self.assertIs(loader.dataset, dataset)
                sampler = loader.batch_sampler
                self.assertIs(sampler.dataset, dataset)
                self.assertEqual(sampler.max_batch_memory, 2)
                self.assertEqual(sampler.max_batch_samples, 2)
                self.assertTrue(sampler.shuffle)
                _assert_store_local_batches(self, sampler)
                batch = next(iter(loader))
                self.assertEqual(batch.tasks, [Task.S2ST, Task.S2ST])
                torch.testing.assert_close(
                    batch.audio_seconds,
                    torch.tensor([0.08, 0.08]),
                )
                self.assertEqual(
                    len(
                        datamodule.diagnostic_samples(
                            [0, 1],
                            split=SampleSplit.TRAIN,
                            loader_name="train",
                        )
                    ),
                    2,
                )

    def test_datamodule_enabled_costs_require_audio_duration_metadata(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch.dict("os.environ", {"ANYDATASET_HOME": str(root / "cache")}):
                output = root / "dataset"
                DatasetWriter(
                    output,
                    dataset_id="toy-speech",
                    split="train",
                    max_shard_samples=2,
                ).write([_raw_sample_without_duration(index) for index in range(2)])
                dataset = AnyDataset(
                    Spec(source=Source.STORE, path=str(output), split="train")
                )
                runtime = _data_runtime()
                config = SpeechConfig(
                    codec="longcat",
                    dataloader=DataLoaderConfig(
                        batch_size=2,
                        num_workers=0,
                        costs=DataLoaderCostsConfig(
                            enabled=True,
                            max_batch_frames=8,
                        ),
                    ),
                )
                datamodule = DataModule(
                    runtime,
                    {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
                )

                with patch(
                    "speech_to_speech.datamodule.module.load_dataset",
                    return_value=dataset,
                ):
                    datamodule.setup()
                    loader = cast(Any, datamodule.train_dataloader())

                with self.assertRaisesRegex(ValueError, "duration metadata"):
                    _ = loader.batch_sampler.costs[0]

    def test_wmt19_loader_uses_default_filter(self):
        dataset = [Mock()]
        view = Mock()
        filtered = Mock()
        filtered.load.return_value = dataset
        view.filter.return_value = filtered
        moss_tts = SimpleNamespace(codec=Mock(return_value=view))
        wmt19 = ModuleType("zhuyin.datasets.wmt19")
        wmt19.moss_tts = moss_tts
        runtime = _data_runtime()

        with patch.dict(
            sys.modules,
            {
                "zhuyin": ModuleType("zhuyin"),
                "zhuyin.datasets": ModuleType("zhuyin.datasets"),
                "zhuyin.datasets.wmt19": wmt19,
            },
        ):
            loaded = load_dataset(DatasetConfig(), runtime)

        self.assertIs(loaded, dataset)
        moss_tts.codec.assert_called_once_with(
            "longcat",
            root=None,
            split="train",
        )
        view.filter.assert_called_once_with("speech_translation_v1")
        filtered.load.assert_called_once_with()

    def test_wmt19_loader_can_disable_filter(self):
        dataset = [Mock()]
        view = Mock()
        filtered = Mock()
        filtered.load.return_value = dataset
        view.filter.return_value = filtered
        moss_tts = SimpleNamespace(codec=Mock(return_value=view))
        wmt19 = ModuleType("zhuyin.datasets.wmt19")
        wmt19.moss_tts = moss_tts
        runtime = _data_runtime()

        with patch.dict(
            sys.modules,
            {
                "zhuyin": ModuleType("zhuyin"),
                "zhuyin.datasets": ModuleType("zhuyin.datasets"),
                "zhuyin.datasets.wmt19": wmt19,
            },
        ):
            loaded = load_dataset(DatasetConfig(filter=None), runtime)

        self.assertIs(loaded, dataset)
        moss_tts.codec.assert_called_once_with(
            "longcat",
            root=None,
            split="train",
        )
        view.filter.assert_called_once_with(None)
        filtered.load.assert_called_once_with()

    def test_toy_settings_reject_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ToyConfig(hidden_size=7, heads=2)
        with self.assertRaisesRegex(ValueError, "toy_samples"):
            DatasetConfig(name=DatasetName.TOY, toy_samples=0)
        codec = SimpleNamespace(
            sample_rate=16_000,
            semantic_feature_dim=4,
            semantic_codebook=torch.zeros(5, 4),
            codebook_sizes=(5, 4),
            acoustic_feature_dim=4,
            acoustic_codebook_sizes=(3,),
            acoustic_codes_to_features=Mock(),
            decode_features=Mock(),
            frame_rate=50.0,
        )
        with self.assertRaisesRegex(ValueError, "LongCat"):
            ToyDataset("longcat", codec)

    def test_split_manifest_dataset_reads_explicit_indices(self):
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "split.json"
            manifest.write_text(
                json.dumps({"version": 1, "splits": {"pilot": [2, 0]}}),
            )
            runtime = _data_runtime()
            config = DatasetConfig(
                name=DatasetName.TOY,
                split_manifest=str(manifest),
                split_label="pilot",
                toy_samples=3,
            )

            dataset = load_dataset(config, runtime)

            self.assertIsInstance(dataset, SplitManifestDataset)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.global_index(0), 2)
            self.assertEqual(dataset.global_index(1), 0)
            first = dataset[0]
            text = first[(Role.SOURCE, Modality.TEXT)].views[TextView.TEXT]
            self.assertEqual(text, "toy source 2")

    def test_split_manifest_rejects_invalid_indices(self):
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "split.json"
            manifest.write_text(
                json.dumps({"version": 1, "splits": {"train": [0, 0]}}),
            )
            config = DatasetConfig(
                name=DatasetName.TOY,
                split_manifest=str(manifest),
                toy_samples=2,
            )

            with self.assertRaisesRegex(ValueError, "repeats"):
                load_dataset(config, _data_runtime())

            manifest.write_text(
                json.dumps({"version": 1, "splits": {"train": [2]}}),
            )
            with self.assertRaisesRegex(IndexError, "outside"):
                load_dataset(config, _data_runtime())

    def test_datamodule_uses_split_manifest_as_store_backed_subset(self):
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "split.json"
            manifest.write_text(
                json.dumps({"version": 1, "splits": {"pilot": [3, 1]}}),
            )
            runtime = _data_runtime()
            config = SpeechConfig(
                codec="longcat",
                dataloader=_loader(2),
                dataset=DatasetConfig(
                    name=DatasetName.TOY,
                    split_manifest=str(manifest),
                    split_label="pilot",
                    toy_samples=4,
                ),
            )
            datamodule = DataModule(
                runtime,
                {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
            )

            datamodule.setup()
            loader = cast(Any, datamodule.train_dataloader())

            self.assertIsInstance(loader.dataset, SplitManifestDataset)
            self.assertEqual(loader.dataset.indices, (3, 1))
            self.assertIs(loader.batch_sampler.dataset, loader.dataset)
            self.assertTrue(loader.batch_sampler.shuffle)
            samples = datamodule.diagnostic_samples(
                [0, 1],
                split=SampleSplit.TRAIN,
                loader_name="train",
            )
            texts = [
                sample[(Role.SOURCE, Modality.TEXT)].views[TextView.TEXT]
                for sample in samples
            ]
            self.assertEqual(texts, ["toy source 3", "toy source 1"])

    def test_split_manifest_builder_binds_audit_fingerprint(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = root / "candidate.json"
            audit = root / "audit.json"
            candidate.write_text(
                json.dumps(
                    {
                        "dataset": "wmt19_tts_codec",
                        "codec": "longcat",
                        "train": [0, 1],
                        "dev": [2],
                        "test": [3],
                    }
                )
            )
            audit.write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "relative_path": "samples.parquet",
                                "sha256": "abc",
                                "parquet": {"num_rows": 4},
                            }
                        ],
                    }
                )
            )

            manifest = build_manifest(
                candidate,
                audit,
                Path("/stable/root"),
                split_method="sequential_no_sample_id",
            )

            self.assertEqual(manifest["dataset_length"], 4)
            self.assertEqual(manifest["split_method"], "sequential_no_sample_id")
            self.assertEqual(manifest["root_fingerprint"], {"samples.parquet": "abc"})
            self.assertEqual(manifest["splits"], {"train": [0, 1], "dev": [2], "test": [3]})

    def test_split_manifest_preserves_map_style_shuffle_groups(self):
        class GroupedDataset(MapStyleABC):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return cast(Any, index)

            def _shuffle(self, **kwargs):
                del kwargs
                yield (0, 1)
                yield (2, 3)

        dataset = SplitManifestDataset(
            GroupedDataset(),
            [3, 0],
            manifest=Path("/stable/split.json"),
            label="train",
        )

        groups = list(
            dataset._shuffle(
                shuffle=True,
                seed=0,
                epoch=0,
                num_replicas=1,
                rank=0,
            )
        )

        self.assertEqual(groups, [(1,), (0,)])

    def test_datamodule_rejects_runtime_codec_mismatch(self):
        config = SpeechConfig(
            codec="unicodec",
            dataloader=_loader(),
        )
        datamodule = DataModule(
            _data_runtime(),
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )

        with self.assertRaisesRegex(ValueError, "same codec"):
            datamodule.setup()

    def test_overfit_datamodule_repeats_only_the_selected_sample(self):
        samples = [object(), object()]
        config = SpeechConfig(
            codec="longcat",
            dataloader=_loader(),
        )
        collator = Mock(side_effect=lambda batch: batch)
        with (
            patch(
                "speech_to_speech.datamodule.module.load_dataset",
                return_value=samples,
            ) as load_dataset,
            patch("speech_to_speech.datamodule.module._collator", return_value=collator),
        ):
            datamodule = DataModule(
                _data_runtime(),
                {
                    "train": LoaderSpec.speech(
                        config,
                        {Task.TTS: 1.0},
                        sample_index=1,
                    )
                },
            )

            datamodule.setup()
            first_epoch = list(datamodule.train_dataloader())
            second_epoch = list(datamodule.train_dataloader())

        load_dataset.assert_called_once()
        self.assertEqual(first_epoch, [[samples[1]]])
        self.assertEqual(second_epoch, [[samples[1]]])

    def test_overfit_datamodule_rejects_runtime_codec_mismatch(self):
        config = SpeechConfig(
            codec="unicodec",
            dataloader=_loader(),
        )
        datamodule = DataModule(
            _data_runtime(),
            {
                "train": LoaderSpec.speech(
                    config,
                    {Task.TTS: 1.0},
                    sample_index=0,
                )
            },
        )

        with self.assertRaisesRegex(ValueError, "same codec"):
            datamodule.setup()

    def test_model_batch_rejects_mixed_execution_signatures(self):
        samples = [
            _sample(Task.ASR),
            _sample(Task.TEXT_AR),
        ]
        with self.assertRaisesRegex(ValueError, "same execution signature"):
            ModelBatch.from_samples(samples, pad_token_id=99)

    def test_model_batch_direct_constructor_maintains_batch_task_invariants(self):
        def batch(tasks: list[Task]) -> ModelBatch:
            predictions = [
                (
                    task.prediction_modality
                    if isinstance(task, Task)
                    else cast(object, task)
                )
                for task in tasks
            ]
            return ModelBatch(
                input_ids=torch.ones(2, 2, dtype=torch.long),
                token_labels=torch.ones(2, 2, dtype=torch.long),
                acoustic_target=None,
                tasks=tasks,
                predictions=predictions,  # type: ignore[arg-type]
                pad_token_id=99,
                generation_prompt_lengths=torch.ones(2, dtype=torch.long),
            )

        cases = (
            ([], ValueError, "one Task per row"),
            ([Task.ASR], ValueError, "one Task per row"),
            (
                [Task.ASR, Task.TEXT_AR],
                ValueError,
                "same execution signature",
            ),
        )

        for tasks, error, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(error, message):
                batch(tasks)

        with self.assertRaisesRegex(TypeError, "Task values"):
            batch([Task.ASR, cast(Task, "asr")])

        with self.assertRaisesRegex(ValueError, "at least one row"):
            ModelBatch(
                input_ids=torch.empty(0, 2, dtype=torch.long),
                token_labels=torch.empty(0, 2, dtype=torch.long),
                acoustic_target=None,
                tasks=[],
                predictions=[],
                pad_token_id=99,
            )

        with self.assertRaisesRegex(TypeError, "signed integer"):
            ModelBatch(
                input_ids=torch.ones(1, 2, dtype=torch.uint64),
                token_labels=torch.ones(1, 2, dtype=torch.long),
                acoustic_target=None,
                tasks=[Task.ASR],
                predictions=[Task.ASR.prediction_modality],
                pad_token_id=99,
            )

    def test_model_batch_accepts_unified_audio_target(self):
        batch = ModelBatch.from_samples([_sample(Task.TTS)], pad_token_id=99)

        self.assertIsNone(batch.acoustic_target)

    def test_model_batch_row_preserves_one_acoustic_target(self):
        batch = ModelBatch.from_samples(
            [
                _target_sample(torch.tensor([[1, 2]])),
                _target_sample(torch.tensor([[3, 4], [5, 6]])),
            ],
            pad_token_id=99,
        )

        row = batch.row(1)

        self.assertEqual(row.tasks, [Task.TTS])
        self.assertTrue(torch.equal(row.input_ids, batch.input_ids[1:2]))
        self.assertIsNotNone(row.acoustic_target)
        if row.acoustic_target is None or batch.acoustic_target is None:
            self.fail("acoustic target was dropped while selecting a row")
        self.assertTrue(
            torch.equal(
                row.acoustic_target["codes"],
                batch.acoustic_target["codes"][1:2],
            )
        )
        with self.assertRaises(IndexError):
            batch.row(2)

    def test_model_batch_owns_acoustic_target_position_constraints(self):
        def batch(position: int, codes: torch.Tensor | None = None) -> ModelBatch:
            return ModelBatch(
                input_ids=torch.tensor([[1, 4]]),
                token_labels=torch.tensor([[-100, 4]]),
                acoustic_target={
                    "semantic_codes": torch.tensor([[[1]]]),
                    "codes": (torch.tensor([[[1, 2]]]) if codes is None else codes),
                    "token_positions": torch.tensor([[position]]),
                },
                tasks=[Task.TTS],
                predictions=[Task.TTS.prediction_modality],
                pad_token_id=99,
            )

        with self.assertRaisesRegex(ValueError, "at least 1"):
            batch(0)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            batch(2)
        with self.assertRaisesRegex(ValueError, "whole padded frame"):
            batch(-1, torch.tensor([[[-1, 2]]]))

    def test_model_batch_rejects_padding_ids_inside_unpadded_acoustic_fields(self):
        samples = {
            "acoustic target codes": _target_sample(torch.tensor([[-1, 2]])),
            "target semantic codes": _target_sample(
                torch.tensor([[1, 2]]),
                semantic_codes=torch.tensor([[-1]]),
            ),
        }

        for name, sample in samples.items():
            with (
                self.subTest(name=name),
                self.assertRaisesRegex(
                    ValueError, f"{name} must contain non-negative codec IDs"
                ),
            ):
                ModelBatch.from_samples([sample], pad_token_id=99)

    def test_model_batch_rejects_malformed_acoustic_code_tensors(self):
        cases = (
            (
                _target_sample(torch.tensor([1, 2])),
                ValueError,
                "acoustic target codes must have shape",
            ),
            (
                _target_sample(torch.empty((0, 2), dtype=torch.long)),
                ValueError,
                "acoustic target codes must contain at least one frame",
            ),
            (
                _target_sample(torch.tensor([[1.0, 2.0]])),
                TypeError,
                "acoustic target codes must contain integer codec IDs",
            ),
            (
                _target_sample(
                    torch.tensor([[1, 2]]),
                    semantic_codes=torch.tensor([1]),
                ),
                ValueError,
                "target semantic codes must have shape",
            ),
            (
                _target_sample(
                    torch.tensor([[1, 2], [2, 1]]),
                    semantic_codes=torch.tensor([[1]]),
                ),
                ValueError,
                "semantic and acoustic codes must share the frame axis",
            ),
        )

        for sample, error, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(error, message):
                ModelBatch.from_samples([sample], pad_token_id=99)

    def test_task_allocation_tracks_weights_across_tiny_batches(self):
        allocation = allocate_tasks([Task.T2ST, Task.TTS], [1.0, 2.0], 6)
        self.assertEqual(allocation.count(Task.T2ST), 2)
        self.assertEqual(allocation.count(Task.TTS), 4)
        collator = Collator(Mock(), {Task.TTS: 1.0, Task.T2ST: 0.0})
        self.assertEqual(collator.tasks, [Task.TTS])

        weights = TaskWeights({Task.T2ST: 1.0, Task.TTS: 9.0})
        tiny_batches = [weights.allocate(1)[0] for _ in range(10)]

        self.assertEqual(tiny_batches.count(Task.T2ST), 1)
        self.assertEqual(tiny_batches.count(Task.TTS), 9)

    def test_task_weights_are_pickleable_for_spawn_workers(self):
        weights = TaskWeights({Task.T2ST: 1.0, Task.TTS: 9.0})
        weights.allocate(1)

        restored = pickle.loads(pickle.dumps(weights))

        self.assertEqual(restored.tasks, [Task.T2ST, Task.TTS])
        self.assertEqual(restored.prediction, None)
        self.assertIsInstance(restored.allocate(1)[0], Task)

    def test_parameter_policy_freezes_explicit_parameter_groups(self):
        model = _StageModel()

        counts = apply_parameter_policy(
            model,
            PARAMETER_POLICY_SPECS[ParameterPolicyName.SPEECH_INTERFACE],
        )

        self.assertGreater(counts[ParameterGroup.BACKBONE], 0)
        self.assertFalse(model.backbone.model.layers[0].weight.requires_grad)
        self.assertTrue(
            model.token_embedding.embeddings["audio"].weight.requires_grad
        )
        self.assertTrue(model.acoustic_decoder.head.weight.requires_grad)
        self.assertFalse(model.acoustic_decoder.decoder.embed_tokens.weight.requires_grad)
        self.assertFalse(model.acoustic_decoder.codebook_embeddings[-1].weight.requires_grad)
        self.assertFalse(model.acoustic_decoder.embedding_projections[-1].weight.requires_grad)

    def test_parameter_policy_callback_applies_on_fit_setup(self):
        model = _StageModel()
        callback = build_parameter_policy(
            default_parameter_policy_config(ParameterPolicyName.SPEECH_INTERFACE)
        )

        callback.setup(Mock(), cast(Any, SimpleNamespace(model=model)), "validate")
        self.assertIsNone(callback.summary)
        self.assertTrue(model.backbone.model.layers[0].weight.requires_grad)

        callback.setup(Mock(), cast(Any, SimpleNamespace(model=model)), "fit")

        self.assertIsNotNone(callback.summary)
        self.assertFalse(model.backbone.model.layers[0].weight.requires_grad)
        self.assertTrue(model.token_embedding.embeddings["audio"].weight.requires_grad)

    def test_partial_qwen_policy_unfreezes_top_layers_and_final_norm(self):
        model = _StageModel()

        apply_parameter_policy(
            model,
            PARAMETER_POLICY_SPECS[
                ParameterPolicyName.SPEECH_INTERFACE_TOP_THIRD
            ],
        )

        self.assertFalse(model.backbone.model.layers[0].weight.requires_grad)
        self.assertFalse(model.backbone.model.layers[1].weight.requires_grad)
        self.assertTrue(model.backbone.model.layers[2].weight.requires_grad)
        self.assertTrue(model.backbone.model.norm.weight.requires_grad)


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
        audio_representation=AudioRepresentation.DECOUPLED,
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
        audio_representation=AudioRepresentation.FULL_CODEC_SEQUENCE,
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


if __name__ == "__main__":
    unittest.main()
