from __future__ import annotations

import json
import multiprocessing
import pickle
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace
from typing import Any, Protocol, cast
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
from anytrain.module.idspace import Layout
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf
from torch import nn

from speech_to_speech.callback import OnDeviceCodecMaterializer
from speech_to_speech.datamodule.collator import Collator, TextCollator, _allocate_tasks
from speech_to_speech.datamodule.dataset import (
    DatasetConfig,
    DatasetName,
    SplitManifestDataset,
    ToyDataset,
    load_dataset,
)
from speech_to_speech.datamodule.joint import LoaderSchedule, ScheduledDataLoader
from speech_to_speech.datamodule.module import Config as DataConfig
from speech_to_speech.datamodule.module import DataModule, LoaderSpec
from speech_to_speech.datamodule.single import SingleCollator
from speech_to_speech.datamodule.text import (
    TextConfig,
    TextDatasetConfig,
    TextDatasetName,
    load_text_dataset,
)
from speech_to_speech.datamodule.types import DataShape
from speech_to_speech.datamodule.parser import (
    _parse_audio_item,
    parse_sample,
    parse_text_sample,
)
from speech_to_speech.datamodule.sample import build_sample
from speech_to_speech.datamodule.single import build_single_sample, parse_single_sample
from speech_to_speech.datamodule.protocol import DataRuntime, DataRuntimeSnapshot
from speech_to_speech.datamodule.types import (
    Language,
    ModelBatch,
    ModelSample,
    RawSingleBatch,
)
from speech_to_speech.model import Config as ModelConfig, ToyConfig
from speech_to_speech.runtime import AudioRepresentation, Config, Runtime
from speech_to_speech.runtime.runtime import audio_tokenizer, dtype
from speech_to_speech.runtime.audio_tokenizer import (
    FlattenedAudioTokenizer,
    NativeAudioTokenizer,
    TorchCodecBPE,
)
from speech_to_speech.stage import (
    PARAMETER_POLICY_SPECS,
    ParameterGroup,
    ParameterPolicyName,
    apply_parameter_policy,
)
from speech_to_speech.task import Task
from scripts._config import overfit as parse_overfit
from scripts.create_split_manifest import build_manifest
from scripts.overfit import (
    _prepare_generation_module,
    build_trainer,
    run,
    runtime_config,
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


class _Event(Protocol):
    def set(self) -> None: ...

    def wait(self, timeout: float | None = None) -> bool: ...


class _Queue(Protocol):
    def put(self, value: list[Task]) -> None: ...

    def get(self, *, timeout: float) -> list[Task]: ...

    def close(self) -> None: ...


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


class _StageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _StageBackbone()
        self.semantic_audio_embedding = nn.Embedding(1, 1)
        self.semantic_audio_adapter = nn.Linear(1, 1)
        self.semantic_audio_output_adapter = nn.Linear(1, 1)
        self.acoustic_decoder = _StageAcousticDecoder()


def _observe_task_updates(
    collator: Collator,
    ready: _Event,
    updated: _Event,
    output: _Queue,
) -> None:
    output.put(collator.tasks)
    ready.set()
    if updated.wait(timeout=5.0):
        output.put(collator.tasks)


class ContractTest(unittest.TestCase):
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
            text_tokenizer=_Tokenizer(10),
            audio_tokenizer=NativeAudioTokenizer(vocab_size=8),
            layout=Layout(text=(0, 10), audio=(10, 20)),
            pad_token_id=0,
            eos_token_id=1,
            boa_token_id=18,
            eoa_token_id=19,
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
        config = parse_overfit(_compose("experiment=unicodec_ddp_smoke"))
        callbacks = [Callback()]
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
        self.assertIs(kwargs["logger"], logger.return_value)
        self.assertEqual(kwargs["callbacks"], callbacks)

    def test_public_configs_support_omegaconf_structured(self):
        runtime_config = OmegaConf.structured(Config)
        model_config = OmegaConf.structured(ModelConfig)

        self.assertIsNone(runtime_config.audio_tokenizer)
        self.assertIsNone(runtime_config.device)
        self.assertEqual(model_config.semantic_audio_adapter, "linear")
        self.assertEqual(model_config.semantic_audio_output_adapter, "linear")

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
                    "acoustic.decoder.layers=1",
                    "acoustic.decoder.heads=1",
                    "acoustic.decoder.ffn_ratio=1",
                )
            )
            with (
                patch("scripts.overfit.pl.seed_everything"),
                patch("scripts.overfit.runtime_config", return_value=Mock()),
                patch("scripts.overfit.Runtime", return_value=runtime),
                patch("scripts.overfit.DataModule", return_value=datamodule),
                patch("scripts.overfit.flow", return_value=(Mock(), model, None)),
                patch("scripts.overfit.apply_parameter_policy") as apply_policy,
                patch(
                    "scripts.overfit.AcousticEvaluation", side_effect=EvaluationReached
                ),
            ):
                with self.assertRaises(EvaluationReached):
                    run(config)
                apply_policy.assert_called_once_with(
                    model,
                    config.parameter_policy.spec(),
                )

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

    def test_runtime_separates_audio_id_capabilities(self):
        rt = Runtime(Config())
        rt.__dict__["text_tokenizer"] = _Tokenizer(10)
        rt.__dict__["audio_tokenizer"] = SimpleNamespace(vocab_size=3)

        self.assertEqual(rt.audio_head_range, (10, 15))
        self.assertEqual(rt.codec_audio_range, (10, 13))
        self.assertEqual(rt.audio_generation_allowed_ids, (10, 11, 12, 14))
        self.assertNotIn(rt.boa_token_id, rt.audio_generation_allowed_ids)
        self.assertTrue(rt.is_codec_audio_id(12))
        self.assertFalse(rt.is_codec_audio_id(rt.eoa_token_id))

    def test_unified_codec_uses_semantic_codes_without_acoustic_side_channel(self):
        item = AudioItem(views={AudioView.UNICODEC: torch.tensor([[1], [2], [3]])})
        semantic, acoustic = _parse_audio_item(item, AudioView.UNICODEC)

        self.assertTrue(torch.equal(semantic, torch.tensor([[1], [2], [3]])))
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

    @patch("speech_to_speech.runtime.runtime.AutoModelForCausalLM.from_pretrained")
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
        config = parse_overfit(
            _compose(
                "runtime=unicodec",
                "model/acoustic=none",
                "runtime.backbone=fake/backbone",
            )
        )

        with patch.dict("os.environ", {"LOCAL_RANK": "1"}):
            result = runtime_config(config)

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

        self.assertIsInstance(batch, RawSingleBatch)
        self.assertEqual(batch.tasks, [Task.TTS])
        self.assertEqual(batch.samples[0].sample_rate, 4)
        self.assertEqual(batch.samples[0].duration_seconds, 1.0)

    def test_on_device_codec_materializer_converts_raw_single_batch(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        runtime.codec = _EncodingCodec()
        raw = SingleCollator(
            runtime,
            {Task.TTS: 1.0},
            encode_missing_codes=True,
        )([_raw_single_waveform_sample()])

        batch = OnDeviceCodecMaterializer(runtime)(raw, device=torch.device("cpu"))

        self.assertIsInstance(batch, ModelBatch)
        self.assertEqual(batch.tasks, [Task.TTS])
        self.assertIsNotNone(batch.acoustic_target)
        self.assertEqual(runtime.codec.calls, [((1, 1, 4), 4)])

    def test_datamodule_can_select_single_shape_without_changing_pair_default(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(10)
        config = DataConfig(
            codec="longcat",
            dataloader={"batch_size": 1, "num_workers": 0},
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

        self.assertIsInstance(batch, RawSingleBatch)
        self.assertEqual(
            datamodule.loader_specs["train"].speech_config.shape,
            DataShape.SINGLE,
        )

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
            text_tokenizer=_ChatTokenizer(10),
            audio_tokenizer=tokenizer,
            layout=Layout(
                text=(0, audio_start),
                audio=(audio_start, audio_start + tokenizer.vocab_size + 2),
            ),
            pad_token_id=0,
            eos_token_id=1,
            boa_token_id=audio_start + tokenizer.vocab_size,
            eoa_token_id=audio_start + tokenizer.vocab_size + 1,
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
                        dataloader={"batch_size": 2, "num_workers": 0},
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

    def test_scheduled_dataloader_can_emit_fixed_joint_steps(self):
        speech = ModelBatch.from_samples([_sample(Task.TTS)], pad_token_id=99)
        mt = ModelBatch.from_samples([_sample(Task.MT)], pad_token_id=99)
        with self.assertRaisesRegex(ValueError, "too small"):
            LoaderSchedule({"speech": 9.0, "mt": 1.0}, batches_per_step=8)
        loader = ScheduledDataLoader(
            {"speech": [speech], "mt": [mt]},
            LoaderSchedule({"speech": 2.0, "mt": 1.0}, batches_per_step=3),
        )

        batch = next(iter(loader))

        self.assertIsInstance(batch, tuple)
        self.assertEqual([item.tasks[0] for item in batch], [Task.TTS, Task.TTS, Task.MT])

    def test_datamodule_sets_up_loaders_and_returns_scheduled_loader(self):
        runtime = _data_runtime()
        runtime.text_tokenizer = _ChatTokenizer(32)
        speech = LoaderSpec.speech(
            DataConfig(
                codec="longcat",
                dataloader={"batch_size": 1, "num_workers": 0},
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
                dataloader={"batch_size": 1, "num_workers": 0},
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
            LoaderSchedule({"speech": 1.0, "mt": 1.0}, batches_per_step=2),
        )

        datamodule.set_loader_weights({"speech": 1.0, "mt": 1.0})
        datamodule.setup("fit")
        loader = datamodule.train_dataloader()
        iterator = iter(loader)

        self.assertEqual(datamodule.schedule.batches_per_step, 2)
        batch = next(iterator)
        self.assertIsInstance(batch, tuple)
        self.assertEqual([item.tasks[0] for item in batch], [Task.TTS, Task.MT])

    def test_datamodule_validates_loader_names(self):
        runtime = _data_runtime()
        speech = LoaderSpec.speech(
            DataConfig(
                codec="longcat",
                dataloader={"batch_size": 1, "num_workers": 0},
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

    def test_datamodule_keeps_schedule_when_loader_weight_update_is_invalid(self):
        runtime = _data_runtime()
        config = DataConfig(
            codec="longcat",
            dataloader={"batch_size": 1, "num_workers": 0},
            dataset=DatasetConfig(name=DatasetName.TOY),
        )
        datamodule = DataModule(
            runtime,
            {"speech": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )
        original = datamodule.schedule

        with self.assertRaisesRegex(ValueError, "missing"):
            datamodule.set_loader_weights({"other": 1.0})

        self.assertIs(datamodule.schedule, original)

    @patch("speech_to_speech.datamodule.module.load_dataset")
    def test_datamodule_setup_loads_dataset_once(self, load_dataset):
        load_dataset.return_value = []
        runtime = _data_runtime()
        config = DataConfig(
            codec="longcat",
            dataloader={"batch_size": 1, "num_workers": 0},
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
        config = DataConfig(
            codec="longcat",
            dataloader={"batch_size": 2, "num_workers": 0},
        )
        datamodule = DataModule(
            runtime,
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )

        datamodule.setup()
        loader = cast(Any, datamodule.train_dataloader())

        self.assertIs(loader.dataset, load_dataset.return_value)
        self.assertEqual(loader.batch_size, 2)

    def test_toy_dataset_uses_codec_shapes_and_value_ranges(self):
        cases = (
            (
                "longcat",
                SimpleNamespace(
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
        config = DataConfig(
            codec="longcat",
            dataloader={"batch_size": 1, "num_workers": 0},
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

        self.assertEqual(len(datamodule.train_samples([0, 1])), 2)
        loader = cast(Any, datamodule.train_dataloader())
        self.assertEqual(loader.batch_size, 1)

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
                config = DataConfig(
                    codec="longcat",
                    dataloader={"batch_size": 2, "num_workers": 0},
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
                self.assertEqual(len(datamodule.train_samples([0, 1])), 2)

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
                config = DataConfig(
                    codec="longcat",
                    dataloader={"batch_size": 2, "num_workers": 0},
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
                self.assertEqual(len(datamodule.train_samples([0, 1])), 2)

    def test_toy_settings_reject_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ToyConfig(hidden_size=7, heads=2)
        with self.assertRaisesRegex(ValueError, "toy_samples"):
            DatasetConfig(name=DatasetName.TOY, toy_samples=0)
        codec = SimpleNamespace(
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
            config = DataConfig(
                codec="longcat",
                dataloader={"batch_size": 2, "num_workers": 0},
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
            samples = datamodule.train_samples([0, 1])
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
                        "dataset_len": 4,
                        "split_candidate": {"method": "sequential_no_sample_id"},
                        "files": [
                            {"relative_path": "samples.parquet", "sha256": "abc"}
                        ],
                    }
                )
            )

            manifest = build_manifest(candidate, audit, Path("/stable/root"))

            self.assertEqual(manifest["dataset_length"], 4)
            self.assertEqual(manifest["split_method"], "sequential_no_sample_id")
            self.assertEqual(manifest["root_fingerprint"], {"samples.parquet": "abc"})
            self.assertEqual(manifest["splits"], {"train": [0, 1], "dev": [2], "test": [3]})

    def test_datamodule_rejects_runtime_codec_mismatch(self):
        config = DataConfig(
            codec="unicodec",
            dataloader={"batch_size": 1, "num_workers": 0},
        )
        datamodule = DataModule(
            _data_runtime(),
            {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
        )

        with self.assertRaisesRegex(ValueError, "same codec"):
            datamodule.setup()

    def test_overfit_datamodule_repeats_only_the_selected_sample(self):
        samples = [object(), object()]
        config = DataConfig(
            codec="longcat",
            dataloader={"batch_size": 1, "num_workers": 0},
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
        config = DataConfig(
            codec="unicodec",
            dataloader={"batch_size": 1, "num_workers": 0},
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
        with self.assertRaisesRegex(ValueError, "same source and target modalities"):
            ModelBatch.from_samples(samples, pad_token_id=99)

    def test_model_batch_direct_constructor_maintains_batch_task_invariants(self):
        def batch(tasks: list[Task]) -> ModelBatch:
            return ModelBatch(
                input_ids=torch.ones(2, 2, dtype=torch.long),
                token_labels=torch.ones(2, 2, dtype=torch.long),
                acoustic_target=None,
                tasks=tasks,
                pad_token_id=99,
            )

        cases = (
            ([], ValueError, "one Task per row"),
            ([Task.ASR], ValueError, "one Task per row"),
            (
                [Task.ASR, Task.TEXT_AR],
                ValueError,
                "same source and target modalities",
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
                pad_token_id=99,
            )

        with self.assertRaisesRegex(TypeError, "signed integer"):
            ModelBatch(
                input_ids=torch.ones(1, 2, dtype=torch.uint64),
                token_labels=torch.ones(1, 2, dtype=torch.long),
                acoustic_target=None,
                tasks=[Task.ASR],
                pad_token_id=99,
            )

    def test_model_batch_accepts_unified_audio_target(self):
        batch = ModelBatch.from_samples([_sample(Task.TTS)], pad_token_id=99)

        self.assertIsNone(batch.acoustic_target)

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

    def test_collator_updates_the_existing_task_weights(self):
        collator = Collator(Mock(), {Task.TTS: 1.0, Task.T2ST: 1.0})
        original = collator
        self.assertEqual(set(collator.tasks), {Task.TTS, Task.T2ST})

        collator.set_task_weights({Task.ASR: 1.0, Task.S2TT: 1.0})

        self.assertIs(collator, original)
        self.assertEqual(set(collator.tasks), {Task.ASR, Task.S2TT})

    def test_collator_task_updates_cross_worker_processes(self):
        runtime = cast(DataRuntime, cast(object, None))
        collator = Collator(runtime, {Task.TTS: 1.0})
        context = multiprocessing.get_context()
        ready = context.Event()
        updated = context.Event()
        output = context.Queue()
        process = context.Process(
            target=_observe_task_updates,
            args=(collator, ready, updated, output),
        )

        process.start()
        try:
            self.assertTrue(ready.wait(timeout=5.0))
            self.assertEqual(output.get(timeout=5.0), [Task.TTS])
            collator.set_task_weights({Task.ASR: 1.0})
            updated.set()
            self.assertEqual(output.get(timeout=5.0), [Task.ASR])
            process.join(timeout=5.0)
            self.assertEqual(process.exitcode, 0)
        finally:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            output.close()

    def test_collator_rejects_invalid_task_weights_before_updating(self):
        collator = Collator(Mock(), {Task.TTS: 1.0})
        cases = (
            ({Task.TTS: -1.0}, "finite and non-negative"),
            ({Task.TTS: float("nan")}, "finite and non-negative"),
            ({Task.TTS: float("inf")}, "finite and non-negative"),
            ({Task.TTS: 0.0}, "finite positive total"),
            (
                {Task.TTS: 1e308, Task.T2ST: 1e308},
                "finite positive total",
            ),
        )

        for weights, message in cases:
            with (
                self.subTest(weights=weights),
                self.assertRaisesRegex(ValueError, message),
            ):
                collator.set_task_weights(weights)
            self.assertEqual(collator.tasks, [Task.TTS])

    def test_fixed_task_allocation_uses_batch_size_and_rejects_tiny_batches(self):
        self.assertEqual(
            _allocate_tasks([Task.T2ST, Task.TTS], [1.0, 2.0], 6),
            [Task.T2ST, Task.T2ST, Task.TTS, Task.TTS, Task.TTS, Task.TTS],
        )
        collator = Collator(Mock(), {Task.TTS: 1.0, Task.T2ST: 0.0})
        self.assertEqual(collator.tasks, [Task.TTS])
        with self.assertRaisesRegex(ValueError, "too small"):
            _allocate_tasks([Task.MT, Task.TTS], [1.0, 9.0], 8)

    def test_parameter_policy_freezes_explicit_parameter_groups(self):
        model = _StageModel()

        counts = apply_parameter_policy(
            model,
            PARAMETER_POLICY_SPECS[ParameterPolicyName.SPEECH_INTERFACE],
        )

        self.assertGreater(counts[ParameterGroup.BACKBONE], 0)
        self.assertFalse(model.backbone.model.layers[0].weight.requires_grad)
        self.assertTrue(model.semantic_audio_embedding.weight.requires_grad)
        self.assertTrue(model.acoustic_decoder.head.weight.requires_grad)
        self.assertFalse(model.acoustic_decoder.decoder.embed_tokens.weight.requires_grad)
        self.assertFalse(model.acoustic_decoder.codebook_embeddings[-1].weight.requires_grad)
        self.assertFalse(model.acoustic_decoder.embedding_projections[-1].weight.requires_grad)

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
    return ModelSample(
        input_ids=torch.tensor([1, 2]),
        token_labels=torch.tensor([-100, 2]),
        acoustic_target=None,
        task=task,
    )


def _target_sample(
    codes: torch.Tensor,
    *,
    semantic_codes: torch.Tensor | None = None,
) -> ModelSample:
    frames = codes.size(0)
    return ModelSample(
        input_ids=torch.tensor([1, 4]),
        token_labels=torch.tensor([-100, 4]),
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
        text_tokenizer=_Tokenizer(10),
        audio_tokenizer=NativeAudioTokenizer(vocab_size=8),
        layout=Layout(text=(0, 10), audio=(10, 20)),
        pad_token_id=0,
        eos_token_id=1,
        boa_token_id=18,
        eoa_token_id=19,
        codec=_longcat_codec(),
    )


def _longcat_codec():
    return SimpleNamespace(
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
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def encode(self, audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
        self.calls.append((tuple(audio.shape), sample_rate))
        return audio.new_tensor([[[0, 2], [1, 3]]], dtype=torch.long)


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
