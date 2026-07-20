from __future__ import annotations

import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock, patch

import torch
from anydataset.types import (
    AudioItem,
    AudioView,
    Modality,
    Role,
    TextItem,
    TextMeta,
    TextView,
)
from hydra import compose, initialize_config_dir
from lightning.pytorch.callbacks import Callback
from omegaconf import DictConfig, OmegaConf

from speech_to_speech.datamodule.collator import Collator
from speech_to_speech.callback import WorldSizeContract
from speech_to_speech.datamodule import DatasetConfig, DatasetName, ToyDataset
from speech_to_speech.datamodule.module import Config as DataConfig
from speech_to_speech.datamodule.module import DataModule
from speech_to_speech.datamodule.parser import _parse_audio_item, parse_sample
from speech_to_speech.datamodule.types import (
    Language,
    ModelBatch,
    ModelSample,
)
from speech_to_speech.callback.stage import Config as StageConfig
from speech_to_speech.callback.stage import StageSwitcher
from speech_to_speech.model import Config as ModelConfig, ToyConfig
from speech_to_speech.runtime import Config, Runtime
from speech_to_speech.runtime.runtime import audio_tokenizer, dtype
from speech_to_speech.runtime.audio_tokenizer import NativeAudioTokenizer, TorchCodecBPE
from speech_to_speech.task import Task
from scripts._config import overfit as parse_overfit
from scripts.overfit import FixedDataModule, build_trainer, run, runtime_config


class _Tokenizer:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def encode(self, text: str, *, add_special_tokens: bool = False):
        self.encoded = (text, add_special_tokens)
        return [1, 2]


class ContractTest(unittest.TestCase):
    def test_trainer_presets_have_one_composable_schema(self):
        configs = [
            _compose(config_name="codec_oracle"),
            _compose(),
            _compose("trainer=ddp"),
            _compose(
                "experiment=acoustic_oracle_ddp_lba_smoke",
                config_name="codec_oracle",
            ),
        ]
        expected = {
            "accelerator",
            "devices",
            "strategy",
            "expected_world_size",
            "use_distributed_sampler",
            "precision",
            "max_epochs",
            "log_every_n_steps",
            "enable_checkpointing",
            "gradient_clip_val",
        }

        for config in configs:
            self.assertEqual(set(config.trainer), expected)

        self.assertEqual(
            configs[2].trainer.strategy,
            "ddp_find_unused_parameters_true",
        )
        self.assertEqual(configs[3].trainer.strategy, "ddp")
        self.assertEqual(configs[3].trainer.precision, "bf16-mixed")
        self.assertTrue(configs[2].trainer.use_distributed_sampler)
        self.assertTrue(configs[3].trainer.use_distributed_sampler)

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
        self.assertIsInstance(kwargs["callbacks"][0], WorldSizeContract)
        self.assertEqual(kwargs["callbacks"][0].expected, 2)
        self.assertEqual(kwargs["callbacks"][1:], callbacks)

    def test_public_configs_support_omegaconf_structured(self):
        runtime_config = OmegaConf.structured(Config)
        model_config = OmegaConf.structured(ModelConfig)

        self.assertIsNone(runtime_config.audio_tokenizer)
        self.assertIsNone(runtime_config.device)
        self.assertEqual(model_config.semantic_audio_adapter, "linear")
        self.assertEqual(model_config.acoustic_prompt_adapter, "linear")

    def test_acoustic_presets_expose_only_supported_options(self):
        flow = _compose()
        rvq = _compose("model/acoustic=rvq")

        self.assertEqual(flow.acoustic.type, "flow")
        self.assertEqual(flow.acoustic.repa.teacher_layer, 9)
        self.assertIn("student_layer", flow.acoustic.repa)
        self.assertNotIn("normalize_features", flow.acoustic)
        self.assertEqual(flow.runtime.codec, "longcat")
        self.assertEqual(flow.model.semantic_audio_adapter, "linear")
        self.assertEqual(rvq.acoustic.type, "rvq")
        self.assertNotIn("repa", rvq.acoustic)

    def test_overfit_acoustic_branch_constructs_evaluation_on_py39(self):
        class EvaluationReached(Exception):
            pass

        runtime = SimpleNamespace(
            layout=Mock(),
            codec=SimpleNamespace(acoustic_codebook_sizes=(1024,)),
            backbone=Mock(),
            flow_matching=Mock(),
        )
        datamodule = Mock()
        datamodule.train_dataloader.return_value = [Mock()]
        with TemporaryDirectory() as output_dir:
            config = parse_overfit(
                _compose(
                    "runtime=longcat_native",
                    f"output_dir={output_dir}",
                    "train.max_steps=1",
                    "acoustic.decoder.layers=1",
                    "acoustic.decoder.heads=1",
                    "acoustic.decoder.ffn_ratio=1",
                )
            )
            with (
                patch("scripts.overfit.pl.seed_everything"),
                patch("scripts.overfit.runtime_config", return_value=Mock()),
                patch("scripts.overfit.init_runtime", return_value=runtime),
                patch("scripts.overfit.FixedDataModule", return_value=datamodule),
                patch("scripts.overfit.flow", return_value=(Mock(), Mock(), None)),
                patch(
                    "scripts.overfit.AcousticEvaluation", side_effect=EvaluationReached
                ),
            ):
                with self.assertRaises(EvaluationReached):
                    run(config)

    def test_task_is_the_modality_source_of_truth(self):
        self.assertIs(Task.S2ST.source_modality, Modality.AUDIO)
        self.assertIs(Task.S2ST.target_modality, Modality.AUDIO)
        self.assertTrue(Task.S2ST.uses_source_role)
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
                "~model/acoustic",
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

    @patch("zhuyin.datasets.wmt19_tts.wmt19_tts_codec")
    def test_datamodule_setup_loads_dataset_once(self, load_dataset):
        load_dataset.return_value = []
        datamodule = DataModule(
            DataConfig(
                codec="longcat",
                dataloader={"batch_size": 1, "num_workers": 0},
            ),
            SimpleNamespace(codec_name="longcat"),
            {Task.TTS: 1.0},
        )

        datamodule.setup()
        datamodule.setup()

        load_dataset.assert_called_once_with(
            codec="longcat",
            root=None,
            split="train",
        )

    def test_toy_dataset_uses_codec_shapes_and_value_ranges(self):
        cases = (
            (
                "longcat",
                SimpleNamespace(
                    semantic_codebook=torch.zeros(5, 4),
                    acoustic_codebook_sizes=(3, 7),
                ),
                AudioView.LONGCAT,
                (5, 3, 7),
            ),
            (
                "unicodec",
                SimpleNamespace(
                    semantic_codebook=torch.zeros(11, 4),
                    acoustic_codebook_sizes=(),
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

    @patch("zhuyin.datasets.wmt19_tts.wmt19_tts_codec")
    def test_datamodule_loads_toy_data_without_prepared_dataset(self, prepared):
        codec = SimpleNamespace(
            semantic_codebook=torch.zeros(5, 4),
            acoustic_codebook_sizes=(3,),
        )
        datamodule = DataModule(
            DataConfig(
                codec="longcat",
                dataloader={"batch_size": 1, "num_workers": 0},
                dataset=DatasetConfig(
                    name=DatasetName.TOY,
                    toy_samples=2,
                    toy_frames=3,
                ),
            ),
            SimpleNamespace(codec_name="longcat", codec=codec),
            {Task.TTS: 1.0},
        )

        datamodule.setup()

        prepared.assert_not_called()
        self.assertEqual(len(datamodule.train_samples([0, 1])), 2)

    def test_toy_settings_reject_invalid_dimensions(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            ToyConfig(hidden_size=7, heads=2)
        with self.assertRaisesRegex(ValueError, "toy_samples"):
            DatasetConfig(name=DatasetName.TOY, toy_samples=0)
        codec = SimpleNamespace(
            semantic_codebook=torch.zeros(5, 4),
            acoustic_codebook_sizes=(),
        )
        with self.assertRaisesRegex(ValueError, "LongCat"):
            ToyDataset("longcat", codec)

    def test_datamodule_rejects_runtime_codec_mismatch(self):
        datamodule = DataModule(
            DataConfig(
                codec="unicodec",
                dataloader={"batch_size": 1, "num_workers": 0},
            ),
            SimpleNamespace(codec_name="longcat"),
            {Task.TTS: 1.0},
        )

        with self.assertRaisesRegex(ValueError, "same codec"):
            datamodule.setup()

    @patch("zhuyin.datasets.wmt19_tts.wmt19_tts_codec")
    def test_overfit_datamodule_repeats_only_the_selected_sample(self, load_dataset):
        samples = [object(), object()]
        load_dataset.return_value = samples
        datamodule = FixedDataModule(
            "longcat",
            SimpleNamespace(codec_name="longcat"),
            {Task.TTS: 1.0},
            sample_index=1,
        )
        datamodule.collator = Mock(side_effect=lambda batch: batch)

        datamodule.setup()
        first_epoch = list(datamodule.train_dataloader())
        second_epoch = list(datamodule.train_dataloader())

        load_dataset.assert_called_once_with(
            codec="longcat",
            root=None,
            split="train",
        )
        self.assertEqual(first_epoch, [[samples[1]]])
        self.assertEqual(second_epoch, [[samples[1]]])

    def test_overfit_datamodule_rejects_runtime_codec_mismatch(self):
        datamodule = FixedDataModule(
            "unicodec",
            SimpleNamespace(codec_name="longcat"),
            {Task.TTS: 1.0},
            sample_index=0,
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
                acoustic_prompt=None,
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
                acoustic_prompt=None,
                acoustic_target=None,
                tasks=[],
                pad_token_id=99,
            )

        with self.assertRaisesRegex(TypeError, "signed integer"):
            ModelBatch(
                input_ids=torch.ones(1, 2, dtype=torch.uint64),
                token_labels=torch.ones(1, 2, dtype=torch.long),
                acoustic_prompt=None,
                acoustic_target=None,
                tasks=[Task.ASR],
                pad_token_id=99,
            )

    def test_model_batch_accepts_unified_audio_target(self):
        batch = ModelBatch.from_samples([_sample(Task.TTS)], pad_token_id=99)

        self.assertIsNone(batch.acoustic_target)

    def test_model_batch_rejects_padding_ids_inside_unpadded_acoustic_fields(self):
        samples = {
            "acoustic prompt codes": ModelSample(
                input_ids=torch.tensor([1, 4]),
                token_labels=torch.tensor([-100, 4]),
                acoustic_prompt={
                    "codes": torch.tensor([[-1, 2]]),
                    "token_positions": torch.tensor([0]),
                },
                acoustic_target=None,
                task=Task.ASR,
            ),
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

    def test_stage_switcher_restores_task_weights_from_current_epoch(self):
        task_weights = [{Task.TTS: 1.0}, {Task.ASR: 1.0}, {Task.TEXT_AR: 1.0}]
        datamodule = SimpleNamespace(set_task_weights=Mock())
        trainer = SimpleNamespace(datamodule=datamodule, current_epoch=3)
        switcher = StageSwitcher(StageConfig(task_weights, epoch_milestones=[2, 4]))

        switcher.on_fit_start(trainer, Mock())
        switcher.on_train_epoch_end(trainer, Mock())

        self.assertEqual(
            datamodule.set_task_weights.call_args_list,
            [unittest.mock.call(task_weights[1]), unittest.mock.call(task_weights[2])],
        )


def _sample(task: Task) -> ModelSample:
    return ModelSample(
        input_ids=torch.tensor([1, 2]),
        token_labels=torch.tensor([-100, 2]),
        acoustic_prompt=None,
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
        acoustic_prompt=None,
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


def _compose(*overrides: str, config_name: str = "overfit") -> DictConfig:
    root = Path(__file__).parents[1]
    with initialize_config_dir(
        version_base=None,
        config_dir=str(root / "configs"),
    ):
        return compose(config_name=config_name, overrides=list(overrides))


def _raw_sample():
    def audio(offset: int) -> AudioItem:
        return AudioItem(
            views={
                AudioView.LONGCAT: torch.tensor(
                    [[offset, offset + 2], [offset + 1, offset + 3]]
                )
            }
        )

    return {
        (Role.SOURCE, Modality.AUDIO): audio(0),
        (Role.SOURCE, Modality.TEXT): TextItem(
            views={TextView.TEXT: "source text"},
            meta={TextMeta.LANG: "zh"},
        ),
        (Role.TARGET, Modality.AUDIO): audio(4),
        (Role.TARGET, Modality.TEXT): TextItem(
            views={TextView.TEXT: "target text"},
            meta={TextMeta.LANG: "en"},
        ),
    }


if __name__ == "__main__":
    unittest.main()
