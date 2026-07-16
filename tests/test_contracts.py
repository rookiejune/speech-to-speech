from __future__ import annotations

import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from types import SimpleNamespace
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
from omegaconf import OmegaConf

from speech_to_speech.datamodule.collator import Collator
from speech_to_speech.datamodule.module import Config as DataConfig
from speech_to_speech.datamodule.module import DataModule
from speech_to_speech.datamodule.types import (
    Language,
    ModelBatch,
    Sample,
    SpeechPair,
    Task,
    _parse_audio_item,
)
from speech_to_speech.callback.stage import Config as StageConfig
from speech_to_speech.callback.stage import StageSwitcher
from speech_to_speech.model import Config as ModelConfig
from speech_to_speech.model import SpeechToSpeechFlowModel
from speech_to_speech.runtime.singleton import Config, Runtime
from speech_to_speech.runtime.singleton import _audio_tokenizer, _dtype
from speech_to_speech.runtime.audio_tokenizer import TorchCodecBPE
from scripts.overfit import FixedDataModule, run, runtime_config


class _Tokenizer:
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def encode(self, text: str, *, add_special_tokens: bool = False):
        self.encoded = (text, add_special_tokens)
        return [1, 2]


class ContractTest(unittest.TestCase):
    def test_public_configs_support_omegaconf_structured(self):
        runtime_config = OmegaConf.structured(Config)
        model_config = OmegaConf.structured(ModelConfig)

        self.assertIsNone(runtime_config.audio_tokenizer)
        self.assertIsNone(runtime_config.device)
        self.assertEqual(model_config.semantic_audio_adapter, "linear")
        self.assertIsNone(model_config.acoustic_decoder_dim)

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
            config = OmegaConf.create(
                {
                    "output_dir": output_dir,
                    "task": "tts",
                    "codec": {"name": "longcat"},
                    "data": {"sample_index": 0},
                    "train": {"seed": 0, "max_steps": 1},
                    "optimizer": {
                        "learning_rate": 2e-5,
                        "weight_decay": 0.01,
                    },
                    "acoustic": {
                        "objective": "flow",
                        "decoder": {
                            "dim": None,
                            "layers": 1,
                            "heads": 1,
                            "ffn_ratio": 1,
                        },
                        "repa": {"weight": None},
                    },
                }
            )
            with patch("scripts.overfit.pl.seed_everything"), patch(
                "scripts.overfit.runtime_config", return_value=Mock()
            ), patch(
                "scripts.overfit.init_runtime", return_value=runtime
            ), patch(
                "scripts.overfit.FixedDataModule", return_value=datamodule
            ), patch.object(
                SpeechToSpeechFlowModel, "__init__", return_value=None
            ), patch(
                "scripts.overfit.Loss"
            ), patch(
                "scripts.overfit.SpeechToSpeech"
            ), patch(
                "scripts.overfit.AcousticEvaluation", side_effect=EvaluationReached
            ):
                with self.assertRaises(EvaluationReached):
                    run(config)

    def test_task_is_the_modality_source_of_truth(self):
        self.assertIs(Task.S2ST.source_modality, Modality.AUDIO)
        self.assertIs(Task.S2ST.target_modality, Modality.AUDIO)
        self.assertTrue(Task.S2ST.paired)
        self.assertIsNone(Task.AUDIO_AR.source_modality)
        self.assertIs(Task.ASR.target_modality, Modality.TEXT)
        self.assertFalse(Task.TTS.paired)

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
        item = AudioItem(
            views={AudioView.UNICODEC: torch.tensor([[1], [2], [3]])}
        )
        with patch(
            "speech_to_speech.datamodule.types.runtime",
            return_value=SimpleNamespace(
                config=SimpleNamespace(audio_view=AudioView.UNICODEC)
            ),
        ):
            semantic, acoustic = _parse_audio_item(item)

        self.assertTrue(torch.equal(semantic, torch.tensor([[1], [2], [3]])))
        self.assertIsNone(acoustic)

    @patch("speech_to_speech.runtime.singleton.AutoModelForCausalLM.from_pretrained")
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
        self.assertIs(_dtype("float16"), torch.float16)
        with self.assertRaisesRegex(ValueError, "unknown torch dtype"):
            _dtype("not_a_dtype")

    def test_overfit_runtime_config_preserves_native_audio_tokenizer(self):
        config = OmegaConf.create(
            {
                "codec": {"name": "unicodec"},
                "runtime": {
                    "backbone": "fake/backbone",
                    "audio_tokenizer": None,
                    "device": "cuda",
                    "dtype": "bfloat16",
                    "attn_implementation": "flash_attention_2",
                },
                "flow": {"method": "midpoint", "nfe": 20, "num_steps": 10},
            }
        )

        result = runtime_config(config)

        self.assertEqual(result.codec, "unicodec")
        self.assertIsNone(result.audio_tokenizer)

    @patch("anytrain.framework.flow_matching.ContinuousFlowRuntime")
    @patch("anytrain.framework.flow_matching.ODESampler")
    def test_runtime_forwards_flow_configuration(self, sampler, flow_runtime):
        configured_sampler = Mock()
        sampler.return_value = configured_sampler
        rt = Runtime(
            Config(flow_method="euler", flow_nfe=7, flow_num_steps=6)
        )

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
        with patch.dict(sys.modules, modules), patch.object(
            TorchCodecBPE, "wrap", return_value=wrapped
        ) as wrap:
            loaded = _audio_tokenizer("~/bpe/longcat/vocab_100k")

        codec_bpe.assert_called_once_with(
            Path("~/bpe/longcat/vocab_100k").expanduser()
        )
        wrap.assert_called_once_with(tokenizer)
        self.assertIs(loaded, wrapped)

    @patch("speech_to_speech.datamodule.types.runtime")
    def test_raw_text_is_encoded_at_the_datamodule_boundary(self, runtime):
        tokenizer = _Tokenizer(10)
        runtime.return_value = SimpleNamespace(
            config=SimpleNamespace(audio_view=AudioView.LONGCAT),
            text_tokenizer=tokenizer,
        )
        raw = _raw_sample()

        pair = SpeechPair.from_raw(raw)

        self.assertTrue(torch.equal(pair.source.text_ids, torch.tensor([1, 2])))
        self.assertIs(pair.source.language, Language.ZH)
        self.assertIs(pair.target.language, Language.EN)
        self.assertEqual(pair.source.acoustic_ids.shape, (2, 1))
        self.assertTrue(
            torch.equal(pair.source.acoustic_ids, torch.tensor([[2], [3]]))
        )
        self.assertEqual(tokenizer.encoded, ("target text", False))

    @patch(
        "speech_to_speech.datamodule.module.runtime",
        return_value=SimpleNamespace(config=SimpleNamespace(codec="longcat")),
    )
    @patch("zhuyin.datasets.wmt19_tts.wmt19_tts_codec")
    def test_datamodule_setup_loads_dataset_once(self, load_dataset, _runtime):
        load_dataset.return_value = []
        datamodule = DataModule(
            DataConfig(
                codec="longcat",
                dataloader={"batch_size": 1, "num_workers": 0},
            ),
            {Task.TTS: 1.0},
        )

        datamodule.setup()
        datamodule.setup()

        load_dataset.assert_called_once_with(codec="longcat")

    @patch(
        "speech_to_speech.datamodule.module.runtime",
        return_value=SimpleNamespace(config=SimpleNamespace(codec="longcat")),
    )
    def test_datamodule_rejects_runtime_codec_mismatch(self, _runtime):
        datamodule = DataModule(
            DataConfig(
                codec="unicodec",
                dataloader={"batch_size": 1, "num_workers": 0},
            ),
            {Task.TTS: 1.0},
        )

        with self.assertRaisesRegex(ValueError, "same codec"):
            datamodule.setup()

    @patch("zhuyin.datasets.wmt19_tts.wmt19_tts_codec")
    def test_overfit_datamodule_repeats_only_the_selected_sample(self, load_dataset):
        samples = [object(), object()]
        load_dataset.return_value = samples
        datamodule = FixedDataModule("longcat", {Task.TTS: 1.0}, sample_index=1)
        datamodule.collator = Mock(side_effect=lambda batch: batch)

        datamodule.setup()
        first_epoch = list(datamodule.train_dataloader())
        second_epoch = list(datamodule.train_dataloader())

        load_dataset.assert_called_once_with(codec="longcat", split="train")
        self.assertEqual(first_epoch, [[samples[1]]])
        self.assertEqual(second_epoch, [[samples[1]]])

    @patch(
        "speech_to_speech.datamodule.types.runtime",
        return_value=SimpleNamespace(pad_token_id=99),
    )
    def test_model_batch_rejects_mixed_execution_signatures(self, _runtime):
        samples = [
            _sample(Task.ASR),
            _sample(Task.TEXT_AR),
        ]
        with self.assertRaisesRegex(ValueError, "same source and target modalities"):
            ModelBatch.from_samples(samples)

    @patch(
        "speech_to_speech.datamodule.types.runtime",
        return_value=SimpleNamespace(pad_token_id=99),
    )
    def test_model_batch_accepts_unified_audio_target(self, _runtime):
        batch = ModelBatch.from_samples([_sample(Task.TTS)])

        self.assertIsNone(batch.acoustic_labels)
        self.assertIsNone(batch.acoustic_label_positions)

    def test_collator_updates_the_existing_strategy(self):
        collator = Collator({Task.TTS: 1.0, Task.T2ST: 1.0})
        original = collator
        self.assertEqual(set(collator.tasks), {Task.TTS, Task.T2ST})

        collator.set_strategy({Task.ASR: 1.0, Task.S2TT: 1.0})

        self.assertIs(collator, original)
        self.assertEqual(set(collator.tasks), {Task.ASR, Task.S2TT})

    def test_stage_switcher_restores_the_strategy_from_current_epoch(self):
        strategies = [{Task.TTS: 1.0}, {Task.ASR: 1.0}, {Task.TEXT_AR: 1.0}]
        datamodule = SimpleNamespace(set_strategy=Mock())
        trainer = SimpleNamespace(datamodule=datamodule, current_epoch=3)
        switcher = StageSwitcher(StageConfig(strategies, milestones=[2, 4]))

        switcher.on_fit_start(trainer, Mock())
        switcher.on_train_epoch_end(trainer, Mock())

        self.assertEqual(
            datamodule.set_strategy.call_args_list,
            [unittest.mock.call(strategies[1]), unittest.mock.call(strategies[2])],
        )


def _sample(task: Task) -> Sample:
    return Sample(
        input_ids=torch.tensor([1, 2]),
        labels=torch.tensor([-100, 2]),
        acoustic_input_ids=None,
        acoustic_input_positions=None,
        semantic_frame_labels=None,
        acoustic_labels=None,
        acoustic_label_positions=None,
        task=task,
    )


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
