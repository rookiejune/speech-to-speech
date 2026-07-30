from __future__ import annotations

import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock, patch

import torch
from anydataset.types import Modality
from anytrain.module.idspace import Layout
from hydra import compose, initialize_config_dir
from torch import Tensor, nn

from scripts import train as train_script
from scripts._config import train as parse_train
from speech_to_speech.callback.logging import TextRetentionLogger
from speech_to_speech.task import Task
from speech_to_speech.loss import TokenObjective
from speech_to_speech.generation import Result
from speech_to_speech.pl_module import Config, SpeechToSpeechModule


class _Tokenizer:
    def apply_chat_template(self, conversation, **kwargs):
        del conversation, kwargs
        return [1, 2]

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return [3, 4]

    def decode(self, token_ids, *, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(8, 4)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _Backbone()
        self.runtime = SimpleNamespace(
            text_tokenizer=_Tokenizer(),
            layout=Layout(text=(0, 8), audio=(8, 10)),
            eos_token_id=7,
        )
        self.token_modalities: list[Modality] = []

    def forward(self, input_ids: Tensor, **kwargs):
        del kwargs
        return SimpleNamespace(
            logits=torch.zeros(*input_ids.shape, 10, device=input_ids.device)
        )

    def token_hidden_states(self, input_ids: Tensor, **kwargs) -> Tensor:
        del kwargs
        return torch.zeros(*input_ids.shape, 4, device=input_ids.device)

    def token_logits(
        self,
        hidden_states: Tensor,
        modality: Modality | None = None,
    ) -> Tensor:
        if modality is None:
            raise ValueError("text evaluation must select a token modality")
        self.token_modalities.append(modality)
        start, end = self.runtime.layout.blocks[modality.value]
        return torch.zeros(
            hidden_states.size(0),
            end - start,
            device=hidden_states.device,
        )


PROBES = {
    "zh_en": {
        "instruction": "Translate Chinese to English.",
        "reference": "reference one",
    },
    "en_zh": {
        "instruction": "Translate English to Chinese.",
        "reference": "reference two",
    },
}


class TextRetentionTest(unittest.TestCase):
    def test_objective_is_registered_as_a_child_module(self):
        model = _Model()
        objective = TokenObjective(model.runtime.layout)

        module = SpeechToSpeechModule(Config(), model=model, objective=objective)

        self.assertIs(dict(module.named_children())["objective"], objective)

    def test_t2tt_uses_source_role_for_text_to_text(self):
        self.assertIs(Task.T2TT.source_modality, Modality.TEXT)
        self.assertIs(Task.T2TT.target_modality, Modality.TEXT)
        self.assertTrue(Task.T2TT.uses_source_role)

    @patch("speech_to_speech.generation.text.generate_responses")
    def test_module_evaluates_greedy_generation_and_text_only_nll(
        self, generate_responses
    ):
        generate_responses.return_value = [
            Result(
                response_ids=torch.tensor([5, 6]),
                audio=None,
            )
            for _ in PROBES
        ]
        model = _Model()
        module = SpeechToSpeechModule(Config(), model=model, objective=Mock())

        results = module.evaluate_text(PROBES, max_new_tokens=16)

        self.assertTrue(module.training)
        requests = generate_responses.call_args.args[0]
        self.assertEqual([request["task"] for request in requests], [Task.T2TT] * 2)
        self.assertEqual(generate_responses.call_args.kwargs["max_new_tokens"], 16)
        self.assertFalse(generate_responses.call_args.kwargs["do_sample"])
        self.assertEqual(results["zh_en"]["generated"], "5 6")
        self.assertAlmostEqual(results["zh_en"]["nll"], math.log(8), places=6)
        self.assertEqual(model.token_modalities, [Modality.TEXT] * len(PROBES))

    def test_callback_records_baseline_and_respects_interval(self):
        evaluate_text = Mock(
            return_value={
                name: {"generated": "decoded text", "nll": math.log(8)}
                for name in PROBES
            }
        )
        module = SimpleNamespace(evaluate_text=evaluate_text)
        experiment = Mock()
        trainer = SimpleNamespace(
            global_step=0,
            is_global_zero=True,
            logger=SimpleNamespace(experiment=experiment),
        )
        logger = TextRetentionLogger(
            PROBES,
            every_n_steps=2,
            max_new_tokens=16,
        )

        logger.on_fit_start(trainer, module)

        evaluate_text.assert_called_once_with(PROBES, max_new_tokens=16)
        self.assertEqual(experiment.add_text.call_count, 2)
        self.assertEqual(experiment.add_scalar.call_count, 4)
        first_delta = experiment.add_scalar.call_args_list[1].args[1]
        self.assertEqual(first_delta, 0.0)
        logged_text = experiment.add_text.call_args_list[0].args[1]
        self.assertIn("Reference: reference one", logged_text)
        self.assertIn("Generated: decoded text", logged_text)

        trainer.global_step = 1
        logger.on_train_batch_end(trainer, module, None, None, 0)
        self.assertEqual(evaluate_text.call_count, 1)

        trainer.global_step = 2
        logger.on_train_batch_end(trainer, module, None, None, 1)
        self.assertEqual(evaluate_text.call_count, 2)

        trainer.is_global_zero = False
        trainer.global_step = 4
        logger.on_train_batch_end(trainer, module, None, None, 2)
        self.assertEqual(evaluate_text.call_count, 2)

    def test_callback_preserves_checkpoint_baseline_on_resume(self):
        evaluate_text = Mock(
            return_value={
                name: {"generated": "resumed text", "nll": 3.0}
                for name in PROBES
            }
        )
        module = SimpleNamespace(evaluate_text=evaluate_text)
        experiment = Mock()
        trainer = SimpleNamespace(
            global_step=20,
            is_global_zero=True,
            logger=SimpleNamespace(experiment=experiment),
        )
        logger = TextRetentionLogger(PROBES, every_n_steps=10)
        logger.load_state_dict(
            {
                "interval": {},
                "baseline_nll": {name: 1.0 for name in PROBES},
            }
        )

        logger.on_fit_start(trainer, module)

        deltas = [
            call.args[1]
            for call in experiment.add_scalar.call_args_list
            if call.args[0].endswith("/nll_delta")
        ]
        self.assertEqual(deltas, [2.0, 2.0])
        self.assertEqual(
            logger.state_dict()["baseline_nll"],
            {name: 1.0 for name in PROBES},
        )


@patch.dict(
    "os.environ",
    {
        "DYNAMIC_HOME": "/tmp/dynamic",
        "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
    },
)
class TextRetentionConfigTest(unittest.TestCase):
    def test_formal_train_enables_a_valid_text_probe(self):
        config = parse_train(_compose_train())

        callback = config.callbacks.text_retention
        self.assertTrue(callback.enabled)
        self.assertEqual(callback.every_n_steps, 10_000)
        self.assertIsNone(callback.every_audio_seconds)
        self.assertEqual(callback.max_new_tokens, 64)
        self.assertEqual(set(callback.probes), {"zh_en"})
        self.assertTrue(callback.probes["zh_en"].instruction)
        self.assertTrue(callback.probes["zh_en"].reference)

    def test_audio_cadence_requires_none_or_a_finite_positive_number(self):
        valid = parse_train(_compose_train())
        for value in (None, 1, 1.5):
            with self.subTest(value=value):
                cast(Any, valid.callbacks.text_retention).every_audio_seconds = value
                valid.callbacks.text_retention.validate()

        cases = (
            (True, TypeError, "number or None"),
            ("10", TypeError, "number or None"),
            (0, ValueError, "finite and positive"),
            (-1, ValueError, "finite and positive"),
            (math.inf, ValueError, "finite and positive"),
            (math.nan, ValueError, "finite and positive"),
        )
        for value, error, message in cases:
            with self.subTest(value=value), self.assertRaisesRegex(error, message):
                config = parse_train(_compose_train())
                cast(Any, config.callbacks.text_retention).every_audio_seconds = value
                config.callbacks.text_retention.validate()

    def test_enabled_text_retention_requires_complete_probes(self):
        empty = parse_train(_compose_train())
        empty.callbacks.text_retention.probes = {}
        with self.assertRaisesRegex(ValueError, "at least one probe"):
            empty.callbacks.text_retention.validate()

        missing_instruction = parse_train(_compose_train())
        missing_instruction.callbacks.text_retention.probes["zh_en"].instruction = ""
        with self.assertRaisesRegex(TypeError, "instruction"):
            missing_instruction.callbacks.text_retention.validate()

        missing_reference = parse_train(_compose_train())
        missing_reference.callbacks.text_retention.probes["zh_en"].reference = ""
        with self.assertRaisesRegex(TypeError, "reference"):
            missing_reference.callbacks.text_retention.validate()

    def test_formal_training_constructs_text_retention_callback(self):
        config = parse_train(_compose_train())
        callback = config.callbacks.text_retention
        callback.every_n_steps = 17
        callback.every_audio_seconds = 12.5
        callback.max_new_tokens = 23
        built = Mock()

        with patch("scripts.train.TextRetentionLogger", return_value=built) as factory:
            callbacks = train_script.training_callbacks(
                config,
                Path("/tmp/output"),
                Mock(),
            )

        self.assertIn(built, callbacks)
        factory.assert_called_once_with(
            {
                name: {
                    "instruction": probe.instruction,
                    "reference": probe.reference,
                }
                for name, probe in callback.probes.items()
            },
            every_n_steps=17,
            every_audio_seconds=12.5,
            max_new_tokens=23,
        )


def _compose_train(*overrides: str):
    root = Path(__file__).parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        return compose(config_name="train", overrides=list(overrides))


if __name__ == "__main__":
    unittest.main()
