from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from anydataset.types import Modality
from anytrain.idspace import Layout
from torch import Tensor, nn

from speech_to_speech.callback.logging import TextRetentionLogger
from speech_to_speech.task import Task
from speech_to_speech.loss import TokenObjective
from speech_to_speech.generation import Result
from speech_to_speech.pl_module import Config, SpeechToSpeech


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

    def forward(self, input_ids: Tensor, **kwargs):
        del kwargs
        return SimpleNamespace(
            logits=torch.zeros(*input_ids.shape, 10, device=input_ids.device)
        )

    def token_hidden_states(self, input_ids: Tensor, **kwargs) -> Tensor:
        del kwargs
        return torch.zeros(*input_ids.shape, 4, device=input_ids.device)

    def token_logits(self, hidden_states: Tensor) -> Tensor:
        return torch.zeros(hidden_states.size(0), 10, device=hidden_states.device)


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

        module = SpeechToSpeech(Config(), model=model, objective=objective)

        self.assertIs(dict(module.named_children())["objective"], objective)

    def test_t2tt_uses_source_role_for_text_to_text(self):
        self.assertIs(Task.T2TT.source_modality, Modality.TEXT)
        self.assertIs(Task.T2TT.target_modality, Modality.TEXT)
        self.assertTrue(Task.T2TT.uses_source_role)

    @patch("speech_to_speech.generation.text.generate")
    def test_module_evaluates_greedy_generation_and_text_only_nll(self, generate):
        generate.return_value = [
            Result(
                response_ids=torch.tensor([5, 6]),
                audio=None,
            )
            for _ in PROBES
        ]
        module = SpeechToSpeech(Config(), model=_Model(), objective=Mock())

        results = module.evaluate_text(PROBES, max_new_tokens=16)

        self.assertTrue(module.training)
        requests = generate.call_args.args[0]
        self.assertEqual([request["task"] for request in requests], [Task.T2TT] * 2)
        self.assertEqual(generate.call_args.kwargs["max_new_tokens"], 16)
        self.assertFalse(generate.call_args.kwargs["do_sample"])
        self.assertEqual(results["zh_en"]["generated"], "5 6")
        self.assertAlmostEqual(results["zh_en"]["nll"], math.log(8), places=6)

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


if __name__ == "__main__":
    unittest.main()
