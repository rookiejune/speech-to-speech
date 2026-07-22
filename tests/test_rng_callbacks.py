from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from torch import Tensor
from anydataset.types import Lang, Modality, Role, TextItem, TextMeta, TextView

from speech_to_speech.generation.evaluation import evaluate
from speech_to_speech.callback.logging.task_sample import TaskSampleLogger
from speech_to_speech.datamodule import ModelBatch
from speech_to_speech.model.acoustic import AcousticRVQDecoder
from speech_to_speech.generation import Result
from speech_to_speech.task import Task


class _RandomGenerationModule:
    def __init__(self, *, use_cuda: bool = False) -> None:
        self.use_cuda = use_cuda

    def generate(self, requests: object, **kwargs: object) -> list[Result]:
        del requests, kwargs
        torch.rand(4)
        if self.use_cuda:
            torch.rand(4, device=torch.device("cuda", torch.cuda.current_device()))
        return [Result(response_ids=torch.tensor([1]), audio=None)]


class _EvaluationModel:
    def __init__(self) -> None:
        self.training = True
        self.parameter = torch.nn.Parameter(torch.zeros(()))
        self.generator_seeds: list[int] = []

    def parameters(self):
        yield self.parameter

    def eval(self) -> _EvaluationModel:
        self.training = False
        return self

    def train(self, mode: bool = True) -> _EvaluationModel:
        self.training = mode
        return self

    def token_hidden_states(self, input_ids: Tensor, **kwargs: object) -> Tensor:
        del input_ids, kwargs
        return torch.zeros(1, 2, 3)

    def target_frame_condition(
        self, hidden_states: Tensor, target_positions: Tensor
    ) -> Tensor:
        del hidden_states, target_positions
        return torch.zeros(1, 2, 3)

    def sample_acoustic_features(
        self,
        condition: Tensor,
        *,
        mask: Tensor | None = None,
        generator: torch.Generator,
    ) -> Tensor:
        del mask
        self.generator_seeds.append(generator.initial_seed())
        return torch.rand(
            (*condition.shape[:2], 1),
            device=condition.device,
            generator=generator,
        )


class _Codec:
    sample_rate = 16_000

    def acoustic_codes_to_features(self, codes: Tensor) -> Tensor:
        return codes.float()

    def decode_features(self, semantic: Tensor, acoustic: Tensor) -> Tensor:
        del semantic
        return acoustic.float().mean().expand(4096).clone()


class RNGCallbackTest(unittest.TestCase):
    def test_task_sample_logger_preserves_cpu_rng(self):
        self.addCleanup(torch.random.set_rng_state, torch.random.get_rng_state())
        logger, trainer = task_sample_logger_fixture()
        module = _RandomGenerationModule()
        torch.manual_seed(123)
        before = torch.random.get_rng_state().clone()

        logger.on_train_batch_start(trainer, module, None, 0)

        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_task_sample_logger_preserves_current_cuda_rng(self):
        self.addCleanup(torch.cuda.set_rng_state, torch.cuda.get_rng_state())
        logger, trainer = task_sample_logger_fixture()
        module = _RandomGenerationModule(use_cuda=True)
        torch.cuda.manual_seed(123)
        before = torch.cuda.get_rng_state().clone()

        logger.on_train_batch_start(trainer, module, None, 0)

        self.assertTrue(torch.equal(torch.cuda.get_rng_state(), before))

    def test_generation_evaluation_uses_local_generators(self):
        self.addCleanup(torch.random.set_rng_state, torch.random.get_rng_state())
        model = _EvaluationModel()
        batch = SimpleNamespace(
            input_ids=torch.ones(1, 2, dtype=torch.long),
            attention_mask=torch.ones(1, 2, dtype=torch.bool),
            acoustic_prompt=None,
            acoustic_prompt_mask=None,
            acoustic_target={
                "semantic_codes": torch.zeros(1, 2, 1, dtype=torch.long),
                "codes": torch.zeros(1, 2, 1, dtype=torch.long),
                "token_positions": torch.zeros(1, 2, dtype=torch.long),
            },
            acoustic_target_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        torch.manual_seed(123)
        before = torch.random.get_rng_state().clone()

        with patch(
            "speech_to_speech.generation.evaluation.device_batch",
            return_value=batch,
        ):
            evaluate(model, batch, _Codec(), seeds=(5, 7))

        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))
        self.assertEqual(model.generator_seeds, [5, 7])
        self.assertTrue(model.training)

    def test_rvq_generator_is_reproducible_without_global_rng_use(self):
        self.addCleanup(torch.random.set_rng_state, torch.random.get_rng_state())
        decoder = AcousticRVQDecoder(
            condition_dim=4,
            codebooks=3,
            codebook_size=8,
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
        ).eval()
        condition = torch.zeros(1, 4, 4)
        torch.manual_seed(123)
        before = torch.random.get_rng_state().clone()

        first = decoder.generate(
            condition, generator=torch.Generator().manual_seed(9)
        )
        second = decoder.generate(
            condition, generator=torch.Generator().manual_seed(9)
        )

        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before))


def task_sample_logger_fixture() -> tuple[TaskSampleLogger, object]:
    batch = ModelBatch(
        input_ids=torch.tensor([[1, 2]]),
        token_labels=torch.tensor([[-100, 2]]),
        acoustic_prompt=None,
        acoustic_target=None,
        tasks=[Task.T2TT],
        pad_token_id=0,
    )
    experiment = SimpleNamespace(add_text=Mock())
    trainer = SimpleNamespace(
        global_step=0,
        is_global_zero=True,
        logger=SimpleNamespace(experiment=experiment),
        datamodule=SimpleNamespace(collator=Mock(return_value=batch)),
    )
    logger = TaskSampleLogger([0], every_n_steps=1)
    logger.samples = [
        {
            (Role.SOURCE, Modality.TEXT): TextItem(
                views={TextView.TEXT: "source"},
                meta={TextMeta.LANG: Lang.ZH},
            ),
            (Role.TARGET, Modality.TEXT): TextItem(
                views={TextView.TEXT: "target"},
                meta={TextMeta.LANG: Lang.EN},
            ),
        }
    ]
    return logger, trainer


if __name__ == "__main__":
    unittest.main()
