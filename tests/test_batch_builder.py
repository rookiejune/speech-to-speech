from __future__ import annotations

import unittest

import torch

from speech_to_speech.datamodule.batch_builder import CausalLMBatchBuilder
from speech_to_speech.types import (
    AutoregressionExample,
    IGNORE_INDEX,
    TranslationExample,
)
from helpers import MockTokenizer, toy_idspace


class CausalLMBatchBuilderTest(unittest.TestCase):
    def test_autoregression_builds_padded_global_batch(self) -> None:
        builder = CausalLMBatchBuilder(toy_idspace(), tokenizer=MockTokenizer())

        batch = builder.autoregression(
            [
                AutoregressionExample(audio_ids=torch.tensor([0, 1, 2])),
                AutoregressionExample(audio_ids=torch.tensor([3, 4])),
            ]
        )

        self.assertEqual(
            batch.input_ids.tolist(),
            [
                [1, 10, 11, 12, 13, 16, 18, 19, 20],
                [1, 10, 11, 12, 13, 16, 21, 22, 0],
            ],
        )
        self.assertEqual(
            batch.attention_mask.tolist(),
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 0],
            ],
        )
        self.assertEqual(
            batch.labels.tolist(),
            [
                [
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    18,
                    19,
                    20,
                    17,
                ],
                [
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    21,
                    22,
                    17,
                    IGNORE_INDEX,
                ],
            ],
        )
        self.assertEqual(batch.logits_to_keep, 4)
        self.assertEqual(
            batch.loss_weights.tolist(),
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0],
            ],
        )

    def test_autoregression_carries_bpe_expansion_loss_weights(self) -> None:
        builder = CausalLMBatchBuilder(toy_idspace(), tokenizer=MockTokenizer())

        batch = builder.autoregression(
            AutoregressionExample(
                audio_ids=torch.tensor([0, 1]),
                audio_weights=torch.tensor([2.0, 3.0]),
            )
        )

        self.assertEqual(batch.labels.tolist()[0][-4:], [IGNORE_INDEX, 18, 19, 17])
        self.assertEqual(batch.loss_weights.tolist()[0][-4:], [0.0, 2.0, 3.0, 1.0])

    def test_translation_uses_chat_template_and_replaces_source_placeholder(self) -> None:
        builder = CausalLMBatchBuilder(toy_idspace(), tokenizer=MockTokenizer())

        batch = builder.translation(
            TranslationExample(
                source_ids=torch.tensor([0, 1]),
                target_ids=torch.tensor([2, 3]),
            )
        )

        self.assertEqual(
            batch.input_ids.tolist(),
            [[1, 10, 11, 16, 18, 19, 17, 12, 13, 16, 20, 21]],
        )
        self.assertEqual(
            batch.labels.tolist(),
            [
                [
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    IGNORE_INDEX,
                    20,
                    21,
                    17,
                ]
            ],
        )
        self.assertEqual(batch.loss_weights.tolist()[0][-4:], [0.0, 1.0, 1.0, 1.0])

    def test_autoregression_generation_uses_boa_without_eoa(self) -> None:
        builder = CausalLMBatchBuilder(toy_idspace(), tokenizer=MockTokenizer())

        batch = builder.autoregression_generation(torch.tensor([0, 1]))

        self.assertEqual(batch.input_ids.tolist(), [[1, 10, 11, 12, 13, 16, 18, 19]])
        self.assertEqual(batch.attention_mask.tolist(), [[1, 1, 1, 1, 1, 1, 1, 1]])

    def test_translation_generation_ends_at_target_boa(self) -> None:
        builder = CausalLMBatchBuilder(toy_idspace(), tokenizer=MockTokenizer())

        batch = builder.translation_generation(torch.tensor([0, 1]))

        self.assertEqual(batch.input_ids.tolist(), [[1, 10, 11, 16, 18, 19, 17, 12, 13, 16]])
        self.assertEqual(batch.attention_mask.tolist(), [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])


if __name__ == "__main__":
    unittest.main()
