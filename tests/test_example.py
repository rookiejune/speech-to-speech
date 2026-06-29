from __future__ import annotations

import unittest

import torch
from anydataset import AnyDataset, Modality, Role, collate_fn

from speech_to_speech.datamodule.example import (
    autoregression_example_from_sample,
    autoregression_examples_from_batch,
    speech_pair_from_sample,
    translation_example_from_sample,
    translation_examples_from_batch,
)
from speech_to_speech.datamodule.schema import (
    SOURCE_AUTOREGRESSION,
    TARGET_AUTOREGRESSION,
    TRANSLATION,
)
from helpers import toy_longcat_sample


class ExampleTest(unittest.TestCase):
    def test_speech_pair_from_sample_reads_source_and_target_longcat(self) -> None:
        source = torch.tensor([[1, 2, 3]])
        target = torch.tensor([[4, 5]])

        pair = speech_pair_from_sample(_sample(source, target))

        self.assertTrue(torch.equal(pair.source_ids, source))
        self.assertTrue(torch.equal(pair.target_ids, target))

    def test_source_autoregression_schema_selects_source_side(self) -> None:
        source = torch.tensor([1, 2, 3])
        target = torch.tensor([4, 5])
        resolved = AnyDataset.resolve_sample(
            _sample(source, target),
            SOURCE_AUTOREGRESSION,
        )

        example = autoregression_example_from_sample(resolved, Role.SOURCE)

        self.assertEqual(set(resolved), {(Role.SOURCE, Modality.AUDIO)})
        self.assertTrue(torch.equal(example.audio_ids, source))

    def test_target_autoregression_schema_selects_target_side(self) -> None:
        source = torch.tensor([1, 2, 3])
        target = torch.tensor([4, 5])
        resolved = AnyDataset.resolve_sample(
            _sample(source, target),
            TARGET_AUTOREGRESSION,
        )

        example = autoregression_example_from_sample(resolved, Role.TARGET)

        self.assertEqual(set(resolved), {(Role.TARGET, Modality.AUDIO)})
        self.assertTrue(torch.equal(example.audio_ids, target))

    def test_translation_schema_preserves_source_target_pair(self) -> None:
        source = torch.tensor([1, 2, 3])
        target = torch.tensor([4, 5])
        resolved = AnyDataset.resolve_sample(_sample(source, target), TRANSLATION)

        example = translation_example_from_sample(resolved)

        self.assertEqual(
            set(resolved),
            {
                (Role.SOURCE, Modality.AUDIO),
                (Role.TARGET, Modality.AUDIO),
            },
        )
        self.assertTrue(torch.equal(example.source_ids, source))
        self.assertTrue(torch.equal(example.target_ids, target))

    def test_examples_from_batch_trim_collated_padding(self) -> None:
        samples = [
            _sample(torch.tensor([1, 2, 3]), torch.tensor([4, 5])),
            _sample(torch.tensor([6]), torch.tensor([7, 8, 9])),
        ]
        batch = collate_fn(TRANSLATION)(samples)

        translations = translation_examples_from_batch(batch)

        self.assertEqual(len(translations), 2)
        self.assertTrue(torch.equal(translations[0].source_ids, torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(translations[0].target_ids, torch.tensor([4, 5])))
        self.assertTrue(torch.equal(translations[1].source_ids, torch.tensor([6])))
        self.assertTrue(torch.equal(translations[1].target_ids, torch.tensor([7, 8, 9])))

    def test_autoregression_examples_from_batch_read_selected_side(self) -> None:
        samples = [
            _sample(torch.tensor([1, 2, 3]), torch.tensor([4, 5])),
            _sample(torch.tensor([6]), torch.tensor([7, 8, 9])),
        ]
        source_batch = collate_fn(SOURCE_AUTOREGRESSION)(samples)
        target_batch = collate_fn(TARGET_AUTOREGRESSION)(samples)

        source_examples = autoregression_examples_from_batch(source_batch, Role.SOURCE)
        target_examples = autoregression_examples_from_batch(target_batch, Role.TARGET)

        self.assertEqual(len(source_examples), 2)
        self.assertEqual(len(target_examples), 2)
        self.assertTrue(torch.equal(source_examples[0].audio_ids, torch.tensor([1, 2, 3])))
        self.assertTrue(torch.equal(source_examples[1].audio_ids, torch.tensor([6])))
        self.assertTrue(torch.equal(target_examples[0].audio_ids, torch.tensor([4, 5])))
        self.assertTrue(torch.equal(target_examples[1].audio_ids, torch.tensor([7, 8, 9])))


def _sample(source: torch.Tensor, target: torch.Tensor):
    return toy_longcat_sample(source, target, include_acoustic=False)


if __name__ == "__main__":
    unittest.main()
