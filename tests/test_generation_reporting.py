from __future__ import annotations

import unittest

import torch

from scripts._generation_reporting import (
    compare_logits,
    first_difference,
    optional_max_abs,
)


class GenerationReportingTest(unittest.TestCase):
    def test_first_difference_reports_value_or_length_mismatch(self):
        self.assertEqual(
            first_difference(torch.tensor([1, 2]), torch.tensor([1, 3])),
            1,
        )
        self.assertEqual(
            first_difference(torch.tensor([1]), torch.tensor([1, 2])),
            1,
        )
        self.assertIsNone(first_difference(torch.tensor([1]), torch.tensor([1])))

    def test_optional_max_abs_requires_matching_shapes(self):
        self.assertIsNone(optional_max_abs(torch.zeros(1), torch.zeros(2)))
        self.assertEqual(optional_max_abs(torch.tensor([1.0]), torch.tensor([3.0])), 2.0)

    def test_compare_logits_rejects_step_count_mismatch(self):
        with self.assertRaisesRegex(ValueError, "same logit steps"):
            compare_logits([torch.zeros(1)], [])


if __name__ == "__main__":
    unittest.main()
