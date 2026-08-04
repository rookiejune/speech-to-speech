from __future__ import annotations

import unittest

import torch

from speech_to_speech.generation.mimo import (
    MimoGenerationOptions,
    MimoGenerationStep,
    generate_mimo,
)


class _Model:
    def mimo_generation_step(
        self,
        text_input_ids: torch.Tensor,
        audio_input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        past_key_values: object | None,
        use_cache: bool,
    ) -> MimoGenerationStep:
        del text_input_ids, audio_input_ids, attention_mask, use_cache
        step = 0 if past_key_values is None else int(past_key_values)
        text_token = (3, 2, 4, 4)[min(step, 3)]
        audio_token = (15, 15, 12, 11)[min(step, 3)]
        text_logits = torch.full((1, 8), -10.0)
        audio_logits = torch.full((1, 16), -10.0)
        text_logits[:, text_token] = 10.0
        audio_logits[:, audio_token] = 10.0
        return MimoGenerationStep(
            text_logits=text_logits,
            audio_logits=audio_logits,
            past_key_values=step + 1,
        )


class _FeatureModel(_Model):
    def __init__(self) -> None:
        self.feature_calls: list[tuple[torch.Tensor | None, torch.Tensor | None]] = []

    def mimo_generation_step(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.feature_calls.append(
            (kwargs.get("audio_features"), kwargs.get("audio_feature_mask"))
        )
        kwargs.pop("audio_features", None)
        kwargs.pop("audio_feature_mask", None)
        return super().mimo_generation_step(*args, **kwargs)


class MimoGenerationTest(unittest.TestCase):
    def test_synchronous_generation_honors_delay_and_independent_endings(self) -> None:
        options = MimoGenerationOptions(
            max_new_tokens=8,
            text_eos_token_id=2,
            audio_eos_token_id=11,
            text_blank_token_id=0,
            audio_blank_token_id=9,
            audio_bos_token_id=10,
            audio_delay_tokens=1,
            do_sample=False,
            use_cache=True,
        )

        result = generate_mimo(
            _Model(),
            torch.tensor([[1]]),
            torch.tensor([[9]]),
            options,
        )

        self.assertTrue(
            torch.equal(result.text_sequences, torch.tensor([[1, 3, 2, 0, 0]]))
        )
        self.assertTrue(
            torch.equal(result.audio_sequences, torch.tensor([[9, 9, 10, 12, 11]]))
        )
        self.assertTrue(bool(result.text_finished.all()))
        self.assertTrue(bool(result.audio_finished.all()))

    def test_rejects_unaligned_prompts(self) -> None:
        options = MimoGenerationOptions(
            max_new_tokens=2,
            text_eos_token_id=2,
            audio_eos_token_id=11,
            text_blank_token_id=0,
            audio_blank_token_id=9,
            audio_bos_token_id=10,
        )
        with self.assertRaisesRegex(ValueError, "aligned"):
            generate_mimo(
                _Model(),
                torch.tensor([[1, 2]]),
                torch.tensor([[9]]),
                options,
            )

    def test_passes_observed_prompt_features_only_during_prefill(self) -> None:
        model = _FeatureModel()
        options = MimoGenerationOptions(
            max_new_tokens=2,
            text_eos_token_id=2,
            audio_eos_token_id=11,
            text_blank_token_id=0,
            audio_blank_token_id=9,
            audio_bos_token_id=10,
            do_sample=False,
            use_cache=True,
        )
        features = torch.ones(1, 1, 3)
        mask = torch.ones(1, 1, dtype=torch.bool)
        generate_mimo(
            model,
            torch.tensor([[1]]),
            torch.tensor([[9]]),
            options,
            prompt_audio_features=features,
            prompt_audio_feature_mask=mask,
        )
        self.assertIsNotNone(model.feature_calls[0][0])
        self.assertIsNone(model.feature_calls[1][0])


if __name__ == "__main__":
    unittest.main()
