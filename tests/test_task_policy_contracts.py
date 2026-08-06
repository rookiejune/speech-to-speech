from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _contracts_helpers import *
from speech_to_speech.task import (
    TARGET_COT,
    uses_source_ctc,
    uses_target_ctc,
)


class TaskPolicyContractTest(unittest.TestCase):
    def test_ctc_routes_follow_audio_transcript_visibility(self):
        source_tasks = {Task.ASR, Task.S2TT, Task.S2ST}
        target_tasks = {Task.AUDIO_AR, Task.T2ST, Task.S2ST}

        for task in Task:
            with self.subTest(task=task, route="source"):
                self.assertEqual(uses_source_ctc(task), task in source_tasks)
            with self.subTest(task=task, route="target"):
                self.assertEqual(uses_target_ctc(task), task in target_tasks)

        self.assertTrue(
            uses_source_ctc(Task.S2ST)
            and not uses_target_ctc(Task.S2ST, trace=TARGET_COT)
        )
        self.assertFalse(uses_target_ctc(Task.TTS))
        self.assertFalse(uses_source_ctc(Task.TTS_VOICE_CLONE))
        self.assertFalse(uses_target_ctc(Task.TTS_VOICE_CLONE))

    def test_task_allocation_tracks_weights_across_tiny_batches(self):
        allocation = allocate_tasks([Task.T2ST, Task.TTS], [1.0, 2.0], 6)
        self.assertEqual(allocation.count(Task.T2ST), 2)
        self.assertEqual(allocation.count(Task.TTS), 4)
        collator = Collator(Mock(), {Task.TTS: 1.0, Task.T2ST: 0.0})
        self.assertEqual(collator.tasks, [Task.TTS])

        weights = TaskWeights({Task.T2ST: 1.0, Task.TTS: 9.0})
        tiny_batches = [weights.allocate(1)[0] for _ in range(10)]

        self.assertEqual(tiny_batches.count(Task.T2ST), 1)
        self.assertEqual(tiny_batches.count(Task.TTS), 9)

    def test_task_weights_are_pickleable_for_spawn_workers(self):
        weights = TaskWeights({Task.T2ST: 1.0, Task.TTS: 9.0})
        weights.allocate(1)

        restored = pickle.loads(pickle.dumps(weights))

        self.assertEqual(restored.tasks, [Task.T2ST, Task.TTS])
        self.assertIsInstance(restored.allocate(1)[0], Task)

    def test_parameter_policy_freezes_explicit_parameter_groups(self):
        model = _StageModel()

        apply_parameter_trainability(
            model,
            ParameterPolicyTrainability(
                PARAMETER_POLICY_SPECS[ParameterPolicyName.SPEECH_INTERFACE]
            ),
        )

        self.assertFalse(model.backbone.layers[0].weight.requires_grad)
        self.assertTrue(model.tokens.audio_embedding.weight.requires_grad)
        self.assertTrue(model.tokens.audio_head.weight.requires_grad)
        self.assertTrue(model.source_audio_encoder.weight.requires_grad)
        self.assertTrue(model.acoustic_decoder.head.weight.requires_grad)
        self.assertFalse(model.acoustic_decoder.decoder.embed_tokens.weight.requires_grad)
        self.assertFalse(model.acoustic_decoder.codebook_embeddings[-1].weight.requires_grad)
        self.assertFalse(model.acoustic_decoder.embedding_projections[-1].weight.requires_grad)

    def test_parameter_policy_callback_applies_on_fit_start(self):
        model = _StageModel()
        callback = build_parameter_policy(
            default_parameter_policy_config(ParameterPolicyName.SPEECH_INTERFACE)
        )

        callback.setup(Mock(), cast(Any, SimpleNamespace(model=model)), "validate")
        self.assertIsNone(callback.summary)
        self.assertTrue(model.backbone.layers[0].weight.requires_grad)

        callback.on_fit_start(Mock(), cast(Any, SimpleNamespace(model=model)))

        self.assertIsNotNone(callback.summary)
        self.assertFalse(model.backbone.layers[0].weight.requires_grad)
        self.assertTrue(model.tokens.audio_embedding.weight.requires_grad)

    def test_partial_qwen_policy_unfreezes_top_layers_and_final_norm(self):
        model = _StageModel()

        apply_parameter_trainability(
            model,
            ParameterPolicyTrainability(
                PARAMETER_POLICY_SPECS[
                    ParameterPolicyName.SPEECH_INTERFACE_TOP_THIRD
                ]
            ),
        )

        self.assertFalse(model.backbone.layers[0].weight.requires_grad)
        self.assertFalse(model.backbone.layers[1].weight.requires_grad)
        self.assertTrue(model.backbone.layers[2].weight.requires_grad)
        self.assertTrue(model.backbone.norm.weight.requires_grad)



if __name__ == "__main__":
    unittest.main()
