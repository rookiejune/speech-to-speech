from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _config_helpers import *
from speech_to_speech.runtime import migrate_config_fields


@patch.dict(
    "os.environ",
    {
        "DYNAMIC_HOME": "/tmp/dynamic",
        "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
    },
)
class ConfigCompositionTest(ConfigTestCase):
    def test_runtime_presets_expose_only_canonical_audio_fields(self):
        for preset in (
            "bicodec",
            "glm4_bicodec",
            "glm4_bicodec_composed",
            "kimi_audio",
            "longcat",
            "longcat_native",
            "qwen2_5_omni_text",
            "stable_codec",
            "unicodec",
        ):
            with self.subTest(preset=preset):
                raw = _compose("overfit", f"runtime={preset}")
                self.assertNotIn("codec", raw.runtime)
                self.assertNotIn("audio_tokenizer", raw.runtime)
                self.assertNotIn("input_audio", raw.runtime)
                self.assertNotIn("vocab_size", raw.runtime.audio_output)
                self.assertNotIn("frame_rate", raw.runtime.audio_output)
                self.assertNotIn("view", raw.runtime.audio_output)
                self.assertIn("detokenizer", raw.runtime.audio_output)

        glm4 = _compose("overfit", "runtime=glm4_bicodec")
        self.assertEqual(set(glm4.runtime.audio_input), {"tokenizer", "bpe"})

        composed = _compose("overfit", "runtime=glm4_bicodec_composed")
        self.assertEqual(set(composed.runtime.audio_input), {"streams"})
        self.assertEqual(
            set(composed.runtime.audio_input.streams),
            {"semantic", "global"},
        )

    def test_runtime_audio_sides_are_regular_overrides(self):
        config = _overfit(
            "runtime=glm4_bicodec",
            "model/acoustic=none",
            "audio_sequence_layout=flattened",
        )

        self.assertIsNotNone(config.runtime.audio_input)
        if config.runtime.audio_input is None:
            self.fail("configured input audio must be independent")
        self.assertEqual(config.runtime.audio_input.tokenizer, "glm4")
        self.assertIsNone(config.runtime.audio_input.bpe)
        self.assertEqual(config.runtime.audio_output.tokenizer, "bicodec")
        self.assertEqual(config.runtime.audio_output.detokenizer, "bicodec")
        self.assertIsNone(config.runtime.audio_output.bpe)
        self.assertEqual(config.datamodule.codec, "bicodec")

    def test_composed_audio_input_parses_to_typed_stream_configs(self):
        config = _overfit(
            "runtime=glm4_bicodec_composed",
            "model/acoustic=none",
            "audio_sequence_layout=flattened",
        )

        self.assertIsNotNone(config.runtime.audio_input)
        if config.runtime.audio_input is None:
            self.fail("configured input audio must be independent")
        self.assertTrue(config.runtime.audio_input.composed)
        self.assertIsNone(config.runtime.audio_input.tokenizer)
        streams = config.runtime.audio_input.streams
        self.assertIsNotNone(streams)
        assert streams is not None
        self.assertEqual(streams["semantic"].tokenizer, "glm4")
        self.assertEqual(streams["global"].tokenizer, "bicodec")

    def test_output_detokenizer_null_is_explicit_codes_only(self):
        config = _overfit(
            "runtime=bicodec",
            "model/acoustic=none",
            "audio_sequence_layout=flattened",
            "runtime.audio_output.detokenizer=null",
        )

        self.assertEqual(config.runtime.audio_output.tokenizer, "bicodec")
        self.assertIsNone(config.runtime.audio_output.detokenizer)

    def test_legacy_runtime_audio_overrides_migrate_to_canonical_sides(self):
        fields = {
            "codec": "bicodec",
            "input_audio": {
                "codec": "glm4",
                "vocab_size": 16_384,
                "frame_rate": 12.5,
            },
            "audio_tokenizer": "/tmp/bicodec-tokenizer",
        }
        with self.assertWarns(FutureWarning):
            migrate_config_fields(fields)

        self.assertEqual(
            fields,
            {
                "audio_input": {
                    "tokenizer": "glm4",
                    "vocab_size": 16_384,
                    "frame_rate": 12.5,
                },
                "audio_output": {
                    "tokenizer": "bicodec",
                    "detokenizer": "bicodec",
                    "bpe": "/tmp/bicodec-tokenizer",
                },
            },
        )

    def test_legacy_null_audio_tokenizer_is_an_explicit_override(self):
        fields = {"codec": "longcat", "audio_tokenizer": None}
        with self.assertWarns(FutureWarning):
            migrate_config_fields(fields)

        self.assertEqual(
            fields,
            {
                "audio_output": {
                    "tokenizer": "longcat",
                    "detokenizer": "longcat",
                    "bpe": None,
                }
            },
        )

    def test_mixed_runtime_audio_fields_reject_conflicts(self):
        with self.assertRaisesRegex(ValueError, "conflicting audio_output.tokenizer"):
            _overfit(
                "runtime=bicodec",
                "model/acoustic=none",
                "audio_sequence_layout=flattened",
                "+runtime.codec=longcat",
                "runtime.audio_output.tokenizer=bicodec",
            )

    def test_roots_parse_to_src_aligned_configs(self):
        flow = _overfit()
        rvq = _overfit("model/acoustic=rvq")
        token = _overfit(
            "runtime=unicodec",
            "model/acoustic=none",
            "audio_sequence_layout=flattened",
        )

        self.assertIsInstance(flow, OverfitFlowConfig)
        self.assertIsInstance(rvq, OverfitRVQConfig)
        self.assertIsInstance(token, OverfitTokenConfig)
        self.assertEqual(token.model.acoustic.type, AcousticType.NONE.value)
        self.assertIsInstance(flow.runtime, RuntimeConfig)
        self.assertIsInstance(flow.model, ModelConfig)
        self.assertIsInstance(flow.pl_module, ModuleConfig)
        self.assertIsInstance(flow.model.ctc, CTCDecoderRoutesConfig)
        self.assertIsInstance(flow.pl_module.ctc, CTCConfig)
        self.assertIsInstance(flow.model.acoustic.decoder, DecoderConfig)
        self.assertEqual(flow.runtime.audio_output.tokenizer, "longcat")
        self.assertEqual(token.runtime.audio_output.tokenizer, "unicodec")
        self.assertIs(flow.model.semantic_audio_adapter, AdapterType.LINEAR)
        self.assertFalse(flow.callbacks.performance.enabled)

    def test_acoustic_generator_initialization_is_explicit(self):
        flow = _overfit("model.acoustic.init_artifact=/tmp/flow-artifact")
        rvq = _overfit(
            "model/acoustic=rvq",
            "model.acoustic.init_artifact=/tmp/rvq-artifact",
        )

        self.assertEqual(flow.model.acoustic.init_artifact, "/tmp/flow-artifact")
        self.assertEqual(rvq.model.acoustic.init_artifact, "/tmp/rvq-artifact")

    def test_toy_smoke_selects_model_and_dataset_without_a_toy_runtime(self):
        config = _overfit("experiment=overfit/toy_smoke")

        self.assertIsInstance(config, OverfitFlowConfig)
        self.assertIsInstance(config.runtime, RuntimeConfig)
        self.assertEqual(config.runtime.audio_output.tokenizer, "longcat")
        self.assertEqual(config.runtime.backbone, "Qwen/Qwen3-0.6B")
        self.assertEqual(config.runtime.device, "cpu")
        self.assertIsInstance(config.model.toy, ToyConfig)
        self.assertEqual(config.model.toy.hidden_size, 32)
        self.assertIs(config.datamodule.dataset.name, DatasetName.TOY)
        self.assertEqual(config.datamodule.dataset.toy_samples, 8)
        self.assertEqual(config.datamodule.dataset.toy_frames, 4)
        self.assertEqual(config.train.max_steps, 2)
        self.assertFalse(config.callbacks.task_sample.enabled)
        self.assertFalse(config.callbacks.evaluation.enabled)

        production = _overfit()
        self.assertIsNone(production.model.toy)
        self.assertIs(production.datamodule.dataset.name, DatasetName.WMT19_TTS)

        selected = _overfit("model=toy", "datamodule/dataset=toy")
        self.assertIsInstance(selected.model.toy, ToyConfig)
        self.assertIs(selected.datamodule.dataset.name, DatasetName.TOY)

        streaming = _overfit("datamodule/dataset=streaming_s2st")
        self.assertIs(streaming.datamodule.dataset.name, DatasetName.STREAMING_S2ST)
        self.assertIsNone(streaming.datamodule.dataset.filter)

    def test_qwen2_5_omni_text_runtime_uses_text_adapter(self):
        config = _overfit(
            "runtime=qwen2_5_omni_text",
            "model/acoustic=none",
        )

        self.assertIs(config.runtime.backbone_type, BackboneType.QWEN2_5_OMNI_TEXT)
        self.assertEqual(config.runtime.audio_output.tokenizer, "bicodec")
        self.assertEqual(config.runtime.backbone, "Qwen/Qwen2.5-Omni-7B")
        self.assertEqual(config.runtime.backbone_module, "")
        self.assertEqual(config.runtime.backbone_body, "base_model")
        self.assertTrue(config.runtime.gradient_checkpointing)

    def test_kimi_audio_runtime_uses_modality_readouts(self):
        config = _overfit(
            "runtime=kimi_audio",
            "model/acoustic=none",
        )

        self.assertIs(config.runtime.backbone_type, BackboneType.KIMI_AUDIO)
        self.assertEqual(config.runtime.audio_output.tokenizer, "bicodec")
        self.assertEqual(config.runtime.backbone, "moonshotai/Kimi-Audio-7B-Instruct")
        self.assertTrue(config.runtime.backbone_trust_remote_code)
        self.assertIn("messages", config.runtime.backbone_chat_template)
        self.assertEqual(config.runtime.backbone_readout, "last_hidden_state[0]")
        self.assertEqual(
            config.runtime.backbone_readouts,
            {"text": "last_hidden_state[0]", "audio": "last_hidden_state[1]"},
        )
        self.assertEqual(config.runtime.backbone_module, "")
        self.assertEqual(config.runtime.backbone_body, "base_model")
        self.assertFalse(config.runtime.backbone_supports_cache_position)
        self.assertTrue(config.runtime.gradient_checkpointing)

    def test_random_backbone_requires_unambiguous_full_training(self):
        random = _overfit("runtime.backbone_initialization=random")

        self.assertIs(
            random.runtime.backbone_initialization,
            BackboneInitialization.RANDOM,
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined with model.toy"):
            _overfit(
                "model=toy",
                "datamodule/dataset=toy",
                "runtime.backbone_initialization=random",
            )
        with self.assertRaisesRegex(ValueError, "fully trainable backbone"):
            _train("runtime.backbone_initialization=random")

        train = _train(
            "runtime.backbone_initialization=random",
            "callback/parameter_policy@callbacks.parameter_policy=full",
            "model.lora=null",
        )
        self.assertIs(
            train.runtime.backbone_initialization,
            BackboneInitialization.RANDOM,
        )

    def test_full_codec_sequence_smoke_is_token_only_comparison(self):
        config = overfit(_compose("overfit", "experiment=overfit/longcat_flattened_smoke"))

        self.assertIsInstance(config, OverfitTokenConfig)
        self.assertEqual(config.runtime.audio_output.tokenizer, "longcat")
        self.assertIs(config.audio_sequence_layout, AudioSequenceLayout.FLATTENED)
        self.assertEqual(config.model.acoustic.type, AcousticType.NONE.value)
        self.assertIsInstance(config.model.toy, ToyConfig)
        self.assertIs(config.datamodule.dataset.name, DatasetName.TOY)
        self.assertEqual(config.run_name, "longcat-flattened-token")
        self.assertEqual(config.train.max_steps, 2)
        self.assertFalse(config.callbacks.task_sample.enabled)
        self.assertFalse(config.callbacks.evaluation.enabled)

    def test_decoupled_semantic_smoke_loads_artifact_config(self):
        with patch.dict(
            "os.environ",
            {"SPEECH_TO_SPEECH_ACOUSTIC_GENERATOR_ARTIFACT": "/tmp/semantic-codec"},
        ):
            config = overfit(
                _compose(
                    "overfit",
                    "experiment=overfit/longcat_semantic_only_smoke",
                )
            )

        self.assertIsInstance(config, OverfitTokenConfig)
        self.assertEqual(config.runtime.audio_output.tokenizer, "longcat")
        self.assertEqual(
            config.runtime.audio_output.acoustic_generator_artifact,
            "/tmp/semantic-codec",
        )
        self.assertEqual(config.runtime.device, "cpu")
        self.assertEqual(config.model.acoustic.type, AcousticType.NONE.value)
        self.assertIsInstance(config.model.toy, ToyConfig)
        self.assertIs(config.datamodule.dataset.name, DatasetName.TOY)

    def test_bicodec_smokes_predict_complete_target_audio(self):
        clone = overfit(_compose("overfit", "experiment=overfit/bicodec_voice_clone_smoke"))
        generate = overfit(_compose("overfit", "experiment=overfit/bicodec_generate_global_smoke"))

        for config in (clone, generate):
            self.assertIsInstance(config, OverfitTokenConfig)
            self.assertEqual(config.runtime.audio_output.tokenizer, "bicodec")
            self.assertIsNone(config.datamodule.dataset.speaker)

        self.assertIs(clone.audio_sequence_layout, AudioSequenceLayout.FLATTENED)
        self.assertIs(generate.audio_sequence_layout, AudioSequenceLayout.FLATTENED)
        self.assertEqual(clone.runtime.audio_input.tokenizer, "glm4")
        self.assertIs(clone.datamodule.dataset.name, DatasetName.TOY)
        self.assertIs(clone.datamodule.shape, DataShape.PAIR)
        self.assertEqual(clone.task, Task.TTS_VOICE_CLONE.value)
        self.assertIs(
            generate.datamodule.dataset.name,
            DatasetName.QWEN_TTS_SPEAKER,
        )
        self.assertIs(generate.datamodule.shape, DataShape.SINGLE)
        self.assertEqual(generate.task, Task.TTS.value)
        self.assertEqual(clone.run_name, "bicodec-voice-clone-full-audio")
        self.assertEqual(generate.run_name, "bicodec-generate-global")
        self.assertIn("bicodec-voice-clone-smoke", clone.output_dir)
        self.assertIn("bicodec-generate-global-smoke", generate.output_dir)
        self.assertIsNone(clone.runtime.audio_output.acoustic_generator_artifact)
        self.assertIsNone(generate.runtime.audio_output.acoustic_generator_artifact)

    def test_root_schema_rejects_unknown_and_foreign_fields(self):
        cases = [
            (overfit, _compose("overfit", "+unknown=1"), "unknown"),
            (
                overfit,
                _compose("overfit", "+model.acoustic.normalize_features=true"),
                "model.acoustic.normalize_features",
            ),
            (
                overfit,
                _compose(
                    "overfit",
                    "model/acoustic=rvq",
                    "+model.acoustic.repa.weight=0.1",
                ),
                "model.acoustic.repa",
            ),
            (
                overfit,
                _compose("overfit", "+train.ckpt_path=/tmp/resume.ckpt"),
                "train.ckpt_path",
            ),
        ]

        for parser, raw, key in cases:
            with self.subTest(key=key):
                with self.assertRaises((ConfigKeyError, ConfigAttributeError)) as raised:
                    parser(raw)
                self.assertIn(key, str(raised.exception))

    def test_unicodec_experiments_close_the_token_training_chain(self):
        cases = [
            (
                "overfit/unicodec",
                100,
                "auto",
                "auto",
                False,
                True,
            ),
            (
                "overfit/unicodec_ddp_smoke",
                2,
                "auto",
                "ddp_find_unused_parameters_true",
                True,
                False,
            ),
        ]

        for experiment, max_steps, devices, strategy, checkpointing, sampler in cases:
            with self.subTest(experiment=experiment):
                config = _overfit(f"experiment={experiment}")

                self.assertIsInstance(config, OverfitTokenConfig)
                self.assertEqual(config.runtime.audio_output.tokenizer, "unicodec")
                self.assertIsNone(config.runtime.audio_output.bpe)
                self.assertEqual(config.train.max_steps, max_steps)
                self.assertEqual(config.trainer.devices, devices)
                self.assertEqual(config.trainer.strategy, strategy)
                self.assertEqual(config.trainer.precision, "bf16-mixed")
                self.assertEqual(config.trainer.max_epochs, -1)
                self.assertEqual(config.trainer.log_every_n_steps, 1)
                self.assertIs(config.trainer.enable_checkpointing, checkpointing)
                self.assertIs(config.trainer.use_distributed_sampler, sampler)
                self.assertTrue(config.callbacks.task_sample.enabled)
                self.assertEqual(config.callbacks.task_sample.every_n_steps, 1)

    def test_parameter_policy_smoke_composes_each_supported_policy(self):
        policies = (
            ParameterPolicyName.FULL,
            ParameterPolicyName.LORA,
            ParameterPolicyName.SPEECH_INTERFACE,
            ParameterPolicyName.SPEECH_INTERFACE_TOP_THIRD,
            ParameterPolicyName.SEMANTIC_ONLY,
            ParameterPolicyName.ACOUSTIC_ONLY,
        )

        for policy in policies:
            with self.subTest(policy=policy.value):
                overrides = [
                    "experiment=train/smoke/parameter_policy",
                    f"callback/parameter_policy@callbacks.parameter_policy={policy.value}",
                ]
                if policy is ParameterPolicyName.LORA:
                    overrides.append("+model/lora@model.lora=qwen")
                else:
                    overrides.append("model.lora=null")
                config = _train(*overrides)

                self.assertIs(config.callbacks.parameter_policy.name, policy)
                self.assertEqual(config.train.max_steps, 2)
                self.assertEqual(config.trainer.accelerator, "cpu")
                self.assertIn(policy.value, config.output_subdir)
                if policy is ParameterPolicyName.ACOUSTIC_ONLY:
                    self.assertIn(
                        ParameterGroup.ALIGNMENT_DECODER,
                        config.callbacks.parameter_policy.frozen_groups,
                    )
                else:
                    self.assertIn(
                        ParameterGroup.ALIGNMENT_DECODER,
                        config.callbacks.parameter_policy.trainable_groups,
                    )

    def test_text_ar_uses_the_text_loader(self):
        loader = LoaderConfig(
            weight=1.0,
            task_weights={"text_ar": 1.0},
        )

        self.assertTrue(loader.is_text)
        self.assertEqual(loader.tasks, {Task.TEXT_AR: 1.0})

    def test_loader_prediction_override_is_removed(self):
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            LoaderConfig(
                weight=1.0,
                task_weights={"s2st": 1.0},
                prediction="parallel",  # type: ignore[call-arg]
            )

    def test_pretraining_framing_rejects_non_ar_tasks(self):
        with self.assertRaisesRegex(ValueError, "only supports AUDIO_AR and TEXT_AR"):
            LoaderConfig(
                weight=1.0,
                task_weights={"asr": 1.0},
                ar_framing="pretraining",
            )

    def test_removed_parallel_groups_are_not_composable(self):
        cases = [
            ("overfit", "codec=unicodec"),
            ("overfit", "sampler=smoke"),
            ("overfit", "optimizer=sft"),
            ("overfit", "init=random"),
            ("overfit", "trainer=overfit"),
        ]

        for config_name, override in cases:
            with self.subTest(config_name=config_name, override=override):
                with self.assertRaises(ConfigCompositionException):
                    _compose(config_name, override)

    def test_public_model_config_parses_domain_enums(self):
        config = _overfit(
            "model.semantic_audio_adapter=mlp",
            "model.audio_output_adapter.type=none",
            "model.fsq_embedding.feature=digit_value",
        )

        self.assertIs(config.model.semantic_audio_adapter, AdapterType.MLP)
        self.assertIs(
            config.model.audio_output_adapter.type,
            AudioOutputAdapterType.NONE,
        )
        self.assertIs(config.model.fsq_embedding.feature, FsqFeature.DIGIT_VALUE)

        with self.assertRaises(ValueError):
            _overfit("model.semantic_audio_adapter=invalid")

    def test_audio_neighbor_smoothing_is_explicit_and_validated(self):
        default = _overfit()
        configured = _overfit("pl_module.audio_neighbor_smoothing=0.05")

        self.assertEqual(default.pl_module.audio_neighbor_smoothing, 0.0)
        self.assertEqual(configured.pl_module.audio_neighbor_smoothing, 0.05)
        with self.assertRaises(ValueError):
            _overfit("pl_module.audio_neighbor_smoothing=1.0")

    def test_ctc_routes_and_decoders_are_structured_and_validated(self):
        default = _overfit()
        configured = _overfit(
            "pl_module.ctc.source.weight=0.25",
            "pl_module.ctc.target.weight=0.5",
            "model.ctc.source.type=linear",
            "model.ctc.source.pool_factor=2",
            "model.ctc.target.type=transformer",
            "model.ctc.target.pool_factor=4",
        )

        for route in (default.pl_module.ctc.source, default.pl_module.ctc.target):
            self.assertIsInstance(route, CTCRouteConfig)
            self.assertEqual(route.weight, 0.0)
        for decoder in (default.model.ctc.source, default.model.ctc.target):
            self.assertIsInstance(decoder, CTCDecoderConfig)
            self.assertIs(decoder.type, CTCDecoderType.IDENTITY)
            self.assertIsNone(decoder.backbone_readout)
            self.assertEqual(decoder.pool_factor, 1)
            self.assertEqual(decoder.layers, 2)
            self.assertEqual(decoder.heads, 8)
            self.assertEqual(decoder.ffn_ratio, 4.0)
            self.assertEqual(decoder.dropout, 0.0)
        self.assertEqual(configured.pl_module.ctc.source.weight, 0.25)
        self.assertEqual(configured.pl_module.ctc.target.weight, 0.5)
        self.assertIs(
            configured.model.ctc.source.type,
            CTCDecoderType.LINEAR,
        )
        self.assertEqual(configured.model.ctc.source.pool_factor, 2)
        self.assertIs(
            configured.model.ctc.target.type,
            CTCDecoderType.TRANSFORMER,
        )
        self.assertEqual(configured.model.ctc.target.pool_factor, 4)
        with self.assertRaises(ValueError):
            _overfit("pl_module.ctc.source.weight=-1")
        with self.assertRaises(ValueError):
            _overfit("model.ctc.target.type=invalid")

    def test_audio_input_adapter_is_structured_and_mlp_by_default(self):
        default = _overfit()
        self.assertIs(
            default.model.audio_input_adapter.type,
            AudioInputAdapterType.MLP,
        )
        configured = _overfit("model.audio_input_adapter.type=transformer")
        self.assertIs(
            configured.model.audio_input_adapter.type,
            AudioInputAdapterType.TRANSFORMER,
        )

    def test_audio_output_adapter_is_structured_and_tied_by_default(self):
        default = _overfit()
        self.assertIs(
            default.model.audio_output_adapter.type,
            AudioOutputAdapterType.NONE,
        )
        configured = _overfit("model.audio_output_adapter.type=mlp")
        self.assertIs(
            configured.model.audio_output_adapter.type,
            AudioOutputAdapterType.MLP,
        )

    def test_lora_model_and_parameter_policy_must_be_selected_together(self):
        config = _lora_overfit()

        self.assertIsInstance(config.model.lora, LoraConfig)
        if config.model.lora is None:
            self.fail("PEFT LoRA config was not composed")
        self.assertEqual(config.model.lora.r, 16)
        self.assertEqual(config.model.lora.lora_alpha, 32)
        self.assertEqual(config.model.lora.init_lora_weights, "pissa")
        self.assertIs(config.callbacks.parameter_policy.name, ParameterPolicyName.LORA)
        self.assertEqual(
            config.callbacks.parameter_policy.trainable_groups,
            [
                ParameterGroup.BACKBONE_ADAPTER,
                ParameterGroup.SEMANTIC_AUDIO_EMBEDDING,
                ParameterGroup.SEMANTIC_AUDIO_ADAPTER,
                ParameterGroup.AUDIO_INPUT_ADAPTER,
                ParameterGroup.AUDIO_OUTPUT,
                ParameterGroup.ALIGNMENT_DECODER,
                ParameterGroup.ACOUSTIC_DECODER,
            ],
        )

        for overrides in (
            ("+model/lora@model.lora=qwen",),
            ("callback/parameter_policy@callbacks.parameter_policy=lora",),
        ):
            with (
                self.subTest(overrides=overrides),
                self.assertRaisesRegex(ValueError, "must be selected together"),
            ):
                _overfit(*overrides)

    def test_lora_muon_requires_pissa_initialization(self):
        config = _lora_overfit("optim.name=muon")
        self.assertEqual(config.optim.name, "muon")
        self.assertEqual(config.model.lora.init_lora_weights, "pissa")

        with self.assertRaisesRegex(ValueError, "pissa initialization"):
            _lora_overfit(
                "model.lora.init_lora_weights=gaussian",
                "optim.name=muon",
            )

    def test_optim_name_is_selectable(self):
        default = _train()
        muon = _train("optim.name=muon")

        self.assertEqual(default.optim.name, "adamw")
        self.assertEqual(muon.optim.name, "muon")
        self.assertEqual(muon.model.lora.init_lora_weights, "pissa")

    def test_lora_training_smoke_accepts_approximate_performance_provider(self):
        with self.assertWarnsRegex(UserWarning, "approximate FLOPs"):
            config = _train(
                "experiment=train/smoke/parameter_policy",
                "+model/lora@model.lora=qwen",
                "callback/parameter_policy@callbacks.parameter_policy=lora",
                "callbacks.performance.enabled=true",
                "callbacks.task_sample.enabled=false",
            )

        self.assertTrue(config.callbacks.performance.enabled)
        self.assertIs(config.callbacks.parameter_policy.name, ParameterPolicyName.LORA)

    def test_lora_rejects_peft_inference_mode_for_training(self):
        with self.assertRaisesRegex(ValueError, "inference_mode=false"):
            _lora_overfit("+model.lora.inference_mode=true")


if __name__ == "__main__":
    unittest.main()
