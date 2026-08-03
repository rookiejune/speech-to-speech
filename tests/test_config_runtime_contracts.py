from __future__ import annotations

# ruff: noqa: F403,F405

import subprocess
import sys
import unittest

from _contracts_helpers import *
from speech_to_speech.model import Config as ModelConfig


class ConfigRuntimeContractTest(unittest.TestCase):
    def test_acoustic_config_import_does_not_load_runtime_models(self):
        code = "\n".join(
            (
                "import sys",
                "sys.modules['flow_matching'] = None",
                "from speech_to_speech.model.acoustic import AcousticType",
                "assert AcousticType.NONE.value == 'none'",
                "assert 'speech_to_speech.model.acoustic.flow' not in sys.modules",
                "assert 'speech_to_speech.model.acoustic.rvq' not in sys.modules",
            )
        )

        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_dataloader_config_validates_loader_values(self):
        invalid = (
            ({"batch_size": 0, "num_workers": 0}, ValueError, "positive"),
            ({"batch_size": True, "num_workers": 0}, TypeError, "integer"),
            ({"batch_size": 1, "num_workers": -1}, ValueError, "non-negative"),
            (
                {"batch_size": 1, "num_workers": 0, "pin_memory": 1},
                TypeError,
                "boolean",
            ),
        )
        for kwargs, error, message in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(error, message):
                DataLoaderConfig(**kwargs)
        costs_invalid = (
            ({"enabled": True}, ValueError, "max_batch_frames"),
            (
                {"enabled": True, "max_batch_frames": 1, "planning_window": 0},
                ValueError,
                "planning_window",
            ),
            ({"enabled": 1}, TypeError, "boolean"),
        )
        for kwargs, error, message in costs_invalid:
            with self.subTest(costs=kwargs), self.assertRaisesRegex(error, message):
                DataLoaderCostsConfig(**kwargs)
        with self.assertRaisesRegex(TypeError, "DataLoaderCostsConfig"):
            DataLoaderConfig(
                batch_size=1,
                num_workers=0,
                costs=cast(DataLoaderCostsConfig, {}),
            )
        structured = OmegaConf.structured(DataLoaderConfig)
        merged = OmegaConf.merge(
            structured,
            {"batch_size": 8, "num_workers": 4, "pin_memory": True},
        )
        loader = OmegaConf.to_object(merged)
        self.assertIsInstance(loader, DataLoaderConfig)
        self.assertIsInstance(loader.costs, DataLoaderCostsConfig)
        self.assertFalse(loader.costs.enabled)
        with self.assertRaisesRegex(ValueError, "text loaders"):
            TextConfig(
                dataloader=DataLoaderConfig(
                    batch_size=1,
                    num_workers=0,
                    costs=DataLoaderCostsConfig(
                        enabled=True,
                        max_batch_frames=8,
                    ),
                )
            )

    def test_datamodule_configs_reject_mapping_loader_config(self):
        loader = cast(DataLoaderConfig, {"batch_size": 1, "num_workers": 0})

        with self.assertRaisesRegex(TypeError, "DataLoaderConfig"):
            SpeechConfig(codec="longcat", dataloader=loader)
        with self.assertRaisesRegex(TypeError, "DataLoaderConfig"):
            TextConfig(dataloader=loader)

    def test_runtime_loads_semantic_codec_artifact_with_existing_backend(self):
        backend = SimpleNamespace(name="longcat")
        support = SimpleNamespace()
        semantic_runtime = SimpleNamespace(sample_rate=24_000, frame_rate=50.0, decode=Mock())
        runtime = Runtime(
            Config(
                codec="longcat",
                semantic_codec_artifact="/tmp/semantic-codec",
            )
        )

        with (
            patch(
                "speech_to_speech.runtime.runtime.load_codec",
                return_value=backend,
            ),
            patch(
                "semantic_acoustic_codec.runtime.artifact.load_artifact",
                return_value=support,
            ) as load,
            patch(
                "semantic_acoustic_codec.runtime.SemanticCodecRuntime",
                return_value=semantic_runtime,
            ) as bind,
        ):
            loaded = runtime.semantic_codec

        self.assertIs(loaded, semantic_runtime)
        load.assert_called_once_with(
            Path("/tmp/semantic-codec"),
            device=None,
        )
        bind.assert_called_once_with(support, backend)

    def test_runtime_rejects_semantic_codec_without_artifact(self):
        runtime = Runtime(Config(codec="longcat"))

        with self.assertRaisesRegex(RuntimeError, "semantic_codec_artifact"):
            _ = runtime.semantic_codec

    def test_semantic_codec_artifact_disables_acoustic_side_channel(self):
        runtime = Runtime(
            Config(
                codec="longcat",
                semantic_codec_artifact="/tmp/semantic-codec",
            )
        )

        with patch("speech_to_speech.runtime.runtime.load_codec") as load:
            self.assertFalse(runtime.acoustic_side_channel)

        load.assert_not_called()

    def test_worker_runtime_snapshot_excludes_model_and_codec(self):
        runtime = SimpleNamespace(
            codec_name="longcat",
            audio_view=AudioView.LONGCAT,
            codec_frame_rate=50.0,
            audio_sequence_layout=AudioSequenceLayout.SEMANTIC,
            semantic_codec_artifact=None,
            acoustic_layout=AcousticLayout.FRAME_ALIGNED,
            acoustic_unit_length=None,
            text_tokenizer=_Tokenizer(10),
            audio_tokenizer=NativeAudioTokenizer(vocab_size=8),
            layout=Layout(text=(0, 10), audio=(10, 20)),
            pad_token_id=0,
            eos_token_id=1,
            boa_token_id=18,
            eoa_token_id=19,
            mask_token_id=20,
            codec=object(),
            backbone=object(),
        )

        snapshot = pickle.loads(pickle.dumps(DataRuntimeSnapshot.from_runtime(runtime)))

        self.assertFalse(hasattr(snapshot, "codec"))
        self.assertFalse(hasattr(snapshot, "backbone"))
        self.assertEqual(snapshot.codec_frame_rate, 50.0)
        self.assertEqual(snapshot.layout.blocks, runtime.layout.blocks)
        self.assertIs(snapshot.layout, snapshot.layout)

    def test_trainer_presets_have_one_composable_schema(self):
        configs = [
            _compose(),
            _compose("trainer=ddp"),
            _compose(
                "experiment=overfit/unicodec_ddp_smoke",
            ),
        ]
        expected = {
            "accelerator",
            "devices",
            "strategy",
            "use_distributed_sampler",
            "precision",
            "max_epochs",
            "log_every_n_steps",
            "enable_checkpointing",
            "gradient_clip_val",
        }

        for config in configs:
            self.assertEqual(set(config.trainer), expected)
            self.assertEqual(config.trainer.devices, "auto")

        self.assertEqual(
            configs[1].trainer.strategy,
            "ddp_find_unused_parameters_true",
        )
        self.assertEqual(
            configs[2].trainer.strategy,
            "ddp_find_unused_parameters_true",
        )
        self.assertEqual(configs[2].trainer.precision, "bf16-mixed")
        self.assertTrue(configs[1].trainer.use_distributed_sampler)
        self.assertFalse(configs[2].trainer.use_distributed_sampler)

    @patch("scripts.overfit.pl.Trainer")
    @patch("scripts.overfit.build_logger")
    def test_overfit_trainer_consumes_the_unicodec_ddp_contract(
        self,
        logger,
        trainer,
    ):
        callbacks = [Callback()]
        with TemporaryDirectory() as output_root:
            config = parse_overfit(
                _compose(
                    "experiment=overfit/unicodec_ddp_smoke",
                    f"repo_output_root={output_root}",
                )
            )
            output_dir = Path(self.id())

            built = build_trainer(config, output_dir, callbacks)

        self.assertIs(built, trainer.return_value)
        kwargs = trainer.call_args.kwargs
        self.assertEqual(kwargs["devices"], "auto")
        self.assertEqual(kwargs["strategy"], "ddp_find_unused_parameters_true")
        self.assertEqual(kwargs["max_epochs"], -1)
        self.assertEqual(kwargs["precision"], "bf16-mixed")
        self.assertFalse(kwargs["use_distributed_sampler"])
        self.assertEqual(kwargs["gradient_clip_val"], 1.0)
        self.assertTrue(kwargs["enable_checkpointing"])
        self.assertEqual(kwargs["num_sanity_val_steps"], 0)
        self.assertIs(kwargs["logger"], logger.return_value)
        self.assertEqual(kwargs["callbacks"], callbacks)

    def test_public_configs_support_omegaconf_structured(self):
        runtime_config = OmegaConf.structured(Config)
        base_model_config = OmegaConf.structured(ModelConfig)
        model_config = OmegaConf.structured(TokenModelConfig)

        self.assertIsNone(runtime_config.audio_tokenizer)
        self.assertIsNone(runtime_config.device)
        self.assertNotIn("acoustic", base_model_config)
        self.assertEqual(model_config.semantic_audio_adapter, "linear")
        self.assertEqual(model_config.audio_input_adapter.type, "mlp")
        self.assertEqual(model_config.audio_output_adapter.type, "none")

    def test_acoustic_presets_expose_only_supported_options(self):
        flow = _compose()
        rvq = _compose("model/acoustic=rvq")
        none = _compose("model/acoustic=none")

        self.assertEqual(flow.model.acoustic.type, "flow")
        self.assertEqual(flow.model.acoustic.repa.teacher_layer, 9)
        self.assertIn("student_layer", flow.model.acoustic.repa)
        self.assertNotIn("normalize_features", flow.model.acoustic)
        self.assertEqual(flow.runtime.codec, "longcat")
        self.assertEqual(flow.model.semantic_audio_adapter, "linear")
        self.assertEqual(rvq.model.acoustic.type, "rvq")
        self.assertNotIn("repa", rvq.model.acoustic)
        self.assertEqual(none.model.acoustic.type, "none")
        self.assertEqual(none.model.acoustic.name, "token")
        self.assertNotIn("decoder", none.model.acoustic)

    def test_overfit_acoustic_branch_constructs_evaluation_on_py39(self):
        class EvaluationReached(Exception):
            pass

        runtime = SimpleNamespace(
            layout=Mock(),
            codec=SimpleNamespace(acoustic_codebook_sizes=(1024,), frame_rate=50.0),
            acoustic_side_channel=True,
            backbone=Mock(),
            flow_matching=Mock(),
        )
        datamodule = Mock()
        datamodule.train_dataloader.return_value = [Mock()]
        model = Mock()
        with TemporaryDirectory() as output_dir:
            config = parse_overfit(
                _compose(
                    "runtime=longcat_native",
                    f"repo_output_root={output_dir}",
                    "output_subdir=contract-test",
                    "train.max_steps=1",
                    "model.acoustic.decoder.layers=1",
                    "model.acoustic.decoder.heads=1",
                    "model.acoustic.decoder.ffn_ratio=1",
                )
            )
            with (
                patch("scripts.overfit.pl.seed_everything"),
                patch("scripts.overfit.runtime_config", return_value=Mock()),
                patch("scripts.overfit.runtime_for_sequence_layout", return_value=runtime),
                patch("scripts.overfit.DataModule", return_value=datamodule),
                patch(
                    "scripts.overfit.build",
                    return_value=(AcousticType.FLOW, Mock(), model),
                ),
                patch(
                    "scripts.overfit.AcousticEvaluation", side_effect=EvaluationReached
                ),
            ):
                with self.assertRaises(EvaluationReached):
                    run(config)

    @patch("scripts.overfit.torch.cuda.set_device")
    def test_post_fit_generation_uses_runtime_device(self, set_device):
        module = Mock()
        module.parameters.return_value = iter(
            [SimpleNamespace(device=torch.device("cuda", 0))]
        )

        device = _prepare_generation_module(module, torch.device("cuda", 0))

        self.assertEqual(device, torch.device("cuda", 0))
        set_device.assert_called_once_with(torch.device("cuda", 0))
        module.to.assert_called_once_with(torch.device("cuda", 0))

    def test_task_is_the_modality_source_of_truth(self):
        self.assertIs(Task.S2ST.source_modality, Modality.AUDIO)
        self.assertIs(Task.S2ST.target_modality, Modality.AUDIO)
        self.assertTrue(Task.S2ST.uses_source_role)
        self.assertIs(Task.MT.source_modality, Modality.TEXT)
        self.assertIs(Task.MT.target_modality, Modality.TEXT)
        self.assertTrue(Task.MT.uses_source_role)
        self.assertIsNone(Task.AUDIO_AR.source_modality)
        self.assertIs(Task.ASR.target_modality, Modality.TEXT)
        self.assertFalse(Task.TTS.uses_source_role)
        self.assertIsNone(Task.PARALLEL_AR.target_modality)
        self.assertIsNone(Task.INTERLEAVED_AR.target_modality)
        self.assertTrue(Task.PARALLEL_AR.prediction_modality.supervises_text)
        self.assertTrue(Task.PARALLEL_AR.prediction_modality.supervises_audio)

    def test_runtime_separates_audio_id_capabilities(self):
        rt = Runtime(Config())
        rt.__dict__["text_tokenizer"] = _Tokenizer(10)
        rt.__dict__["audio_tokenizer"] = SimpleNamespace(vocab_size=3)

        self.assertEqual(rt.audio_head_range, (10, 16))
        self.assertEqual(rt.codec_audio_range, (10, 13))
        self.assertEqual(rt.audio_generation_allowed_ids, (10, 11, 12, 14))
        self.assertEqual(rt.mask_token_id, 15)
        self.assertNotIn(rt.boa_token_id, rt.audio_generation_allowed_ids)
        self.assertNotIn(rt.mask_token_id, rt.audio_generation_allowed_ids)
        self.assertTrue(rt.is_codec_audio_id(12))
        self.assertFalse(rt.is_codec_audio_id(rt.eoa_token_id))

    def test_unified_codec_uses_semantic_codes_without_acoustic_side_channel(self):
        item = AudioItem(views={AudioView.UNICODEC: torch.tensor([[1], [2], [3]])})
        semantic, acoustic = _parse_audio_item(item, AudioView.UNICODEC)

        self.assertTrue(torch.equal(semantic, torch.tensor([[1], [2], [3]])))
        self.assertIsNone(acoustic)

    def test_stable_codec_uses_full_codes_without_acoustic_side_channel(self):
        codes = torch.tensor([[1], [2], [3]])
        item = AudioItem(views={AudioView.STABLE: codes})

        semantic, acoustic = _parse_audio_item(item, AudioView.STABLE)

        self.assertIs(semantic, codes)
        self.assertIsNone(acoustic)

    def test_bicodec_runtime_rejects_fixed_length_structured_codec(self):
        with self.assertRaisesRegex(ValueError, "semantic_codec_artifact"):
            Runtime(Config(codec="bicodec"))

    def test_parser_rejects_non_codec_audio_views(self):
        item = AudioItem(
            views={AudioView.WAVEFORM: torch.zeros(2, 2)},
            meta={},
        )

        with self.assertRaisesRegex(ValueError, "unsupported codec audio view"):
            _parse_audio_item(item, AudioView.WAVEFORM)

    @patch("speech_to_speech.runtime.backbone.hf.AutoModelForCausalLM.from_pretrained")
    def test_backbone_loading_forwards_runtime_configuration(self, from_pretrained):
        backbone = GradientCheckpointingBackbone()
        from_pretrained.return_value = backbone
        rt = Runtime(
            Config(
                backbone="fake/backbone",
                device="cuda",
                dtype="bfloat16",
                attn_implementation="flash_attention_2",
                gradient_checkpointing=True,
            )
        )

        loaded = rt.backbone

        from_pretrained.assert_called_once_with(
            "fake/backbone",
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self.assertEqual(backbone.gradient_checkpointing_calls, 1)
        self.assertEqual(
            backbone.gradient_checkpointing_kwargs,
            {"gradient_checkpointing_kwargs": {"use_reentrant": False}},
        )
        self.assertEqual(backbone.input_require_grads_calls, 1)
        self.assertFalse(backbone.config.use_cache)
        self.assertEqual(backbone.moves, ["cuda"])
        self.assertIs(loaded, backbone)

    def test_runtime_dtype_is_explicit(self):
        self.assertIs(dtype("float16"), torch.float16)
        with self.assertRaisesRegex(ValueError, "unknown torch dtype"):
            dtype("not_a_dtype")

    def test_overfit_runtime_config_preserves_native_audio_tokenizer(self):
        with TemporaryDirectory() as output_root:
            config = parse_overfit(
                _compose(
                    "runtime=unicodec",
                    "model/acoustic=none",
                    "audio_sequence_layout=flattened",
                    "runtime.backbone=fake/backbone",
                    f"repo_output_root={output_root}",
                )
            )

        with patch.dict("os.environ", {"LOCAL_RANK": "1"}):
            result = runtime_config(config.runtime)

        self.assertEqual(result.codec, "unicodec")
        self.assertIsNone(result.audio_tokenizer)
        self.assertEqual(result.device, "cuda:1")

    @patch("anytrain.framework.flow_matching.ContinuousFlowRuntime")
    @patch("anytrain.framework.flow_matching.ODESampler")
    def test_runtime_forwards_flow_configuration(self, sampler, flow_runtime):
        configured_sampler = Mock()
        sampler.return_value = configured_sampler
        rt = Runtime(Config(flow_method="euler", flow_nfe=7, flow_num_steps=6))

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
        with (
            patch.dict(sys.modules, modules),
            patch.object(TorchCodecBPE, "wrap", return_value=wrapped) as wrap,
        ):
            loaded = audio_tokenizer("~/bpe/longcat/vocab_100k")

        codec_bpe.assert_called_once_with(Path("~/bpe/longcat/vocab_100k").expanduser())
        wrap.assert_called_once_with(tokenizer)
        self.assertIs(loaded, wrapped)


if __name__ == "__main__":
    unittest.main()
