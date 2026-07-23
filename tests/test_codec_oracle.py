from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from anydataset.types import AudioItem, AudioView, Modality, Role
from anytrain.idspace import Layout
from hydra import compose, initialize_config_dir
from lightning.fabric.utilities.data import _replace_dunder_methods
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.utilities.data import _update_dataloader
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler

from scripts._config import CodecOracleConfig, codec_oracle
from scripts.codec_oracle import training_callbacks
from speech_to_speech.codec_oracle import (
    AcousticFlowModel,
    AcousticFlowScreening,
    AcousticRVQModel,
    AcousticRVQScreening,
    DataConfig,
    DataModule,
    Initialization,
    LBAConfig,
    Logger as OracleLogger,
    Objective,
    collate,
    codes,
    single_batch_loader,
    training_item,
)
from speech_to_speech.codec_oracle.factory import (
    build_flow,
    build_rvq,
    process_device,
)
from speech_to_speech.loss.types import LossItem
from speech_to_speech.model import (
    AcousticFlow,
    AcousticRVQDecoder,
    AdapterType,
    DecoderConfig,
)
from speech_to_speech.runtime.audio_tokenizer import NativeAudioTokenizer


class CodecOracleTest(unittest.TestCase):
    def test_experiment_precision_matches_bfloat16_runtime(self):
        config = _config()

        self.assertEqual(config.trainer.precision, "bf16-mixed")
        self.assertIsNone(config.runtime.audio_tokenizer)

    def test_codec_oracle_allows_explicit_bpe_tokenizer(self):
        config = _config("runtime.audio_tokenizer=/tmp/longcat-bpe")

        self.assertEqual(config.runtime.audio_tokenizer, "/tmp/longcat-bpe")

    @patch("speech_to_speech.codec_oracle.factory.torch.cuda.set_device")
    def test_process_device_preserves_explicit_index(self, set_device):
        with patch.dict("os.environ", {"LOCAL_RANK": "0"}):
            explicit = process_device("cuda:3")
        with patch.dict("os.environ", {"LOCAL_RANK": "2"}):
            local = process_device("cuda")

        self.assertEqual(explicit, torch.device("cuda:3"))
        self.assertEqual(local, torch.device("cuda:2"))
        self.assertEqual(
            [call.args[0] for call in set_device.call_args_list],
            [torch.device("cuda:3"), torch.device("cuda:2")],
        )

    def test_training_callbacks_archive_all_periodic_checkpoints(self):
        config = _config(
            "trainer.enable_checkpointing=false",
            "callbacks.grad_norm.enabled=false",
            "callbacks.nonfinite.enabled=false",
            "callbacks.performance.enabled=false",
        )
        oracle = Callback()

        callbacks = training_callbacks(config, oracle, Path(self.id()))

        self.assertEqual(callbacks, [oracle])
        self.assertFalse(any(isinstance(x, ModelCheckpoint) for x in callbacks))

        config = _config(
            "trainer.enable_checkpointing=true",
            "callbacks.grad_norm.enabled=false",
            "callbacks.nonfinite.enabled=false",
        )
        with patch("scripts.codec_oracle.ModelCheckpoint") as checkpoint:
            callbacks = training_callbacks(config, Callback(), Path(self.id()))

        checkpoint.assert_called_once()
        kwargs = checkpoint.call_args.kwargs
        self.assertEqual(kwargs["every_n_train_steps"], 10_000)
        self.assertEqual(kwargs["save_top_k"], -1)
        self.assertTrue(kwargs["save_last"])
        self.assertFalse(kwargs["auto_insert_metric_name"])
        self.assertIn(checkpoint.return_value, callbacks)

    @patch("scripts.codec_oracle.TrainingFlops")
    @patch("scripts.codec_oracle.PerformanceCallback")
    def test_training_callbacks_configure_dynamic_mfu(
        self,
        performance,
        training_flops,
    ):
        config = _config("callbacks.performance.hardware_peak_flops=123.0")
        oracle = Callback()

        callbacks = training_callbacks(config, oracle, Path(self.id()))

        performance.assert_called_once_with(
            model_flops_per_batch=training_flops.return_value,
            hardware_peak_flops=123.0,
            log_every_n_steps=100,
            warmup_steps=20,
            measure_window_steps=100,
            sync_cuda=True,
            sync_distributed=True,
        )
        self.assertIs(callbacks[0], performance.return_value)
        self.assertIs(callbacks[1], oracle)

    def test_lba_loader_supports_lightning_sampler_injection(self):
        data = DataConfig(
            batch_size=3,
            num_workers=0,
            pin_memory=True,
            persistent_workers=True,
            lba=LBAConfig(enabled=True, max_batch_seconds=6.0),
        )
        datamodule = DataModule(
            data,
            "longcat",
            frame_rate=2.0,
            output_dir=Path(self.id()),
        )
        dataset = [_sample(2)]
        datamodule.dataset = dataset

        with _replace_dunder_methods(DataLoader, "dataset"):
            loader = datamodule.train_dataloader()
        sampler = DistributedSampler(
            dataset,
            num_replicas=2,
            rank=0,
            shuffle=False,
        )
        updated = _update_dataloader(loader, sampler)

        self.assertEqual(type(loader).__name__, "LBA")
        self.assertIs(loader.dataset, dataset)
        self.assertIsInstance(loader.sampler, RandomSampler)
        self.assertEqual(loader.batch_size, 3)
        self.assertEqual(loader.num_workers, 0)
        self.assertTrue(loader.pin_memory)
        self.assertFalse(loader.persistent_workers)
        self.assertEqual(loader.max_padded_length, 12)
        self.assertTrue(callable(loader.collate_fn))
        self.assertTrue(callable(loader.len_fn))
        self.assertIs(updated.sampler, sampler)
        self.assertEqual(updated.max_padded_length, 12)
        self.assertTrue(callable(updated.len_fn))

    def test_collate_pads_variable_length_codec_sequences(self):
        data = _config("codec_oracle.data.max_seconds=2.0").codec_oracle.data
        batch = collate(
            [_sample(3), _sample(1)],
            codec="longcat",
            data=data,
            frame_rate=2.0,
        )

        self.assertEqual(tuple(batch["codes"].shape), (2, 3, 4))
        self.assertTrue(
            torch.equal(batch["mask"], torch.tensor([[1, 1, 1], [1, 0, 0]]).bool())
        )
        self.assertTrue((batch["codes"][1, 1:] == -1).all())
        self.assertTrue(
            torch.equal(
                batch["semantic_tokens"],
                torch.tensor([[0, 4, 8], [0, 0, 0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch["semantic_token_spans"],
                torch.tensor([[1, 1, 1], [1, 0, 0]]),
            )
        )

    def test_collate_encodes_bpe_tokens_and_spans(self):
        batch = collate(
            [_sample(3)],
            codec="longcat",
            data=DataConfig(),
            frame_rate=2.0,
            audio_tokenizer=_BpeTokenizer(),
        )

        expected = _sample(3)[(Role.TARGET, Modality.AUDIO)].views[AudioView.LONGCAT]
        self.assertTrue(torch.equal(batch["codes"], expected.unsqueeze(0)))
        self.assertTrue(
            torch.equal(batch["semantic_tokens"], torch.tensor([[0, 1]]))
        )
        self.assertTrue(
            torch.equal(batch["semantic_token_spans"], torch.tensor([[2, 1]]))
        )

    def test_training_item_rejects_misaligned_bpe_spans(self):
        with self.assertRaisesRegex(ValueError, "cover every oracle frame"):
            training_item(
                torch.tensor([[0, 1], [1, 2], [2, 3]]),
                audio_tokenizer=_BrokenBpeTokenizer(),
            )

    def test_default_data_keeps_the_full_prepared_sequence(self):
        value = codes(
            _sample(5),
            codec="longcat",
            data=DataConfig(),
            frame_rate=2.0,
        )

        self.assertEqual(tuple(value.shape), (5, 4))

    def test_lba_budget_is_a_hard_sample_limit(self):
        data = DataConfig(
            max_seconds=None,
            overlong="error",
            lba=LBAConfig(enabled=True, max_batch_seconds=2.0),
        )
        with self.assertRaisesRegex(ValueError, "hard limit"):
            codes(_sample(5), codec="longcat", data=data, frame_rate=2.0)

        truncated = codes(
            _sample(5),
            codec="longcat",
            data=DataConfig(
                max_seconds=None,
                overlong="truncate",
                lba=LBAConfig(enabled=True, max_batch_seconds=2.0),
            ),
            frame_rate=2.0,
        )
        self.assertEqual(tuple(truncated.shape), (4, 4))

    @patch("speech_to_speech.codec_oracle.data.wmt19_tts_codec")
    def test_duration_filter_removes_overlong_samples(self, dataset):
        dataset.return_value = [_sample(2), _sample(5), _sample(3)]
        module = DataModule(
            DataConfig(
                overlong="filter",
                num_workers=0,
                lba=LBAConfig(enabled=True, max_batch_seconds=2.0),
            ),
            "longcat",
            frame_rate=2.0,
            output_dir=Path(self.id()),
        )

        with self.assertWarnsRegex(UserWarning, "filtered 1"):
            module.setup()

        self.assertEqual(len(module.dataset), 2)
        self.assertEqual(module.filtered_samples, 1)

    def test_data_rejects_invalid_duration_limits(self):
        for value in (0.0, -1.0, float("nan")):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "max_seconds"),
            ):
                DataConfig(max_seconds=value)

        with self.assertRaisesRegex(TypeError, "max_seconds"):
            DataConfig(max_seconds=True)

    def test_single_batch_loader_keeps_discrete_training_inputs(self):
        codes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        flow = next(iter(single_batch_loader(codes)))
        self.assertEqual(tuple(flow["codes"].shape), (1, 2, 4))
        self.assertEqual(tuple(flow["mask"].shape), (1, 2))
        self.assertEqual(tuple(flow["semantic_tokens"].shape), (1, 2))
        self.assertEqual(tuple(flow["semantic_token_spans"].shape), (1, 2))
        self.assertTrue(flow["mask"].all())
        self.assertFalse(flow["codes"].is_floating_point())

    def test_random_embedding_is_deterministic_without_changing_global_rng(self):
        codebook = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        torch.manual_seed(17)
        state = torch.random.get_rng_state()

        first = Initialization.RANDOM.weight(codebook, seed=3)
        after = torch.random.get_rng_state()
        second = Initialization.RANDOM.weight(codebook, seed=3)

        self.assertTrue(torch.equal(state, after))
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(torch.equal(first, codebook))

    @patch("speech_to_speech.codec_oracle.factory.condition_dim", return_value=4)
    @patch("speech_to_speech.codec_oracle.factory.AcousticFlowScreening")
    @patch("speech_to_speech.codec_oracle.factory.AcousticFlowModel")
    def test_build_flow_consumes_model_and_oracle_configs(
        self,
        model,
        screening,
        condition_dim,
    ):
        config = _config(
            "model.semantic_audio_adapter=mlp",
            "codec_oracle.decoder.layers=3",
            "codec_oracle.normalize_features=false",
        )
        target = torch.tensor([[[2.0], [4.0]]])
        codec = SimpleNamespace(
            acoustic_codes_to_features=Mock(return_value=target),
            semantic_codebook=torch.arange(8, dtype=torch.float32).reshape(4, 2),
            frame_rate=25.0,
        )
        runtime = SimpleNamespace(codec=codec, flow_matching=Mock())
        codes = torch.tensor([[1, 2], [3, 4]])

        built, _ = build_flow(
            config,
            codes,
            Initialization.CODEC,
            runtime,
            torch.device("cpu"),
        )

        self.assertIs(built, screening.return_value)
        condition_dim.assert_called_once_with(config)
        model.assert_called_once_with(
            adapter=AdapterType.MLP,
            runtime=runtime,
            condition_dim=4,
            flow_runtime=runtime.flow_matching,
            decoder=config.codec_oracle.decoder,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        self.assertTrue(
            torch.equal(
                screening.call_args.kwargs["target_mean"],
                torch.zeros(1, 1, 1),
            )
        )
        self.assertTrue(
            torch.equal(
                screening.call_args.kwargs["target_std"],
                torch.ones(1, 1, 1),
            )
        )

    @patch("speech_to_speech.codec_oracle.factory.condition_dim", return_value=4)
    @patch("speech_to_speech.codec_oracle.factory.AcousticRVQScreening")
    @patch("speech_to_speech.codec_oracle.factory.AcousticRVQModel")
    def test_build_rvq_consumes_model_and_oracle_configs(
        self,
        model,
        screening,
        condition_dim,
    ):
        config = _config(
            "codec_oracle=rvq",
            "model.semantic_audio_adapter=mlp",
            "codec_oracle.decoder.layers=3",
        )
        codec = SimpleNamespace(
            acoustic_codebook_sizes=(7, 9),
            semantic_codebook=torch.arange(8, dtype=torch.float32).reshape(4, 2),
            frame_rate=25.0,
        )
        runtime = SimpleNamespace(codec=codec)
        codes = torch.tensor([[1, 2, 3], [2, 3, 4]])

        built, metadata = build_rvq(
            config,
            codes,
            Initialization.CODEC,
            runtime,
            torch.device("cpu"),
        )

        self.assertIs(built, screening.return_value)
        condition_dim.assert_called_once_with(config)
        model.assert_called_once_with(
            adapter=AdapterType.MLP,
            runtime=runtime,
            condition_dim=4,
            decoder=config.codec_oracle.decoder,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
        self.assertEqual(metadata["objective"], "rvq")
        self.assertEqual(metadata["acoustic_codebook_sizes"], [7, 9])

    def test_lightweight_oracle_models_do_not_register_a_backbone(self):
        codec = SimpleNamespace(
            semantic_codebook=torch.arange(32, dtype=torch.float32).reshape(8, 4),
            acoustic_codebook_sizes=(7, 9),
            acoustic_feature_dim=4,
        )
        runtime = SimpleNamespace(
            codec=codec,
            audio_tokenizer=NativeAudioTokenizer(vocab_size=8),
        )
        decoder = DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2)

        flow = AcousticFlowModel(
            adapter=AdapterType.LINEAR,
            runtime=runtime,
            condition_dim=4,
            flow_runtime=_Flow(),
            decoder=decoder,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        rvq = AcousticRVQModel(
            adapter=AdapterType.LINEAR,
            runtime=runtime,
            condition_dim=4,
            decoder=decoder,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        for model in (flow, rvq):
            with self.subTest(model=type(model).__name__):
                keys = tuple(model.state_dict())
                self.assertFalse(any("backbone" in key for key in keys))
                self.assertFalse(hasattr(model, "backbone"))
                self.assertEqual(
                    tuple(model.semantic_condition(torch.tensor([[1, 2]])).shape),
                    (1, 2, 4),
                )

    def test_bpe_condition_repeats_token_embeddings_by_span(self):
        codec = SimpleNamespace(
            semantic_codebook=torch.eye(4),
            acoustic_codebook_sizes=(7,),
            acoustic_feature_dim=4,
        )
        model = AcousticFlowModel(
            adapter=None,
            runtime=SimpleNamespace(codec=codec, audio_tokenizer=_BpeTokenizer()),
            condition_dim=4,
            flow_runtime=_Flow(),
            decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        with torch.no_grad():
            model.semantic_audio_embedding.weight.copy_(
                torch.arange(12, dtype=torch.float32).reshape(3, 4)
            )

        condition = model.semantic_condition(
            torch.tensor([[0, 1, 0]]),
            torch.tensor([[2, 1, 0]]),
            frames=4,
        )

        self.assertTrue(
            torch.equal(
                condition,
                torch.tensor(
                    [
                        [
                            [0.0, 1.0, 2.0, 3.0],
                            [0.0, 1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0, 7.0],
                            [0.0, 0.0, 0.0, 0.0],
                        ]
                    ]
                ),
            )
        )

    def test_acoustic_flow_screening_uses_formal_model_target_latent(self):
        calls: list[torch.Tensor] = []

        def dequantize(codes: torch.Tensor) -> torch.Tensor:
            calls.append(codes.clone())
            return codes.sum(dim=-1, keepdim=True).float()

        model = _OracleModel(dequantize)
        module = AcousticFlowScreening(
            model,
            initialization=Initialization.CODEC,
            seed=0,
            flow_runtime=_Flow(),
            learning_rate=1e-3,
            weight_decay=0.0,
            target_mean=torch.zeros(1, 1, 1),
            target_std=torch.ones(1, 1, 1),
        )
        batch = {
            "codes": torch.tensor([[[1, 3, 4], [2, 5, 6]]]),
            "mask": torch.ones((1, 2), dtype=torch.bool),
        }

        output = module.training_step(batch, 0)
        output["loss"].backward()

        self.assertIs(module.model, model)
        parameter_names = dict(module.named_parameters())
        self.assertIn("model.acoustic_flow.decoder.input.weight", parameter_names)
        self.assertNotIn("decoder.input.weight", parameter_names)
        self.assertEqual(len(calls), 1)
        self.assertTrue(torch.equal(calls[0], batch["codes"][..., 1:]))
        self.assertIsNotNone(module.model.semantic_audio_embedding.weight.grad)
        self.assertFalse(module.model.unused.weight.requires_grad)
        optimized = {
            id(parameter)
            for group in module.configure_optimizers().param_groups
            for parameter in group["params"]
        }
        trainable = {
            id(parameter)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(optimized, trainable)

    def test_acoustic_flow_screening_replaces_padding_before_target_latent(self):
        calls: list[torch.Tensor] = []

        def dequantize(codes: torch.Tensor) -> torch.Tensor:
            calls.append(codes.clone())
            return codes.sum(dim=-1, keepdim=True).float()

        module = AcousticFlowScreening(
            _OracleModel(dequantize),
            initialization=Initialization.CODEC,
            seed=0,
            flow_runtime=_Flow(),
            learning_rate=1e-3,
            weight_decay=0.0,
            target_mean=torch.zeros(1, 1, 1),
            target_std=torch.ones(1, 1, 1),
        )
        batch = {
            "codes": torch.tensor([[[1, 3, 4], [-1, -1, -1]]]),
            "mask": torch.tensor([[True, False]]),
        }

        output = module.training_step(batch, 0)

        self.assertTrue(torch.equal(calls[0][0, 1], torch.tensor([0, 0])))
        self.assertTrue(torch.isfinite(output["loss"]))

    def test_acoustic_rvq_screening_closes_discrete_training_and_sampling(self):
        model = _RVQOracleModel()
        module = AcousticRVQScreening(
            model,
            initialization=Initialization.CODEC,
            seed=0,
            learning_rate=1e-3,
            weight_decay=0.0,
        )
        batch = {
            "codes": torch.tensor([[[1, 3, 4], [2, 5, 6], [-1, -1, -1]]]),
            "mask": torch.tensor([[True, True, False]]),
        }

        output = module.training_step(batch, 0)
        output["loss"].backward()

        self.assertTrue(torch.isfinite(output["loss"]))
        details = output["rvq"].details or {}
        self.assertEqual(set(details), {"codebook_0", "codebook_1", "frames"})
        self.assertTrue(torch.equal(details["frames"], torch.tensor([2.0])))
        self.assertFalse(
            module.model.acoustic_decoder.decoder.embed_tokens.weight.requires_grad
        )
        self.assertFalse(
            module.model.acoustic_decoder.codebook_embeddings[-1].weight.requires_grad
        )
        self.assertFalse(module.model.unused.weight.requires_grad)
        self.assertTrue(
            all(
                parameter.grad is not None
                for parameter in module.parameters()
                if parameter.requires_grad
            )
        )
        optimizer = module.configure_optimizers()
        optimized = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertTrue(
            all(
                parameter.requires_grad
                for group in optimizer.param_groups
                for parameter in group["params"]
            )
        )
        trainable = {
            id(parameter)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        self.assertEqual(optimized, trainable)

        sampled = module.sample(torch.tensor([[1, 2]]), seed=3)
        self.assertEqual(tuple(sampled.shape), (1, 2, 2))
        self.assertTrue((sampled[..., 0] < 7).all())
        self.assertTrue((sampled[..., 1] < 9).all())

    def test_rvq_oracle_logger_records_codebook_metrics_and_waveform(self):
        experiment = Mock()
        strategy = Mock()
        strategy.reduce.side_effect = lambda value, *, reduce_op: value
        trainer = SimpleNamespace(
            global_step=1,
            is_global_zero=True,
            logger=SimpleNamespace(experiment=experiment),
            strategy=strategy,
        )
        codec = SimpleNamespace(
            decode_features=Mock(
                side_effect=lambda semantic, features: (
                    semantic[..., 0].float() + features[..., 0]
                )
            )
        )
        logger = OracleLogger(
            objective=Objective.RVQ,
            codec=codec,
            codes=torch.tensor([[1, 2, 3], [2, 4, 5]]),
            output_dir=Path(self.id()),
            sample_rate=16_000,
            seed=0,
            sample_every_n_steps=1,
            histogram_every_n_steps=1,
            save_audio=False,
            metadata={},
        )
        module = _RVQLoggerModule()
        item = LossItem(
            loss=torch.tensor([1.0]),
            details={
                "codebook_0": torch.tensor([0.5]),
                "codebook_1": torch.tensor([1.5]),
            },
        )

        logger.on_train_batch_end(
            trainer,
            module,
            {"loss": torch.tensor(1.0), "rvq": item},
            None,
            0,
        )

        self.assertEqual(logger.samples[0]["step"], 1)
        self.assertEqual(logger._losses.count, 1)
        strategy.reduce.assert_not_called()
        self.assertIn("code_accuracy", logger.samples[0])
        self.assertIn("codebook_0_accuracy", logger.samples[0])
        scalar_tags = [call.args[0] for call in experiment.add_scalar.call_args_list]
        self.assertIn("oracle/sample_code_accuracy", scalar_tags)
        self.assertIn("oracle/sample_feature_mse", scalar_tags)
        histogram_tags = [
            call.args[0] for call in experiment.add_histogram.call_args_list
        ]
        self.assertEqual(
            histogram_tags,
            ["rvq/codebook_0_loss", "rvq/codebook_1_loss"],
        )
        experiment.add_audio.assert_called_once()
        codec.decode_features.assert_called_once()

    def test_oracle_loss_window_reduces_once_at_train_end(self):
        strategy = Mock()
        strategy.reduce.side_effect = lambda value, *, reduce_op: value
        trainer = SimpleNamespace(
            global_step=3,
            is_global_zero=True,
            strategy=strategy,
            world_size=1,
        )
        module = SimpleNamespace(device=torch.device("cpu"))
        with TemporaryDirectory() as directory:
            logger = OracleLogger(
                objective=Objective.FLOW,
                codec=Mock(),
                codes=torch.ones((1, 2), dtype=torch.long),
                output_dir=Path(directory),
                sample_rate=16_000,
                seed=0,
                sample_every_n_steps=100,
                histogram_every_n_steps=100,
                save_audio=False,
                metadata={},
            )
            for value in (1.0, 2.0, 4.0):
                logger.on_train_batch_end(
                    trainer,
                    module,
                    {"loss": torch.tensor(value)},
                    None,
                    0,
                )
            with patch.object(logger, "_sample"):
                logger.on_train_end(trainer, module)

            report = json.loads((Path(directory) / "metrics.json").read_text())

        strategy.reduce.assert_called_once()
        self.assertEqual(strategy.reduce.call_args.kwargs, {"reduce_op": "sum"})
        self.assertEqual(report["steps"], 3)
        self.assertEqual(report["loss"]["first"], 1.0)
        self.assertEqual(report["loss"]["last"], 4.0)


class _Flow:
    def training_sample(self, target: torch.Tensor, *, x_0=None):
        del x_0
        return SimpleNamespace(
            x_t=torch.zeros_like(target),
            velocity=torch.ones_like(target),
            t=torch.zeros(target.size(0)),
        )

    def sample(self, model, noise, **kwargs):
        del model, kwargs
        return SimpleNamespace(final=torch.zeros_like(noise))


class _OracleModel(nn.Module):
    def __init__(self, dequantize) -> None:
        super().__init__()
        self.layout = Layout(text=(0, 4), audio=(4, 14))
        self.semantic_audio_embedding = nn.Embedding.from_pretrained(
            torch.randn(8, 4),
            freeze=False,
        )
        self.semantic_audio_adapter = nn.Identity()
        self.acoustic_flow = AcousticFlow(4, 1, _Flow())
        self.unused = nn.Linear(4, 4)
        self.dequantize = dequantize

    @property
    def acoustic_decoder(self):
        return self.acoustic_flow.decoder

    def target_frame_label_condition(self, token_labels, positions):
        return self.semantic_audio_embedding(token_labels - 4)

    def semantic_condition(self, semantic_codes, spans=None, *, frames=None):
        del spans, frames
        return self.semantic_audio_adapter(
            self.semantic_audio_embedding(semantic_codes)
        )

    def acoustic_target_latent(self, labels):
        return self.dequantize(labels)


class _RVQOracleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layout = Layout(text=(0, 4), audio=(4, 14))
        self.semantic_audio_embedding = nn.Embedding.from_pretrained(
            torch.randn(8, 4),
            freeze=False,
        )
        self.semantic_audio_adapter = nn.Identity()
        self.acoustic_decoder = AcousticRVQDecoder(
            4,
            2,
            (7, 9),
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
        )
        self.unused = nn.Linear(4, 4)

    def target_frame_label_condition(self, token_labels, positions):
        del positions
        return self.semantic_audio_embedding(token_labels - 4)

    def semantic_condition(self, semantic_codes, spans=None, *, frames=None):
        del spans, frames
        return self.semantic_audio_adapter(
            self.semantic_audio_embedding(semantic_codes)
        )

    def acoustic_code_features(self, codes):
        values = codes.float().sum(dim=-1, keepdim=True)
        return values.expand(*values.shape[:-1], 4)


class _RVQLoggerModule:
    device = torch.device("cpu")

    def sample(self, semantic_codes, *, seed, spans=None, frames=None):
        del semantic_codes, seed, spans, frames
        return torch.tensor([[[2, 3], [4, 0]]])

    def features(self, acoustic_codes):
        return acoustic_codes.float().sum(dim=-1, keepdim=True)


class _BpeTokenizer:
    vocab_size = 3

    def encode(self, frames):
        del frames
        return torch.tensor([0, 1], dtype=torch.long)

    def decode(self, token_ids):
        values = (
            token_ids.detach().cpu().tolist()
            if isinstance(token_ids, torch.Tensor)
            else token_ids
        )
        spans = self.frame_spans(token_ids)
        span_values = spans.detach().cpu().tolist() if isinstance(spans, torch.Tensor) else spans
        return [
            (int(value) % 4,)
            for value, span in zip(values, span_values)
            for _ in range(int(span))
        ]

    def frame_spans(self, token_ids):
        size = token_ids.numel() if isinstance(token_ids, torch.Tensor) else len(token_ids)
        spans = [2, 1, 1][:size]
        if isinstance(token_ids, torch.Tensor):
            return torch.tensor(spans, dtype=torch.long, device=token_ids.device)
        return spans


class _BrokenBpeTokenizer(_BpeTokenizer):
    def frame_spans(self, token_ids):
        size = token_ids.numel() if isinstance(token_ids, torch.Tensor) else len(token_ids)
        spans = [1] * size
        if isinstance(token_ids, torch.Tensor):
            return torch.tensor(spans, dtype=torch.long, device=token_ids.device)
        return spans


def _sample(frames: int):
    return {
        (Role.TARGET, Modality.AUDIO): AudioItem(
            views={AudioView.LONGCAT: torch.arange(frames * 4).reshape(frames, 4)}
        )
    }


def _config(*overrides: str) -> CodecOracleConfig:
    root = Path(__file__).parents[1]
    with initialize_config_dir(version_base=None, config_dir=str(root / "configs")):
        return codec_oracle(
            compose(config_name="codec_oracle", overrides=list(overrides))
        )


if __name__ == "__main__":
    unittest.main()
