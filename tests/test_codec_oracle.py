from __future__ import annotations

import unittest
from pathlib import Path
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
from scripts.codec_oracle import (
    build_flow,
    build_rvq,
    process_device,
    training_callbacks,
)
from speech_to_speech.codec_oracle import (
    AcousticFlowScreening,
    AcousticRVQScreening,
    DataConfig,
    DataModule,
    Initialization,
    LBAConfig,
    Logger as OracleLogger,
    Objective,
    collate,
    single_batch_loader,
)
from speech_to_speech.loss.types import LossItem
from speech_to_speech.model import AcousticFlow, AcousticRVQDecoder, AdapterType


class CodecOracleTest(unittest.TestCase):
    def test_experiment_precision_matches_bfloat16_runtime(self):
        config = _config()

        self.assertEqual(config.trainer.precision, "bf16-mixed")
        self.assertIsNone(config.runtime.audio_tokenizer)

    @patch("scripts.codec_oracle.torch.cuda.set_device")
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
        )

        callbacks = training_callbacks(config, Callback(), Path(self.id()))

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

    def test_single_batch_loader_keeps_discrete_training_inputs(self):
        codes = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        flow = next(iter(single_batch_loader(codes)))
        self.assertEqual(tuple(flow["codes"].shape), (1, 2, 4))
        self.assertEqual(tuple(flow["mask"].shape), (1, 2))
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

    @patch("scripts.codec_oracle.AcousticFlowScreening")
    @patch("scripts.codec_oracle.SpeechToSpeechFlowModel")
    def test_build_flow_consumes_model_and_oracle_configs(self, model, screening):
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
        built_model_config = model.call_args.args[0]
        self.assertIs(built_model_config.semantic_audio_adapter, AdapterType.MLP)
        self.assertEqual(model.call_args.kwargs["decoder"].layers, 3)
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

    @patch("scripts.codec_oracle.AcousticRVQScreening")
    @patch("scripts.codec_oracle.SpeechToSpeechRVQModel")
    def test_build_rvq_consumes_model_and_oracle_configs(self, model, screening):
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
        )

        self.assertIs(built, screening.return_value)
        built_model_config = model.call_args.args[0]
        self.assertIs(built_model_config.semantic_audio_adapter, AdapterType.MLP)
        self.assertEqual(model.call_args.kwargs["decoder"].layers, 3)
        self.assertEqual(metadata["objective"], "rvq")
        self.assertEqual(metadata["acoustic_codebook_sizes"], [7, 9])

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
            id(parameter) for parameter in module.parameters() if parameter.requires_grad
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
            "codes": torch.tensor(
                [[[1, 3, 4], [2, 5, 6], [-1, -1, -1]]]
            ),
            "mask": torch.tensor([[True, True, False]]),
        }

        output = module.training_step(batch, 0)
        output["loss"].backward()

        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertEqual(
            set(output["rvq"].details or {}),
            {"codebook_0", "codebook_1"},
        )
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
            id(parameter) for parameter in module.parameters() if parameter.requires_grad
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
                side_effect=lambda semantic, features: semantic[..., 0].float()
                + features[..., 0]
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
        self.assertEqual(logger.losses, [1.0])
        strategy.reduce.assert_called_once()
        self.assertEqual(strategy.reduce.call_args.kwargs, {"reduce_op": "mean"})
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

    def acoustic_code_features(self, codes):
        values = codes.float().sum(dim=-1, keepdim=True)
        return values.expand(*values.shape[:-1], 4)


class _RVQLoggerModule:
    device = torch.device("cpu")

    def sample(self, semantic_codes, *, seed):
        del semantic_codes, seed
        return torch.tensor([[[2, 3], [4, 0]]])

    def features(self, acoustic_codes):
        return acoustic_codes.float().sum(dim=-1, keepdim=True)


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
