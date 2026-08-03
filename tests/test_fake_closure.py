from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from anydataset.types import AudioView
from anytrain.codec import AcousticLayout
from anytrain.module.idspace import Layout
from speech_to_speech.prediction import PredictionModality
from semantic_acoustic_codec.config import (
    DecoderConfig as SharedDecoderConfig,
    Route,
    RVQPredictor,
)
from semantic_acoustic_codec.model import (
    AcousticRVQDecoder as SharedRVQDecoder,
    FMFeatureGenerator,
    RVQCodeGenerator,
)
from semantic_acoustic_codec.runtime import (
    SamplingConfig,
)
from semantic_acoustic_codec.runtime.artifact import (
    AcousticGeneratorArtifact,
    AcousticGeneratorSpec,
)
from torch import Tensor, nn

from speech_to_speech.datamodule.collate.collator import Collator
from speech_to_speech.datamodule.dataset.speech import ToyDataset
from speech_to_speech.loss.module import FlowObjective
from speech_to_speech.model import ToyConfig
from semantic_acoustic_codec.model.dit import DiTDecoder
from speech_to_speech.model.acoustic.flow import FlowModel
from speech_to_speech.model.acoustic.rvq import RVQModel
from speech_to_speech.model.base import Config as ModelConfig
from speech_to_speech.model.audio_output import (
    AudioOutputAdapterConfig,
    AudioOutputAdapterType,
)
from speech_to_speech.generation import (
    decode_generated_audio,
    decode_generated_codes,
)
from speech_to_speech.runtime import AudioSequenceLayout
from speech_to_speech.runtime.audio_tokenizer import NativeAudioTokenizer
from speech_to_speech.task import Task


class _TextTokenizer:
    _placeholder = "$$$PLACEHOLDER$$$"

    def __len__(self) -> int:
        return 32

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == self._placeholder:
            return [7, 8]
        return [4 + sum(text.encode("utf-8")) % 4, 9]

    def apply_chat_template(self, conversation, **kwargs) -> str | list[int]:
        content = conversation[0]["content"]
        rendered = f"<user>{content}</user><assistant>"
        if not kwargs["tokenize"]:
            return rendered
        if self._placeholder in content:
            return [1, 2, 17, 8, 3]
        return [1, 2, 3]


class _Codec:
    acoustic_feature_dim = 4
    acoustic_codebook_sizes = (16,)
    semantic_feature_dim = 4
    codebook_sizes = (8, 16)
    frame_rate = 50.0
    sample_rate = 16_000

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self.semantic_codebook = torch.randn(8, 4, generator=generator)

    def acoustic_codes_to_features(self, acoustic_codes: Tensor) -> Tensor:
        if acoustic_codes.dim() != 3 or acoustic_codes.size(-1) != 1:
            raise ValueError("fake codec expects [batch, frame, 1] acoustic codes.")
        values = acoustic_codes[..., 0].to(dtype=torch.float64)
        return torch.stack((values, values.square(), values + 1, values * 0.5), dim=-1)

    def encode(self, audio: Tensor, sample_rate: int) -> Tensor:
        del sample_rate
        return audio.new_zeros(audio.shape[0], 1, 2, dtype=torch.long)

    def decode(self, codes: Tensor) -> Tensor:
        return codes[..., 0].float()

    def decode_features(
        self, semantic_codes: Tensor, acoustic_features: Tensor
    ) -> Tensor:
        if semantic_codes.dim() != 3:
            raise ValueError(
                "fake codec expects [batch, frame, codebook] semantic codes."
            )
        if semantic_codes.shape[:2] != acoustic_features.shape[:2]:
            raise ValueError("fake semantic codes and acoustic features must align.")
        semantic = semantic_codes.to(dtype=acoustic_features.dtype).sum(dim=-1)
        return semantic + acoustic_features[..., 0]


class _ModuleCodec(_Codec, nn.Module):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        _Codec.__init__(self)
        self.backend_weight = nn.Parameter(torch.ones(1))


class _FlowRuntime:
    def __init__(self) -> None:
        self.sampled = False

    def training_sample(self, x_1: Tensor, *, x_0: Tensor | None = None):
        del x_0
        x_0 = torch.zeros_like(x_1)
        return SimpleNamespace(
            x_t=x_1 * 0.5,
            velocity=x_1 - x_0,
            t=torch.full((x_1.size(0),), 0.5, device=x_1.device),
        )

    def sample(self, model: nn.Module, x_0: Tensor, **model_extras: object):
        del model, model_extras
        self.sampled = True
        return SimpleNamespace(final=torch.zeros_like(x_0))


class _Teacher:
    feature_dim = 3

    def __call__(
        self,
        semantic_codes: Tensor,
        acoustic_codes: Tensor,
        mask: Tensor,
    ) -> Tensor:
        del semantic_codes, acoustic_codes
        return torch.ones(mask.shape + (self.feature_dim,), device=mask.device)


class _Runtime:
    def __init__(self) -> None:
        self.config = SimpleNamespace(audio_view=AudioView.LONGCAT)
        self.codec_name = "longcat"
        self.audio_view = AudioView.LONGCAT
        self.codec_frame_rate = 50.0
        self.audio_sequence_layout = AudioSequenceLayout.SEMANTIC
        self.semantic_codec_artifact = None
        self.acoustic_layout = AcousticLayout.FRAME_ALIGNED
        self.acoustic_unit_length = None
        self.text_tokenizer = _TextTokenizer()
        self.audio_tokenizer = NativeAudioTokenizer(vocab_size=8)
        self.codec = _Codec()
        self.layout = Layout(text=(0, 32), audio=(32, 43))
        self.flow_matching = _FlowRuntime()
        self.pad_token_id = 0
        self.eos_token_id = 10
        self.boa_token_id = 40
        self.eoa_token_id = 41
        self.mask_token_id = 42

    @property
    def codec_audio_range(self) -> tuple[int, int]:
        return 32, 40


class FakeClosureTest(unittest.TestCase):
    def test_acoustic_model_does_not_own_runtime_codec(self):
        rt = _Runtime()
        rt.codec = _ModuleCodec()

        model = FlowModel(
            _model_config(),
            runtime=rt,
            decoder={"hidden_dim": 4, "layers": 1, "heads": 1, "ffn_ratio": 2},
        )

        self.assertIs(model.acoustic_codec, rt.codec)
        self.assertNotIn("acoustic_codec", model._modules)
        self.assertFalse(
            any(name.startswith("acoustic_codec.") for name, _ in model.named_parameters())
        )

    def test_flow_model_loads_generator_and_adapts_hidden_condition(self):
        rt = _Runtime()
        config = SharedDecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2)
        generator = FMFeatureGenerator(6, 4, config)
        _fill(generator.core.decoder, 0.125)
        artifact = _artifact(Route.FM, config, generator, condition_dim=6)

        model = FlowModel(
            _model_config(),
            runtime=rt,
            decoder={
                "hidden_dim": 4,
                "layers": 1,
                "heads": 1,
                "ffn_ratio": 2,
            },
            initialization=artifact,
        )

        self.assertEqual(model.acoustic_condition.hidden_dim, 4)
        self.assertEqual(model.acoustic_condition.condition_dim, 6)
        self.assertEqual(model.acoustic_decoder.decoder.condition_dim, 6)
        self.assertIsInstance(model.acoustic_decoder, DiTDecoder)
        _assert_state_equal(self, model.acoustic_decoder, generator.core)

    def test_rvq_model_loads_codebook_ar_generator(self):
        rt = _Runtime()
        config = SharedDecoderConfig(
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
            rvq_predictor=RVQPredictor.CODEBOOK_AR,
        )
        generator = RVQCodeGenerator(6, (16,), config)
        self.assertIsInstance(generator.core, SharedRVQDecoder)
        _fill(generator.core, 0.25)
        artifact = _artifact(Route.RVQ, config, generator, condition_dim=6)

        model = RVQModel(
            _model_config(),
            runtime=rt,
            decoder={
                "hidden_dim": 4,
                "layers": 1,
                "heads": 1,
                "ffn_ratio": 2,
            },
            initialization=artifact,
        )

        self.assertEqual(model.acoustic_condition.condition_dim, 6)
        self.assertEqual(model.acoustic_decoder.condition_dim, 6)
        _assert_state_equal(self, model.acoustic_decoder, generator.core)

    def test_rvq_model_rejects_mtp_initialization(self):
        rt = _Runtime()
        config = SharedDecoderConfig(
            hidden_dim=4,
            layers=1,
            heads=1,
            ffn_ratio=2,
            rvq_predictor=RVQPredictor.MTP,
            mtp_layers=1,
            mtp_heads=1,
        )
        generator = RVQCodeGenerator(6, (16,), config)
        artifact = _artifact(Route.RVQ, config, generator, condition_dim=6)

        with self.assertRaisesRegex(ValueError, "codebook_ar"):
            RVQModel(
                _model_config(),
                runtime=rt,
                decoder={
                    "hidden_dim": 4,
                    "layers": 1,
                    "heads": 1,
                    "ffn_ratio": 2,
                },
                initialization=artifact,
            )

    def test_flow_model_uses_runtime_sampler(self):
        rt = _Runtime()
        model = FlowModel(
            _model_config(),
            runtime=rt,
        )
        output = model.sample_acoustic_features(torch.zeros(2, 3, 4))

        self.assertTrue(rt.flow_matching.sampled)
        self.assertEqual(output.shape, (2, 3, 4))

    def test_flow_repa_config_closes_model_and_objective(self):
        rt = _Runtime()
        batch = Collator(rt, {Task.TTS: 1.0})([_dataset(rt)[0]])
        model = FlowModel(
            _model_config(),
            runtime=rt,
            decoder={
                "hidden_dim": 4,
                "layers": 1,
                "heads": 1,
                "ffn_ratio": 2,
            },
            repa={"feature_dim": 3, "student_layer": 1},
        )
        objective = FlowObjective(
            rt.layout,
            rt.flow_matching,
            repa={"weight": 0.1, "teacher": _Teacher()},
        )

        outputs = objective(batch, model)

        self.assertIn("repa", outputs)
        self.assertTrue(torch.isfinite(outputs["loss"]))
        self.assertEqual(model.acoustic_decoder.decoder.feature_layer, 1)
        self.assertIsNotNone(model.acoustic_decoder.decoder.feature_projection)
        self.assertIsInstance(model.acoustic_decoder, DiTDecoder)

    def test_rvq_model_generates_acoustic_features(self):
        torch.manual_seed(0)
        rt = _Runtime()
        model = RVQModel(
            _model_config(),
            runtime=rt,
            decoder={
                "hidden_dim": 4,
                "layers": 1,
                "heads": 1,
                "ffn_ratio": 2,
            },
        ).eval()
        self.assertIsInstance(model.acoustic_decoder, SharedRVQDecoder)

        def audio_logits(hidden_states: Tensor, local_ids=None) -> Tensor:
            self.assertIsNone(local_ids)
            start, end = rt.layout.blocks["audio"]
            logits = hidden_states.new_full(
                (*hidden_states.shape[:-1], end - start),
                float("-inf"),
            )
            logits[..., 0] = 0
            return logits

        with patch.object(model, "semantic_audio_logits", side_effect=audio_logits):
            generation = model.generate_audio_features(
                torch.tensor([[1, 2]]),
                max_new_tokens=2,
                do_sample=False,
                use_cache=False,
            )

        self.assertTrue(
            torch.equal(generation["sequence"], torch.tensor([[1, 2, 32, 32]]))
        )
        self.assertEqual(
            generation["features"].shape,
            (1, 2, rt.codec.acoustic_feature_dim),
        )
        self.assertTrue(torch.equal(generation["frame_counts"], torch.tensor([2])))
        self.assertTrue(torch.isfinite(generation["features"]).all())

    def test_all_tasks_build_expected_model_batches(self):
        rt = _Runtime()
        samples = list(_dataset(rt))
        for task in Task:
            with self.subTest(task=task.value):
                batch = Collator(rt, {task: 1.0})(samples)

                self.assertEqual(batch.tasks, [task, task])
                self.assertEqual(batch.input_ids.shape, batch.token_labels.shape)
                self.assertEqual(
                    batch.acoustic_target is not None,
                    task.prediction_modality.supervises_audio,
                )
                if task.prediction_modality is PredictionModality.AUDIO:
                    supervised = batch.token_labels[0].ne(-100).nonzero().flatten()
                    first = int(supervised[0])
                    last = int(supervised[-1])
                    self.assertEqual(
                        int(batch.input_ids[0, first - 1]), rt.boa_token_id
                    )
                    self.assertEqual(int(batch.input_ids[0, last]), rt.eoa_token_id)

    def test_all_task_paths_forward_backward_and_update_parameters(self):
        for task in Task:
            with self.subTest(task=task.value):
                torch.manual_seed(0)
                rt = _Runtime()
                batch = Collator(rt, {task: 1.0})(list(_dataset(rt)))
                model = FlowModel(
                    _model_config(),
                    runtime=rt,
                )
                loss = FlowObjective(rt.layout, rt.flow_matching)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                before = {
                    name: parameter.detach().clone()
                    for name, parameter in model.named_parameters()
                }

                outputs = loss(batch, model)
                optimizer.zero_grad()
                outputs["loss"].backward()
                optimizer.step()

                self.assertTrue(torch.isfinite(outputs["loss"]))
                self.assertEqual(
                    "flow_matching" in outputs,
                    task.prediction_modality.supervises_audio,
                )
                self.assertTrue(
                    any(
                        not torch.equal(before[name], parameter.detach())
                        for name, parameter in model.named_parameters()
                    )
                )

    def test_fake_semantic_and_acoustic_outputs_decode_to_waveform(self):
        rt = _Runtime()
        batch = Collator(rt, {Task.TTS: 1.0})([_dataset(rt)[0]])
        model = FlowModel(
            _model_config(),
            runtime=rt,
        )
        labels = batch.token_labels[0]
        start, end = rt.codec_audio_range
        semantic = labels[labels.ge(start) & labels.lt(end)][None]
        assert batch.acoustic_target is not None
        features = model.acoustic_target_latent(batch.acoustic_target["codes"])
        self.assertEqual(
            features.dtype,
            model.backbone.get_input_embeddings().weight.dtype,
        )

        waveform = decode_generated_audio(
            semantic,
            features,
            codec=rt.codec,
            audio_tokenizer=rt.audio_tokenizer,
            audio_token_range=rt.codec_audio_range,
        )

        self.assertEqual(waveform.shape, (1, 3))
        self.assertTrue(torch.isfinite(waveform).all())
        decoded_codes = decode_generated_codes(
            semantic,
            batch.acoustic_target["codes"],
            codec=rt.codec,
            audio_tokenizer=rt.audio_tokenizer,
            audio_token_range=rt.codec_audio_range,
        )
        self.assertTrue(torch.equal(decoded_codes, waveform))


def _model_config() -> ModelConfig:
    return ModelConfig(
        semantic_audio_adapter=None,
        audio_output_adapter=AudioOutputAdapterConfig(
            type=AudioOutputAdapterType.NONE
        ),
        toy=ToyConfig(
            hidden_size=4,
            intermediate_size=8,
            layers=1,
            heads=1,
            max_position_embeddings=64,
        ),
    )


def _dataset(runtime: _Runtime) -> ToyDataset:
    return ToyDataset(
        runtime.codec_name,
        runtime.codec,
        samples=2,
        frames=3,
    )


def _artifact(
    route: Route,
    decoder: SharedDecoderConfig,
    generator: FMFeatureGenerator | RVQCodeGenerator,
    *,
    condition_dim: int,
) -> AcousticGeneratorArtifact:
    return AcousticGeneratorArtifact(
        generator=generator,
        spec=AcousticGeneratorSpec(
            route=route,
            condition_dim=condition_dim,
            decoder=decoder,
            backend_name="fake",
            sample_rate=16_000,
            frame_rate=50.0,
            semantic_frame_rate=50.0,
            semantic_vocab_size=8,
            semantic_embedding_dim=4,
            acoustic_feature_dim=4,
            acoustic_codebook_sizes=(16,),
            acoustic_layout=AcousticLayout.FRAME_ALIGNED,
            acoustic_unit_length=None,
            feature_mean=(0.0, 0.0, 0.0, 0.0),
            feature_std=(1.0, 1.0, 1.0, 1.0),
            sampling=SamplingConfig(),
        ),
    )


def _fill(module: nn.Module, value: float) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(value)


def _assert_state_equal(
    test: unittest.TestCase,
    actual: nn.Module,
    expected: nn.Module,
) -> None:
    test.assertEqual(actual.state_dict().keys(), expected.state_dict().keys())
    for key, value in expected.state_dict().items():
        test.assertTrue(torch.equal(actual.state_dict()[key], value), key)


if __name__ == "__main__":
    unittest.main()
