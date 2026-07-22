from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any, cast

import torch
from anytrain.idspace import Layout
from lightning import pytorch as pl
from lightning.fabric.utilities.throughput import measure_flops
from torch import Tensor, nn

from speech_to_speech._flops import (
    adapter,
    flow_decoder,
    linear,
    qwen_backbone,
    rvq_decoder,
)
from speech_to_speech.datamodule.types import AcousticPrompt, AcousticTarget, ModelBatch
from speech_to_speech.loss import FlowObjective, LossItem, RVQObjective, TokenObjective
from speech_to_speech.model import (
    AdapterType,
    Config as ModelConfig,
    DecoderConfig,
    FlowModel,
    RVQModel,
    TokenModel,
    ToyConfig,
)
from speech_to_speech.performance import TrainingFlops
from speech_to_speech.pl_module import Config as ModuleConfig, SpeechToSpeechModule
from speech_to_speech.task import Task


class TrainingFlopsTest(unittest.TestCase):
    def test_linear_formula_matches_lightning_meta_measurement(self):
        with torch.device("meta"):
            module = nn.Linear(3, 5, bias=False)
            inputs = torch.randn(7, 3)
        measured = measure_flops(module, lambda: module(inputs))
        self.assertEqual(measured, linear(module, 7))

    def test_token_path_uses_audio_rows_padding_lengths_and_valid_labels(self):
        module = _module(_token_model(), TokenObjective(_layout()))
        batch = _batch(
            input_ids=torch.tensor([[1, 7, 8, 2], [1, 7, 0, 0]]),
            labels=torch.tensor([[-100, 7, 8, -100], [-100, 9, -100, -100]]),
            tasks=[Task.TTS, Task.TTS],
        )
        expected = _token_expected(cast(TokenModel, module.model), batch)
        self.assertEqual(_flops(module, batch), expected)

        # Changing a padded token changes the dense attention work, while the
        # padded B*S projection shape stays fixed.
        longer = _batch(
            input_ids=torch.tensor([[1, 7, 8, 2], [1, 7, 2, 0]]),
            labels=batch.token_labels,
            tasks=[Task.TTS, Task.TTS],
        )
        self.assertGreater(_flops(module, longer), _flops(module, batch))

        # The semantic input adapter follows actual audio IDs, not B*S.
        text_id = _batch(
            input_ids=torch.tensor([[1, 2, 8, 2], [1, 2, 0, 0]]),
            labels=batch.token_labels,
            tasks=[Task.TTS, Task.TTS],
        )
        self.assertGreater(_flops(module, batch), _flops(module, text_id))

        # The token head follows valid shifted labels only.
        fewer_labels = _batch(
            input_ids=batch.input_ids,
            labels=torch.tensor([[-100, 7, -100, -100], [-100, 9, -100, -100]]),
            tasks=[Task.TTS, Task.TTS],
        )
        self.assertGreater(_flops(module, batch), _flops(module, fewer_labels))

    def test_acoustic_prompt_adapter_uses_padded_token_shape(self):
        model = _token_model()
        module = _module(model, TokenObjective(_layout()))
        base = _batch(
            input_ids=torch.tensor([[1, 7, 8, 2], [1, 7, 0, 0]]),
            labels=torch.tensor([[-100, 7, 8, -100], [-100, 9, -100, -100]]),
            tasks=[Task.S2ST, Task.S2ST],
        )
        prompt = _prompt()
        with_prompt = _batch(
            input_ids=base.input_ids,
            labels=base.token_labels,
            tasks=base.tasks,
            acoustic_prompt=prompt,
        )
        difference = _flops(module, with_prompt) - _flops(module, base)
        expected = 3 * adapter(
            model.acoustic_prompt_adapter,
            rows=8,
            in_features=4,
            out_features=4,
            name="test prompt",
        )
        self.assertEqual(difference, expected)

    def test_text_head_counts_only_the_layout_slice(self):
        model = _token_model()
        model.backbone.lm_head = nn.Linear(4, 9, bias=False)
        module = _module(model, TokenObjective(_layout()))
        batch = _batch(
            input_ids=torch.tensor([[1, 2, 3, 4], [1, 2, 0, 0]]),
            labels=torch.tensor([[-100, 2, 3, -100], [-100, 4, -100, -100]]),
            tasks=[Task.ASR, Task.ASR],
        )

        expected = _token_expected(model, batch)

        self.assertEqual(_flops(module, batch), expected)

    def test_flow_uses_padded_target_frames(self):
        model = _flow_model()
        module = _module(model, FlowObjective(_layout(), _flow_runtime()))
        sparse = _batch(
            input_ids=torch.tensor([[1, 7, 8, 2], [1, 7, 0, 0]]),
            labels=torch.tensor([[-100, 7, 8, -100], [-100, 9, -100, -100]]),
            tasks=[Task.TTS, Task.TTS],
            acoustic_target=_target(
                torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
            ),
        )
        dense = _batch(
            input_ids=sparse.input_ids,
            labels=sparse.token_labels,
            tasks=sparse.tasks,
            acoustic_target=_target(torch.ones((2, 3), dtype=torch.bool)),
        )
        expected = _token_expected(model, sparse) + 3 * flow_decoder(
            model.acoustic_decoder,
            batch=2,
            frames=3,
        )
        self.assertEqual(_flops(module, sparse), expected)
        self.assertEqual(_flops(module, dense), _flops(module, sparse))

    def test_rvq_uses_valid_target_frames(self):
        model = _rvq_model()
        module = _module(model, RVQObjective(_layout()))
        sparse_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
        dense_mask = torch.ones((2, 3), dtype=torch.bool)
        sparse = _batch(
            input_ids=torch.tensor([[1, 7, 8, 2], [1, 7, 0, 0]]),
            labels=torch.tensor([[-100, 7, 8, -100], [-100, 9, -100, -100]]),
            tasks=[Task.TTS, Task.TTS],
            acoustic_target=_target(sparse_mask),
        )
        dense = _batch(
            input_ids=sparse.input_ids,
            labels=sparse.token_labels,
            tasks=sparse.tasks,
            acoustic_target=_target(dense_mask),
        )
        expected = _token_expected(model, sparse) + 3 * rvq_decoder(
            model.acoustic_decoder,
            valid_frames=int(sparse_mask.sum().item()),
        )
        self.assertEqual(_flops(module, sparse), expected)
        self.assertGreater(_flops(module, dense), _flops(module, sparse))

    def test_rejects_unsupported_paths_and_mismatched_outputs(self):
        model = _token_model()
        module = _module(model, TokenObjective(_layout()))
        batch = _batch(
            input_ids=torch.tensor([[1, 7, 8, 2], [1, 7, 0, 0]]),
            labels=torch.tensor([[-100, 7, 8, -100], [-100, 9, -100, -100]]),
            tasks=[Task.TTS, Task.TTS],
        )
        with self.assertRaisesRegex(ValueError, "outputs do not match"):
            _flops(module, batch, outputs={"loss": torch.tensor(1.0)})

        model.backbone.config._attn_implementation = "sdpa"
        with self.assertRaisesRegex(ValueError, "FlashAttention 2"):
            _flops(module, batch)

        model.backbone.config._attn_implementation = "flash_attention_2"
        next(iter(model.semantic_audio_adapter.parameters())).requires_grad_(False)
        with self.assertRaisesRegex(ValueError, "full model to be trainable"):
            _flops(module, batch)

        model = _token_model()
        module = _module(model, TokenObjective(_layout()))
        trainer = cast(pl.Trainer, SimpleNamespace(callbacks=[_grad_logger()]))
        with self.assertRaisesRegex(ValueError, "GradLogger"):
            TrainingFlops()(
                trainer=trainer,
                pl_module=module,
                outputs=_outputs(),
                batch=batch,
                batch_idx=0,
            )

        flow_model = _flow_model()
        flow_model.acoustic_decoder.repa_student_layer = 1
        flow_module = _module(flow_model, FlowObjective(_layout(), _flow_runtime()))
        flow_batch = _batch(
            input_ids=batch.input_ids,
            labels=batch.token_labels,
            tasks=batch.tasks,
            acoustic_target=_target(torch.ones((2, 3), dtype=torch.bool)),
        )
        with self.assertRaisesRegex(ValueError, "REPA"):
            _flops(flow_module, flow_batch)


def _layout() -> Layout:
    return Layout(text=(0, 7), audio=(7, 12))


class _Tokenizer:
    vocab_size = 3

    def decode(self, token_ids: Any) -> list[tuple[int, ...]]:
        values = list(token_ids)
        return [(int(value) % 3,) for value in values]

    def frame_spans(self, token_ids: Any) -> list[int]:
        return [1 for _ in token_ids]


class _Codec:
    acoustic_feature_dim = 4
    acoustic_codebook_sizes = (5, 6)
    semantic_codebook = torch.arange(12, dtype=torch.float32).reshape(3, 4)


def _flow_runtime() -> Any:
    return SimpleNamespace()


class _Runtime:
    layout = _layout()
    codec = _Codec()
    audio_tokenizer = _Tokenizer()
    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    boa_token_id = 10
    eoa_token_id = 11
    flow_matching = _flow_runtime()


def _model_config() -> ModelConfig:
    return ModelConfig(
        semantic_audio_adapter=AdapterType.LINEAR,
        semantic_audio_output_adapter=AdapterType.LINEAR,
        acoustic_prompt_adapter=AdapterType.LINEAR,
        toy=ToyConfig(
            hidden_size=4,
            intermediate_size=8,
            layers=1,
            heads=1,
            max_position_embeddings=32,
        ),
    )


def _token_model() -> TokenModel:
    model = TokenModel(_model_config(), runtime=cast(Any, _Runtime()))
    model.backbone.config._attn_implementation = "flash_attention_2"
    return model


def _flow_model() -> FlowModel:
    model = FlowModel(
        _model_config(),
        runtime=cast(Any, _Runtime()),
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
    )
    model.backbone.config._attn_implementation = "flash_attention_2"
    return model


def _rvq_model() -> RVQModel:
    model = RVQModel(
        _model_config(),
        runtime=cast(Any, _Runtime()),
        decoder=DecoderConfig(hidden_dim=4, layers=1, heads=1, ffn_ratio=2),
    )
    model.backbone.config._attn_implementation = "flash_attention_2"
    return model


def _module(model: nn.Module, objective: nn.Module) -> SpeechToSpeechModule[Any]:
    return SpeechToSpeechModule(
        ModuleConfig(),
        model=cast(Any, model),
        objective=cast(Any, objective),
    )


def _batch(
    *,
    input_ids: Tensor,
    labels: Tensor,
    tasks: list[Task],
    acoustic_prompt: AcousticPrompt | None = None,
    acoustic_target: AcousticTarget | None = None,
) -> ModelBatch:
    return ModelBatch(
        input_ids=input_ids,
        token_labels=labels,
        acoustic_prompt=acoustic_prompt,
        acoustic_target=acoustic_target,
        tasks=tasks,
        pad_token_id=0,
    )


def _prompt() -> AcousticPrompt:
    positions = torch.tensor([[1, 2], [1, -1]])
    codes = torch.zeros((2, 2, 2), dtype=torch.long)
    codes[positions < 0] = -1
    return {
        "codes": codes,
        "token_positions": positions,
    }


def _target(mask: Tensor) -> AcousticTarget:
    codes = torch.zeros((*mask.shape, 2), dtype=torch.long).masked_fill(
        ~mask[..., None], -1
    )
    return {
        "semantic_codes": torch.zeros((*mask.shape, 1), dtype=torch.long).masked_fill(
            ~mask[..., None], -1
        ),
        "codes": codes,
        "token_positions": torch.where(mask, torch.tensor(1), torch.tensor(-1)),
    }


def _token_expected(model: TokenModel, batch: ModelBatch) -> int:
    core = model.backbone.base_model
    input_ids = batch.input_ids
    hidden = core.config.hidden_size
    embedding = model.semantic_audio_embedding
    audio_start, audio_end = model.layout.blocks["audio"]
    rows = int((input_ids.ge(audio_start) & input_ids.lt(audio_end)).sum())
    forward = adapter(
        model.semantic_audio_adapter,
        rows=rows,
        in_features=embedding.embedding_dim,
        out_features=hidden,
        name="test semantic",
    )
    if batch.acoustic_prompt is not None:
        forward += adapter(
            model.acoustic_prompt_adapter,
            rows=input_ids.numel(),
            in_features=4,
            out_features=hidden,
            name="test prompt",
        )
    forward += qwen_backbone(
        core,
        batch=input_ids.size(0),
        sequence=input_ids.size(1),
        lengths=batch.attention_mask.sum(dim=1),
    )
    valid = batch.token_labels[:, 1:].ne(-100)
    count = int(valid.sum())
    modality = batch.tasks[0].target_modality
    start, end = model.layout.blocks[modality.value]
    if modality.value == "audio":
        forward += adapter(
            model.semantic_audio_output_adapter,
            rows=count,
            in_features=hidden,
            out_features=embedding.embedding_dim,
            name="test output",
        )
        forward += 2 * count * embedding.embedding_dim * embedding.num_embeddings
    else:
        forward += 2 * count * hidden * (end - start)
    return 3 * forward


def _outputs(*, acoustic: str | None = None) -> dict[str, Any]:
    output: dict[str, Any] = {
        "loss": torch.tensor(1.0),
        "token": LossItem(torch.ones(2), None),
    }
    if acoustic is not None:
        output[acoustic] = LossItem(torch.ones(2), None)
    return output


def _grad_logger() -> Any:
    from speech_to_speech.callback.logging import GradLogger

    return GradLogger(("token", "token"), "unused")


def _flops(
    module: SpeechToSpeechModule[Any],
    batch: ModelBatch,
    *,
    outputs: dict[str, Any] | None = None,
) -> float:
    if outputs is None:
        objective = module.objective
        name = (
            "flow_matching"
            if type(objective) is FlowObjective and batch.acoustic_target is not None
            else "rvq"
            if type(objective) is RVQObjective and batch.acoustic_target is not None
            else None
        )
        outputs = _outputs(acoustic=name)
    trainer = cast(pl.Trainer, SimpleNamespace(callbacks=[]))
    return TrainingFlops()(
        trainer=trainer,
        pl_module=module,
        outputs=outputs,
        batch=batch,
        batch_idx=0,
    )


if __name__ == "__main__":
    unittest.main()
