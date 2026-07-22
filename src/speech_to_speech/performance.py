from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

import torch
from anydataset.types import Modality
from anytrain.perf import training_flops_from_forward
from lightning import pytorch as pl
from lightning.pytorch.callbacks import Callback
from torch import Tensor, nn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from transformers import Qwen3ForCausalLM, Qwen3Model

from ._flops import (
    adapter,
    flow_decoder,
    qwen_backbone,
    rvq_decoder,
)
from .callback.logging import GradLogger
from .datamodule.types import ModelBatch
from .loss import FlowObjective, LossItem, RVQObjective, TokenObjective
from .model import FlowModel, RVQModel, TokenModel
from .pl_module import SpeechToSpeechModule


class _Trainer(Protocol):
    callbacks: list[Callback]


class TrainingFlops:
    """Estimate local-rank FLOPs for one standard joint-training batch.

    The analytical estimate counts dense projections and attention matrix
    multiplications, then uses the conventional two-forward-equivalents
    estimate for backward. Lookup, scatter, normalization, activation, loss,
    and frozen codec feature extraction are outside this model-MFU boundary.
    """

    def __call__(
        self,
        *,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> float:
        del batch_idx
        if type(pl_module) is not SpeechToSpeechModule:
            raise TypeError("training FLOPs require SpeechToSpeechModule.")
        if not isinstance(batch, ModelBatch):
            raise TypeError("training FLOPs require a ModelBatch.")
        callbacks = cast(_Trainer, cast(object, trainer)).callbacks
        if any(isinstance(callback, GradLogger) for callback in callbacks):
            raise ValueError(
                "training FLOPs do not support GradLogger because it adds "
                "extra autograd work."
            )

        model = cast(nn.Module, cast(object, pl_module.model))
        objective = cast(nn.Module, cast(object, pl_module.objective))
        expected = {"loss", "token"}
        if type(objective) is TokenObjective:
            if type(model) is not TokenModel:
                raise TypeError("TokenObjective FLOPs require the standard TokenModel.")
        elif type(objective) is FlowObjective:
            if type(model) is not FlowModel:
                raise TypeError(
                    "FlowObjective FLOPs require the standard FlowModel."
                )
            _flow_objective(cast(FlowObjective, objective), model)
            if batch.acoustic_target is not None:
                expected.add("flow_matching")
        elif type(objective) is RVQObjective:
            if type(model) is not RVQModel:
                raise TypeError(
                    "RVQObjective FLOPs require the standard RVQModel."
                )
            if batch.acoustic_target is not None:
                expected.add("rvq")
        else:
            raise TypeError(
                f"training FLOPs do not support objective {type(objective).__name__}."
            )

        _outputs(outputs, expected)
        token_model = cast(TokenModel, model)
        core = _backbone(token_model)
        _trainable(model)
        forward = _token_path(token_model, core, batch)

        if isinstance(model, FlowModel):
            target = _target(batch, model)
            if target is not None:
                _, mask = target
                decoder = model.acoustic_decoder
                feature_dim = model.runtime.codec.acoustic_feature_dim
                if decoder.latent_dim != feature_dim:
                    raise ValueError(
                        "Flow decoder latent size does not match the codec feature size."
                    )
                forward += flow_decoder(
                    decoder,
                    batch=mask.size(0),
                    frames=mask.size(1),
                )
        elif isinstance(model, RVQModel):
            target = _target(batch, model)
            if target is not None:
                _, mask = target
                decoder = model.acoustic_decoder
                sizes = tuple(model.runtime.codec.acoustic_codebook_sizes)
                if tuple(decoder.codebook_sizes) != sizes:
                    raise ValueError(
                        "RVQ decoder codebooks do not match the runtime codec."
                    )
                forward += rvq_decoder(
                    decoder,
                    valid_frames=int(mask.sum().item()),
                )

        return training_flops_from_forward(float(forward), backward_multiplier=2.0)


def _flow_objective(objective: FlowObjective, model: nn.Module) -> None:
    if objective.repa_weight is not None or objective.repa_teacher is not None:
        raise ValueError("training FLOPs do not support REPA.")
    decoder = cast(FlowModel, model).acoustic_decoder
    if decoder.repa_projection is not None or decoder.repa_student_layer is not None:
        raise ValueError("training FLOPs do not support REPA.")


def _outputs(outputs: Any, expected: set[str]) -> None:
    if not isinstance(outputs, Mapping):
        raise TypeError("training FLOPs outputs must be a mapping.")
    keys = set(outputs)
    if keys != expected:
        raise ValueError(
            "training FLOPs outputs do not match the active objective branch: "
            f"expected {sorted(expected)}, got {sorted(keys)}."
        )
    if not isinstance(outputs["loss"], Tensor):
        raise TypeError("training FLOPs loss output must be a tensor.")
    for name in expected - {"loss"}:
        if not isinstance(outputs[name], LossItem):
            raise TypeError(f"training FLOPs output {name!r} must be a LossItem.")


def _backbone(model: TokenModel) -> Qwen3Model:
    backbone = cast(object, model.backbone)
    if type(backbone) is not Qwen3ForCausalLM:
        raise TypeError("training FLOPs require a standard Qwen3ForCausalLM backbone.")
    backbone = cast(Qwen3ForCausalLM, backbone)
    core = backbone.base_model
    if type(core) is not Qwen3Model:
        raise TypeError("training FLOPs require a standard Qwen3Model backbone body.")
    if core.config._attn_implementation != "flash_attention_2":
        raise ValueError("training FLOPs require Qwen3 FlashAttention 2.")
    return core


def _trainable(model: nn.Module) -> None:
    replaced_linear = next(
        (
            type(module).__name__
            for module in model.modules()
            if isinstance(module, nn.Linear)
            and type(module) not in {nn.Linear, NonDynamicallyQuantizableLinear}
        ),
        None,
    )
    if replaced_linear is not None:
        raise TypeError(
            "training FLOPs do not support replaced Linear modules such as "
            f"{replaced_linear}."
        )
    allowed_frozen: set[str] = set()
    if isinstance(model, RVQModel):
        last = model.acoustic_decoder.codebooks - 1
        allowed_frozen = {
            "acoustic_decoder.decoder.embed_tokens.weight",
            *(
                name
                for name, _ in model.named_parameters()
                if name.startswith(
                    (
                        f"acoustic_decoder.codebook_embeddings.{last}.",
                        f"acoustic_decoder.embedding_projections.{last}.",
                    )
                )
            ),
        }
    frozen = sorted(
        name
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad and name not in allowed_frozen
    )
    if frozen:
        raise ValueError(
            "training FLOPs require the full model to be trainable; frozen "
            f"parameters include {frozen[0]!r}."
        )


def _token_path(model: TokenModel, core: Qwen3Model, batch: ModelBatch) -> int:
    input_ids = batch.input_ids
    if input_ids.dim() != 2 or input_ids.size(0) < 1 or input_ids.size(1) < 1:
        raise ValueError("training FLOPs input ids must have shape [B, S].")
    batch_size, sequence = input_ids.shape
    attention_mask = batch.attention_mask
    lengths = attention_mask.sum(dim=1)
    if not bool(lengths.gt(0).all()):
        raise ValueError("each training FLOPs input row must contain a valid token.")

    embedding = model.semantic_audio_embedding
    if type(embedding) is not nn.Embedding:
        raise TypeError("training FLOPs require a semantic audio embedding.")
    hidden = core.config.hidden_size
    audio_start, audio_end = model.layout.blocks[Modality.AUDIO.value]
    if embedding.num_embeddings != audio_end - audio_start:
        raise ValueError(
            "semantic audio embedding rows do not match the audio layout block."
        )
    audio_rows = int((input_ids.ge(audio_start) & input_ids.lt(audio_end)).sum().item())
    forward = adapter(
        model.semantic_audio_adapter,
        rows=audio_rows,
        in_features=embedding.embedding_dim,
        out_features=hidden,
        name="semantic audio adapter",
    )
    if batch.acoustic_prompt is not None:
        forward += adapter(
            model.acoustic_prompt_adapter,
            rows=batch_size * sequence,
            in_features=model.runtime.codec.acoustic_feature_dim,
            out_features=hidden,
            name="acoustic prompt adapter",
        )
    forward += qwen_backbone(
        core,
        batch=batch_size,
        sequence=sequence,
        lengths=lengths,
    )
    forward += _token_head(model, batch)
    return forward


def _token_head(model: TokenModel, batch: ModelBatch) -> int:
    labels = batch.token_labels[:, 1:]
    valid = labels.ne(-100)
    if not bool(valid.any(dim=1).all()):
        raise ValueError(
            "each training FLOPs token-label row must contain a valid target."
        )
    rows = int(valid.sum().item())
    modality = batch.tasks[0].target_modality
    start, end = model.layout.blocks[modality.value]
    if bool((valid & (labels.lt(start) | labels.ge(end))).any()):
        raise ValueError(
            f"training FLOPs labels contain an id outside the {modality.value} block."
        )

    hidden = model.backbone.config.hidden_size
    if modality is Modality.TEXT:
        output = model.backbone.get_output_embeddings()
        if type(output) is not nn.Linear or output.in_features != hidden:
            raise ValueError(f"text token head must be Linear({hidden}, vocab_size).")
        if output.out_features < end - start:
            raise ValueError("text token head does not cover the text layout block.")
        # VocabularyHeadMixin selects only the text-layout rows from this head.
        return 2 * rows * hidden * (end - start)
    if modality is Modality.AUDIO:
        embedding = model.semantic_audio_embedding
        forward = adapter(
            model.semantic_audio_output_adapter,
            rows=rows,
            in_features=hidden,
            out_features=embedding.embedding_dim,
            name="semantic audio output adapter",
        )
        return forward + 2 * rows * embedding.embedding_dim * embedding.num_embeddings
    raise ValueError(f"training FLOPs do not support modality {modality.value!r}.")


def _target(
    batch: ModelBatch,
    model: FlowModel | RVQModel,
) -> tuple[Tensor, Tensor] | None:
    target = batch.acoustic_target
    if target is None:
        return None
    codes = target["codes"]
    mask = batch.acoustic_target_mask
    if mask is None:
        raise RuntimeError("training FLOPs acoustic target mask is unavailable.")
    if codes.dim() != 3 or mask.dim() != 2 or codes.shape[:2] != mask.shape:
        raise ValueError(
            "training FLOPs acoustic codes and mask must have shapes "
            "[B, F, Q] and [B, F]."
        )
    if mask.dtype != torch.bool:
        raise TypeError("training FLOPs acoustic target mask must be boolean.")
    if not bool(mask.any(dim=1).all()):
        raise ValueError(
            "each training FLOPs acoustic target row must contain a valid frame."
        )
    codebooks = tuple(model.runtime.codec.acoustic_codebook_sizes)
    if codes.size(-1) != len(codebooks):
        raise ValueError(
            f"training FLOPs target has {codes.size(-1)} acoustic codebooks; "
            f"the runtime requires {len(codebooks)}."
        )
    return codes, mask


__all__ = ["TrainingFlops"]
