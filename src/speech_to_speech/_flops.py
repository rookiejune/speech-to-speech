from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from transformers import Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
)

from .model.acoustic.dit import AcousticDiT, DiTBlock, TimeEmbedding
from .model.acoustic.rvq import AcousticRVQDecoder
from .model.adapter import MLPAdapter


def adapter(
    module: nn.Module,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    name: str,
) -> int:
    """Count the matrix multiplications in a supported feature adapter."""
    if type(module) is nn.Identity:
        if in_features != out_features:
            raise ValueError(f"{name} identity cannot change dimensions.")
        return 0
    if type(module) is nn.Linear:
        require_linear(module, in_features, out_features, name)
        return linear(module, rows)
    if type(module) is MLPAdapter:
        intermediate = module.gate_proj.out_features
        require_linear(module.gate_proj, in_features, intermediate, f"{name} gate")
        require_linear(module.up_proj, in_features, intermediate, f"{name} up")
        require_linear(module.down_proj, intermediate, out_features, f"{name} down")
        return sum(
            linear(projection, rows)
            for projection in (module.gate_proj, module.up_proj, module.down_proj)
        )
    raise TypeError(f"{name} uses an unsupported module: {type(module).__name__}.")


def flow_decoder(decoder: AcousticDiT, *, batch: int, frames: int) -> int:
    """Count a standard dense AcousticDiT forward pass."""
    if type(decoder) is not AcousticDiT:
        raise TypeError("Flow FLOPs require the standard AcousticDiT decoder.")
    if batch < 1 or frames < 1:
        raise ValueError("Flow FLOPs batch and frame dimensions must be positive.")
    if decoder.repa_projection is not None or decoder.repa_student_layer is not None:
        raise ValueError("Flow FLOPs do not support a REPA decoder.")

    rows = batch * frames
    hidden = decoder.input.out_features
    latent = decoder.latent_dim
    condition = decoder.condition.in_features
    require_linear(decoder.input, latent, hidden, "Flow input")
    require_linear(decoder.output, hidden, latent, "Flow output")
    require_linear(decoder.condition, condition, hidden, "Flow condition")

    forward = linear(decoder.input, rows)
    forward += linear(decoder.condition, rows)
    forward += linear(decoder.output, rows)
    forward += _time(decoder.time, batch, hidden)
    for block in decoder.blocks:
        if type(block) is not DiTBlock:
            raise TypeError("Flow FLOPs require standard DiTBlock layers.")
        forward += _dit_block(block, batch, frames, hidden)
    return forward


def rvq_decoder(decoder: AcousticRVQDecoder, *, valid_frames: int) -> int:
    """Count a standard frame-packed AcousticRVQDecoder forward pass."""
    if type(decoder) is not AcousticRVQDecoder:
        raise TypeError("RVQ FLOPs require the standard AcousticRVQDecoder model.")
    if valid_frames < 1:
        raise ValueError("RVQ FLOPs valid frame count must be positive.")

    codebooks = decoder.codebooks
    hidden = decoder.hidden_dim
    condition = decoder.condition_dim
    forward = _projection(
        decoder.condition,
        rows=valid_frames,
        in_features=condition,
        out_features=hidden,
        name="RVQ condition",
    )

    if (
        len(decoder.codebook_embeddings) != codebooks
        or len(decoder.embedding_projections) != codebooks
    ):
        raise ValueError("RVQ embedding modules do not match the decoder codebooks.")
    for index, (embedding, projection, size) in enumerate(
        zip(
            decoder.codebook_embeddings,
            decoder.embedding_projections,
            decoder.codebook_sizes,
        )
    ):
        if type(embedding) is not nn.Embedding or embedding.weight.shape != (
            size,
            decoder.embedding_dim,
        ):
            raise ValueError("RVQ codebook embedding shape does not match the decoder.")
        cost = _projection(
            projection,
            rows=valid_frames,
            in_features=decoder.embedding_dim,
            out_features=hidden,
            name=f"RVQ codebook {index} projection",
        )
        if index + 1 < codebooks:
            forward += cost

    core = decoder.decoder
    if core.config.hidden_size != hidden:
        raise ValueError("RVQ Qwen decoder dimensions do not match its configuration.")
    forward += qwen_backbone(
        core,
        batch=valid_frames,
        sequence=codebooks,
        lengths=(codebooks,) * valid_frames,
    )

    if len(decoder.heads) != codebooks:
        raise ValueError("RVQ output heads do not match the decoder codebooks.")
    for index, (head, size) in enumerate(zip(decoder.heads, decoder.codebook_sizes)):
        if type(head) is not nn.Linear:
            raise TypeError("RVQ FLOPs require linear output heads.")
        require_linear(head, hidden, size, f"RVQ codebook {index} head")
        forward += linear(head, valid_frames)
    return forward


def qwen_backbone(
    core: Qwen3Model,
    *,
    batch: int,
    sequence: int,
    lengths: Sequence[int] | Tensor,
) -> int:
    """Count dense projections and full causal attention in a Qwen3 backbone.

    Projections run over the padded ``batch * sequence`` shape. Attention uses
    the supplied valid length of each row, matching an unpadded fused kernel.
    """
    if type(core) is not Qwen3Model:
        raise TypeError("Qwen FLOPs require a Qwen3Model backbone.")
    if batch < 1 or sequence < 1:
        raise ValueError("Qwen FLOPs batch and sequence dimensions must be positive.")
    config = core.config
    hidden = config.hidden_size
    if len(core.layers) != config.num_hidden_layers:
        raise ValueError("Qwen decoder depth does not match its configuration.")
    layer_types = config.layer_types
    if (
        not isinstance(layer_types, list)
        or len(layer_types) != len(core.layers)
        or any(layer_type != "full_attention" for layer_type in layer_types)
    ):
        raise ValueError("Qwen FLOPs support full causal attention layers only.")
    if core.gradient_checkpointing:
        raise ValueError("Qwen FLOPs do not support gradient checkpointing.")

    attention_lengths = _lengths(lengths, batch=batch, sequence=sequence)
    rows = batch * sequence
    query_width = config.num_attention_heads * config.head_dim
    key_value_width = config.num_key_value_heads * config.head_dim
    forward = 0
    for layer in core.layers:
        if type(layer) is not Qwen3DecoderLayer:
            raise TypeError("Qwen FLOPs require standard Qwen3DecoderLayer layers.")
        attention = layer.self_attn
        mlp = layer.mlp
        if type(attention) is not Qwen3Attention or type(mlp) is not Qwen3MLP:
            raise TypeError(
                "Qwen FLOPs require standard Qwen3 attention and MLP layers."
            )
        require_linear(attention.q_proj, hidden, query_width, "Qwen query")
        require_linear(attention.k_proj, hidden, key_value_width, "Qwen key")
        require_linear(attention.v_proj, hidden, key_value_width, "Qwen value")
        require_linear(attention.o_proj, query_width, hidden, "Qwen attention output")
        require_linear(mlp.gate_proj, hidden, config.intermediate_size, "Qwen MLP gate")
        require_linear(mlp.up_proj, hidden, config.intermediate_size, "Qwen MLP up")
        require_linear(mlp.down_proj, config.intermediate_size, hidden, "Qwen MLP down")
        forward += sum(
            linear(projection, rows)
            for projection in (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
                attention.o_proj,
                mlp.gate_proj,
                mlp.up_proj,
                mlp.down_proj,
            )
        )
        forward += 2 * query_width * attention_lengths
    return forward


def linear(module: nn.Linear, rows: int) -> int:
    """Count multiply-adds for a dense linear projection."""
    if type(module) is not nn.Linear:
        raise TypeError("linear FLOPs require an exact nn.Linear module.")
    return _linear(module, rows)


def require_linear(
    module: nn.Module,
    in_features: int,
    out_features: int,
    name: str,
) -> None:
    """Require an exact Linear shape before applying an analytical formula."""
    if type(module) is not nn.Linear or (
        module.in_features,
        module.out_features,
    ) != (in_features, out_features):
        raise ValueError(f"{name} must be Linear({in_features}, {out_features}).")


def _projection(
    module: nn.Module,
    *,
    rows: int,
    in_features: int,
    out_features: int,
    name: str,
) -> int:
    if type(module) is nn.Identity:
        if in_features != out_features:
            raise ValueError(f"{name} identity cannot change dimensions.")
        return 0
    if type(module) is not nn.Linear:
        raise TypeError(f"{name} uses an unsupported module: {type(module).__name__}.")
    require_linear(module, in_features, out_features, name)
    return linear(module, rows)


def _time(module: nn.Module, rows: int, hidden: int) -> int:
    if type(module) is not TimeEmbedding:
        raise TypeError("Flow FLOPs require the standard TimeEmbedding module.")
    projection = module.projection
    if len(projection) != 3:
        raise TypeError("Flow time embedding uses an unsupported projection.")
    input_projection = projection[0]
    activation = projection[1]
    output_projection = projection[2]
    if (
        type(input_projection) is not nn.Linear
        or type(activation) is not nn.SiLU
        or type(output_projection) is not nn.Linear
    ):
        raise TypeError("Flow time embedding uses an unsupported projection.")
    require_linear(input_projection, hidden, hidden * 4, "Flow time input")
    require_linear(output_projection, hidden * 4, hidden, "Flow time output")
    return linear(input_projection, rows) + linear(output_projection, rows)


def _dit_block(block: DiTBlock, batch: int, frames: int, hidden: int) -> int:
    rows = batch * frames
    require_linear(block.film, hidden, hidden * 6, "Flow FiLM")
    attention = block.attention
    if (
        type(attention) is not nn.MultiheadAttention
        or not attention.batch_first
        or attention.embed_dim != hidden
        or attention.kdim != hidden
        or attention.vdim != hidden
        or attention.in_proj_weight is None
        or attention.in_proj_weight.shape != (hidden * 3, hidden)
    ):
        raise TypeError("Flow FLOPs require standard dense self-attention.")
    if type(attention.out_proj) is not NonDynamicallyQuantizableLinear or (
        attention.out_proj.in_features,
        attention.out_proj.out_features,
    ) != (hidden, hidden):
        raise ValueError(f"Flow attention output must be Linear({hidden}, {hidden}).")
    ffn = block.ffn
    if len(ffn) != 3:
        raise TypeError("Flow FLOPs require the standard DiT feed-forward network.")
    ffn_input = ffn[0]
    activation = ffn[1]
    ffn_output = ffn[2]
    if (
        type(ffn_input) is not nn.Linear
        or type(activation) is not nn.GELU
        or type(ffn_output) is not nn.Linear
        or ffn_input.in_features != hidden
        or ffn_output.in_features != ffn_input.out_features
        or ffn_output.out_features != hidden
    ):
        raise TypeError("Flow FLOPs require the standard DiT feed-forward network.")
    return (
        linear(block.film, rows)
        + 2 * rows * attention.in_proj_weight.numel()
        + _linear(attention.out_proj, rows)
        + linear(ffn_input, rows)
        + linear(ffn_output, rows)
        + 4 * batch * frames * frames * hidden
    )


def _linear(module: nn.Linear, rows: int) -> int:
    return 2 * rows * module.in_features * module.out_features


def _lengths(
    lengths: Sequence[int] | Tensor,
    *,
    batch: int,
    sequence: int,
) -> int:
    if isinstance(lengths, Tensor):
        if lengths.dim() != 1 or lengths.numel() != batch:
            raise ValueError("Qwen attention lengths must have shape [batch].")
        if (
            lengths.dtype == torch.bool
            or torch.is_floating_point(lengths)
            or torch.is_complex(lengths)
        ):
            raise TypeError("Qwen attention lengths must use an integer dtype.")
        if bool(((lengths < 1) | (lengths > sequence)).any()):
            raise ValueError("Qwen attention lengths must be in [1, sequence].")
        return int((lengths * (lengths + 1)).sum().item())

    if len(lengths) != batch:
        raise ValueError("Qwen attention lengths must contain one value per batch row.")
    if any(
        isinstance(length, bool)
        or not isinstance(length, int)
        or not 1 <= length <= sequence
        for length in lengths
    ):
        raise ValueError("Qwen attention lengths must be integers in [1, sequence].")
    return sum(length * (length + 1) for length in lengths)


__all__ = [
    "adapter",
    "flow_decoder",
    "linear",
    "qwen_backbone",
    "require_linear",
    "rvq_decoder",
]
