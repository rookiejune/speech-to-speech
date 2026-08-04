from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from anytrain.module.dit import DiTBlock, SequenceAttention, TimeEmbedding
from semantic_acoustic_codec.model.dit import DiTDecoder
from semantic_acoustic_codec.model.rvq import AcousticRVQDecoder
from torch import Tensor, nn
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from transformers import Qwen3Model
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3MLP,
)

from ..model._helper import MLPAdapter
from ..model.audio_input import AudioInputAdapterType, AudioInputTower
from ..model.audio_output import AudioOutputAdapter, AudioOutputAdapterType


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
        return _linear_sum(
            (
                module.gate_proj,
                module.up_proj,
                module.down_proj,
            ),
            rows,
        )
    if type(module) is AudioOutputAdapter:
        if module.config.type is AudioOutputAdapterType.TRANSFORMER:
            return audio_output_transformer(module, rows=rows)
        return adapter(
            module.adapter,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            name=name,
        )
    raise TypeError(f"{name} uses an unsupported module: {type(module).__name__}.")


def audio_output_transformer(module: AudioOutputAdapter, *, rows: int) -> int:
    """Count a pointwise-row estimate for causal audio output transformer layers.

    Teacher-forcing FLOPs use the selected valid label rows as the sequence
    budget proxy, matching the sparse CE path that still pays for a full-sequence
    adapter forward in practice via denser ops; this keeps the estimate finite
    and deterministic without requiring padded sequence geometry here.
    """
    if module.layers is None:
        raise TypeError("audio output transformer FLOPs require transformer layers.")
    if type(module.input_projection) is not nn.Linear:
        raise TypeError("audio output transformer must use a linear input projection.")
    hidden = module.out_features
    require_linear(module.input_projection, module.in_features, hidden, "audio output projection")
    total = linear(module.input_projection, rows)
    for layer in module.layers:
        # Q, K, V, out projections + 2-layer FFN, ignoring attention score matmuls'
        # sequence-length term for the sparse-row proxy.
        total += 4 * 2 * rows * hidden * hidden
        ffn = layer.ffn
        if not isinstance(ffn, nn.Sequential) or len(ffn) < 4:
            raise TypeError("audio output transformer FFN shape is unsupported.")
        up = ffn[0]
        down = ffn[3]
        if type(up) is not nn.Linear or type(down) is not nn.Linear:
            raise TypeError("audio output transformer FFN must use linear layers.")
        total += linear(up, rows) + linear(down, rows)
    return total


def audio_input_tower(
    tower: AudioInputTower,
    *,
    batch: int,
    frames: int,
) -> int:
    """Count a dense source-audio input tower forward pass.

    ``AudioInputTower`` receives a padded ``[B, F, D]`` tensor and applies its
    adapter before masking the result. The estimate therefore counts all
    ``B * F`` rows; the mask prevents padding from becoming context but does
    not make the standard PyTorch encoder an unpadded operation.
    """
    if type(tower) is not AudioInputTower:
        raise TypeError("audio input FLOPs require the standard AudioInputTower.")
    if batch < 1 or frames < 1:
        raise ValueError("audio input FLOPs batch and frame dimensions must be positive.")

    rows = batch * frames
    in_features = tower.in_features
    out_features = tower.out_features
    if tower.config.type is AudioInputAdapterType.MLP:
        if type(tower.input_projection) is not nn.Identity:
            raise TypeError("audio input MLP must not use an input projection.")
        return adapter(
            tower.adapter,
            rows=rows,
            in_features=in_features,
            out_features=out_features,
            name="audio input adapter",
        )

    if tower.config.type is not AudioInputAdapterType.TRANSFORMER:
        raise ValueError("audio input FLOPs require an enabled MLP or transformer tower.")
    if type(tower.input_projection) is not nn.Linear:
        raise TypeError("audio input transformer must use a linear input projection.")
    require_linear(
        tower.input_projection,
        in_features,
        out_features,
        "audio input input projection",
    )
    encoder = tower.adapter
    if type(encoder) is not nn.TransformerEncoder:
        raise TypeError("audio input transformer must use nn.TransformerEncoder.")

    forward = linear(tower.input_projection, rows)
    if len(encoder.layers) != tower.config.layers:
        raise ValueError("audio input transformer depth does not match its config.")
    for layer in encoder.layers:
        forward += _audio_input_encoder_layer(
            layer,
            rows=rows,
            batch=batch,
            frames=frames,
            hidden=out_features,
            heads=tower.config.heads,
        )
    return forward


def _audio_input_encoder_layer(
    layer: nn.Module,
    *,
    rows: int,
    batch: int,
    frames: int,
    hidden: int,
    heads: int,
) -> int:
    if type(layer) is not nn.TransformerEncoderLayer:
        raise TypeError("audio input FLOPs require standard TransformerEncoderLayer layers.")
    if not layer.self_attn.batch_first:
        raise TypeError("audio input transformer must use batch-first attention.")
    attention = layer.self_attn
    if attention.embed_dim != hidden or attention.num_heads != heads:
        raise ValueError("audio input transformer attention dimensions do not match config.")
    if attention.head_dim * attention.num_heads != hidden:
        raise ValueError("audio input transformer heads do not cover hidden size.")
    in_projection = attention.in_proj_weight
    if in_projection.shape != (3 * hidden, hidden):
        raise ValueError("audio input transformer QKV projection shape is unsupported.")
    output_projection = attention.out_proj
    if type(output_projection) not in {
        nn.Linear,
        NonDynamicallyQuantizableLinear,
    }:
        raise TypeError("audio input transformer output projection must be linear.")
    if (
        output_projection.in_features,
        output_projection.out_features,
    ) != (hidden, hidden):
        raise ValueError("audio input transformer output projection shape is unsupported.")

    linear1 = layer.linear1
    linear2 = layer.linear2
    if type(linear1) is not nn.Linear or type(linear2) is not nn.Linear:
        raise TypeError("audio input transformer FFN projections must be linear.")
    if linear1.in_features != hidden or linear2.out_features != hidden:
        raise ValueError("audio input transformer FFN dimensions are unsupported.")
    if linear2.in_features != linear1.out_features:
        raise ValueError("audio input transformer FFN projections do not align.")

    # MultiheadAttention stores Q/K/V in one [3H, H] projection. Its two
    # attention matrix products are dense because this tower is not causal.
    return (
        2 * rows * hidden * (3 * hidden)
        + _linear_module(output_projection, rows)
        + linear(linear1, rows)
        + linear(linear2, rows)
        + 4 * batch * frames * frames * hidden
    )


def _linear_module(module: nn.Module, rows: int) -> int:
    if type(module) not in {nn.Linear, NonDynamicallyQuantizableLinear}:
        raise TypeError("linear FLOPs require an exact linear module.")
    return _linear(cast(nn.Linear, module), rows)


def flow_decoder(decoder: DiTDecoder, *, batch: int, frames: int) -> int:
    """Count a standard dense SAC ``DiTDecoder`` forward pass."""
    if type(decoder) is not DiTDecoder:
        raise TypeError("Flow FLOPs require the standard SAC DiTDecoder.")
    if batch < 1 or frames < 1:
        raise ValueError("Flow FLOPs batch and frame dimensions must be positive.")
    core = decoder.decoder
    if core.feature_projection is not None or core.feature_layer is not None:
        raise ValueError("Flow FLOPs do not support a REPA decoder.")

    rows = batch * frames
    hidden = core.input.out_features
    latent = core.latent_dim
    condition_projection = core.condition
    condition = core.condition_dim
    if condition_projection is None or condition is None:
        raise RuntimeError("Flow condition projection is not configured.")
    require_linear(core.input, latent, hidden, "Flow input")
    require_linear(core.output, hidden, latent, "Flow output")
    require_linear(condition_projection, condition, hidden, "Flow condition")

    forward = linear(core.input, rows)
    forward += linear(condition_projection, rows)
    forward += linear(core.output, rows)
    forward += _time(core.time, batch, hidden)
    for block in core.blocks:
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
    if type(core) is not Qwen3Model:
        raise TypeError("RVQ FLOPs require a Qwen3Model decoder core.")
    qwen_core = cast(Qwen3Model, core)
    if qwen_core.config.hidden_size != hidden:
        raise ValueError("RVQ Qwen decoder dimensions do not match its configuration.")
    forward += qwen_backbone(
        qwen_core,
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
        forward += _qwen_layer(
            layer,
            rows=rows,
            hidden=hidden,
            intermediate=config.intermediate_size,
            query_width=query_width,
            key_value_width=key_value_width,
            attention_lengths=attention_lengths,
        )
    return forward


def _qwen_layer(
    layer: nn.Module,
    *,
    rows: int,
    hidden: int,
    intermediate: int,
    query_width: int,
    key_value_width: int,
    attention_lengths: int,
) -> int:
    if type(layer) is not Qwen3DecoderLayer:
        raise TypeError("Qwen FLOPs require standard Qwen3DecoderLayer layers.")
    attention = layer.self_attn
    mlp = layer.mlp
    if type(attention) is not Qwen3Attention or type(mlp) is not Qwen3MLP:
        raise TypeError("Qwen FLOPs require standard Qwen3 attention and MLP layers.")
    require_linear(attention.q_proj, hidden, query_width, "Qwen query")
    require_linear(attention.k_proj, hidden, key_value_width, "Qwen key")
    require_linear(attention.v_proj, hidden, key_value_width, "Qwen value")
    require_linear(attention.o_proj, query_width, hidden, "Qwen attention output")
    require_linear(mlp.gate_proj, hidden, intermediate, "Qwen MLP gate")
    require_linear(mlp.up_proj, hidden, intermediate, "Qwen MLP up")
    require_linear(mlp.down_proj, intermediate, hidden, "Qwen MLP down")
    return (
        _linear_sum(
            (
                attention.q_proj,
                attention.k_proj,
                attention.v_proj,
                attention.o_proj,
                mlp.gate_proj,
                mlp.up_proj,
                mlp.down_proj,
            ),
            rows,
        )
        + 2 * query_width * attention_lengths
    )


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
    if block.cross_attention:
        raise TypeError("Flow FLOPs support self-attention DiT blocks only.")
    require_linear(block.film, hidden, hidden * 6, "Flow FiLM")
    return (
        linear(block.film, rows)
        + _dit_attention(
            block.attention,
            rows=rows,
            batch=batch,
            frames=frames,
            hidden=hidden,
        )
        + _dit_ffn(block.ffn, rows=rows, hidden=hidden)
    )


def _dit_attention(
    attention: nn.Module,
    *,
    rows: int,
    batch: int,
    frames: int,
    hidden: int,
) -> int:
    if (
        type(attention) is not SequenceAttention
        or attention.hidden_dim != hidden
        or attention.heads * attention.head_dim != hidden
    ):
        raise TypeError("Flow FLOPs require standard dense self-attention.")
    for name, projection in (
        ("query", attention.query),
        ("key", attention.key),
        ("value", attention.value),
        ("output", attention.output),
    ):
        require_linear(projection, hidden, hidden, f"Flow attention {name}")
    return (
        _linear_sum(
            (
                attention.query,
                attention.key,
                attention.value,
                attention.output,
            ),
            rows,
        )
        + 4 * batch * frames * frames * hidden
    )


def _dit_ffn(ffn: nn.Module, *, rows: int, hidden: int) -> int:
    if type(ffn) is not nn.Sequential or len(ffn) != 3:
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
    return _linear_sum((ffn_input, ffn_output), rows)


def _linear(module: nn.Linear, rows: int) -> int:
    return 2 * rows * module.in_features * module.out_features


def _linear_sum(modules: Sequence[nn.Linear], rows: int) -> int:
    return sum(linear(module, rows) for module in modules)


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
        isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= sequence
        for length in lengths
    ):
        raise ValueError("Qwen attention lengths must be integers in [1, sequence].")
    return sum(length * (length + 1) for length in lengths)


__all__ = [
    "adapter",
    "audio_input_tower",
    "flow_decoder",
    "linear",
    "qwen_backbone",
    "require_linear",
    "rvq_decoder",
]
