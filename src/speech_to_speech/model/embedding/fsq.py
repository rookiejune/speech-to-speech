from __future__ import annotations

import math
from dataclasses import dataclass
from math import prod
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..._compat import StrEnum, auto
from ..._compat import register


class FsqFeature(StrEnum):
    DIGIT_VALUE = auto()
    DIGIT_ONEHOT = auto()


@dataclass(frozen=True)
class FsqEmbeddingConfig:
    feature: FsqFeature = FsqFeature.DIGIT_ONEHOT

    def __post_init__(self) -> None:
        if not isinstance(self.feature, FsqFeature):
            raise TypeError("FSQ embedding feature must be an FsqFeature.")


@dataclass(frozen=True)
class FsqNeighbors:
    """Immediate lattice neighbors for a batch of local audio token IDs."""

    token_ids: Tensor
    weights: Tensor
    valid: Tensor


FsqLevelValues = tuple[tuple[tuple[float, ...], ...], ...]


class FsqEmbedding(nn.Module):
    """Factorized embedding for packed multi-stage FSQ product indices.

    ``digit_onehot`` learns an unrestricted row for every level of every digit,
    then adds the centered digit rows and one stage offset. ``digit_value`` is
    the lower-rank baseline and requires canonical level values from the codec.
    Marker and special rows remain free embeddings. Lookup and logits stay
    factorized; ``.weight`` is only a compatibility materialization path.
    """

    topology: Tensor
    level_values: Tensor
    radix_order = "first_fastest"

    def __init__(
        self,
        *,
        codebook_sizes: tuple[int, ...],
        fsq_levels: tuple[tuple[int, ...], ...],
        num_embeddings: int,
        embedding_dim: int,
        target_rms: float,
        config: FsqEmbeddingConfig | None = None,
        level_values: FsqLevelValues | None = None,
    ) -> None:
        super().__init__()
        free = _validate_embedding_shape(
            codebook_sizes,
            fsq_levels,
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        if (
            isinstance(target_rms, bool)
            or not isinstance(target_rms, (int, float))
            or not math.isfinite(target_rms)
            or target_rms <= 0
        ):
            raise ValueError("FSQ target row RMS must be finite and positive.")

        self.config = FsqEmbeddingConfig() if config is None else config
        self.codebook_sizes = codebook_sizes
        self.fsq_levels = fsq_levels
        self._num_embeddings = num_embeddings
        self._embedding_dim = embedding_dim
        self._bases = tuple(_radix_basis(stage) for stage in fsq_levels)
        register(self, "topology", _topology_state(codebook_sizes, fsq_levels))

        values = _validate_level_values(fsq_levels, level_values, self.config.feature)
        self._value_slices: tuple[tuple[slice, ...], ...] = ()
        if values is not None:
            flat, slices = _flatten_level_values(values)
            self._value_slices = slices
            register(self, "level_values", flat)

        self.offsets = nn.ParameterList()
        self.tables = nn.ModuleList()
        self.slopes = nn.ModuleList()
        for levels in fsq_levels:
            offset = nn.Parameter(torch.empty(embedding_dim))
            nn.init.normal_(offset, std=embedding_dim**-0.5)
            self.offsets.append(offset)

            tables = nn.ParameterList()
            slopes = nn.ParameterList()
            for level in levels:
                if self.config.feature is FsqFeature.DIGIT_ONEHOT:
                    table = nn.Parameter(torch.empty(level, embedding_dim))
                    nn.init.normal_(table, std=embedding_dim**-0.5)
                    with torch.no_grad():
                        table.sub_(table.mean(dim=0, keepdim=True))
                    tables.append(table)
                else:
                    slope = nn.Parameter(torch.empty(embedding_dim))
                    nn.init.normal_(slope, std=embedding_dim**-0.5)
                    slopes.append(slope)
            self.tables.append(tables)
            self.slopes.append(slopes)

        with torch.no_grad():
            for stage in range(len(codebook_sizes)):
                self._scale_stage(stage, target_rms)

        self.free = nn.Embedding(free, embedding_dim)
        nn.init.normal_(self.free.weight, std=embedding_dim**-0.5)
        if free > 0:
            with torch.no_grad():
                _scale_to_rms(self.free.weight, target_rms)

    @property
    def num_embeddings(self) -> int:
        return self._num_embeddings

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def weight(self) -> Tensor:
        return self._materialize()

    def forward(self, input_ids: Tensor) -> Tensor:
        return self.rows(input_ids)

    def rows(self, input_ids: Tensor) -> Tensor:
        """Return vocabulary rows without materializing the complete table."""
        self._validate_input_ids(input_ids)
        flat_ids = input_ids.reshape(-1)
        output = self.offsets[0].new_empty(flat_ids.numel(), self._embedding_dim)

        start = 0
        for stage, size in enumerate(self.codebook_sizes):
            mask = flat_ids.ge(start) & flat_ids.lt(start + size)
            positions = mask.nonzero(as_tuple=False).flatten()
            if positions.numel() == 0:
                start += size
                continue
            product_ids = flat_ids.index_select(0, positions) - start
            output = output.index_copy(
                0,
                positions,
                self._code_rows(product_ids, stage=stage),
            )
            start += size

        if self.free.num_embeddings > 0:
            mask = flat_ids.ge(start)
            positions = mask.nonzero(as_tuple=False).flatten()
            free_ids = flat_ids.index_select(0, positions) - start
            output = output.index_copy(0, positions, self.free(free_ids))
        return output.view(*input_ids.shape, self._embedding_dim)

    def logits(self, hidden: Tensor, local_ids: Tensor | None = None) -> Tensor:
        """Compute tied logits without constructing a vocabulary-by-hidden table."""
        if local_ids is not None:
            if local_ids.dim() != 1:
                raise ValueError("local_ids must be a one-dimensional tensor.")
            weight = self.rows(local_ids)
            return F.linear(hidden.to(dtype=weight.dtype), weight)

        values = hidden.to(dtype=self.offsets[0].dtype)
        logits = [
            self._stage_logits(values, stage=stage) for stage in range(len(self.codebook_sizes))
        ]
        if self.free.num_embeddings > 0:
            logits.append(F.linear(values, self.free.weight))
        return torch.cat(logits, dim=-1)

    def code_embedding(self, product_ids: Tensor, *, stage: int) -> Tensor:
        """Embed raw product indices for one residual stage."""
        if stage < 0 or stage >= len(self.codebook_sizes):
            raise ValueError(f"FSQ stage {stage} is out of range.")
        size = self.codebook_sizes[stage]
        if bool((product_ids < 0).any()) or bool((product_ids >= size).any()):
            raise ValueError("product id is outside the FSQ codebook.")
        return self._code_rows(product_ids, stage=stage)

    def neighbors(self, local_ids: Tensor) -> FsqNeighbors:
        """Return normalized immediate +/-1 digit neighbors within each stage."""
        if local_ids.dim() != 1:
            raise ValueError("FSQ neighbor ids must be one-dimensional.")
        self._validate_input_ids(local_ids)
        width = 2 * max(len(levels) for levels in self.fsq_levels)
        token_ids = local_ids.new_zeros((local_ids.numel(), width))
        valid = torch.zeros_like(token_ids, dtype=torch.bool)

        start = 0
        for stage, (size, levels, basis) in enumerate(
            zip(self.codebook_sizes, self.fsq_levels, self._bases)
        ):
            stage_mask = local_ids.ge(start) & local_ids.lt(start + size)
            positions = stage_mask.nonzero(as_tuple=False).flatten()
            product_ids = local_ids.index_select(0, positions) - start
            digits = _unpack(product_ids, levels, basis)
            for digit, (level, stride) in enumerate(zip(levels, basis)):
                selected = digits[:, digit]
                lower = selected.gt(0)
                upper = selected.lt(level - 1)
                token_ids[positions, 2 * digit] = start + product_ids - stride
                token_ids[positions, 2 * digit + 1] = start + product_ids + stride
                valid[positions, 2 * digit] = lower
                valid[positions, 2 * digit + 1] = upper
            start += size

        counts = valid.sum(dim=-1, keepdim=True)
        token_ids.masked_fill_(~valid, 0)
        weights = valid.to(dtype=torch.float32) / counts.clamp_min(1)
        return FsqNeighbors(token_ids=token_ids, weights=weights, valid=valid)

    def _code_rows(self, product_ids: Tensor, *, stage: int) -> Tensor:
        flat = product_ids.reshape(-1)
        digits = _unpack(flat, self.fsq_levels[stage], self._bases[stage])
        values = self.offsets[stage].expand(flat.numel(), -1)
        if self.config.feature is FsqFeature.DIGIT_ONEHOT:
            for digit, table in enumerate(self._tables(stage)):
                values = values + F.embedding(digits[:, digit], _centered(table))
        else:
            for digit, slope in enumerate(self._slopes(stage)):
                level = self._level_values(stage, digit).index_select(0, digits[:, digit])
                values = values + level[:, None] * slope
        return values.view(*product_ids.shape, self._embedding_dim)

    def _stage_logits(self, hidden: Tensor, *, stage: int) -> Tensor:
        scores: list[Tensor] = []
        if self.config.feature is FsqFeature.DIGIT_ONEHOT:
            scores.extend(F.linear(hidden, _centered(table)) for table in self._tables(stage))
        else:
            for digit, slope in enumerate(self._slopes(stage)):
                factor = F.linear(hidden, slope.unsqueeze(0))
                scores.append(factor * self._level_values(stage, digit))

        combined = scores[0]
        for score in scores[1:]:
            combined = (score.unsqueeze(-1) + combined.unsqueeze(-2)).flatten(-2)
        offset = F.linear(hidden, self.offsets[stage].unsqueeze(0))
        return combined + offset

    def _materialize(self) -> Tensor:
        rows = [
            self._code_rows(
                torch.arange(size, device=self.offsets[stage].device),
                stage=stage,
            )
            for stage, size in enumerate(self.codebook_sizes)
        ]
        code = torch.cat(rows, dim=0)
        if self.free.num_embeddings == 0:
            return code
        return torch.cat([code, self.free.weight], dim=0)

    def _level_values(self, stage: int, digit: int) -> Tensor:
        values = cast(Tensor, self.level_values)
        return values[self._value_slices[stage][digit]]

    def _scale_stage(self, stage: int, target_rms: float) -> None:
        if self.config.feature is FsqFeature.DIGIT_ONEHOT:
            energy = self.offsets[stage].float().square().mean()
            for table in self._tables(stage):
                energy = energy + _centered(table).float().square().mean()
            parameters = self._tables(stage)
        else:
            mean = self.offsets[stage].float().clone()
            variance = torch.zeros_like(mean)
            parameters = self._slopes(stage)
            for digit, slope in enumerate(parameters):
                level = self._level_values(stage, digit).float()
                factor = slope.float()
                mean = mean + level.mean() * factor
                variance = variance + level.var(unbiased=False) * factor.square()
            energy = (mean.square() + variance).mean()
        scale = target_rms / math.sqrt(float(energy))
        self.offsets[stage].mul_(scale)
        for parameter in parameters:
            parameter.mul_(scale)

    def _tables(self, stage: int) -> nn.ParameterList:
        return cast(nn.ParameterList, self.tables[stage])

    def _slopes(self, stage: int) -> nn.ParameterList:
        return cast(nn.ParameterList, self.slopes[stage])

    def _validate_input_ids(self, input_ids: Tensor) -> None:
        validation_weight = self.offsets[0].detach()[:1].expand(self._num_embeddings, 1)
        F.embedding(input_ids, validation_weight)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        topology_key = f"{prefix}topology"
        topology = state_dict.get(topology_key)
        expected_topology = _topology_state(self.codebook_sizes, self.fsq_levels)
        if topology is not None and not _same_state(topology, expected_topology):
            error_msgs.append(
                f'FSQ topology at "{topology_key}" does not match the configured '
                f"codebook_sizes={self.codebook_sizes!r}, "
                f"fsq_levels={self.fsq_levels!r}."
            )

        if self.config.feature is FsqFeature.DIGIT_VALUE:
            values_key = f"{prefix}level_values"
            values = state_dict.get(values_key)
            expected_values = cast(Tensor, self.level_values)
            if values is not None and not _same_state(values, expected_values):
                error_msgs.append(f'FSQ level values at "{values_key}" do not match the codec.')
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


def reference_rms(reference: Tensor, *, chunk_rows: int = 2_048) -> float:
    """Measure tensor RMS with chunk-local FP32 copies instead of one full copy."""
    if reference.numel() == 0:
        raise ValueError("RMS reference tensor must not be empty.")
    if chunk_rows < 1:
        raise ValueError("RMS chunk size must be positive.")
    values = reference.detach()
    rows = values.reshape(1, -1) if values.dim() == 0 else values.reshape(values.size(0), -1)
    total = torch.zeros((), device=values.device, dtype=torch.float64)
    for start in range(0, rows.size(0), chunk_rows):
        chunk = rows[start : start + chunk_rows]
        total += chunk.float().square().sum().to(dtype=torch.float64)
    return math.sqrt(float(total / values.numel()))


def _topology_state(
    codebook_sizes: tuple[int, ...],
    fsq_levels: tuple[tuple[int, ...], ...],
) -> Tensor:
    values = [len(codebook_sizes)]
    for size, levels in zip(codebook_sizes, fsq_levels):
        values.extend((size, len(levels), *levels))
    return torch.tensor(values, dtype=torch.int64)


def _validate_embedding_shape(
    codebook_sizes: tuple[int, ...],
    fsq_levels: tuple[tuple[int, ...], ...],
    *,
    num_embeddings: int,
    embedding_dim: int,
) -> int:
    if embedding_dim < 1:
        raise ValueError("FSQ embedding dimension must be positive.")
    if len(codebook_sizes) != len(fsq_levels):
        raise ValueError("fsq_levels must align with codebook_sizes.")
    if not codebook_sizes:
        raise ValueError("FSQ embedding requires at least one codebook.")
    for size, stage in zip(codebook_sizes, fsq_levels):
        if size < 1:
            raise ValueError("codebook sizes must be positive.")
        if not stage or any(level < 2 for level in stage):
            raise ValueError("FSQ levels must be at least 2.")
        if prod(stage) != size:
            raise ValueError(f"FSQ levels {stage} must multiply to codebook size {size}.")

    free = num_embeddings - sum(codebook_sizes)
    if free < 0:
        raise ValueError("num_embeddings must cover codebook rows plus free marker rows.")
    return free


def _validate_level_values(
    fsq_levels: tuple[tuple[int, ...], ...],
    values: FsqLevelValues | None,
    feature: FsqFeature,
) -> FsqLevelValues | None:
    if feature is FsqFeature.DIGIT_ONEHOT:
        return None
    if values is None:
        raise ValueError(
            "digit_value FSQ embedding requires canonical level values from the codec."
        )
    if len(values) != len(fsq_levels):
        raise ValueError("FSQ level values must align with stages.")
    normalized: list[tuple[tuple[float, ...], ...]] = []
    for stage_levels, stage_values in zip(fsq_levels, values):
        if len(stage_values) != len(stage_levels):
            raise ValueError("FSQ level values must align with stage digits.")
        digits: list[tuple[float, ...]] = []
        for level, digit_values in zip(stage_levels, stage_values):
            digit = tuple(float(value) for value in digit_values)
            if len(digit) != level:
                raise ValueError("FSQ digit values must match their level count.")
            if any(not math.isfinite(value) for value in digit):
                raise ValueError("FSQ digit values must be finite.")
            if any(left >= right for left, right in zip(digit, digit[1:])):
                raise ValueError("FSQ digit values must be strictly increasing.")
            digits.append(digit)
        normalized.append(tuple(digits))
    return tuple(normalized)


def _flatten_level_values(
    values: FsqLevelValues,
) -> tuple[Tensor, tuple[tuple[slice, ...], ...]]:
    flat: list[float] = []
    stages: list[tuple[slice, ...]] = []
    for stage in values:
        digits: list[slice] = []
        for digit in stage:
            start = len(flat)
            flat.extend(digit)
            digits.append(slice(start, len(flat)))
        stages.append(tuple(digits))
    return torch.tensor(flat, dtype=torch.float32), tuple(stages)


def _radix_basis(levels: tuple[int, ...]) -> tuple[int, ...]:
    basis = [1]
    for level in levels[:-1]:
        basis.append(basis[-1] * level)
    return tuple(basis)


def _unpack(
    product_ids: Tensor,
    levels: tuple[int, ...],
    basis: tuple[int, ...],
) -> Tensor:
    level_tensor = product_ids.new_tensor(levels)
    basis_tensor = product_ids.new_tensor(basis)
    return (product_ids[:, None] // basis_tensor) % level_tensor


def _product_level_indices(vocab: int, levels: tuple[int, ...]) -> Tensor:
    return _unpack(
        torch.arange(vocab, dtype=torch.int64),
        levels,
        _radix_basis(levels),
    )


def _centered(table: Tensor) -> Tensor:
    return table - table.mean(dim=0, keepdim=True)


def _scale_to_rms(values: Tensor, target_rms: float) -> None:
    current = math.sqrt(float(values.float().square().mean()))
    values.mul_(target_rms / current)


def _same_state(actual: Tensor, expected: Tensor) -> bool:
    return (
        actual.dtype is expected.dtype
        and actual.shape == expected.shape
        and torch.equal(actual.detach().cpu(), expected.detach().cpu())
    )
