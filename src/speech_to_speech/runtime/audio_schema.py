from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

from torch import Tensor

from .._tensor import is_signed_integer_dtype

if TYPE_CHECKING:
    from .audio_tokenizer.contract import AudioTokenizer


TokenRange = tuple[int, int]


@dataclass(frozen=True)
class AudioTokenCandidates:
    """Compact legal-next set in codec-local token-ID space."""

    marker_ids: tuple[int, ...] = ()
    token_ranges: tuple[TokenRange, ...] = ()
    allows_eoa: bool = False

    def __post_init__(self) -> None:
        for marker_id in self.marker_ids:
            _non_negative_id(marker_id, "audio candidate marker")
        if len(self.marker_ids) != len(set(self.marker_ids)):
            raise ValueError("audio candidate marker ids must be unique.")
        for start, end in self.token_ranges:
            _token_range(start, end)
        if not isinstance(self.allows_eoa, bool):
            raise TypeError("audio candidate allows_eoa must be a bool.")


@dataclass(frozen=True)
class AudioTokenBlock:
    """One marker followed by one or more ordered payload-range cycles."""

    name: str
    marker_id: int | None
    token_ranges: tuple[TokenRange, ...]
    min_repeats: int = 1
    max_repeats: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("audio token block name must be non-empty.")
        if self.marker_id is not None:
            _non_negative_id(self.marker_id, "audio token block marker")
        if not self.token_ranges:
            raise ValueError("audio token blocks require at least one token range.")
        for start, end in self.token_ranges:
            _token_range(start, end)
        if (
            isinstance(self.min_repeats, bool)
            or not isinstance(self.min_repeats, int)
            or self.min_repeats < 1
        ):
            raise ValueError("audio token block min_repeats must be positive.")
        if self.max_repeats is not None and (
            isinstance(self.max_repeats, bool)
            or not isinstance(self.max_repeats, int)
            or self.max_repeats < self.min_repeats
        ):
            raise ValueError(
                "audio token block max_repeats must be at least min_repeats."
            )

    def contract_state(self) -> dict[str, object]:
        return {
            "name": self.name,
            "marker_id": self.marker_id,
            "token_ranges": [list(bounds) for bounds in self.token_ranges],
            "min_repeats": self.min_repeats,
            "max_repeats": self.max_repeats,
        }


@dataclass(frozen=True)
class AudioGrammarVariant:
    """One complete codec-private serialization accepted inside an audio span."""

    name: str
    blocks: tuple[AudioTokenBlock, ...]
    equal_repeat_groups: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("audio grammar variant name must be non-empty.")
        if not self.blocks:
            raise ValueError("audio grammar variants require at least one block.")
        names = tuple(block.name for block in self.blocks)
        if len(names) != len(set(names)):
            raise ValueError("audio grammar block names must be unique within a variant.")
        markers = tuple(
            block.marker_id for block in self.blocks if block.marker_id is not None
        )
        if len(markers) != len(set(markers)):
            raise ValueError("audio grammar marker ids must be unique within a variant.")
        if any(block.marker_id is None for block in self.blocks[1:]):
            raise ValueError(
                "audio grammar blocks after the first require explicit markers."
            )
        for marker in markers:
            if any(
                start <= marker < end
                for block in self.blocks
                for start, end in block.token_ranges
            ):
                raise ValueError("audio grammar markers must not overlap payload ranges.")
        grouped: set[str] = set()
        for group in self.equal_repeat_groups:
            if len(group) < 2 or len(group) != len(set(group)):
                raise ValueError(
                    "audio grammar equal-repeat groups require distinct block names."
                )
            if any(name not in names for name in group):
                raise ValueError(
                    "audio grammar equal-repeat groups must reference variant blocks."
                )
            reference_index = names.index(group[0])
            if any(names.index(name) <= reference_index for name in group[1:]):
                raise ValueError(
                    "audio grammar equal-repeat references must precede dependents."
                )
            overlap = grouped.intersection(group)
            if overlap:
                raise ValueError(
                    "audio grammar blocks may belong to only one equal-repeat group."
                )
            grouped.update(group)

    def contract_state(self) -> dict[str, object]:
        return {
            "name": self.name,
            "blocks": [block.contract_state() for block in self.blocks],
            "equal_repeat_groups": [
                list(group) for group in self.equal_repeat_groups
            ],
        }


@dataclass(frozen=True)
class AudioTokenGrammar:
    """Codec-private marker/range/order grammar for local audio token IDs.

    BOA, the audio-schema selector, and EOA belong to the outer modality grammar.
    This object validates only the codec payload between schema selection and EOA.
    """

    name: str
    variants: tuple[AudioGrammarVariant, ...]
    default_variant: str
    generation_variants: tuple[str, ...]
    prompt_continuations: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("audio token grammar name must be non-empty.")
        if not self.variants:
            raise ValueError("audio token grammar requires at least one variant.")
        names = tuple(variant.name for variant in self.variants)
        if len(names) != len(set(names)):
            raise ValueError("audio token grammar variant names must be unique.")
        if self.default_variant not in names:
            raise ValueError("audio token grammar default variant must be registered.")
        if not self.generation_variants:
            raise ValueError("audio token grammar requires generation variants.")
        if len(self.generation_variants) != len(set(self.generation_variants)):
            raise ValueError("audio grammar generation variants must be unique.")
        if any(name not in names for name in self.generation_variants):
            raise ValueError(
                "audio grammar generation variants must reference registered variants."
            )
        if self.default_variant not in self.generation_variants:
            raise ValueError(
                "audio grammar default variant must be a generation variant."
            )
        prompt_variants: set[str] = set()
        for continuation in self.prompt_continuations:
            if len(continuation) != 2:
                raise ValueError(
                    "audio grammar prompt continuations require variant pairs."
                )
            prompt_variant, generation_variant = continuation
            if prompt_variant not in names:
                raise ValueError(
                    "audio grammar prompt continuations must reference registered "
                    "prompt variants."
                )
            if generation_variant not in self.generation_variants:
                raise ValueError(
                    "audio grammar prompt continuations must target generation "
                    "variants."
                )
            if prompt_variant in prompt_variants:
                raise ValueError(
                    "audio grammar prompt variants may have only one continuation."
                )
            prompt_variants.add(prompt_variant)

    @property
    def private_marker_ids(self) -> tuple[int, ...]:
        values = {
            block.marker_id
            for variant in self.variants
            for block in variant.blocks
            if block.marker_id is not None
        }
        return tuple(sorted(values))

    def contract_state(self) -> dict[str, object]:
        return {
            "grammar": self.name,
            "default_variant": self.default_variant,
            "generation_variants": list(self.generation_variants),
            "prompt_continuations": [
                list(continuation) for continuation in self.prompt_continuations
            ],
            "private_marker_ids": list(self.private_marker_ids),
            "variants": [variant.contract_state() for variant in self.variants],
        }

    def viable_variants(
        self,
        token_ids: Sequence[int] | Tensor,
        *,
        variants: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        """Return selected generation variants that can extend this prefix."""
        values = _local_ids(token_ids)
        return tuple(
            variant.name
            for variant in self._selected_variants(variants)
            if _is_viable(values, variant)
        )

    def next_candidates(
        self,
        token_ids: Sequence[int] | Tensor,
        *,
        variants: Sequence[str] | None = None,
    ) -> AudioTokenCandidates:
        """Return legal markers/ranges/EOA without expanding the codec vocabulary."""
        values = _local_ids(token_ids)
        candidates: list[AudioTokenCandidates] = []
        errors: list[ValueError] = []
        for variant in self._selected_variants(variants):
            try:
                candidates.append(_variant_candidates(values, variant))
            except ValueError as error:
                errors.append(error)
        if not candidates:
            detail = "" if not errors else f" {errors[0]}"
            raise ValueError(
                "audio codec prefix does not match any selected generation "
                f"variant.{detail}"
            )
        return _merge_candidates(candidates)

    def generation_variant(
        self,
        prompt_payloads: Sequence[Sequence[int] | Tensor],
    ) -> str:
        """Resolve the response variant from complete codec-local prompt spans."""
        if isinstance(prompt_payloads, (str, bytes)) or not isinstance(
            prompt_payloads, Sequence
        ):
            raise TypeError("audio prompt payloads must be a sequence of payloads.")
        continuations = dict(self.prompt_continuations)
        selected: set[str] = set()
        for index, payload in enumerate(prompt_payloads):
            variant = self._prompt_variant(_local_ids(payload), index=index)
            continuation = continuations.get(variant.name)
            if continuation is not None:
                selected.add(continuation)
        if not selected:
            return self.default_variant
        if len(selected) != 1:
            labels = ", ".join(sorted(selected))
            raise ValueError(
                "audio prompt payloads require conflicting generation variants: "
                f"{labels}."
            )
        return next(iter(selected))

    def validate_prefix(
        self,
        token_ids: Sequence[int] | Tensor,
        *,
        variant: str | None = None,
    ) -> None:
        """Reject a local codec-token prefix that cannot complete the variant."""
        _parse(_local_ids(token_ids), self._variant(variant))

    def is_complete(
        self,
        token_ids: Sequence[int] | Tensor,
        *,
        variant: str | None = None,
    ) -> bool:
        """Return whether EOA may legally follow this local codec payload."""
        return _parse(_local_ids(token_ids), self._variant(variant))

    def validate_complete(
        self,
        token_ids: Sequence[int] | Tensor,
        *,
        variant: str | None = None,
    ) -> None:
        if not self.is_complete(token_ids, variant=variant):
            raise ValueError("audio codec payload ended before its grammar was complete.")

    def validate_next(
        self,
        token_ids: Sequence[int] | Tensor,
        next_token_id: int,
        *,
        eoa_token_id: int,
        variant: str | None = None,
    ) -> None:
        """Validate one proposed local codec token or the outer EOA token."""
        next_id = _non_negative_id(next_token_id, "next audio token")
        eoa_id = _non_negative_id(eoa_token_id, "audio EOA token")
        values = _local_ids(token_ids)
        if next_id == eoa_id:
            if not _parse(values, self._variant(variant)):
                raise ValueError(
                    "audio EOA is not allowed before the codec grammar is complete."
                )
            return
        _parse((*values, next_id), self._variant(variant))

    def _variant(self, name: str | None) -> AudioGrammarVariant:
        selected = self.default_variant if name is None else name
        if not isinstance(selected, str) or not selected:
            raise ValueError("audio grammar variant must be a non-empty string.")
        for variant in self.variants:
            if variant.name == selected:
                return variant
        available = ", ".join(variant.name for variant in self.variants)
        raise ValueError(
            f"unknown audio grammar variant {selected!r}; available: {available}."
        )

    def _selected_variants(
        self,
        names: Sequence[str] | None,
    ) -> tuple[AudioGrammarVariant, ...]:
        selected = self.generation_variants if names is None else names
        if isinstance(selected, (str, bytes)):
            raise TypeError("audio grammar variants must be a sequence of names.")
        values = tuple(selected)
        if not values:
            raise ValueError("audio grammar variant selection must not be empty.")
        if len(values) != len(set(values)):
            raise ValueError("audio grammar variant selection must be unique.")
        return tuple(self._variant(name) for name in values)

    def _prompt_variant(
        self,
        token_ids: tuple[int, ...],
        *,
        index: int,
    ) -> AudioGrammarVariant:
        matches = tuple(
            variant
            for variant in self.variants
            if _is_complete_variant(token_ids, variant)
        )
        if not matches:
            raise ValueError(
                f"audio prompt payload {index} does not match a complete codec "
                "grammar variant."
            )
        block_count = max(len(variant.blocks) for variant in matches)
        specific = tuple(
            variant for variant in matches if len(variant.blocks) == block_count
        )
        if len(specific) != 1:
            labels = ", ".join(variant.name for variant in specific)
            raise ValueError(
                f"audio prompt payload {index} is ambiguous across variants: "
                f"{labels}."
            )
        return specific[0]


@dataclass(frozen=True)
class AudioTokenSpec:
    """Immutable identity and executable grammar of one codec token schema."""

    schema_id: str
    codec_name: str
    sequence_layout: str
    tokenizer: AudioTokenizer
    grammar: AudioTokenGrammar
    tokenizer_state_sha256: str

    @classmethod
    def create(
        cls,
        *,
        codec_name: str,
        sequence_layout: str,
        tokenizer: AudioTokenizer,
    ) -> AudioTokenSpec:
        if not isinstance(codec_name, str) or not codec_name:
            raise ValueError("audio token schema codec_name must be non-empty.")
        if not isinstance(sequence_layout, str) or not sequence_layout:
            raise ValueError("audio token schema sequence_layout must be non-empty.")
        state = tokenizer.contract_state()
        if not isinstance(state, Mapping):
            raise TypeError("audio tokenizer contract_state() must return a mapping.")
        tokenizer_grammar = state.get("grammar")
        if not isinstance(tokenizer_grammar, str) or not tokenizer_grammar:
            raise ValueError("audio tokenizer contract requires a non-empty grammar.")
        grammar = tokenizer.grammar
        if not isinstance(grammar, AudioTokenGrammar):
            raise TypeError("audio tokenizer grammar must be an AudioTokenGrammar.")
        serialized = json.dumps(
            {
                "tokenizer": state,
                "codec_grammar": grammar.contract_state(),
            },
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        return cls(
            schema_id=f"{codec_name}:{tokenizer_grammar}:{digest[:12]}",
            codec_name=codec_name,
            sequence_layout=sequence_layout,
            tokenizer=tokenizer,
            grammar=grammar,
            tokenizer_state_sha256=digest,
        )

    @property
    def selector(self) -> str:
        return f"<audio_schema:{self.schema_id}>"

    def contract_state(self) -> dict[str, object]:
        return {
            "grammar": "audio-token-spec-v2",
            "schema_id": self.schema_id,
            "codec_name": self.codec_name,
            "sequence_layout": self.sequence_layout,
            "selector": self.selector,
            "tokenizer_grammar": self.tokenizer.contract_state()["grammar"],
            "tokenizer_state_sha256": self.tokenizer_state_sha256,
            "codec_grammar": self.grammar.contract_state(),
        }

    def viable_variants(
        self,
        token_ids: Sequence[int] | Tensor,
        *,
        variants: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        return self.grammar.viable_variants(token_ids, variants=variants)

    def next_candidates(
        self,
        token_ids: Sequence[int] | Tensor,
        *,
        variants: Sequence[str] | None = None,
    ) -> AudioTokenCandidates:
        return self.grammar.next_candidates(token_ids, variants=variants)

    def generation_variant(
        self,
        prompt_payloads: Sequence[Sequence[int] | Tensor],
    ) -> str:
        return self.grammar.generation_variant(prompt_payloads)

    def validate_prefix(
        self,
        token_ids: Sequence[int] | Tensor,
        *,
        variant: str | None = None,
    ) -> None:
        self.grammar.validate_prefix(token_ids, variant=variant)

    def allows_eoa(
        self,
        token_ids: Sequence[int] | Tensor,
        *,
        variant: str | None = None,
    ) -> bool:
        return self.grammar.is_complete(token_ids, variant=variant)

    def validate_complete(
        self,
        token_ids: Sequence[int] | Tensor,
        *,
        variant: str | None = None,
    ) -> None:
        self.grammar.validate_complete(token_ids, variant=variant)

    def validate_next(
        self,
        token_ids: Sequence[int] | Tensor,
        next_token_id: int,
        *,
        eoa_token_id: int,
        variant: str | None = None,
    ) -> None:
        self.grammar.validate_next(
            token_ids,
            next_token_id,
            eoa_token_id=eoa_token_id,
            variant=variant,
        )


@dataclass(frozen=True)
class AudioTokenRegistry:
    """Append-only schema registry; a runtime currently selects one default."""

    specs: tuple[AudioTokenSpec, ...]
    default_schema_id: str

    def __post_init__(self) -> None:
        if not self.specs:
            raise ValueError("audio token registry requires at least one schema.")
        ids = tuple(spec.schema_id for spec in self.specs)
        if len(ids) != len(set(ids)):
            raise ValueError("audio token registry schema ids must be unique.")
        if self.default_schema_id not in ids:
            raise ValueError("audio token registry default schema is not registered.")

    @property
    def default(self) -> AudioTokenSpec:
        return self.get(self.default_schema_id)

    def get(self, schema_id: str) -> AudioTokenSpec:
        if not isinstance(schema_id, str) or not schema_id:
            raise ValueError("audio schema id must be a non-empty string.")
        for spec in self.specs:
            if spec.schema_id == schema_id:
                return spec
        raise KeyError(f"unknown audio token schema: {schema_id!r}.")


def _parse(token_ids: tuple[int, ...], variant: AudioGrammarVariant) -> bool:
    return _variant_candidates(token_ids, variant).allows_eoa


def _variant_candidates(
    token_ids: tuple[int, ...],
    variant: AudioGrammarVariant,
) -> AudioTokenCandidates:
    cursor = 0
    repeats: dict[str, int] = {}
    equal_reference = {
        name: group[0]
        for group in variant.equal_repeat_groups
        for name in group[1:]
    }
    for block_index, block in enumerate(variant.blocks):
        if block.marker_id is not None:
            if cursor == len(token_ids):
                return AudioTokenCandidates(marker_ids=(block.marker_id,))
            if token_ids[cursor] != block.marker_id:
                raise ValueError(
                    f"audio codec block {block.name!r} must begin with marker "
                    f"{block.marker_id}."
                )
            cursor += 1

        count = 0
        pattern_index = 0
        next_marker = (
            variant.blocks[block_index + 1].marker_id
            if block_index + 1 < len(variant.blocks)
            else None
        )
        effective_min = block.min_repeats
        effective_max = block.max_repeats
        reference = equal_reference.get(block.name)
        if reference is not None:
            reference_count = repeats[reference]
            effective_min = max(effective_min, reference_count)
            effective_max = reference_count

        while cursor < len(token_ids):
            token_id = token_ids[cursor]
            if (
                pattern_index == 0
                and count >= effective_min
                and next_marker is not None
                and token_id == next_marker
            ):
                break
            if effective_max is not None and count >= effective_max:
                break
            start, end = block.token_ranges[pattern_index]
            if not start <= token_id < end:
                raise ValueError(
                    f"audio codec block {block.name!r} expects a token in "
                    f"[{start}, {end}), got {token_id}."
                )
            cursor += 1
            pattern_index += 1
            if pattern_index == len(block.token_ranges):
                pattern_index = 0
                count += 1

        repeats[block.name] = count
        if cursor == len(token_ids):
            if pattern_index != 0:
                return AudioTokenCandidates(
                    token_ranges=(block.token_ranges[pattern_index],)
                )
            can_repeat = effective_max is None or count < effective_max
            ranges = (block.token_ranges[0],) if can_repeat else ()
            if block_index + 1 < len(variant.blocks):
                markers = (
                    (next_marker,)
                    if count >= effective_min and next_marker is not None
                    else ()
                )
                return AudioTokenCandidates(
                    marker_ids=markers,
                    token_ranges=ranges,
                )
            return AudioTokenCandidates(
                token_ranges=ranges,
                allows_eoa=count >= effective_min,
            )
        if pattern_index != 0 or count < effective_min:
            raise ValueError(f"audio codec block {block.name!r} has incomplete payload.")

    if cursor != len(token_ids):
        raise ValueError(
            f"audio codec payload contains an unexpected token at index {cursor}."
        )
    raise AssertionError("audio codec candidate parser did not terminate at the prefix.")


def _is_viable(token_ids: tuple[int, ...], variant: AudioGrammarVariant) -> bool:
    try:
        _variant_candidates(token_ids, variant)
    except ValueError:
        return False
    return True


def _is_complete_variant(
    token_ids: tuple[int, ...],
    variant: AudioGrammarVariant,
) -> bool:
    try:
        return _variant_candidates(token_ids, variant).allows_eoa
    except ValueError:
        return False


def _merge_candidates(
    candidates: Sequence[AudioTokenCandidates],
) -> AudioTokenCandidates:
    markers = tuple(
        sorted({marker for candidate in candidates for marker in candidate.marker_ids})
    )
    ranges = sorted(
        {
            bounds
            for candidate in candidates
            for bounds in candidate.token_ranges
        }
    )
    merged: list[TokenRange] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return AudioTokenCandidates(
        marker_ids=markers,
        token_ranges=tuple(merged),
        allows_eoa=any(candidate.allows_eoa for candidate in candidates),
    )


def _local_ids(token_ids: Sequence[int] | Tensor) -> tuple[int, ...]:
    if isinstance(token_ids, Tensor):
        if not is_signed_integer_dtype(token_ids.dtype):
            raise TypeError("audio codec token ids must use a signed integer dtype.")
        if token_ids.dim() != 1:
            raise ValueError("audio codec token ids must have shape [tokens].")
        values = token_ids.detach().cpu().tolist()
    else:
        values = list(token_ids)
    return tuple(
        _non_negative_id(value, "audio codec token") for value in values
    )


def _non_negative_id(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer.")
    token_id = int(value)
    if token_id < 0:
        raise ValueError(f"{name} must be non-negative.")
    return token_id


def _token_range(start: object, end: object) -> None:
    left = _non_negative_id(start, "audio token range start")
    right = _non_negative_id(end, "audio token range end")
    if right <= left:
        raise ValueError("audio token ranges must be non-empty.")


__all__ = [
    "AudioGrammarVariant",
    "AudioTokenBlock",
    "AudioTokenCandidates",
    "AudioTokenGrammar",
    "AudioTokenRegistry",
    "AudioTokenSpec",
    "TokenRange",
]
