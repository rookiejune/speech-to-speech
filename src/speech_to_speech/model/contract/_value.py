from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from anydataset.types import Modality

from ._protocol import ContractStateProvider, ConfigStateProvider
from .core import canonical_value


HF_NON_ARCHITECTURE_CONFIG_FIELDS = frozenset(
    {
        "_attn_implementation",
        "_attn_implementation_internal",
        "_commit_hash",
        "_name_or_path",
        "architectures",
        "attn_implementation",
        "auto_map",
        "bad_words_ids",
        "begin_suppress_tokens",
        "diversity_penalty",
        "do_sample",
        "dtype",
        "early_stopping",
        "encoder_no_repeat_ngram_size",
        "exponential_decay_length_penalty",
        "finetuning_task",
        "forced_bos_token_id",
        "forced_eos_token_id",
        "gradient_checkpointing",
        "id2label",
        "initializer_range",
        "label2id",
        "length_penalty",
        "max_length",
        "min_length",
        "no_repeat_ngram_size",
        "num_beam_groups",
        "num_beams",
        "num_return_sequences",
        "output_attentions",
        "output_hidden_states",
        "output_scores",
        "problem_type",
        "remove_invalid_values",
        "repetition_penalty",
        "return_dict",
        "return_dict_in_generate",
        "suppress_tokens",
        "task_specific_params",
        "temperature",
        "tokenizer_class",
        "top_k",
        "top_p",
        "torch_dtype",
        "transformers_version",
        "typical_p",
        "use_cache",
    }
)

def contract_state(value: ContractStateProvider) -> dict[str, Any]:
    state = canonical_value(value.contract_state())
    if not isinstance(state, dict):
        raise TypeError("contract_state() must return a mapping.")
    state_grammar(state)
    return state


def state_grammar(state: Mapping[str, Any]) -> str:
    grammar = state.get("grammar")
    if not isinstance(grammar, str) or not grammar:
        raise ValueError("contract state requires a non-empty grammar.")
    return grammar




def backbone_config_state(config: object) -> dict[str, Any]:
    if isinstance(config, ConfigStateProvider):
        state = config.to_dict()
    else:
        try:
            state = vars(config)
        except TypeError as error:
            raise TypeError(
                "backbone config must expose to_dict() or attributes."
            ) from error
    if not isinstance(state, Mapping):
        raise TypeError("backbone config state must be a mapping.")
    canonical = semantic_config_value(state)
    if not isinstance(canonical, dict):
        raise TypeError("backbone config state must resolve to a mapping.")
    return canonical


def semantic_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        result = {
            semantic_config_key(key): semantic_config_value(item)
            for key, item in value.items()
            if not (
                isinstance(key, str)
                and key in HF_NON_ARCHITECTURE_CONFIG_FIELDS
            )
        }
        return {key: result[key] for key in sorted(result)}
    if isinstance(value, (list, tuple)):
        return [semantic_config_value(item) for item in value]
    return canonical_value(value)


def semantic_config_key(value: object) -> str:
    if isinstance(value, str):
        if value.startswith(("int:", "str:")):
            return f"str:{value}"
        return value
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("backbone config mapping keys must be strings or integers.")
    return f"int:{value}"




def token_block(
    value: object,
    modality: Modality,
) -> tuple[int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 2
    ):
        raise TypeError(f"token layout block {modality.value!r} must contain two ids.")
    start = non_negative_int(value[0], f"{modality.value} token block start")
    end = positive_int(value[1], f"{modality.value} token block end")
    if end <= start:
        raise ValueError(f"{modality.value} token block must be non-empty.")
    return start, end


def config_positive_int(
    config: Mapping[str, Any],
    key: str,
    name: str,
) -> int:
    return positive_int(config.get(key), name)


def positive_sizes(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of integers.")
    sizes = tuple(positive_int(item, f"{name} size") for item in value)
    if not sizes:
        raise ValueError(f"{name} must not be empty.")
    return sizes


def positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


def non_negative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def optional_sha256(value: object, name: str) -> str | None:
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest.")
    return value




def qualified_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"

__all__ = [
    "backbone_config_state",
    "config_positive_int",
    "contract_state",
    "non_negative_int",
    "optional_sha256",
    "positive_int",
    "positive_sizes",
    "qualified_name",
    "state_grammar",
    "token_block",
]
