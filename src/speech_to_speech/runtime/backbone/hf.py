from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cached_property, partial
from inspect import Parameter, signature
from numbers import Integral
from typing import Protocol, cast

import torch
from anydataset.types import Modality
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer

from ..types import Backbone, BackboneOutput, BackboneReadout, TextTokenizer
from .adapter import BackboneBodyAdapter, BackboneExtra, BackboneOutputView
from .config import AdapterConfig, BackboneInitialization, BackboneType
from .kimi import (
    KimiRawTokenizer,
    KimiTokenizerAdapter,
    call_kimi_body,
    should_checkpoint_kimi_body,
)


@dataclass(frozen=True)
class HuggingFaceBackboneAdapter:
    config: AdapterConfig

    @cached_property
    def text_tokenizer(self) -> TextTokenizer:
        if self.config.type is BackboneType.QWEN2_5_OMNI_TEXT:
            processor = AutoProcessor.from_pretrained(
                self.config.path,
                trust_remote_code=self.config.trust_remote_code,
            )
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise TypeError("Qwen2.5-Omni processor must expose a text tokenizer.")
        elif self.config.type is BackboneType.KIMI_AUDIO:
            raw_tokenizer = AutoTokenizer.from_pretrained(
                self.config.path,
                trust_remote_code=self.config.trust_remote_code,
            )
            tokenizer = KimiTokenizerAdapter(
                cast(KimiRawTokenizer, raw_tokenizer),
                chat_template=self.config.chat_template,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.path,
                trust_remote_code=self.config.trust_remote_code,
            )
        if (
            self.config.chat_template is not None
            and not isinstance(tokenizer, KimiTokenizerAdapter)
        ):
            setattr(tokenizer, "chat_template", self.config.chat_template)
        bind_chat_bos(tokenizer)
        return cast(TextTokenizer, cast(object, tokenizer))

    @cached_property
    def root_model(self) -> nn.Module:
        kwargs = {}
        if self.config.dtype is not None:
            kwargs["dtype"] = dtype(self.config.dtype)
        if self.config.attn_implementation is not None:
            kwargs["attn_implementation"] = self.config.attn_implementation
        model = (
            self._pretrained(**kwargs)
            if self.config.initialization is BackboneInitialization.PRETRAINED
            else self._random(**kwargs)
        )
        if self.config.type is BackboneType.KIMI_AUDIO:
            model = _prepare_kimi_body(model)
        if self.config.gradient_checkpointing:
            if self.config.type is BackboneType.KIMI_AUDIO:
                _disable_cache(cast(nn.Module, model))
            else:
                _enable_gradient_checkpointing(model)
        if self.config.device is not None:
            model = cast(nn.Module, cast(object, model)).to(self.config.device)
        return cast(nn.Module, cast(object, model))

    @cached_property
    def model(self) -> Backbone:
        return cast(Backbone, cast(object, _path(self.root_model, self.config.module)))

    @cached_property
    def hidden_size(self) -> int:
        return _hidden_size(getattr(self.model, "config", None))

    @property
    def has_modality_readouts(self) -> bool:
        return self.body.has_modality_readouts

    @cached_property
    def body(self) -> BackboneBodyAdapter:
        target = _path(cast(object, self.model), self.config.body)
        if not callable(target):
            raise TypeError(f"backbone body path {self.config.body!r} is not callable.")
        body = cast(Callable[..., BackboneOutput], target)
        if self.config.type is BackboneType.KIMI_AUDIO:
            body = _kimi_body_callable(
                self.model,
                body,
                enabled=self.config.gradient_checkpointing,
            )
        return BackboneBodyAdapter(
            body,
            readout=BackboneReadout(self.config.readout),
            supports_cache_position=self.config.supports_cache_position,
            modality_readouts=_modality_readouts(self.config.readouts),
        )

    def input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def contract_state(self) -> Mapping[str, object]:
        return {
            "grammar": "huggingface-backbone-v1",
            "type": self.config.type.value,
            "source": self.config.path,
            "module": self.config.module,
            "body": self.config.body,
            "execution": self.body.contract_state(),
        }

    def encode(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        output_hidden_states: bool,
        past_key_values: Cache | None = None,
        use_cache: bool = False,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        modality: Modality | None = None,
        extra: BackboneExtra | None = None,
    ) -> BackboneOutputView:
        return self.body.encode(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_ids=position_ids,
            cache_position=cache_position,
            modality=modality,
            extra=extra,
        )

    def _pretrained(self, **kwargs: object) -> object:
        if self.config.type is BackboneType.QWEN2_5_OMNI_TEXT:
            config = AutoConfig.from_pretrained(
                self.config.path,
                trust_remote_code=self.config.trust_remote_code,
            )
            return _omni_text_model_factory().from_pretrained(
                self.config.path,
                config=_omni_text_config(config),
                key_mapping={r"^thinker\.model\.": ""},
                trust_remote_code=self.config.trust_remote_code,
                **kwargs,
            )
        return AutoModel.from_pretrained(
            self.config.path,
            trust_remote_code=self.config.trust_remote_code,
            **kwargs,
        )

    def _random(self, **kwargs: object) -> object:
        config = AutoConfig.from_pretrained(
            self.config.path,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.config.type is BackboneType.QWEN2_5_OMNI_TEXT:
            return _omni_text_model_factory()._from_config(
                _omni_text_config(config),
                **kwargs,
            )
        return AutoModel.from_config(
            config,
            trust_remote_code=self.config.trust_remote_code,
            **kwargs,
        )


def create(config: AdapterConfig) -> HuggingFaceBackboneAdapter:
    return HuggingFaceBackboneAdapter(config)


class _OmniModelFactory(Protocol):
    def from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        **kwargs: object,
    ) -> object: ...

    def _from_config(self, config: object, **kwargs: object) -> object: ...


def _omni_text_model_factory() -> _OmniModelFactory:
    try:
        from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerTextModel
    except ImportError as error:
        raise RuntimeError(
            "Qwen2.5-Omni text backbone requires a transformers build that "
            "exposes Qwen2_5OmniThinkerTextModel."
        ) from error
    return cast(_OmniModelFactory, Qwen2_5OmniThinkerTextModel)


def _omni_text_config(config: object) -> PretrainedConfig:
    thinker = getattr(config, "thinker_config", None)
    text = getattr(thinker, "text_config", None)
    if not isinstance(text, PretrainedConfig):
        raise TypeError(
            "Qwen2.5-Omni config must expose thinker_config.text_config."
        )
    return text


def _prepare_kimi_body(model: object) -> nn.Module:
    if not isinstance(model, nn.Module):
        raise TypeError("Kimi-Audio model body must be a torch.nn.Module.")
    for name in ("lm_head", "mimo_output"):
        if getattr(model, name, None) is not None:
            raise TypeError(
                "Kimi-Audio backbone must load the base AutoModel without "
                f"the {name} output head."
            )
    vq_adaptor = getattr(model, "vq_adaptor", None)
    if vq_adaptor is not None and not isinstance(vq_adaptor, nn.Module):
        raise TypeError("Kimi-Audio vq_adaptor must be a torch.nn.Module or None.")
    if hasattr(model, "vq_adaptor"):
        setattr(model, "vq_adaptor", None)
    return model


def _modality_readouts(
    readouts: Mapping[str, str],
) -> dict[Modality, BackboneReadout]:
    return {
        Modality(modality): BackboneReadout(path)
        for modality, path in readouts.items()
    }


def _kimi_body_callable(
    model: object,
    body: Callable[..., object],
    *,
    enabled: bool,
) -> Callable[..., BackboneOutput]:
    if not isinstance(model, nn.Module):
        raise TypeError("Kimi-Audio model body must be a torch.nn.Module.")

    def call(**kwargs: object) -> BackboneOutput:
        kwargs["return_dict"] = True
        output = call_kimi_body(
            body,
            checkpointed=should_checkpoint_kimi_body(model, enabled),
            **kwargs,
        )
        return cast(BackboneOutput, output)

    return call


def _path(root: object, path: str) -> object:
    current = root
    for item in (part for part in path.split(".") if part):
        if not hasattr(current, item):
            raise AttributeError(f"backbone object has no attribute {item!r}.")
        current = getattr(current, item)
    return current


def _hidden_size(config: object) -> int:
    for item in (
        config,
        getattr(config, "text_config", None),
        getattr(getattr(config, "thinker_config", None), "text_config", None),
    ):
        value = getattr(item, "hidden_size", None)
        if isinstance(value, bool) or not isinstance(value, Integral):
            continue
        size = int(value)
        if size <= 0:
            raise ValueError("backbone hidden size must be positive.")
        return size
    raise AttributeError("backbone config does not expose a text hidden size.")


def _enable_gradient_checkpointing(model: object) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError(
            "runtime.gradient_checkpointing requires a Hugging Face "
            "backbone module."
        )

    backbones = _gradient_checkpointing_backbones(model)
    if backbones:
        for backbone in backbones:
            _enable_backbone_gradient_checkpointing(backbone)
    else:
        _enable_layer_gradient_checkpointing(model)
    _enable_input_require_grads(model)
    _disable_cache(model)


def _gradient_checkpointing_backbones(model: nn.Module) -> tuple[PreTrainedModel, ...]:
    if isinstance(model, PreTrainedModel):
        return (model,)

    backbones: list[PreTrainedModel] = []

    def visit(module: nn.Module) -> None:
        for child in module.children():
            if isinstance(child, PreTrainedModel):
                backbones.append(child)
            else:
                visit(child)

    visit(model)
    return tuple(backbones)


def _enable_backbone_gradient_checkpointing(model: PreTrainedModel) -> None:
    if not model.supports_gradient_checkpointing:
        raise TypeError(
            "backbone does not support Hugging Face gradient checkpointing; "
            "disable runtime.gradient_checkpointing for this backbone."
        )
    if not callable(model.gradient_checkpointing_enable):
        raise TypeError(
            "backbone does not expose gradient_checkpointing_enable(); "
            "disable runtime.gradient_checkpointing for this backbone."
        )
    if not _accepts_gradient_checkpointing_kwargs(model.gradient_checkpointing_enable):
        raise TypeError(
            "backbone gradient_checkpointing_enable() must accept "
            "gradient_checkpointing_kwargs; upgrade transformers or disable "
            "runtime.gradient_checkpointing for this backbone."
        )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )


def _accepts_gradient_checkpointing_kwargs(callback: Callable[..., object]) -> bool:
    parameters = signature(callback).parameters.values()
    return any(
        parameter.kind is Parameter.VAR_KEYWORD
        or (
            parameter.name == "gradient_checkpointing_kwargs"
            and parameter.kind
            in (Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY)
        )
        for parameter in parameters
    )


def _enable_layer_gradient_checkpointing(model: nn.Module) -> None:
    modules = tuple(
        module
        for module in model.modules()
        if isinstance(module, GradientCheckpointingLayer)
    )
    if not modules:
        raise TypeError(
            "backbone does not expose Hugging Face gradient checkpointing layers; "
            "disable runtime.gradient_checkpointing for this backbone."
        )
    gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
    for module in modules:
        setattr(module, "_gradient_checkpointing_func", gradient_checkpointing_func)
        module.gradient_checkpointing = True


def _enable_input_require_grads(model: nn.Module) -> None:
    hook = getattr(model, "enable_input_require_grads", None)
    if callable(hook):
        hook()
        return

    for module in model.children():
        hook = getattr(module, "enable_input_require_grads", None)
        if callable(hook):
            hook()
            continue
        _enable_input_require_grads(module)


def _disable_cache(model: nn.Module) -> None:
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None and hasattr(config, "use_cache"):
            config.use_cache = False


# Qwen chat turn start; stock HF leaves bos_token unset while keeping this in vocab.
_CHAT_BOS_TOKEN = "<|im_start|>"


def bind_chat_bos(tokenizer: object) -> None:
    """Expose chat turn-start as bos when the tokenizer leaves bos unset."""
    if getattr(tokenizer, "bos_token_id", None) is not None:
        return
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return
    try:
        token_id = convert(_CHAT_BOS_TOKEN)
    except (AttributeError, NotImplementedError):
        return
    if isinstance(token_id, bool) or not isinstance(token_id, Integral):
        return
    token_id = int(token_id)
    if token_id < 0:
        return
    unk = getattr(tokenizer, "unk_token_id", None)
    if unk is not None and token_id == int(unk):
        return
    setattr(tokenizer, "bos_token", _CHAT_BOS_TOKEN)


def dtype(value: str) -> torch.dtype:
    try:
        result = getattr(torch, value)
    except AttributeError as error:
        raise ValueError(f"unknown torch dtype: {value}") from error
    if not isinstance(result, torch.dtype):
        raise ValueError(f"unknown torch dtype: {value}")
    return result


__all__ = [
    "HuggingFaceBackboneAdapter",
    "bind_chat_bos",
    "create",
    "dtype",
]
