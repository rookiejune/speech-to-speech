from __future__ import annotations

import unittest
from collections.abc import Callable, Collection, Sequence
from types import SimpleNamespace
from typing import Literal, cast
from unittest.mock import patch

import torch
from anydataset.types import Modality
from torch import Tensor, nn

from speech_to_speech.runtime.backbone import hf
from speech_to_speech.runtime.backbone.config import (
    AdapterConfig,
    BackboneInitialization,
    BackboneType,
)
from speech_to_speech.runtime.backbone.kimi import KimiTokenizerAdapter


_PROJECT_CHAT_TEMPLATE = (
    '{{ messages | map(attribute="content") | join("\\n") }}'
    '{% if add_generation_prompt %}{{ "\\n" }}{% endif %}'
)


class _RawTokenizer:
    special_tokens = {"[BOS]": 0, "[EOS]": 1, "<pad>": 2}
    bos_id = 0
    eos_id = 1
    pad_id = 2
    vocab_size = 16

    def encode(
        self,
        text: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Literal["all"] | Collection[str] = (),
        disallowed_special: Collection[str] = (),
    ) -> list[int]:
        del text, allowed_special, disallowed_special
        values = [4, 5]
        if bos:
            values.insert(0, self.bos_id)
        if eos:
            values.append(self.eos_id)
        return values

    def decode(self, token_ids: Sequence[int]) -> str:
        return ",".join(str(token_id) for token_id in token_ids)


class _KimiBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4, use_cache=True)
        self.embed_tokens = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.mimo_layers = nn.ModuleList([nn.Linear(4, 4)])
        self.vq_adaptor: nn.Module | None = nn.Linear(4, 4)
        self.calls: list[dict[str, object]] = []

    @property
    def base_model(self) -> _KimiBody:
        return self

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None,
        output_hidden_states: bool,
        past_key_values: object | None,
        use_cache: bool,
        position_ids: Tensor | None,
        return_dict: bool,
    ) -> SimpleNamespace:
        self.calls.append(
            {
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "output_hidden_states": output_hidden_states,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "position_ids": position_ids,
                "return_dict": return_dict,
            }
        )
        return SimpleNamespace(
            last_hidden_state=(inputs_embeds + 1, inputs_embeds + 2),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


def _config(**kwargs: object) -> AdapterConfig:
    values: dict[str, object] = {
        "type": BackboneType.KIMI_AUDIO,
        "path": "fake/kimi-audio",
        "trust_remote_code": True,
        "chat_template": _PROJECT_CHAT_TEMPLATE,
        "readout": "last_hidden_state[0]",
        "readouts": {
            "text": "last_hidden_state[0]",
            "audio": "last_hidden_state[1]",
        },
        "supports_cache_position": False,
        "module": "",
        "body": "base_model",
    }
    values.update(kwargs)
    return AdapterConfig(**values)  # type: ignore[arg-type]


class KimiHuggingFaceBackboneTest(unittest.TestCase):
    def test_loads_remote_tokenizer_and_wraps_keyword_api(self) -> None:
        raw = _RawTokenizer()
        adapter = hf.HuggingFaceBackboneAdapter(_config())

        with (
            patch.object(hf.AutoTokenizer, "from_pretrained", return_value=raw) as load,
            patch.object(hf.AutoProcessor, "from_pretrained") as load_processor,
        ):
            tokenizer = adapter.text_tokenizer

        load.assert_called_once_with("fake/kimi-audio", trust_remote_code=True)
        load_processor.assert_not_called()
        self.assertIsInstance(tokenizer, KimiTokenizerAdapter)
        kimi_tokenizer = cast(KimiTokenizerAdapter, tokenizer)
        self.assertIs(kimi_tokenizer.raw, raw)
        self.assertEqual(
            kimi_tokenizer.encode("hello", add_special_tokens=True),
            [0, 4, 5, 1],
        )
        self.assertEqual(
            kimi_tokenizer.apply_chat_template(
                [{"role": "user", "content": "hello"}],
                add_generation_prompt=True,
            ),
            "hello\n",
        )

    def test_pretrained_loads_base_body_without_output_heads(self) -> None:
        body = _KimiBody()
        input_embeddings = body.get_input_embeddings()
        retained_parameters = {
            name: parameter
            for name, parameter in body.named_parameters()
            if not name.startswith("vq_adaptor.")
        }
        adapter = hf.HuggingFaceBackboneAdapter(
            _config(
                dtype="bfloat16",
                attn_implementation="flash_attention_2",
            )
        )

        with (
            patch.object(
                hf.AutoModel,
                "from_pretrained",
                return_value=body,
            ) as load,
            patch.object(hf.AutoModel, "from_config") as load_random,
            patch.object(hf.AutoModelForCausalLM, "from_pretrained") as load_causal,
        ):
            loaded = adapter.model

        load.assert_called_once_with(
            "fake/kimi-audio",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        load_random.assert_not_called()
        load_causal.assert_not_called()
        self.assertIs(adapter.root_model, body)
        self.assertIs(loaded, body)
        self.assertIsNone(body.vq_adaptor)
        self.assertIs(adapter.input_embeddings(), input_embeddings)
        loaded_parameters = dict(body.named_parameters())
        self.assertEqual(tuple(loaded_parameters), tuple(retained_parameters))
        for name, parameter in retained_parameters.items():
            self.assertIs(loaded_parameters[name], parameter)
        self.assertIn("layers.0.weight", body.state_dict())
        self.assertIn("mimo_layers.0.weight", body.state_dict())
        self.assertFalse(any(name.startswith("vq_adaptor.") for name in body.state_dict()))

    def test_random_initialization_uses_remote_base_auto_model(self) -> None:
        body = _KimiBody()
        remote_config = SimpleNamespace(hidden_size=4)
        adapter = hf.HuggingFaceBackboneAdapter(
            _config(initialization=BackboneInitialization.RANDOM)
        )

        with (
            patch.object(
                hf.AutoConfig,
                "from_pretrained",
                return_value=remote_config,
            ) as load_config,
            patch.object(
                hf.AutoModel,
                "from_config",
                return_value=body,
            ) as load,
            patch.object(hf.AutoModelForCausalLM, "from_config") as load_causal,
        ):
            loaded = adapter.model

        load_config.assert_called_once_with(
            "fake/kimi-audio",
            trust_remote_code=True,
        )
        load.assert_called_once_with(
            remote_config,
            trust_remote_code=True,
        )
        load_causal.assert_not_called()
        self.assertIs(loaded, body)
        self.assertIsNone(body.vq_adaptor)

    def test_encode_routes_checkpointing_readouts_and_cache_contract(self) -> None:
        body = _KimiBody()
        adapter = hf.HuggingFaceBackboneAdapter(
            _config(gradient_checkpointing=True)
        )
        checkpoint_flags: list[bool] = []
        call_kwargs: list[dict[str, object]] = []

        def call_body(
            body: Callable[..., object],
            *,
            checkpointed: bool,
            **kwargs: object,
        ) -> object:
            call_kwargs.append(kwargs)
            checkpoint_flags.append(checkpointed)
            return body(**kwargs)

        with patch.object(
            hf.AutoModel,
            "from_pretrained",
            return_value=body,
        ):
            model = cast(nn.Module, adapter.model)

        inputs = torch.randn(1, 2, 4, requires_grad=True)
        cache_position = torch.arange(inputs.size(1))
        with patch.object(hf, "call_kimi_body", side_effect=call_body):
            model.train()
            with torch.enable_grad():
                text = adapter.encode(
                    inputs_embeds=inputs,
                    attention_mask=torch.ones(1, 2, dtype=torch.long),
                    output_hidden_states=False,
                    cache_position=cache_position,
                    modality=Modality.TEXT,
                ).last_hidden_state

            model.eval()
            with torch.enable_grad():
                audio_eval = adapter.encode(
                    inputs_embeds=inputs,
                    attention_mask=None,
                    output_hidden_states=False,
                    cache_position=cache_position,
                    modality=Modality.AUDIO,
                ).last_hidden_state

            model.train()
            with torch.no_grad():
                audio_no_grad = adapter.encode(
                    inputs_embeds=inputs,
                    attention_mask=None,
                    output_hidden_states=False,
                    cache_position=cache_position,
                    modality=Modality.AUDIO,
                ).last_hidden_state

        self.assertTrue(adapter.has_modality_readouts)
        self.assertTrue(torch.equal(text, inputs + 1))
        self.assertTrue(torch.equal(audio_eval, inputs + 2))
        self.assertTrue(torch.equal(audio_no_grad, inputs + 2))
        self.assertEqual(checkpoint_flags, [True, False, False])
        self.assertTrue(all("cache_position" not in kwargs for kwargs in call_kwargs))
        self.assertTrue(
            all(kwargs["output_hidden_states"] is False for kwargs in call_kwargs)
        )
        self.assertTrue(all(kwargs["return_dict"] is True for kwargs in call_kwargs))
        self.assertFalse(body.config.use_cache)


if __name__ == "__main__":
    unittest.main()
