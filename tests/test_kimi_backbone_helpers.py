from __future__ import annotations

import unittest
from collections.abc import Collection, Mapping, Sequence
from types import SimpleNamespace
from typing import Literal
from unittest.mock import patch

import torch
from peft import LoraConfig, inject_adapter_in_model
from torch import Tensor, nn

from speech_to_speech.runtime.backbone.kimi import (
    KimiTokenizerAdapter,
    call_kimi_body,
    checkpoint_kimi_body,
    kimi_body,
    remove_kimi_output_heads,
    should_checkpoint_kimi_body,
)


_PROJECT_CHAT_TEMPLATE = (
    '{{ messages | map(attribute="content") | join("\\n") }}'
    '{% if add_generation_prompt %}{{ "\\n" }}{% endif %}'
)


class _RawTokenizer:
    special_tokens = {
        "[BOS]": 0,
        "[EOS]": 1,
        "<pad>": 2,
        "<audio>": 3,
    }
    bos_id = 0
    eos_id = 1
    pad_id = 2
    vocab_size = 16

    def __init__(self) -> None:
        self.encode_calls: list[dict[str, object]] = []
        self.decode_calls: list[list[int]] = []

    def encode(
        self,
        text: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Literal["all"] | Collection[str] = (),
        disallowed_special: Collection[str] = (),
    ) -> list[int]:
        self.encode_calls.append(
            {
                "text": text,
                "bos": bos,
                "eos": eos,
                "allowed_special": allowed_special,
                "disallowed_special": disallowed_special,
            }
        )
        values = [4, 5]
        if bos:
            values.insert(0, self.bos_id)
        if eos:
            values.append(self.eos_id)
        return values

    def decode(self, token_ids: Sequence[int]) -> str:
        values = list(token_ids)
        self.decode_calls.append(values)
        return ",".join(str(value) for value in values)

    def convert_tokens_to_ids(self, tokens: str | Sequence[str]) -> int | list[int]:
        if isinstance(tokens, str):
            return self.special_tokens[tokens]
        return [self.special_tokens[token] for token in tokens]


class _ChatRawTokenizer(_RawTokenizer):
    def __init__(self) -> None:
        super().__init__()
        self.chat_call: tuple[Sequence[Mapping[str, str]], dict[str, object]] | None = None

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        **kwargs: object,
    ) -> str | list[int]:
        self.chat_call = (conversation, kwargs)
        return [4, 5] if kwargs["tokenize"] else "rendered"


class _StatefulRawTokenizer(_RawTokenizer):
    def __init__(
        self,
        token_id: int,
        *,
        chat_template: str | None = None,
    ) -> None:
        super().__init__()
        self.token_id = token_id
        self.chat_template = chat_template

    def encode(
        self,
        text: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Literal["all"] | Collection[str] = (),
        disallowed_special: Collection[str] = (),
    ) -> list[int]:
        super().encode(
            text,
            bos=bos,
            eos=eos,
            allowed_special=allowed_special,
            disallowed_special=disallowed_special,
        )
        values = [4, self.token_id]
        if bos:
            values.insert(0, self.bos_id)
        if eos:
            values.append(self.eos_id)
        return values


class _ContractRawTokenizer(_StatefulRawTokenizer):
    def contract_state(self) -> Mapping[str, object]:
        return {
            "grammar": "fixture-kimi-raw-v1",
            "asset": "fixture-tokenizer-v1",
        }


class _RankedRawTokenizer(_RawTokenizer):
    def __init__(self, *, first_rank: int) -> None:
        super().__init__()
        self.tokenizer = SimpleNamespace(
            _mergeable_ranks={
                b"first": first_rank,
                b"second": 1 - first_rank,
            },
            _pat_str=r"\w+|[^\w\s]+",
            _special_tokens={"<audio>": 3},
        )


class _BrokenTokenizerBase(_RawTokenizer):
    chat_template = None

    def convert_tokens_to_ids(self, tokens: str | Sequence[str]) -> int | list[int]:
        del tokens
        raise AttributeError("missing inherited tokenizer state")

    def apply_chat_template(
        self,
        conversation: Sequence[Mapping[str, str]],
        **kwargs: object,
    ) -> str:
        del conversation, kwargs
        raise AssertionError("inherited template must not be called")


class _RemoteTokenizerWithoutTemplate(_BrokenTokenizerBase):
    pass


class KimiTokenizerAdapterTest(unittest.TestCase):
    def test_adapts_special_tokens_and_encode_decode_signatures(self) -> None:
        raw = _RawTokenizer()
        tokenizer = KimiTokenizerAdapter(raw)

        self.assertEqual(len(tokenizer), 16)
        self.assertEqual(tokenizer.vocab_size, 16)
        self.assertEqual(tokenizer.bos_token_id, 0)
        self.assertEqual(tokenizer.eos_token_id, 1)
        self.assertEqual(tokenizer.pad_token_id, 2)
        self.assertEqual(tokenizer.special_tokens_map["bos_token"], "[BOS]")
        self.assertEqual(tokenizer.special_tokens_map["eos_token"], "[EOS]")
        self.assertEqual(tokenizer.special_tokens_map["pad_token"], "<pad>")
        self.assertEqual(
            tokenizer.special_tokens_map["additional_special_tokens"],
            ("<audio>",),
        )

        self.assertEqual(tokenizer.encode("hello"), [4, 5])
        self.assertEqual(tokenizer.encode("hello", add_special_tokens=True), [0, 4, 5, 1])
        self.assertEqual(raw.encode_calls[0]["bos"], False)
        self.assertEqual(raw.encode_calls[0]["eos"], False)
        self.assertEqual(raw.encode_calls[1]["bos"], True)
        self.assertEqual(raw.encode_calls[1]["eos"], True)
        self.assertEqual(raw.encode_calls[1]["allowed_special"], "all")

        self.assertEqual(tokenizer.decode([0, 4, 3, 5, 1]), "4,5")
        self.assertEqual(raw.decode_calls[-1], [4, 5])
        self.assertEqual(
            tokenizer.decode([0, 4, 3, 5, 1], skip_special_tokens=False),
            "0,4,3,5,1",
        )
        self.assertEqual(tokenizer.convert_tokens_to_ids("<audio>"), 3)
        self.assertEqual(tokenizer.convert_tokens_to_ids(["[BOS]", "[EOS]"]), [0, 1])

    def test_rejects_non_one_dimensional_token_sequences(self) -> None:
        tokenizer = KimiTokenizerAdapter(_RawTokenizer())

        with self.assertRaisesRegex(TypeError, "must be an integer"):
            tokenizer.decode([[4, 5]])  # type: ignore[list-item]
        with self.assertRaisesRegex(ValueError, "one-dimensional"):
            tokenizer.decode(torch.ones(1, 2, dtype=torch.long))  # type: ignore[arg-type]

    def test_chat_template_absence_is_explicit(self) -> None:
        tokenizer = KimiTokenizerAdapter(_RawTokenizer())

        with self.assertRaisesRegex(NotImplementedError, "official chat template"):
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "hello"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            )

    def test_delegates_chat_template_when_remote_tokenizer_provides_one(self) -> None:
        raw = _ChatRawTokenizer()
        tokenizer = KimiTokenizerAdapter(raw)
        conversation = [{"role": "user", "content": "hello"}]

        self.assertEqual(
            tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=False,
            ),
            "rendered",
        )
        self.assertIsNotNone(raw.chat_call)
        assert raw.chat_call is not None
        self.assertIs(raw.chat_call[0], conversation)
        self.assertEqual(raw.chat_call[1]["add_generation_prompt"], True)
        self.assertEqual(
            tokenizer.apply_chat_template(conversation, tokenize=True),
            [4, 5],
        )

    def test_bypasses_broken_inherited_huggingface_tokenizer_helpers(self) -> None:
        tokenizer = KimiTokenizerAdapter(_RemoteTokenizerWithoutTemplate())

        self.assertEqual(tokenizer.convert_tokens_to_ids("<audio>"), 3)
        with self.assertRaisesRegex(KeyError, "has no token"):
            tokenizer.convert_tokens_to_ids("not-special")
        with self.assertRaisesRegex(NotImplementedError, "official chat template"):
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "hello"}],
            )

    def test_configured_project_template_replaces_missing_official_template(self) -> None:
        tokenizer = KimiTokenizerAdapter(
            _RemoteTokenizerWithoutTemplate(),
            chat_template=_PROJECT_CHAT_TEMPLATE,
        )
        conversation = [{"role": "user", "content": "hello"}]

        self.assertEqual(
            tokenizer.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            ),
            "hello\n",
        )
        self.assertEqual(
            tokenizer.apply_chat_template(
                conversation,
                tokenize=True,
                add_generation_prompt=True,
            ),
            [4, 5],
        )

    def test_contract_state_tracks_same_class_raw_encoding_behavior(self) -> None:
        first = KimiTokenizerAdapter(_StatefulRawTokenizer(5))
        second = KimiTokenizerAdapter(_StatefulRawTokenizer(6))

        self.assertNotEqual(first.contract_state(), second.contract_state())

    def test_contract_state_tracks_tiktoken_mergeable_ranks(self) -> None:
        first = KimiTokenizerAdapter(_RankedRawTokenizer(first_rank=0))
        second = KimiTokenizerAdapter(_RankedRawTokenizer(first_rank=1))

        self.assertNotEqual(first.contract_state(), second.contract_state())

    def test_contract_state_tracks_raw_chat_template(self) -> None:
        first = KimiTokenizerAdapter(
            _StatefulRawTokenizer(5, chat_template="first raw template")
        )
        second = KimiTokenizerAdapter(
            _StatefulRawTokenizer(5, chat_template="second raw template")
        )

        self.assertNotEqual(first.contract_state(), second.contract_state())

    def test_contract_state_prefers_explicit_raw_contract(self) -> None:
        raw = _ContractRawTokenizer(5)
        tokenizer = KimiTokenizerAdapter(raw)

        state = tokenizer.contract_state()

        raw_state = state["raw_state"]
        self.assertIsInstance(raw_state, Mapping)
        assert isinstance(raw_state, Mapping)
        self.assertEqual(raw_state["grammar"], "explicit-contract-state-v1")
        self.assertEqual(raw.encode_calls, [])


class _KimiBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([nn.Linear(4, 4)])
        self.mimo_layers = nn.ModuleList([nn.Linear(4, 4)])

    def forward(self, inputs_embeds: Tensor, *, use_cache: bool = False) -> Tensor:
        del use_cache
        return self.mimo_layers[0](self.layers[0](inputs_embeds))


class _KimiRoot(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _KimiBody()
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.mimo_output = nn.Linear(4, 8, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens


class KimiResourceHelpersTest(unittest.TestCase):
    def test_removes_both_output_heads_without_reparenting_body(self) -> None:
        root = _KimiRoot()
        body = root.model
        body_state_names = tuple(body.state_dict())

        self.assertIs(kimi_body(root), body)
        self.assertEqual(
            remove_kimi_output_heads(root),
            ("lm_head", "mimo_output"),
        )

        self.assertIs(root.model, body)
        self.assertIsNone(root.lm_head)
        self.assertIsNone(root.mimo_output)
        self.assertEqual(tuple(body.state_dict()), body_state_names)
        self.assertIn("model.layers.0.weight", root.state_dict())
        self.assertIn("model.mimo_layers.0.weight", root.state_dict())
        self.assertNotIn("lm_head.weight", root.state_dict())
        self.assertNotIn("mimo_output.weight", root.state_dict())
        self.assertEqual(remove_kimi_output_heads(root), ())

    def test_body_resolver_rejects_missing_remote_body(self) -> None:
        with self.assertRaisesRegex(TypeError, "callable model body"):
            kimi_body(nn.Linear(2, 2))


class _CheckpointBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.calls = 0

    def forward(self, inputs_embeds: Tensor, *, use_cache: bool) -> dict[str, Tensor]:
        if use_cache:
            raise AssertionError("test body must not receive use_cache=True")
        self.calls += 1
        return {"last_hidden_state": torch.sin(inputs_embeds * self.scale)}


class _ProjectionLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4, bias=False)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.q_proj(inputs)


class _LoRACheckpointBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_ProjectionLayer()])

    def forward(
        self,
        inputs_embeds: Tensor,
        *,
        use_cache: bool,
        return_dict: bool,
    ) -> SimpleNamespace:
        if use_cache or not return_dict:
            raise AssertionError("checkpointed Kimi calls must use a structured output")
        hidden = self.layers[0](inputs_embeds)
        return SimpleNamespace(last_hidden_state=(hidden, hidden))


class KimiCheckpointHelpersTest(unittest.TestCase):
    def test_disabled_call_does_not_checkpoint(self) -> None:
        body = _CheckpointBody()
        inputs = torch.randn(2, 4, requires_grad=True)

        output = checkpoint_kimi_body(
            body,
            enabled=False,
            kwargs={"inputs_embeds": inputs, "use_cache": False},
        )

        self.assertEqual(body.calls, 1)
        output["last_hidden_state"].sum().backward()
        self.assertEqual(body.calls, 1)

    def test_non_reentrant_checkpoint_recomputes_structured_output(self) -> None:
        body = _CheckpointBody()
        inputs = torch.randn(2, 4, requires_grad=True)

        output = checkpoint_kimi_body(
            body,
            enabled=True,
            kwargs={"inputs_embeds": inputs},
        )
        self.assertEqual(body.calls, 1)
        output["last_hidden_state"].sum().backward()

        self.assertGreaterEqual(body.calls, 2)
        self.assertIsNotNone(body.scale.grad)

    def test_checkpoint_call_forces_non_reentrant_and_disables_cache(self) -> None:
        sentinel = object()
        body = _CheckpointBody()
        inputs = torch.randn(1, 4)

        with patch(
            "speech_to_speech.runtime.backbone.kimi.checkpoint",
            return_value=sentinel,
        ) as mocked:
            result = checkpoint_kimi_body(
                body,
                enabled=True,
                kwargs={"inputs_embeds": inputs},
            )

        self.assertIs(result, sentinel)
        mocked.assert_called_once_with(
            body,
            use_reentrant=False,
            inputs_embeds=inputs,
            use_cache=False,
        )

    def test_checkpoint_rejects_cache_state(self) -> None:
        body = _CheckpointBody()
        inputs = torch.randn(1, 4)

        with self.assertRaisesRegex(ValueError, "use_cache=False"):
            checkpoint_kimi_body(
                body,
                enabled=True,
                kwargs={"inputs_embeds": inputs, "use_cache": True},
            )
        with self.assertRaisesRegex(ValueError, "past_key_values"):
            checkpoint_kimi_body(
                body,
                enabled=True,
                kwargs={"inputs_embeds": inputs, "past_key_values": object()},
            )

    def test_keyword_call_and_training_gate(self) -> None:
        body = _CheckpointBody()
        inputs = torch.randn(1, 4)

        output = call_kimi_body(
            body,
            checkpointed=False,
            inputs_embeds=inputs,
            use_cache=False,
        )
        self.assertEqual(tuple(output["last_hidden_state"].shape), (1, 4))
        self.assertTrue(should_checkpoint_kimi_body(body, True))
        body.eval()
        self.assertFalse(should_checkpoint_kimi_body(body, True))
        body.train()
        with torch.no_grad():
            self.assertFalse(should_checkpoint_kimi_body(body, True))

    def test_non_reentrant_checkpoint_keeps_external_lora_gradients(self) -> None:
        body = _LoRACheckpointBody()
        inject_adapter_in_model(
            LoraConfig(r=1, target_modules=["q_proj"]),
            body,
            adapter_name="speech",
        )

        output = checkpoint_kimi_body(
            body,
            enabled=True,
            kwargs={
                "inputs_embeds": torch.randn(1, 2, 4),
                "return_dict": True,
            },
        )
        output.last_hidden_state[0].sum().backward()

        gradients = {
            name: parameter.grad
            for name, parameter in body.named_parameters()
            if ".lora_B." in name
        }
        self.assertTrue(gradients)
        self.assertTrue(all(gradient is not None for gradient in gradients.values()))


if __name__ == "__main__":
    unittest.main()
