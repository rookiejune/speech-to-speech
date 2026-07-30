from __future__ import annotations

import unittest

from speech_to_speech.runtime.runtime import bind_chat_bos, text_special_id


class _SpecialTokenizer:
    def __init__(
        self,
        *,
        pad_token_id: int | None = 0,
        bos_token_id: int | None = 1,
        eos_token_id: int | None = 2,
        special_tokens_map: dict[str, str] | None = None,
        encoded: dict[str, list[int]] | None = None,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.special_tokens_map = special_tokens_map or {}
        self._encoded = encoded or {}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(self._encoded[text])


class _BindableTokenizer:
    def __init__(
        self,
        *,
        bos_token_id: int | None,
        token_ids: dict[str, int],
        unk_token_id: int | None = None,
    ) -> None:
        self.bos_token_id = bos_token_id
        self.bos_token: str | None = None
        self.unk_token_id = unk_token_id
        self._token_ids = token_ids

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_ids[token]


class TextSpecialIdTest(unittest.TestCase):
    def test_reads_tokenizer_attributes(self):
        tokenizer = _SpecialTokenizer(pad_token_id=10, bos_token_id=11, eos_token_id=12)

        self.assertEqual(text_special_id(tokenizer, "pad_token_id"), 10)
        self.assertEqual(text_special_id(tokenizer, "bos_token_id"), 11)
        self.assertEqual(text_special_id(tokenizer, "eos_token_id"), 12)

    def test_falls_back_to_special_tokens_map(self):
        tokenizer = _SpecialTokenizer(
            pad_token_id=None,
            bos_token_id=None,
            eos_token_id=None,
            special_tokens_map={
                "pad_token": "<pad>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
            },
            encoded={"<pad>": [3], "<bos>": [4], "<eos>": [5]},
        )

        self.assertEqual(text_special_id(tokenizer, "pad_token_id"), 3)
        self.assertEqual(text_special_id(tokenizer, "bos_token_id"), 4)
        self.assertEqual(text_special_id(tokenizer, "eos_token_id"), 5)

    def test_rejects_missing_and_multi_token_mappings(self):
        missing = _SpecialTokenizer(pad_token_id=None, special_tokens_map={})
        multi = _SpecialTokenizer(
            bos_token_id=None,
            special_tokens_map={"bos_token": "<bos>"},
            encoded={"<bos>": [1, 2]},
        )

        with self.assertRaisesRegex(ValueError, "missing pad_token_id"):
            text_special_id(missing, "pad_token_id")
        with self.assertRaisesRegex(ValueError, "must map to one id"):
            text_special_id(multi, "bos_token_id")


class BindChatBosTest(unittest.TestCase):
    def test_binds_im_start_when_bos_missing(self):
        tokenizer = _BindableTokenizer(
            bos_token_id=None,
            token_ids={"<|im_start|>": 151644},
        )

        bind_chat_bos(tokenizer)

        self.assertEqual(tokenizer.bos_token, "<|im_start|>")

    def test_leaves_existing_bos_unchanged(self):
        tokenizer = _BindableTokenizer(
            bos_token_id=1,
            token_ids={"<|im_start|>": 151644},
        )

        bind_chat_bos(tokenizer)

        self.assertIsNone(tokenizer.bos_token)

    def test_skips_unknown_im_start(self):
        tokenizer = _BindableTokenizer(
            bos_token_id=None,
            token_ids={"<|im_start|>": 0},
            unk_token_id=0,
        )

        bind_chat_bos(tokenizer)

        self.assertIsNone(tokenizer.bos_token)


if __name__ == "__main__":
    unittest.main()
