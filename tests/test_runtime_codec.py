from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from speech_to_speech.runtime.codec import LongCatCodec, load_codec


class RuntimeCodecTest(unittest.TestCase):
    def test_load_codec_longcat_constructs_adapter(self) -> None:
        backend = SimpleNamespace(
            decoders={"16k_4codebooks": SimpleNamespace(latent_dim=32)},
        )

        with patch(
            "speech_to_speech.runtime.codec.load_frame",
            return_value=backend,
        ) as load_frame:
            codec = load_codec("longcat", device="cuda")

        adapter = cast(LongCatCodec, codec)
        self.assertIsInstance(adapter, LongCatCodec)
        self.assertIs(adapter.codec, backend)
        load_frame.assert_called_once_with("longcat", device="cuda")


if __name__ == "__main__":
    unittest.main()
