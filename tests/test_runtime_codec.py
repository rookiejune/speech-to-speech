from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from speech_to_speech.runtime.codec import load_codec


class RuntimeCodecTest(unittest.TestCase):
    def test_load_codec_longcat_uses_anytrain_backend(self) -> None:
        backend = SimpleNamespace(name="longcat")

        with patch(
            "speech_to_speech.runtime.codec.load_semantic_acoustic",
            return_value=backend,
        ) as load_codec_backend:
            codec = load_codec("longcat", device="cuda")

        self.assertIs(codec, backend)
        load_codec_backend.assert_called_once_with("longcat", device="cuda")


if __name__ == "__main__":
    unittest.main()
