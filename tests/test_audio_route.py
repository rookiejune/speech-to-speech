from __future__ import annotations

import unittest
from typing import cast

from speech_to_speech.audio_route import (
    BICODEC_PREDICT_ACOUSTIC,
    BICODEC_REUSE_PROMPT_ACOUSTIC,
    AudioStream,
    Config,
    Decode,
    Output,
    Prompt,
    PromptSource,
    StreamSource,
)


class AudioRouteTest(unittest.TestCase):
    def test_stream_order_is_canonical_not_declaration_order(self):
        prompt = Prompt(
            source=PromptSource.SOURCE,
            streams=(AudioStream.SEMANTIC, AudioStream.ACOUSTIC),
        )
        output = Output(
            streams=(AudioStream.SEMANTIC, AudioStream.ACOUSTIC),
        )

        expected = (AudioStream.ACOUSTIC, AudioStream.SEMANTIC)
        self.assertEqual(prompt.canonical_streams, expected)
        self.assertEqual(output.canonical_streams, expected)

    def test_stream_declarations_reject_duplicates(self):
        with self.assertRaisesRegex(ValueError, "prompt streams.*duplicates"):
            Prompt(
                source=PromptSource.REFERENCE,
                streams=(AudioStream.SEMANTIC, AudioStream.SEMANTIC),
            )
        with self.assertRaisesRegex(ValueError, "output streams.*duplicates"):
            Output(streams=(AudioStream.ACOUSTIC, AudioStream.ACOUSTIC))

    def test_route_rejects_decode_stream_missing_from_prompt(self):
        with self.assertRaisesRegex(ValueError, "prompt does not provide acoustic"):
            Config(
                prompt=Prompt(
                    source=PromptSource.REFERENCE,
                    streams=(AudioStream.SEMANTIC,),
                ),
                output=Output(streams=(AudioStream.SEMANTIC,)),
                decode=Decode(
                    semantic=StreamSource.OUTPUT,
                    acoustic=StreamSource.PROMPT,
                ),
            )

    def test_route_rejects_decode_stream_missing_from_output(self):
        with self.assertRaisesRegex(ValueError, "output does not provide semantic"):
            Config(
                prompt=Prompt(
                    source=PromptSource.REFERENCE,
                    streams=(AudioStream.ACOUSTIC,),
                ),
                output=Output(streams=(AudioStream.ACOUSTIC,)),
                decode=Decode(
                    semantic=StreamSource.OUTPUT,
                    acoustic=StreamSource.OUTPUT,
                ),
            )

    def test_generator_availability_is_validated_by_composition(self):
        route = Config(
            prompt=Prompt(source=PromptSource.SOURCE, streams=()),
            output=Output(streams=()),
            decode=Decode(
                semantic=StreamSource.GENERATOR,
                acoustic=StreamSource.GENERATOR,
            ),
        )

        self.assertIs(route.decode.semantic, StreamSource.GENERATOR)
        self.assertIs(route.decode.acoustic, StreamSource.GENERATOR)

    def test_contract_values_are_strict(self):
        with self.assertRaisesRegex(TypeError, "prompt source"):
            Prompt(
                source=cast(PromptSource, "reference"),
                streams=(AudioStream.SEMANTIC,),
            )

    def test_stream_lists_are_normalized_for_omegaconf(self):
        output = Output(
            streams=cast(tuple[AudioStream, ...], [AudioStream.SEMANTIC])
        )

        self.assertEqual(output.streams, (AudioStream.SEMANTIC,))

    def test_bicodec_reuse_prompt_acoustic_preset(self):
        route = BICODEC_REUSE_PROMPT_ACOUSTIC

        self.assertIs(route.prompt.source, PromptSource.REFERENCE)
        self.assertEqual(
            route.prompt.canonical_streams,
            (AudioStream.ACOUSTIC, AudioStream.SEMANTIC),
        )
        self.assertEqual(route.output.streams, (AudioStream.SEMANTIC,))
        self.assertIs(route.decode.semantic, StreamSource.OUTPUT)
        self.assertIs(route.decode.acoustic, StreamSource.PROMPT)

    def test_bicodec_predict_acoustic_preset(self):
        route = BICODEC_PREDICT_ACOUSTIC

        self.assertEqual(
            route.output.canonical_streams,
            (AudioStream.ACOUSTIC, AudioStream.SEMANTIC),
        )
        self.assertIs(route.decode.semantic, StreamSource.OUTPUT)
        self.assertIs(route.decode.acoustic, StreamSource.OUTPUT)


if __name__ == "__main__":
    unittest.main()
