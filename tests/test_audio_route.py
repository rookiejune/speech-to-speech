from __future__ import annotations

import unittest
from typing import cast

from speech_to_speech.audio_route import (
    BICODEC_GENERATE_GLOBAL,
    BICODEC_REUSE_PROMPT_GLOBAL,
    AudioStream,
    Config,
    Decode,
    Output,
    Prompt,
    PromptSource,
    StreamSource,
)


class AudioRouteTest(unittest.TestCase):
    def test_global_stream_is_distinct_from_frame_acoustic(self):
        self.assertIsNot(AudioStream.GLOBAL, AudioStream.ACOUSTIC)
        self.assertIs(AudioStream("global"), AudioStream.GLOBAL)
        self.assertIs(AudioStream("acoustic"), AudioStream.ACOUSTIC)

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
        with self.assertRaisesRegex(ValueError, "global.*acoustic"):
            Output(streams=(AudioStream.GLOBAL, AudioStream.ACOUSTIC))

    def test_route_rejects_decode_stream_missing_from_prompt(self):
        with self.assertRaisesRegex(ValueError, "prompt does not provide .*global"):
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

    def test_stream_lists_are_normalized(self):
        output = Output(
            streams=cast(tuple[AudioStream, ...], [AudioStream.SEMANTIC])
        )
        self.assertEqual(output.streams, (AudioStream.SEMANTIC,))

    def test_bicodec_reuse_prompt_global_preset(self):
        route = BICODEC_REUSE_PROMPT_GLOBAL

        self.assertIs(route.prompt.source, PromptSource.REFERENCE)
        self.assertEqual(route.prompt.canonical_streams, (AudioStream.GLOBAL,))
        self.assertEqual(route.output.canonical_streams, (AudioStream.SEMANTIC,))
        self.assertIs(route.decode.semantic, StreamSource.OUTPUT)
        self.assertIs(route.decode.acoustic, StreamSource.PROMPT)

    def test_bicodec_generate_global_preset(self):
        route = BICODEC_GENERATE_GLOBAL

        self.assertIs(route.prompt.source, PromptSource.SOURCE)
        self.assertEqual(route.prompt.canonical_streams, ())
        self.assertEqual(
            route.output.canonical_streams,
            (AudioStream.GLOBAL, AudioStream.SEMANTIC),
        )
        self.assertIs(route.decode.semantic, StreamSource.OUTPUT)
        self.assertIs(route.decode.acoustic, StreamSource.OUTPUT)


if __name__ == "__main__":
    unittest.main()
