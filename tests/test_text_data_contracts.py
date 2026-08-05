from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _contracts_helpers import *
from speech_to_speech.datamodule.builder import build_speech_sample
from speech_to_speech.datamodule.contract import DataRuntime
from speech_to_speech.datamodule.loader import ARFraming
from speech_to_speech.datamodule.sample import Speech, Text, TextPair


class TextDataContractTest(unittest.TestCase):
    def test_text_parser_ignores_audio_fields(self):
        tokenizer = _Tokenizer(10)
        runtime = SimpleNamespace(text_tokenizer=tokenizer)

        pair = parse_text_sample(_raw_text_sample(), runtime)

        self.assertTrue(torch.equal(pair.source.text_token_ids, torch.tensor([1, 2])))
        self.assertIs(pair.source.language, Language.ZH)
        self.assertIs(pair.target.language, Language.EN)
        self.assertEqual(tokenizer.encoded, ("target text", False))

    def test_text_collator_builds_mt_batches_without_audio_runtime(self):
        runtime = _text_runtime()

        batch = TextCollator(runtime, {Task.MT: 1.0})([_raw_text_sample()])

        self.assertEqual(batch.tasks, [Task.MT])
        self.assertIsNone(batch.acoustic_target)
        prompt_length = int(batch.generation_prompt_lengths[0])
        response = torch.tensor(
            [
                runtime.control_token_id(ControlToken.MT_BEGIN),
                runtime.control_token_id(ControlToken.LANG_EN),
                1,
                2,
                runtime.control_token_id(ControlToken.MT_END),
            ]
        )
        self.assertTrue(batch.token_labels[0, :prompt_length].eq(-100).all())
        self.assertTrue(
            torch.equal(batch.input_ids[0, prompt_length:], response)
        )
        self.assertTrue(torch.equal(batch.token_labels[0, prompt_length:], response))
        self.assertEqual(batch.target_languages, ["en"])

    def test_text_collator_selects_mt_control_from_target_language(self):
        runtime = _text_runtime()
        raw = {
            (Role.SOURCE, Modality.TEXT): TextItem(
                views={TextView.TEXT: "source text"},
                meta={TextMeta.LANG: Lang.EN},
            ),
            (Role.TARGET, Modality.TEXT): TextItem(
                views={TextView.TEXT: "target text"},
                meta={TextMeta.LANG: Lang.ZH},
            ),
        }

        batch = TextCollator(runtime, {Task.MT: 1.0})([raw])

        prompt_length = int(batch.generation_prompt_lengths[0].item())
        response = torch.tensor(
            [
                runtime.control_token_id(ControlToken.MT_BEGIN),
                runtime.control_token_id(ControlToken.LANG_ZH),
                1,
                2,
                runtime.control_token_id(ControlToken.MT_END),
            ]
        )
        self.assertTrue(batch.token_labels[0, :prompt_length].eq(-100).all())
        self.assertTrue(torch.equal(batch.input_ids[0, prompt_length:], response))
        self.assertTrue(torch.equal(batch.token_labels[0, prompt_length:], response))
        self.assertEqual(batch.target_languages, ["zh"])

    def test_mt_builder_requires_target_language(self):
        pair = TextPair(
            source=Text(torch.tensor([1, 2]), Language.EN),
            target=Text(
                torch.tensor([1, 2]),
                cast(Language, SimpleNamespace(code=None)),
            ),
        )

        with self.assertRaisesRegex(ValueError, "target language"):
            build_speech_sample(
                cast(Speech, pair.source),
                cast(Speech, pair.target),
                Task.MT,
                cast(DataRuntime, _text_runtime()),
                prompt="translate $$$PLACEHOLDER$$$ now",
            )

    def test_text_collator_rejects_audio_tasks(self):
        runtime = _text_runtime()

        with self.assertRaisesRegex(ValueError, "text-only"):
            TextCollator(runtime, {Task.TTS: 1.0})

    @patch("speech_to_speech.datamodule.collate.build_text_sample")
    @patch("speech_to_speech.datamodule.collate.parse_text_sample")
    def test_text_collator_forwards_explicit_trace(self, parse, build):
        runtime = SimpleNamespace()
        raw = Mock()
        parsed = Mock()
        built = Mock()
        parse.return_value = parsed
        build.return_value = built
        collator = TextCollator(
            runtime,
            {Task.TEXT_AR: 1.0},
            trace="direct",
        )

        self.assertEqual(collator._model_samples([raw]), [built])
        parse.assert_called_once_with(raw, runtime)
        build.assert_called_once_with(
            parsed,
            Task.TEXT_AR,
            runtime,
            ar_framing=ARFraming.INSTRUCTION,
            tasks=None,
            trace="direct",
        )

    def test_text_ar_pretraining_collator_uses_bos_without_chat_template(self):
        tokenizer = _ChatTokenizer(32)
        tokenizer.bos_token_id = 3
        tokenizer.apply_chat_template = Mock(
            side_effect=AssertionError("pretraining must not render chat prompts")
        )
        runtime = _text_runtime(tokenizer)

        batch = TextCollator(
            runtime,
            {Task.TEXT_AR: 1.0},
            ar_framing=ARFraming.PRETRAINING,
        )([_raw_text_sample()])

        self.assertEqual(batch.tasks, [Task.TEXT_AR])
        self.assertEqual(batch.target_languages, [None])
        self.assertEqual(int(batch.generation_prompt_lengths[0]), 1)
        self.assertEqual(int(batch.input_ids[0, 0]), tokenizer.bos_token_id)
        self.assertEqual(int(batch.token_labels[0, 0]), -100)
        self.assertTrue(
            torch.equal(
                batch.token_labels[0, 1:],
                torch.tensor([1, 2, runtime.eos_token_id]),
            )
        )
        tokenizer.apply_chat_template.assert_not_called()

    @patch("anydataset.presets.WMT19")
    def test_text_dataset_config_loads_anydataset_wmt19(self, wmt19):
        config = TextDatasetConfig(
            name=TextDatasetName.WMT19,
            split="validation",
            source_lang="de",
            target_lang="en",
        )

        loaded = load_text_dataset(config)

        self.assertIs(loaded, wmt19.return_value)
        wmt19.assert_called_once_with(
            split="validation",
            source_lang="de",
            target_lang="en",
        )

    def test_text_datamodule_reads_toy_text_without_codec_runtime(self):
        runtime = _text_runtime()
        datamodule = DataModule(
            runtime,
            {
                "mt": LoaderSpec.text(
                    TextConfig(
                        dataloader=_loader(2),
                        dataset=TextDatasetConfig(
                            name=TextDatasetName.TOY,
                            toy_samples=2,
                        ),
                    ),
                    {Task.MT: 1.0},
                )
            },
        )

        datamodule.setup()
        batch = next(iter(datamodule.train_dataloader()))

        self.assertEqual(batch.input_ids.size(0), 2)
        self.assertEqual(batch.tasks, [Task.MT, Task.MT])
        self.assertIsNone(batch.acoustic_target)

    def test_text_validation_dataloader_limits_samples(self):
        runtime = _text_runtime()
        text_config = TextConfig(
            dataloader=_loader(4),
            dataset=TextDatasetConfig(
                name=TextDatasetName.TOY,
                toy_samples=5,
            ),
        )
        datamodule = DataModule(
            runtime,
            {"mt": LoaderSpec.text(text_config, {Task.MT: 1.0})},
            validation=LoaderSpec.text(
                text_config,
                {Task.MT: 1.0},
                max_samples=2,
            ),
        )

        datamodule.setup()
        batches = list(datamodule.val_dataloader())

        self.assertEqual(sum(batch.input_ids.size(0) for batch in batches), 2)
        self.assertTrue(
            all(task is Task.MT for batch in batches for task in batch.tasks)
        )



def _text_runtime(tokenizer=None):
    lexical_text_vocab_size = 32
    controls = ControlTokenLookup(lexical_text_vocab_size)
    text_end = lexical_text_vocab_size + len(ControlToken)
    return SimpleNamespace(
        text_tokenizer=_ChatTokenizer(32) if tokenizer is None else tokenizer,
        layout=Layout(text=(0, text_end), audio=(text_end, text_end + 4)),
        lexical_text_vocab_size=lexical_text_vocab_size,
        control_token_ids=controls.ids,
        control_token_id=controls,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=31,
    )


if __name__ == "__main__":
    unittest.main()
