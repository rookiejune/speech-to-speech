from __future__ import annotations

import json
import os
import unittest
from typing import cast
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from anydataset.types import Lang, Modality, Role, TextItem, TextMeta, TextView
from anytrain.module.idspace import Layout

from _contracts_helpers import _ChatTokenizer
from _config_helpers import _train
from speech_to_speech.datamodule.collate.collator import TextCollator, pack_text_samples
from speech_to_speech.datamodule.config import DataLoaderConfig
from speech_to_speech.datamodule.dataset.text import TextConfig
from speech_to_speech.datamodule.module import DataModule, LoaderSpec
from speech_to_speech.datamodule.dataset.text import (
    DocumentTextDataset,
    JsonlTextDataset,
    TextDatasetConfig,
    TextDatasetName,
    load_text_dataset,
)
from speech_to_speech.datamodule.loader import ARFraming
from speech_to_speech.datamodule.protocol import TextRuntime
from speech_to_speech.datamodule.types import ModelBatch, ModelSample
from speech_to_speech.task import PredictionModality
from speech_to_speech.task import Task


class _PackingTokenizer(_ChatTokenizer):
    bos_token_id: int | None = 3


class GeneralTextDatasetTest(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "DYNAMIC_HOME": "/tmp/dynamic",
            "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
            "SPEECH_TO_SPEECH_TEXT_CORPUS": "/tmp/corpus.jsonl",
        },
    )
    def test_train_config_exposes_general_corpus_and_packing_fields(self) -> None:
        config = _train("experiment=train/kimi_audio/ar_pretrain")

        self.assertIs(config.text_datamodule.dataset.name, TextDatasetName.GENERAL)
        self.assertEqual(config.text_datamodule.dataset.path, "/tmp/corpus.jsonl")
        self.assertEqual(config.text_datamodule.max_tokens, 4096)
        self.assertTrue(config.text_datamodule.pack_documents)

    def test_jsonl_accepts_string_and_language_records(self) -> None:
        with TemporaryDirectory() as root:
            path = Path(root) / "corpus.jsonl"
            path.write_text(
                '"first document"\n'
                + json.dumps({"text": "第二个文档", "lang": "zh"})
                + "\n",
                encoding="utf-8",
            )

            dataset = JsonlTextDataset(path)
            first = cast(TextItem, dataset[0][(Role.TARGET, Modality.TEXT)])
            second = cast(TextItem, dataset[1][(Role.TARGET, Modality.TEXT)])

            self.assertEqual(len(dataset), 2)
            self.assertEqual(
                first.views[TextView.TEXT],
                "first document",
            )
            self.assertIs(
                first.meta[TextMeta.LANG],
                Lang.EN,
            )
            self.assertIs(
                second.meta[TextMeta.LANG],
                Lang.ZH,
            )

    def test_document_reader_is_one_sample_and_loader_auto_detects_jsonl(self) -> None:
        with self.assertRaisesRegex(ValueError, "require[s]? dataset.path"):
            load_text_dataset(TextDatasetConfig(name=TextDatasetName.GENERAL))

        with TemporaryDirectory() as root:
            document = Path(root) / "book.txt"
            document.write_text("a whole document\nwith two lines", encoding="utf-8")
            loaded_document = load_text_dataset(
                TextDatasetConfig(
                    name=TextDatasetName.GENERAL,
                    path=document,
                )
            )
            self.assertIsInstance(loaded_document, DocumentTextDataset)
            self.assertEqual(len(cast(DocumentTextDataset, loaded_document)), 1)

            jsonl = Path(root) / "book.jsonl"
            jsonl.write_text('"one"\n"two"\n', encoding="utf-8")
            loaded_jsonl = load_text_dataset(
                TextDatasetConfig(
                    name=TextDatasetName.GENERAL,
                    path=jsonl,
                )
            )
            self.assertIsInstance(loaded_jsonl, JsonlTextDataset)
            self.assertEqual(len(cast(JsonlTextDataset, loaded_jsonl)), 2)

    def test_jsonl_validation_is_strict(self) -> None:
        with TemporaryDirectory() as root:
            path = Path(root) / "bad.jsonl"
            path.write_text("\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 1.*blank"):
                JsonlTextDataset(path)

            path.write_text(json.dumps({"text": "ok", "extra": 1}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown fields"):
                JsonlTextDataset(path)

            path.write_text(
                json.dumps({"text": "bonjour", "lang": "fr"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid field 'lang'"):
                JsonlTextDataset(path)

            with self.assertRaises(FileNotFoundError):
                JsonlTextDataset(Path(root) / "missing.jsonl")

            with self.assertRaisesRegex(ValueError, "unknown text dataset encoding"):
                JsonlTextDataset(path, encoding="definitely-not-an-encoding")

            path.write_bytes(b"\xff\n")
            with self.assertRaisesRegex(ValueError, "not valid 'utf-8'"):
                JsonlTextDataset(path)

    def test_pretraining_collator_packs_token_budget(self) -> None:
        with TemporaryDirectory() as root:
            path = Path(root) / "corpus.jsonl"
            path.write_text('"one"\n"two"\n', encoding="utf-8")
            dataset = JsonlTextDataset(path)
            tokenizer = _PackingTokenizer(64)
            runtime = cast(
                TextRuntime,
                SimpleNamespace(
                    text_tokenizer=tokenizer,
                    layout=Layout(text=(0, 64), audio=(64, 68)),
                    pad_token_id=0,
                    eos_token_id=63,
                ),
            )

            batch = TextCollator(
                runtime,
                {Task.TEXT_AR: 1.0},
                ar_framing=ARFraming.PRETRAINING,
                max_tokens=7,
                pack_documents=True,
            )([dataset[0], dataset[1]])

            self.assertEqual(tuple(batch.input_ids.shape), (1, 7))
            prompt_lengths = batch.generation_prompt_lengths
            self.assertIsNotNone(prompt_lengths)
            self.assertEqual(int(cast(torch.Tensor, prompt_lengths)[0]), 1)
            self.assertEqual(int(batch.input_ids[0, 0]), 3)
            self.assertEqual(batch.supervised_token_count, 6)
            self.assertTrue(torch.equal(batch.token_labels[0, :1], torch.tensor([-100])))

    def test_general_corpus_runs_through_text_loader(self) -> None:
        with TemporaryDirectory() as root:
            path = Path(root) / "corpus.jsonl"
            path.write_text('"one"\n"two"\n', encoding="utf-8")
            tokenizer = _PackingTokenizer(64)
            runtime = cast(
                TextRuntime,
                SimpleNamespace(
                    text_tokenizer=tokenizer,
                    layout=Layout(text=(0, 64), audio=(64, 68)),
                    pad_token_id=0,
                    eos_token_id=63,
                ),
            )
            config = TextConfig(
                dataloader=DataLoaderConfig(batch_size=2, num_workers=0),
                dataset=TextDatasetConfig(
                    name=TextDatasetName.GENERAL,
                    path=path,
                    format="jsonl",
                ),
                max_tokens=7,
                pack_documents=True,
            )
            datamodule = DataModule(
                runtime,
                {
                    "text": LoaderSpec.text(
                        config,
                        {Task.TEXT_AR: 1.0},
                        prediction=PredictionModality.TEXT,
                        ar_framing=ARFraming.PRETRAINING,
                    )
                },
            )

            datamodule.setup()
            batch = cast(ModelBatch, next(iter(datamodule.train_dataloader())))

            self.assertEqual(tuple(batch.input_ids.shape), (1, 7))
            self.assertEqual(batch.tasks, [Task.TEXT_AR])

    def test_packing_splits_long_documents_and_preserves_eos(self) -> None:
        sample = ModelSample.from_sequence(
            torch.tensor([3, 10, 11, 12, 13, 63]),
            torch.tensor([-100, 10, 11, 12, 13, 63]),
            task=Task.TEXT_AR,
            prediction=PredictionModality.TEXT,
            generation_prompt_length=1,
        )

        packed = pack_text_samples([sample], max_tokens=4)

        self.assertEqual(len(packed), 2)
        self.assertEqual([row.input_ids.numel() for row in packed], [4, 4])
        self.assertTrue(all(int(row.input_ids[-1]) == 63 for row in packed))
        self.assertTrue(all(int(row.token_labels[0]) == -100 for row in packed))


if __name__ == "__main__":
    unittest.main()
