from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _contracts_helpers import *


class SplitManifestContractTest(unittest.TestCase):
    def test_split_manifest_dataset_reads_explicit_indices(self):
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "split.json"
            manifest.write_text(
                json.dumps({"version": 1, "splits": {"pilot": [2, 0]}}),
            )
            runtime = _data_runtime()
            config = DatasetConfig(
                name=DatasetName.TOY,
                split_manifest=str(manifest),
                split_label="pilot",
                toy_samples=3,
            )

            dataset = load_dataset(config, runtime)

            self.assertIsInstance(dataset, SplitManifestDataset)
            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.global_index(0), 2)
            self.assertEqual(dataset.global_index(1), 0)
            first = dataset[0]
            text = first[(Role.SOURCE, Modality.TEXT)].views[TextView.TEXT]
            self.assertEqual(text, "toy source 2")
    def test_split_manifest_rejects_invalid_indices(self):
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "split.json"
            manifest.write_text(
                json.dumps({"version": 1, "splits": {"train": [0, 0]}}),
            )
            config = DatasetConfig(
                name=DatasetName.TOY,
                split_manifest=str(manifest),
                toy_samples=2,
            )

            with self.assertRaisesRegex(ValueError, "repeats"):
                load_dataset(config, _data_runtime())

            manifest.write_text(
                json.dumps({"version": 1, "splits": {"train": [2]}}),
            )
            with self.assertRaisesRegex(IndexError, "outside"):
                load_dataset(config, _data_runtime())
    def test_datamodule_uses_split_manifest_as_store_backed_subset(self):
        with TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "split.json"
            manifest.write_text(
                json.dumps({"version": 1, "splits": {"pilot": [3, 1]}}),
            )
            runtime = _data_runtime()
            config = SpeechConfig(
                codec="longcat",
                dataloader=_loader(2),
                dataset=DatasetConfig(
                    name=DatasetName.TOY,
                    split_manifest=str(manifest),
                    split_label="pilot",
                    toy_samples=4,
                ),
            )
            datamodule = DataModule(
                runtime,
                {"train": LoaderSpec.speech(config, {Task.TTS: 1.0})},
            )

            datamodule.setup()
            loader = cast(Any, datamodule.train_dataloader())

            self.assertIsInstance(loader.dataset, SplitManifestDataset)
            self.assertEqual(loader.dataset.indices, (3, 1))
            self.assertIs(loader.batch_sampler.dataset, loader.dataset)
            self.assertTrue(loader.batch_sampler.shuffle)
            samples = datamodule.diagnostic_samples(
                [0, 1],
                split=SampleSplit.TRAIN,
                loader_name="train",
            )
            texts = [
                sample[(Role.SOURCE, Modality.TEXT)].views[TextView.TEXT]
                for sample in samples
            ]
            self.assertEqual(texts, ["toy source 3", "toy source 1"])
    def test_split_manifest_builder_binds_audit_fingerprint(self):
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            candidate = root / "candidate.json"
            audit = root / "audit.json"
            candidate.write_text(
                json.dumps(
                    {
                        "dataset": "wmt19_tts_codec",
                        "codec": "longcat",
                        "train": [0, 1],
                        "dev": [2],
                        "test": [3],
                    }
                )
            )
            audit.write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "relative_path": "samples.parquet",
                                "sha256": "abc",
                                "parquet": {"num_rows": 4},
                            }
                        ],
                    }
                )
            )

            manifest = build_manifest(
                candidate,
                audit,
                Path("/stable/root"),
                split_method="sequential_no_sample_id",
            )

            self.assertEqual(manifest["dataset_length"], 4)
            self.assertEqual(manifest["split_method"], "sequential_no_sample_id")
            self.assertEqual(manifest["root_fingerprint"], {"samples.parquet": "abc"})
            self.assertEqual(manifest["splits"], {"train": [0, 1], "dev": [2], "test": [3]})
    def test_split_manifest_preserves_map_style_shuffle_groups(self):
        class GroupedDataset(MapStyleABC):
            def __len__(self):
                return 4

            def __getitem__(self, index):
                return cast(Any, index)

            def _shuffle(self, **kwargs):
                del kwargs
                yield (0, 1)
                yield (2, 3)

        dataset = SplitManifestDataset(
            GroupedDataset(),
            [3, 0],
            manifest=Path("/stable/split.json"),
            label="train",
        )

        groups = list(
            dataset._shuffle(
                shuffle=True,
                seed=0,
                epoch=0,
                num_replicas=1,
                rank=0,
            )
        )

        self.assertEqual(groups, [(1,), (0,)])


if __name__ == "__main__":
    unittest.main()
