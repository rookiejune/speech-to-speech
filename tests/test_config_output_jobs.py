from __future__ import annotations

# ruff: noqa: F403,F405

import unittest

from _config_helpers import *


@patch.dict(
    "os.environ",
    {
        "DYNAMIC_HOME": "/tmp/dynamic",
        "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
    },
)
class ConfigOutputJobTest(ConfigTestCase):
    def test_training_outputs_use_one_tensorboard_root(self):
        configs = (
            _overfit(),
            _overfit("experiment=overfit/unicodec"),
        )

        for config in configs:
            with self.subTest(output_subdir=config.output_subdir):
                root = Path(config.repo_output_root)
                self.assertEqual(
                    Path(config.output_dir),
                    root / config.output_subdir,
                )
                self.assertEqual(
                    Path(config.logging.save_dir),
                    root / "tensorboard",
                )
                self.assertEqual(config.logging.run_name, config.output_subdir)

        csv = _overfit("experiment=overfit/toy_smoke")
        self.assertEqual(csv.logging.save_dir, csv.output_dir)
        self.assertEqual(csv.logging.run_name, "csv")

    def test_repo_output_root_prefers_the_project_training_root(self):
        with patch.dict(
            "os.environ",
            {"SPEECH_TO_SPEECH_TRAIN_ROOT": "/tmp/speech-train"},
        ):
            overfit_config = _overfit()

        self.assertEqual(overfit_config.repo_output_root, "/tmp/speech-train")

    def test_repo_output_root_falls_back_to_the_dynamic_train_root(self):
        with patch.dict(
            "os.environ",
            {
                "DYNAMIC_HOME": "/tmp/dynamic",
                "SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer",
            },
            clear=True,
        ):
            config = _overfit()

        self.assertEqual(config.repo_output_root, "/tmp/dynamic/train/speech-to-speech")

    def test_missing_training_root_fails_without_dynamic_home(self):
        with (
            patch.dict(
                "os.environ",
                {"SPEECH_TO_SPEECH_AUDIO_TOKENIZER": "/tmp/audio-tokenizer"},
                clear=True,
            ),
            self.assertRaisesRegex(InterpolationResolutionError, "DYNAMIC_HOME"),
        ):
            _overfit()

    def test_output_subdir_cannot_escape_the_repo_output_root(self):
        for override in ("output_subdir=/tmp/run", "output_subdir=../run"):
            with self.subTest(override=override):
                with self.assertRaisesRegex(ValueError, "output_subdir"):
                    _overfit(override)

        with self.assertRaisesRegex(ValueError, "output_dir must equal"):
            _overfit("output_dir=/tmp/other")

    def test_overfit_run_name_preserves_composition_and_decoder_depth(self):
        cases = [
            ((), "flow-8l"),
            (("model/acoustic=rvq",), "rvq-8l"),
            (("model.acoustic.decoder.layers=3",), "flow-3l"),
            (
                (
                    "runtime=unicodec",
                    "model/acoustic=none",
                    "audio_sequence_layout=flattened",
                ),
                "token",
            ),
        ]

        for overrides, expected in cases:
            with self.subTest(expected=expected):
                config = overfit(_compose("overfit", *overrides))
                self.assertEqual(config.run_name, expected)
                self.assertEqual(Path(config.output_dir).name, expected)

    def test_overfit_jobs_use_the_token_safe_run_name(self):
        root = Path(__file__).parents[1]
        jobs = {"01_tts.sh": "tts", "02_s2st.sh": "s2st"}

        for filename, task in jobs.items():
            with self.subTest(job=filename):
                source = (root / "jobs" / "002" / filename).read_text()
                match = re.search(r'output_subdir="([^"]+)"', source)
                self.assertIsNotNone(match)
                subdir = match.group(1).replace(r"\${", "${")
                config = overfit(
                    _compose(
                        "overfit",
                        "runtime=unicodec",
                        "model/acoustic=none",
                        "audio_sequence_layout=flattened",
                        f"task={task}",
                        "repo_output_root=/tmp/train",
                        f"output_subdir={subdir}",
                    )
                )
                self.assertEqual(
                    config.output_dir,
                    f"/tmp/train/002-single-batch-overfit/{task}/token",
                )
                self.assertEqual(
                    config.logging.save_dir,
                    "/tmp/train/tensorboard",
                )
                self.assertEqual(
                    config.logging.run_name,
                    f"002-single-batch-overfit/{task}/token",
                )

    def test_training_jobs_override_root_and_relative_subdir(self):
        root = Path(__file__).parents[1]
        jobs = [*sorted((root / "jobs" / "002").glob("*.sh"))]
        jobs.extend(sorted((root / "jobs" / "005").glob("*.sh")))

        for path in jobs:
            with self.subTest(job=path.name):
                source = path.read_text()
                self.assertIn(
                    'repo_output_root="${SPEECH_TO_SPEECH_TRAIN_ROOT}"',
                    source,
                )
                match = re.search(r'output_subdir="([^"]+)"', source)
                self.assertIsNotNone(match)
                self.assertFalse(Path(match.group(1)).is_absolute())
                self.assertNotRegex(source, r"\boutput_dir=")

    def test_staged_joint_job_uses_train_entry(self):
        root = Path(__file__).parents[1]
        source = (root / "jobs" / "011" / "03_staged_joint_train.sh").read_text()

        self.assertIn("scripts/train.py", source)
        self.assertNotIn("scripts/overfit.py", source)
        self.assertIn('SPEECH_TO_SPEECH_STEP_MODE:-fused_joint', source)
        self.assertIn('trainer="staged_static_ddp"', source)
        self.assertIn('trainer="staged_ddp"', source)
        self.assertIn('"loader_plan.step_mode=${step_mode}"', source)
        self.assertIn("fdu_stage_data_args datamodule.dataset.root", source)
        self.assertIn("SPEECH_TO_SPEECH_EXPERIMENT:-train/staged_joint/stage_1", source)
        self.assertIn('"experiment=${experiment}"', source)
        self.assertIn('job_reject_overrides experiment task loader_plan -- "$@"', source)

    def test_job_wrappers_source_existing_project_environment(self):
        root = Path(__file__).parents[1]
        env = root / "jobs" / "env.sh"
        jobs = sorted(path for path in (root / "jobs").rglob("*.sh") if path != env)

        self.assertTrue(env.is_file())
        self.assertTrue(env.stat().st_mode & 0o111)
        self.assertTrue(jobs)
        self.assertFalse((root / "jobs" / "013" / "fdu_env.sh").exists())
        for path in jobs:
            with self.subTest(job=str(path.relative_to(root))):
                source = path.read_text()
                jobs_dir = next(parent for parent in path.parents if parent.name == "jobs")

                self.assertEqual(jobs_dir / "env.sh", env)
                self.assertIn(
                    'JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
                    source,
                )
                self.assertIn('source "${JOB_DIR%/jobs/*}/jobs/env.sh"', source)
                self.assertNotIn("workspace/jobs/env.sh", source)
                self.assertTrue(path.stat().st_mode & 0o111)
                self.assertNotRegex(
                    source,
                    r"/(?:home|mnt|Users)/|hf-mirror|Qwen3-0\.6B|HF_HOME|ANYTRAIN_HOME",
                )

    def test_project_jobs_env_owns_speech_settings(self):
        root = Path(__file__).parents[1]
        project_env = (root / "jobs" / "env.sh").read_text()
        workspace_env = (root / "../workspace" / "jobs" / "env.sh").resolve().read_text()

        self.assertIn('source "${REPOS_ROOT}/workspace/jobs/env.sh"', project_env)
        for name in (
            "SPEECH_TO_SPEECH_ROOT",
            "SPEECH_TO_SPEECH_PYTHON",
            "SPEECH_TO_SPEECH_TRAIN_ROOT",
            "SPEECH_TO_SPEECH_AUDIO_TOKENIZER",
            "job_reject_overrides",
            "fdu_stage_data_args",
            "fdu_qwen_root",
        ):
            with self.subTest(name=name):
                self.assertIn(name, project_env)
                self.assertNotIn(name, workspace_env)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", workspace_env)

    def test_jobs_default_the_training_root_to_dynamic_home_train(self):
        root = Path(__file__).parents[1]
        source = (root / "jobs" / "env.sh").read_text()

        self.assertIn(
            'SPEECH_TO_SPEECH_TRAIN_ROOT="${SPEECH_TO_SPEECH_TRAIN_ROOT:-${DYNAMIC_HOME}/train/speech-to-speech}"',
            source,
        )

    def test_unicodec_jobs_require_a_compatible_python(self):
        root = Path(__file__).parents[1]
        env = (root / "jobs" / "env.sh").read_text()
        jobs = {
            "02_unicodec.sh": ("overfit/unicodec", "overfit"),
            "05_unicodec_ddp.sh": ("overfit/unicodec_ddp_smoke", "ddp-smoke"),
        }

        self.assertNotIn(
            "SPEECH_TO_SPEECH_UNICODEC_PYTHON=",
            env,
        )
        for filename, (experiment, output_name) in jobs.items():
            with self.subTest(job=filename):
                source = (root / "jobs" / "005" / filename).read_text()
                config = overfit(_compose("overfit", f"experiment={experiment}"))
                self.assertIn(
                    "SPEECH_TO_SPEECH_UNICODEC_PYTHON:?Set ",
                    source,
                )
                self.assertTrue(config.output_subdir.endswith(f"/{output_name}"))
                self.assertIn(
                    f'output_subdir="{config.output_subdir}"',
                    source,
                )

    def test_unicodec_smoke_jobs_select_complete_experiments(self):
        root = Path(__file__).parents[1]
        jobs = {
            "02_unicodec.sh": "overfit/unicodec",
            "05_unicodec_ddp.sh": "overfit/unicodec_ddp_smoke",
        }

        for filename, expected in jobs.items():
            with self.subTest(job=filename):
                source = (root / "jobs" / "005" / filename).read_text()
                self.assertEqual(
                    re.findall(r"\bexperiment=([a-z0-9_/]+)", source),
                    [expected],
                )


if __name__ == "__main__":
    unittest.main()
