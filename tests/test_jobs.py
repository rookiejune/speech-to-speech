from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
JOBS = ROOT / "jobs"


def _environment(**overrides: str) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "REPOS_ROOT",
        "WORKSPACE_ROOT",
        "SPEECH_TO_SPEECH_ROOT",
        "SPEECH_TO_SPEECH_EXPERIMENT",
        "SPEECH_TO_SPEECH_PYTHON",
        "SPEECH_TO_SPEECH_ACOUSTIC_GENERATOR_ARTIFACT",
        "SPEECH_TO_SPEECH_STAGE_DATA_ROOT",
        "SPEECH_TO_SPEECH_STEP_MODE",
        "CUDA_VISIBLE_DEVICES",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "WORKSPACE_SKIP_CONDA_ACTIVATE": "1",
            "WORKSPACE_PYTHON": sys.executable,
            "LOCATION": "fudan",
            "STATIC_HOME": "/private/tmp/speech-to-speech-jobs-static",
            "DYNAMIC_HOME": "/private/tmp/speech-to-speech-jobs-dynamic",
            "PYTHONPYCACHEPREFIX": "/private/tmp/speech-to-speech-jobs-pycache",
            "SPEECH_TO_SPEECH_ACOUSTIC_GENERATOR_ARTIFACT": "/private/tmp/speech-to-speech-sac-artifact",
            "SPEECH_TO_SPEECH_STAGE_DATA_ROOT": "/private/tmp/speech-to-speech-jobs-data",
        }
    )
    environment.update(overrides)
    return environment


def _run(
    path: Path,
    *args: str,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(path), *args],
        cwd=ROOT,
        env=environment or _environment(),
        text=True,
        capture_output=True,
        check=False,
    )


class JobsTest(unittest.TestCase):
    def test_identity_guard_rejects_hydra_override_forms(self) -> None:
        command = (
            'source "$1"; shift; '
            'job_reject_overrides experiment task loader_plan -- "$@"'
        )
        valid = subprocess.run(
            ["bash", "-c", command, "jobs-test", str(JOBS / "env.sh"), "train.max_steps=1"],
            cwd=ROOT,
            env=_environment(),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(valid.returncode, 0, valid.stderr)

        for override in (
            "experiment=other",
            "+task=s2st",
            "++loader_plan.loaders.extra.weight=1.0",
            "~task",
            "experiment@_global_=other",
        ):
            with self.subTest(override=override):
                result = subprocess.run(
                    ["bash", "-c", command, "jobs-test", str(JOBS / "env.sh"), override],
                    cwd=ROOT,
                    env=_environment(),
                    text=True,
                    capture_output=True,
                    check=False,
                )

                self.assertEqual(result.returncode, 2)
                self.assertIn("cannot be overridden", result.stderr)
                self.assertIn(override, result.stderr)

    def test_staged_wrapper_rejects_identity_arguments_before_training(self) -> None:
        for override in (
            "experiment=other",
            "model.acoustic.init_artifact=null",
        ):
            with self.subTest(override=override):
                result = _run(
                    JOBS / "011" / "03_staged_joint_train.sh",
                    override,
                )

                self.assertEqual(result.returncode, 2)
                self.assertIn(override.split("=", 1)[0], result.stderr)
                self.assertIn("cannot be overridden", result.stderr)

    def test_staged_wrapper_rejects_unknown_step_mode(self) -> None:
        result = _run(
            JOBS / "011" / "03_staged_joint_train.sh",
            environment=_environment(SPEECH_TO_SPEECH_STEP_MODE="other"),
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("SPEECH_TO_SPEECH_STEP_MODE", result.stderr)

    def test_staged_wrapper_defaults_modes_and_windows_by_stage(self) -> None:
        cases = (
            (None, "stage_0", "serial_joint", "staged_ddp", "2"),
            ("train/staged_joint/stage_1", "stage_1", "fused_joint", "staged_static_ddp", "3"),
            ("train/staged_joint/stage_2", "stage_2", "fused_joint", "staged_static_ddp", "5"),
            ("train/staged_joint/stage_3", "stage_3", "fused_joint", "staged_static_ddp", "6"),
        )

        for experiment, stage, mode, trainer, window in cases:
            overrides = {"SPEECH_TO_SPEECH_PYTHON": "/bin/echo"}
            if experiment is not None:
                overrides["SPEECH_TO_SPEECH_EXPERIMENT"] = experiment
            with self.subTest(stage=stage):
                result = _run(
                    JOBS / "011" / "03_staged_joint_train.sh",
                    environment=_environment(**overrides),
                )

                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f"experiment=train/staged_joint/{stage}", result.stdout)
                self.assertIn(f"trainer={trainer}", result.stdout)
                self.assertIn(f"loader_plan.step_mode={mode}", result.stdout)
                self.assertIn(
                    "model.acoustic.init_artifact=/private/tmp/speech-to-speech-sac-artifact",
                    result.stdout,
                )
                self.assertIn(
                    f"loader_plan.accumulate_grad_batches={window}",
                    result.stdout,
                )

    def test_staged_wrapper_explicit_serial_mode_uses_find_unused_trainer(self) -> None:
        result = _run(
            JOBS / "011" / "03_staged_joint_train.sh",
            environment=_environment(
                SPEECH_TO_SPEECH_EXPERIMENT="train/staged_joint/stage_3",
                SPEECH_TO_SPEECH_STEP_MODE="serial_joint",
                SPEECH_TO_SPEECH_PYTHON="/bin/echo",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("trainer=staged_ddp", result.stdout)
        self.assertIn("loader_plan.step_mode=serial_joint", result.stdout)
        self.assertIn("loader_plan.accumulate_grad_batches=6", result.stdout)

    def test_staged_wrapper_stage_0_fused_mode_uses_find_unused_trainer(self) -> None:
        result = _run(
            JOBS / "011" / "03_staged_joint_train.sh",
            environment=_environment(
                SPEECH_TO_SPEECH_STEP_MODE="fused_joint",
                SPEECH_TO_SPEECH_PYTHON="/bin/echo",
            ),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("trainer=staged_ddp", result.stdout)
        self.assertIn("loader_plan.step_mode=fused_joint", result.stdout)
        self.assertIn("loader_plan.accumulate_grad_batches=2", result.stdout)

    def test_staged_wrapper_rejects_unknown_experiment(self) -> None:
        result = _run(
            JOBS / "011" / "03_staged_joint_train.sh",
            environment=_environment(SPEECH_TO_SPEECH_EXPERIMENT="train/staged_joint/stage_4"),
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("must be", result.stderr)

    def test_staged_wrapper_requires_external_generator_artifact(self) -> None:
        result = _run(
            JOBS / "011" / "03_staged_joint_train.sh",
            environment=_environment(SPEECH_TO_SPEECH_ACOUSTIC_GENERATOR_ARTIFACT=""),
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SPEECH_TO_SPEECH_ACOUSTIC_GENERATOR_ARTIFACT", result.stderr)


if __name__ == "__main__":
    unittest.main()
