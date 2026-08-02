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
        "SPEECH_TO_SPEECH_STAGE",
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
            'job_reject_overrides experiment task stage stage_id -- "$@"'
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
            "++stage=stage_4",
            "stage_id=stage_4",
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
        result = _run(
            JOBS / "011" / "03_staged_joint_train.sh",
            "experiment=other",
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("experiment", result.stderr)
        self.assertIn("cannot be overridden", result.stderr)

    def test_staged_wrapper_rejects_stage_zero(self) -> None:
        result = _run(
            JOBS / "011" / "03_staged_joint_train.sh",
            environment=_environment(SPEECH_TO_SPEECH_STAGE="stage_0"),
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("must be", result.stderr)


if __name__ == "__main__":
    unittest.main()
