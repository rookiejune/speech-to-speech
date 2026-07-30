#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${JOB_DIR%/jobs/*}/jobs/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${SPEECH_TO_SPEECH_ROOT}"
"${SPEECH_TO_SPEECH_PYTHON}" scripts/overfit.py \
  experiment=overfit \
  task=tts \
  repo_output_root="${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  output_subdir="002-single-batch-overfit/tts/\${run_name}" \
  "$@"
