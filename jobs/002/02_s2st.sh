#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${SPEECH_TO_SPEECH_ROOT}"
"${SPEECH_TO_SPEECH_PYTHON}" scripts/overfit.py \
  experiment=overfit \
  task=s2st \
  output_dir="${SPEECH_TO_SPEECH_TRAIN_ROOT}/002-single-batch-overfit/s2st" \
  "$@"
