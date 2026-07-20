#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","codec":"unicodec"}'
"${SPEECH_TO_SPEECH_UNICODEC_PYTHON}" scripts/overfit.py \
  experiment=unicodec_overfit \
  repo_output_root="${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  output_subdir="005-codec-screening/unicodec/formal" \
  "$@"
