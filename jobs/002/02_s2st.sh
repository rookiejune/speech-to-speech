#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${SPEECH_TO_SPEECH_ROOT}"
"${SPEECH_TO_SPEECH_PYTHON}" scripts/overfit.py \
  --task s2st \
  --audio-tokenizer "${SPEECH_TO_SPEECH_AUDIO_TOKENIZER}" \
  --output-dir "${SPEECH_TO_SPEECH_TRAIN_ROOT}/002-single-batch-overfit/s2st" \
  "$@"
