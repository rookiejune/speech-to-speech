#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${JOB_DIR%/jobs/*}/jobs/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${SPEECH_TO_SPEECH_ROOT}"

EXTRA_ARGS=()
if [[ -n "${SPEECH_TO_SPEECH_STAGE_DATA_ROOT:-}" ]]; then
  EXTRA_ARGS+=(--data-root "${SPEECH_TO_SPEECH_STAGE_DATA_ROOT}")
fi
if [[ -n "${SPEECH_TO_SPEECH_SPLIT_MANIFEST:-}" ]]; then
  EXTRA_ARGS+=(--split-manifest "${SPEECH_TO_SPEECH_SPLIT_MANIFEST}")
fi
if [[ -n "${SPEECH_TO_SPEECH_SPLIT_LABEL:-}" ]]; then
  EXTRA_ARGS+=(--split-label "${SPEECH_TO_SPEECH_SPLIT_LABEL}")
fi

"${SPEECH_TO_SPEECH_PYTHON}" scripts/generation_smoke.py \
  --audio-tokenizer "${SPEECH_TO_SPEECH_AUDIO_TOKENIZER}" \
  --output-dir "${SPEECH_TO_SPEECH_TRAIN_ROOT}/004-real-cached-generation" \
  "${EXTRA_ARGS[@]}" \
  "$@"
