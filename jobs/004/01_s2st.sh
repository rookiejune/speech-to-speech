#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${JOB_DIR%/jobs/*}/jobs/env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

cd "${SPEECH_TO_SPEECH_ROOT}"

HYDRA_ARGS=(
  "runtime.audio_output.bpe=${SPEECH_TO_SPEECH_AUDIO_TOKENIZER}"
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}"
)
if [[ -n "${SPEECH_TO_SPEECH_STAGE_DATA_ROOT:-}" ]]; then
  HYDRA_ARGS+=("datamodule.dataset.root=${SPEECH_TO_SPEECH_STAGE_DATA_ROOT}")
fi
if [[ -n "${SPEECH_TO_SPEECH_SPLIT_MANIFEST:-}" ]]; then
  HYDRA_ARGS+=("datamodule.dataset.split_manifest=${SPEECH_TO_SPEECH_SPLIT_MANIFEST}")
fi
if [[ -n "${SPEECH_TO_SPEECH_SPLIT_LABEL:-}" ]]; then
  HYDRA_ARGS+=("datamodule.dataset.split_label=${SPEECH_TO_SPEECH_SPLIT_LABEL}")
fi

"${SPEECH_TO_SPEECH_PYTHON}" scripts/generation_smoke.py \
  "${HYDRA_ARGS[@]}" \
  "$@"
