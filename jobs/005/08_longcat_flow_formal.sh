#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","codec":"longcat","objective":"flow","profile":"formal"}'
"${SPEECH_TO_SPEECH_PYTHON}" scripts/codec_oracle.py \
  repo_output_root="${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  output_subdir="005-codec-oracle/longcat/flow-\${codec_oracle.decoder.layers}l/\${codec_oracle.initialization}" \
  "$@"
