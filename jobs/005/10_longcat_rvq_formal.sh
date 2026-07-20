#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","codec":"longcat","objective":"rvq","profile":"formal"}'
"${SPEECH_TO_SPEECH_PYTHON}" scripts/codec_oracle.py \
  codec_oracle=rvq \
  output_dir="${SPEECH_TO_SPEECH_TRAIN_ROOT}/005-codec-oracle/longcat/rvq-\${codec_oracle.decoder.layers}l/\${codec_oracle.initialization}" \
  "$@"
