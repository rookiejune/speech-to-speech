#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONUNBUFFERED=1

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","codec":"unicodec","strategy":"ddp","lba":false}'
"${SPEECH_TO_SPEECH_UNICODEC_PYTHON}" scripts/overfit.py \
  experiment=unicodec_ddp_smoke \
  output_dir="${SPEECH_TO_SPEECH_TRAIN_ROOT}/005-codec-screening/unicodec/formal-ddp" \
  "$@"
