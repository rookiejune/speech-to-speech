#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

export ANYTRAIN_DEBUG="${ANYTRAIN_DEBUG:-True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONUNBUFFERED=1

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","codec":"unicodec","strategy":"ddp","lba":false}'
"${SPEECH_TO_SPEECH_UNICODEC_PYTHON}" scripts/overfit.py \
  trainer=ddp \
  trainer.max_epochs=-1 \
  trainer.precision=bf16-mixed \
  codec=unicodec \
  runtime=native \
  task=tts \
  output_dir="${SPEECH_TO_SPEECH_TRAIN_ROOT}/005-codec-screening/unicodec/formal-ddp" \
  "$@"
