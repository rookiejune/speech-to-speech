#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"

export ANYTRAIN_DEBUG="${ANYTRAIN_DEBUG:-True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONUNBUFFERED=1

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","codec":"unicodec","strategy":"ddp","lba":true}'
"${SPEECH_TO_SPEECH_UNICODEC_PYTHON}" scripts/codec_oracle.py \
  experiment=acoustic_oracle_ddp_lba \
  codec=unicodec \
  output_dir="${SPEECH_TO_SPEECH_TRAIN_ROOT}/005-codec-oracle-ddp-lba/unicodec/\${init.name}" \
  "$@"
