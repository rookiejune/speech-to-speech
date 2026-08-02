#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${JOB_DIR%/jobs/*}/jobs/env.sh"
: "${SPEECH_TO_SPEECH_UNICODEC_PYTHON:?Set SPEECH_TO_SPEECH_UNICODEC_PYTHON to a fairseq-compatible Python executable}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONUNBUFFERED=1

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","codec":"unicodec","strategy":"ddp"}'
"${SPEECH_TO_SPEECH_UNICODEC_PYTHON}" scripts/overfit.py \
  experiment=overfit/unicodec_ddp_smoke \
  repo_output_root="${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  output_subdir="005-codec-screening/unicodec/ddp-smoke" \
  "$@"
