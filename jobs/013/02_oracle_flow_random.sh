#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/fdu_env.sh"

fdu_oracle_data_args

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","experiment":"fdu_oracle_flow_random_smoke","objective":"flow","initialization":"random"}'
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/codec_oracle.py \
  experiment=fdu_oracle_flow_random_smoke \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "${FDU_DATA_ARGS[@]}" \
  "$@"
