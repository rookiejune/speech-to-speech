#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/fdu_env.sh"

qwen_root="$(fdu_qwen_root)"
fdu_stage_data_args data.dataset.root

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","experiment":"fdu_stage_2_acoustic_none_smoke","stage":"stage_2","objective":"token","entry":"train"}'
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/train.py \
  experiment=fdu_stage_2_acoustic_none_smoke \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "${FDU_DATA_ARGS[@]}" \
  "$@"
