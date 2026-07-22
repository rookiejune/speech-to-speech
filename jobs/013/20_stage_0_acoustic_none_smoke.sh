#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/fdu_env.sh"

qwen_root="$(fdu_qwen_root)"
fdu_stage_data_args data.root

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","experiment":"fdu_stage_0_acoustic_none_smoke","stage":"stage_0","objective":"token","entry":"overfit"}'
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/overfit.py \
  experiment=fdu_stage_0_acoustic_none_smoke \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "${FDU_DATA_ARGS[@]}" \
  "$@"
