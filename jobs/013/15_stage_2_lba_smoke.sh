#!/usr/bin/env bash
set -euo pipefail

REPOS_ROOT="${REPOS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${REPOS_ROOT}/workspace/jobs/fudan/speech_to_speech_env.sh"

qwen_root="$(fdu_qwen_root)"
fdu_stage_data_args data.dataset.root

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","experiment":"fdu_stage_2_lba_smoke","stage":"stage_2","objective":"rvq","entry":"train","lba":true}'
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/train.py \
  experiment=fdu_stage_2_lba_smoke \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "${FDU_DATA_ARGS[@]}" \
  "$@"
