#!/usr/bin/env bash
set -euo pipefail

REPOS_ROOT="${REPOS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${REPOS_ROOT}/workspace/jobs/fudan/speech_to_speech_env.sh"

qwen_root="$(fdu_qwen_root)"

stage="${SPEECH_TO_SPEECH_STAGE:-stage_1}"
visible_devices="${CUDA_VISIBLE_DEVICES:-${SPEECH_TO_SPEECH_STAGE_GPUS:-0,1}}"

fdu_stage_data_args data.dataset.root

cd "${SPEECH_TO_SPEECH_ROOT}"
echo "{\"event\":\"job.launch\",\"entry\":\"scripts/train.py\",\"stage\":\"${stage}\",\"devices\":\"${visible_devices}\"}"
CUDA_VISIBLE_DEVICES="${visible_devices}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/train.py \
  "stage=${stage}" \
  "trainer=static_ddp" \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "${FDU_DATA_ARGS[@]}" \
  "$@"
