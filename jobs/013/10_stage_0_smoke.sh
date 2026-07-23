#!/usr/bin/env bash
set -euo pipefail

REPOS_ROOT="${REPOS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${REPOS_ROOT}/workspace/jobs/fudan/speech_to_speech_env.sh"

qwen_root="$(fdu_qwen_root)"
fdu_stage_data_args data.root

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","experiment":"fdu_stage_0_smoke","stage":"stage_0","objective":"rvq","entry":"overfit"}'
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/overfit.py \
  experiment=fdu_stage_0_smoke \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "${FDU_DATA_ARGS[@]}" \
  "$@"
