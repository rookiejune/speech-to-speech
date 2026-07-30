#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${JOB_DIR%/jobs/*}/jobs/env.sh"

qwen_root="$(fdu_qwen_root)"

stage="${SPEECH_TO_SPEECH_STAGE:-stage_1}"
case "$stage" in
  stage_1 | stage_2 | stage_3 | stage_4)
    ;;
  *)
    echo "SPEECH_TO_SPEECH_STAGE must be stage_1 through stage_4, got: $stage" >&2
    exit 2
    ;;
esac
experiment="train/staged_joint_${stage}"
visible_devices="${CUDA_VISIBLE_DEVICES:-${SPEECH_TO_SPEECH_STAGE_GPUS:-0,1}}"
job_reject_overrides experiment task stage -- "$@"

fdu_stage_data_args data.dataset.root

cd "${SPEECH_TO_SPEECH_ROOT}"
echo "{\"event\":\"job.launch\",\"entry\":\"scripts/train.py\",\"experiment\":\"${experiment}\",\"stage\":\"${stage}\",\"devices\":\"${visible_devices}\"}"
CUDA_VISIBLE_DEVICES="${visible_devices}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/train.py \
  "experiment=${experiment}" \
  "trainer=staged_static_ddp" \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "${FDU_DATA_ARGS[@]}" \
  "$@"
