#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${JOB_DIR%/jobs/*}/jobs/env.sh"

qwen_root="$(fdu_qwen_root)"

experiment="${SPEECH_TO_SPEECH_EXPERIMENT:-train/staged_joint/stage_1}"
case "$experiment" in
  train/staged_joint/stage_1 | train/staged_joint/stage_2 | train/staged_joint/stage_3 | train/staged_joint/stage_4)
    ;;
  *)
    echo "SPEECH_TO_SPEECH_EXPERIMENT must be train/staged_joint/stage_1 through stage_4, got: $experiment" >&2
    exit 2
    ;;
esac
visible_devices="${CUDA_VISIBLE_DEVICES:-${SPEECH_TO_SPEECH_EXPERIMENT_GPUS:-0,1}}"
job_reject_overrides experiment task loader_plan -- "$@"

fdu_stage_data_args datamodule.dataset.root

cd "${SPEECH_TO_SPEECH_ROOT}"
echo "{\"event\":\"job.launch\",\"entry\":\"scripts/train.py\",\"experiment\":\"${experiment}\",\"devices\":\"${visible_devices}\"}"
CUDA_VISIBLE_DEVICES="${visible_devices}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/train.py \
  "experiment=${experiment}" \
  "trainer=staged_static_ddp" \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "${FDU_DATA_ARGS[@]}" \
  "$@"
