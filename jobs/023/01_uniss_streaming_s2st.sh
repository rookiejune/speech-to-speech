#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${JOB_DIR%/jobs/*}/jobs/env.sh"

visible_devices="${CUDA_VISIBLE_DEVICES:?the scheduler must assign all generation and training GPUs}"
qwen_root="$(fdu_qwen_root)"
job_reject_overrides experiment runtime datamodule.source loader_plan trainer.max_epochs -- "$@"

cd "${SPEECH_TO_SPEECH_ROOT}"
echo "{\"event\":\"job.launch\",\"entry\":\"scripts/train.py\",\"experiment\":\"train/uniss_streaming_s2st\",\"assigned_devices\":\"${visible_devices}\"}"
CUDA_VISIBLE_DEVICES="${visible_devices}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/train.py \
  "experiment=train/uniss_streaming_s2st" \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "$@"
