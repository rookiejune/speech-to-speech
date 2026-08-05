#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${JOB_DIR%/jobs/*}/jobs/env.sh"

: "${SPEECH_TO_SPEECH_STREAM_ROOT:?set the immutable streaming snapshot root}"
: "${SPEECH_TO_SPEECH_STREAM_ID:?set the stable logical stream id}"
: "${SPEECH_TO_SPEECH_STREAM_EXPECTED_SAMPLES:?set the selected 2N seed count}"
: "${SPEECH_TO_SPEECH_PRODUCER_CUDA_VISIBLE_DEVICES:?set producer GPU ids in model-stage order}"

visible_devices="${CUDA_VISIBLE_DEVICES:-${SPEECH_TO_SPEECH_STREAM_TRAIN_GPUS:?set training GPU ids}}"
qwen_root="$(fdu_qwen_root)"
job_reject_overrides experiment datamodule.streaming loader_plan trainer.max_epochs -- "$@"

cd "${SPEECH_TO_SPEECH_ROOT}"
echo "{\"event\":\"job.launch\",\"entry\":\"scripts/train.py\",\"experiment\":\"train/streaming_s2st\",\"train_devices\":\"${visible_devices}\",\"producer_devices\":\"${SPEECH_TO_SPEECH_PRODUCER_CUDA_VISIBLE_DEVICES}\"}"
CUDA_VISIBLE_DEVICES="${visible_devices}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/train.py \
  "experiment=train/streaming_s2st" \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "$@"
