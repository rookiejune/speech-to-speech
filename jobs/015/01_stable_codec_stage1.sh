#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${JOB_DIR%/jobs/*}/jobs/env.sh"

data_root="${SPEECH_TO_SPEECH_STABLE_DATA_ROOT:?set SPEECH_TO_SPEECH_STABLE_DATA_ROOT}"
split_manifest="${SPEECH_TO_SPEECH_STABLE_SPLIT_MANIFEST:?set SPEECH_TO_SPEECH_STABLE_SPLIT_MANIFEST}"
qwen_root="$(fdu_qwen_root)"
visible_devices="${CUDA_VISIBLE_DEVICES:?the scheduler or caller must assign the training GPUs}"
stable_python="${SPEECH_TO_SPEECH_STABLE_PYTHON:?set SPEECH_TO_SPEECH_STABLE_PYTHON to a stable-codec-compatible Python}"

cd "${SPEECH_TO_SPEECH_ROOT}"
echo "{\"event\":\"job.launch\",\"entry\":\"scripts/train.py\",\"experiment\":\"train/stable_codec/stage_1\",\"codec\":\"stable_codec\",\"devices\":\"${visible_devices}\"}"
"${stable_python}" scripts/train.py \
  "experiment=train/stable_codec/stage_1" \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "datamodule.dataset.root=${data_root}" \
  "datamodule.dataset.split_manifest=${split_manifest}" \
  "$@"
