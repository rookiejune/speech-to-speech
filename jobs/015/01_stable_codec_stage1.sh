#!/usr/bin/env bash
set -euo pipefail

REPOS_ROOT="${REPOS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${REPOS_ROOT}/workspace/jobs/env.sh"

data_root="${SPEECH_TO_SPEECH_STABLE_DATA_ROOT:?set SPEECH_TO_SPEECH_STABLE_DATA_ROOT}"
split_manifest="${SPEECH_TO_SPEECH_STABLE_SPLIT_MANIFEST:?set SPEECH_TO_SPEECH_STABLE_SPLIT_MANIFEST}"
qwen_root="$(fdu_qwen_root)"
visible_devices="${CUDA_VISIBLE_DEVICES:-${SPEECH_TO_SPEECH_STABLE_GPUS:-0,1}}"
stable_python="${SPEECH_TO_SPEECH_STABLE_PYTHON:?set SPEECH_TO_SPEECH_STABLE_PYTHON to a stable-codec-compatible Python}"

cd "${SPEECH_TO_SPEECH_ROOT}"
echo "{\"event\":\"job.launch\",\"entry\":\"scripts/train.py\",\"experiment\":\"train/stable_codec_stage1_train\",\"codec\":\"stable_codec\",\"devices\":\"${visible_devices}\"}"
CUDA_VISIBLE_DEVICES="${visible_devices}" "${stable_python}" scripts/train.py \
  "experiment=train/stable_codec_stage1_train" \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "data.dataset.root=${data_root}" \
  "data.dataset.split_manifest=${split_manifest}" \
  "$@"
