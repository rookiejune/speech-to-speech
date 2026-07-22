#!/usr/bin/env bash
set -euo pipefail

export LOCATION="${LOCATION:-fudan}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

if [[ -z "${SPEECH_TO_SPEECH_PYTHON:-}" && -x /home/zhuyin/anaconda3/envs/py312/bin/python ]]; then
  export SPEECH_TO_SPEECH_PYTHON=/home/zhuyin/anaconda3/envs/py312/bin/python
fi

source "$(dirname "${BASH_SOURCE[0]}")/../env.sh"
export HF_HOME="${HF_HOME:-${STATIC_HOME}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export ANYTRAIN_HOME="${ANYTRAIN_HOME:-${STATIC_HOME}/.anytrain}"

default_qwen_root="${HF_HUB_CACHE}/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
if [[ -z "${SPEECH_TO_SPEECH_STAGE_QWEN_ROOT:-}" && -d "${default_qwen_root}" ]]; then
  qwen_root="${default_qwen_root}"
else
  qwen_root="${SPEECH_TO_SPEECH_STAGE_QWEN_ROOT:-Qwen/Qwen3-0.6B}"
fi

data_args=()
if [[ -n "${SPEECH_TO_SPEECH_STAGE_DATA_ROOT:-}" ]]; then
  data_args=("data.dataset.root=${SPEECH_TO_SPEECH_STAGE_DATA_ROOT}")
fi

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","experiment":"fdu_stage_4_acoustic_none_smoke","stage":"stage_4","objective":"token"}'
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/train.py \
  experiment=fdu_stage_4_acoustic_none_smoke \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "${data_args[@]}" \
  "$@"
