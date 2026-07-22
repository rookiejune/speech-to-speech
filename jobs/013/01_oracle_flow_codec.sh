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

data_args=()
if [[ -n "${SPEECH_TO_SPEECH_ORACLE_DATA_ROOT:-}" ]]; then
  data_args=("codec_oracle.data.root=${SPEECH_TO_SPEECH_ORACLE_DATA_ROOT}")
fi

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","experiment":"fdu_oracle_flow_codec_smoke","objective":"flow","initialization":"codec"}'
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/codec_oracle.py \
  experiment=fdu_oracle_flow_codec_smoke \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "${data_args[@]}" \
  "$@"
