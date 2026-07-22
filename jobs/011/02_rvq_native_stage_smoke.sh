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
  data_args=("data.root=${SPEECH_TO_SPEECH_STAGE_DATA_ROOT}")
fi

launcher_dir="${SPEECH_TO_SPEECH_TRAIN_ROOT}/011-rvq-native-stage-smoke/launcher"
mkdir -p "${launcher_dir}"

run_stage() {
  local stage="$1"
  local experiment="011_rvq_native_stage_${stage}_smoke"
  local status
  shift

  (
    set +e
    cd "${SPEECH_TO_SPEECH_ROOT}" || exit 2
    CUDA_VISIBLE_DEVICES="${SPEECH_TO_SPEECH_STAGE_GPU:-0}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/overfit.py \
      "experiment=${experiment}" \
      "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
      "runtime.backbone=${qwen_root}" \
      "${data_args[@]}" \
      "$@"
    status=$?
    printf '%s\n' "${status}" >"${launcher_dir}/stage_${stage}.exit"
    exit "${status}"
  ) >"${launcher_dir}/stage_${stage}.log" 2>&1
  status=$?
  return "${status}"
}

echo '{"event":"job.launch","experiment":"011_rvq_native_stage_smoke","stages":"0,1,2,3,4","codec":"longcat","tokenizer":"native","objective":"rvq"}'
echo "{\"event\":\"job.paths\",\"output_root\":\"${SPEECH_TO_SPEECH_TRAIN_ROOT}\",\"launcher_dir\":\"${launcher_dir}\"}"

overall_status=0
for stage in 0 1 2 3 4; do
  if ! run_stage "${stage}" "$@"; then
    overall_status=1
  fi
done

printf '%s\n' "${overall_status}" >"${launcher_dir}/overall.exit"
exit "${overall_status}"
