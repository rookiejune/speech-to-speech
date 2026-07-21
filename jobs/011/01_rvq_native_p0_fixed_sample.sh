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
if [[ -z "${SPEECH_TO_SPEECH_P0_QWEN_ROOT:-}" && -d "${default_qwen_root}" ]]; then
  qwen_root="${default_qwen_root}"
else
  qwen_root="${SPEECH_TO_SPEECH_P0_QWEN_ROOT:-Qwen/Qwen3-0.6B}"
fi

data_args=()
if [[ -n "${SPEECH_TO_SPEECH_P0_DATA_ROOT:-}" ]]; then
  data_args=("data.root=${SPEECH_TO_SPEECH_P0_DATA_ROOT}")
fi

launcher_dir="${SPEECH_TO_SPEECH_TRAIN_ROOT}/011-qwen-rvq-native-p0-fixed-sample/launcher"
mkdir -p "${launcher_dir}"

run_task() {
  local task="$1"
  local gpu="$2"
  shift 2

  (
    set +e
    cd "${SPEECH_TO_SPEECH_ROOT}" || exit 2
    CUDA_VISIBLE_DEVICES="${gpu}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/overfit.py \
      experiment=011_qwen_rvq_native_p0_fixed_sample \
      "task=${task}" \
      "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
      "runtime.backbone=${qwen_root}" \
      "${data_args[@]}" \
      "$@"
    local status=$?
    printf '%s\n' "${status}" >"${launcher_dir}/${task}.exit"
    exit "${status}"
  ) >"${launcher_dir}/${task}.log" 2>&1 &
  printf '%s\n' "$!" >"${launcher_dir}/${task}.pid"
}

echo '{"event":"job.launch","experiment":"011_qwen_rvq_native_p0_fixed_sample","task":"tts+s2st","codec":"longcat","tokenizer":"native","objective":"rvq"}'
echo "{\"event\":\"job.paths\",\"output_root\":\"${SPEECH_TO_SPEECH_TRAIN_ROOT}\",\"launcher_dir\":\"${launcher_dir}\"}"

run_task tts "${SPEECH_TO_SPEECH_P0_TTS_GPU:-1}" "$@"
tts_pid="$(cat "${launcher_dir}/tts.pid")"
run_task s2st "${SPEECH_TO_SPEECH_P0_S2ST_GPU:-2}" "$@"
s2st_pid="$(cat "${launcher_dir}/s2st.pid")"

set +e
wait "${tts_pid}"
tts_status=$?
wait "${s2st_pid}"
s2st_status=$?
set -e

overall_status=0
if [[ "${tts_status}" -ne 0 || "${s2st_status}" -ne 0 ]]; then
  overall_status=1
fi
printf '%s\n' "${overall_status}" >"${launcher_dir}/overall.exit"
exit "${overall_status}"
