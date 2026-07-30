#!/usr/bin/env bash
set -euo pipefail

SPEECH_TO_SPEECH_JOBS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SPEECH_TO_SPEECH_ROOT="${SPEECH_TO_SPEECH_ROOT:-$(cd -- "$SPEECH_TO_SPEECH_JOBS_DIR/.." && pwd)}"
export REPOS_ROOT="${REPOS_ROOT:-$(cd -- "$SPEECH_TO_SPEECH_ROOT/.." && pwd)}"

source "${REPOS_ROOT}/workspace/jobs/env.sh"

FDU_DATA_ARGS=()

job_reject_overrides() {
  local -a keys=()
  local separator=0
  local arg override key

  while (( $# > 0 )); do
    if [[ "$1" == "--" ]]; then
      separator=1
      shift
      break
    fi
    keys+=("$1")
    shift
  done
  if (( separator == 0 || ${#keys[@]} == 0 )); then
    echo 'job_reject_overrides requires identity keys followed by --' >&2
    return 2
  fi

  for arg in "$@"; do
    override="${arg%%=*}"
    if [[ "${override:0:1}" == "~" ]]; then
      override="${override:1}"
    fi
    while [[ "$override" == +* ]]; do
      override="${override#+}"
    done
    for key in "${keys[@]}"; do
      if [[ "$override" == "$key" \
        || "$override" == "${key}."* \
        || "$override" == "${key}@"* ]]; then
        printf 'job identity %s cannot be overridden through "$@": %s\n' \
          "$key" "$arg" >&2
        return 2
      fi
    done
  done
}

fdu_stage_data_args() {
  local key="${1:-data.dataset.root}"
  FDU_DATA_ARGS=()
  if [[ -n "${SPEECH_TO_SPEECH_STAGE_DATA_ROOT:-}" ]]; then
    FDU_DATA_ARGS=("${key}=${SPEECH_TO_SPEECH_STAGE_DATA_ROOT}")
  fi
}

fdu_qwen_root() {
  local env_name="${1:-SPEECH_TO_SPEECH_STAGE_QWEN_ROOT}"
  local override="${!env_name-}"
  local default_qwen_root="${HF_HUB_CACHE}/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca"
  if [[ -z "${override}" && -d "${default_qwen_root}" ]]; then
    printf '%s\n' "${default_qwen_root}"
  else
    printf '%s\n' "${override:-Qwen/Qwen3-0.6B}"
  fi
}

export SPEECH_TO_SPEECH_PYTHON="${SPEECH_TO_SPEECH_PYTHON:-$WORKSPACE_PYTHON}"
export SPEECH_TO_SPEECH_TRAIN_ROOT="${SPEECH_TO_SPEECH_TRAIN_ROOT:-${DYNAMIC_HOME}/train/speech-to-speech}"
export SPEECH_TO_SPEECH_AUDIO_TOKENIZER="${SPEECH_TO_SPEECH_AUDIO_TOKENIZER:-${STATIC_HOME}/bpe/longcat/vocab_100k_minfreq_0_maxlen_none_codes_8192}"
SPEECH_TO_SPEECH_PYTHONPATH="$SPEECH_TO_SPEECH_ROOT/src:$REPOS_ROOT/semantic-acoustic-codec/src:$REPOS_ROOT/third_party/length-based-batching-adapter/src"
export PYTHONPATH="$SPEECH_TO_SPEECH_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
