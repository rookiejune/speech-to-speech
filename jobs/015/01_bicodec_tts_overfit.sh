#!/usr/bin/env bash
set -euo pipefail

REPOS_ROOT="${REPOS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${REPOS_ROOT}/workspace/jobs/fudan/speech_to_speech_env.sh"

if [[ -n "${SPARK_TTS_ROOT:-}" ]]; then
  export PYTHONPATH="${SPARK_TTS_ROOT}:${PYTHONPATH:-}"
fi

qwen_root="$(fdu_qwen_root)"
bicodec_data_root="${WMT19_QWEN_TTS_ROLE_SPEAKER_BICODEC_ROOT:-${DYNAMIC_HOME}/datasets/wmt19_qwen_tts_role_speaker_bicodec/train_0_1000}"

bicodec_args=(
  "data.root=${bicodec_data_root}"
  "runtime.backbone=${qwen_root}"
)

cd "${SPEECH_TO_SPEECH_ROOT}"
echo '{"event":"job.launch","experiment":"bicodec_tts_overfit","task":"tts","codec":"bicodec","entry":"overfit"}'
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/overfit.py \
  experiment=bicodec_tts_overfit \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "${bicodec_args[@]}" \
  "$@"
