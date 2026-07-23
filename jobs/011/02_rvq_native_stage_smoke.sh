#!/usr/bin/env bash
set -euo pipefail

REPOS_ROOT="${REPOS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
source "${REPOS_ROOT}/workspace/jobs/fudan/speech_to_speech_env.sh"

qwen_root="$(fdu_qwen_root)"
fdu_stage_data_args data.root

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
      "${FDU_DATA_ARGS[@]}" \
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
