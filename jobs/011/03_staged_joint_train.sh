#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${JOB_DIR%/jobs/*}/jobs/env.sh"

qwen_root="$(fdu_qwen_root)"

experiment="${SPEECH_TO_SPEECH_EXPERIMENT:-train/staged_joint/stage_0}"
case "$experiment" in
  train/staged_joint/stage_0)
    accumulate_grad_batches="2"
    default_step_mode="serial_joint"
    ;;
  train/staged_joint/stage_1)
    accumulate_grad_batches="3"
    default_step_mode="fused_joint"
    ;;
  train/staged_joint/stage_2)
    accumulate_grad_batches="5"
    default_step_mode="fused_joint"
    ;;
  train/staged_joint/stage_3)
    accumulate_grad_batches="6"
    default_step_mode="fused_joint"
    ;;
  *)
    echo "SPEECH_TO_SPEECH_EXPERIMENT must be train/staged_joint/stage_0 through stage_3, got: $experiment" >&2
    exit 2
    ;;
esac
visible_devices="${CUDA_VISIBLE_DEVICES:?the scheduler or caller must assign the training GPUs}"
step_mode="${SPEECH_TO_SPEECH_STEP_MODE:-${default_step_mode}}"
case "$step_mode" in
  fused_joint)
    if [[ "$experiment" == "train/staged_joint/stage_0" ]]; then
      trainer="staged_ddp"
    else
      trainer="staged_static_ddp"
    fi
    ;;
  serial_joint)
    trainer="staged_ddp"
    ;;
  *)
    echo "SPEECH_TO_SPEECH_STEP_MODE must be fused_joint or serial_joint, got: $step_mode" >&2
    exit 2
    ;;
esac
job_reject_overrides experiment task loader_plan model.acoustic.init_artifact -- "$@"
generator_artifact="${SPEECH_TO_SPEECH_ACOUSTIC_GENERATOR_ARTIFACT:?set SPEECH_TO_SPEECH_ACOUSTIC_GENERATOR_ARTIFACT to an artifact exported by semantic-acoustic-generator}"

fdu_stage_data_args datamodule.dataset.root

cd "${SPEECH_TO_SPEECH_ROOT}"
echo "{\"event\":\"job.launch\",\"entry\":\"scripts/train.py\",\"experiment\":\"${experiment}\",\"step_mode\":\"${step_mode}\",\"trainer\":\"${trainer}\",\"devices\":\"${visible_devices}\"}"
args=(
  "experiment=${experiment}" \
  "trainer=${trainer}" \
  "loader_plan.step_mode=${step_mode}" \
  "model.acoustic.init_artifact=${generator_artifact}" \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "${FDU_DATA_ARGS[@]}" \
)
if [[ -n "${accumulate_grad_batches}" ]]; then
  args+=("loader_plan.accumulate_grad_batches=${accumulate_grad_batches}")
fi
"${SPEECH_TO_SPEECH_PYTHON}" scripts/train.py \
  "${args[@]}" \
  "$@"
