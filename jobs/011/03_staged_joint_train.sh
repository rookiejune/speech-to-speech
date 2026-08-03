#!/usr/bin/env bash
set -euo pipefail

JOB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${JOB_DIR%/jobs/*}/jobs/env.sh"

qwen_root="$(fdu_qwen_root)"

experiment="${SPEECH_TO_SPEECH_EXPERIMENT:-train/staged_joint/stage_1}"
case "$experiment" in
  train/staged_joint/stage_1 | train/staged_joint/stage_2 | train/staged_joint/stage_3 | train/staged_joint/stage_4)
    ;;
  *)
    echo "SPEECH_TO_SPEECH_EXPERIMENT must be train/staged_joint/stage_1 through stage_4, got: $experiment" >&2
    exit 2
    ;;
esac
visible_devices="${CUDA_VISIBLE_DEVICES:-${SPEECH_TO_SPEECH_EXPERIMENT_GPUS:-0,1}}"
step_mode="${SPEECH_TO_SPEECH_STEP_MODE:-fused_joint}"
case "$step_mode" in
  fused_joint)
    trainer="staged_static_ddp"
    case "$experiment" in
      train/staged_joint/stage_1)
        accumulate_grad_batches="2"
        ;;
      train/staged_joint/stage_2)
        accumulate_grad_batches="3"
        ;;
      train/staged_joint/stage_3)
        accumulate_grad_batches="5"
        ;;
      train/staged_joint/stage_4)
        accumulate_grad_batches="6"
        ;;
    esac
    ;;
  serial_joint)
    trainer="staged_ddp"
    case "$experiment" in
      train/staged_joint/stage_1)
        accumulate_grad_batches="2"
        ;;
      train/staged_joint/stage_2)
        accumulate_grad_batches="3"
        ;;
      train/staged_joint/stage_3)
        accumulate_grad_batches="5"
        ;;
      train/staged_joint/stage_4)
        accumulate_grad_batches="6"
        ;;
    esac
    ;;
  *)
    echo "SPEECH_TO_SPEECH_STEP_MODE must be fused_joint or serial_joint, got: $step_mode" >&2
    exit 2
    ;;
esac
job_reject_overrides experiment task loader_plan -- "$@"

fdu_stage_data_args datamodule.dataset.root

cd "${SPEECH_TO_SPEECH_ROOT}"
echo "{\"event\":\"job.launch\",\"entry\":\"scripts/train.py\",\"experiment\":\"${experiment}\",\"step_mode\":\"${step_mode}\",\"trainer\":\"${trainer}\",\"devices\":\"${visible_devices}\"}"
args=(
  "experiment=${experiment}" \
  "trainer=${trainer}" \
  "loader_plan.step_mode=${step_mode}" \
  "repo_output_root=${SPEECH_TO_SPEECH_TRAIN_ROOT}" \
  "runtime.backbone=${qwen_root}" \
  "${FDU_DATA_ARGS[@]}" \
)
if [[ -n "${accumulate_grad_batches}" ]]; then
  args+=("loader_plan.accumulate_grad_batches=${accumulate_grad_batches}")
fi
CUDA_VISIBLE_DEVICES="${visible_devices}" "${SPEECH_TO_SPEECH_PYTHON}" scripts/train.py \
  "${args[@]}" \
  "$@"
