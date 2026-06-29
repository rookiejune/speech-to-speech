#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

cd "$S2S_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$S2S_PYTHON" scripts/train.py \
  experiment=wmt19_quality_muon \
  tasks=s0_ar_dominant \
  trainer.name=wmt19-quality-001-s0-lora-muon \
  trainer.default_root_dir="$S2S_TRAIN_ROOT" \
  "$@"
