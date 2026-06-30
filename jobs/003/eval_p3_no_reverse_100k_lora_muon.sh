#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

cd "$S2S_ROOT"

"$S2S_PYTHON" scripts/evaluate_free_running.py \
  config \
  experiment=wmt19_quality_100k_muon \
  tasks=s2_translation_weighted \
  tasks.weights.target_to_source=0.0 \
  train.acoustic_loss_weight=0.01 \
  trainer.name=wmt19-quality-003-p3-no-reverse-100k-lora-muon \
  "$@"
