#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

cd "$S2S_ROOT"

"$S2S_PYTHON" scripts/train.py \
  experiment=wmt19_quality_100k_muon \
  tasks=s2_translation_weighted \
  train.acoustic_loss_weight=0.01 \
  trainer.name=wmt19-quality-003-p2-s2-weak-reverse-100k-lora-muon \
  "$@"
