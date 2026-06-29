#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

cd "$S2S_ROOT"

"$S2S_PYTHON" scripts/train.py \
  experiment=wmt19_quality_muon \
  tasks=s1_bidirectional_mixed \
  trainer.name=wmt19-quality-001-s1-lora-muon \
  ${S2S_TRAIN_OVERRIDES[@]+"${S2S_TRAIN_OVERRIDES[@]}"} \
  "$@"
