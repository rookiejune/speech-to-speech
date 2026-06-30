#!/usr/bin/env bash
set -euo pipefail

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/env.sh"

cd "$S2S_ROOT"

"$S2S_PYTHON" scripts/train.py \
  experiment=wmt19_acoustic_target_embed_100k_muon \
  trainer.name=wmt19-acoustic-004-target-embed-100k-lora-muon \
  "$@"
