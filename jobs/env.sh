#!/usr/bin/env bash
set -euo pipefail

S2S_JOBS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export S2S_ROOT="${S2S_ROOT:-$(cd -- "$S2S_JOBS_DIR/.." && pwd)}"
export REPOS_ROOT="${REPOS_ROOT:-$(cd -- "$S2S_ROOT/.." && pwd)}"

source "$REPOS_ROOT/workspace/jobs/env.sh"

export PYTHONPATH="$S2S_ROOT/src:$PYTHONPATH"
export S2S_PYTHON="${S2S_PYTHON:-${WORKSPACE_PYTHON:-python}}"
