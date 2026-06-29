# Speech-to-Speech Jobs

`jobs/env.sh` sources the shared workspace environment from
`../workspace/jobs/env.sh`, then adds only speech-to-speech specific variables.
Each job wrapper sets the GPU slot and Hydra overrides, then calls the real
entrypoint `scripts/train.py "$@"`.

Common variables:

| Variable | Meaning |
| --- | --- |
| `S2S_ROOT` | This repository root. Defaults to the parent of `jobs/`. |
| `REPOS_ROOT` | Parent directory containing `workspace/`, `third_party/`, and this repo. |
| `PYTHONPATH` | Prepends this repo's `src` to the workspace-level Python path. |
| `S2S_PYTHON` | Python executable. Defaults to `$WORKSPACE_PYTHON`. |
| `S2S_TRAIN_ROOT` | Training output root. Defaults to `$DYNAMIC_HOME/train/speech-to-speech`. |
| `CUDA_VISIBLE_DEVICES` | Per-job GPU selection; each wrapper sets a default but lets callers override it. |

Shared cache variables such as `STATIC_HOME`, `DYNAMIC_HOME`, `HF_HOME`,
`ANYDATASET_HOME`, and `BPE_CACHE_DIR` are documented in
`workspace/jobs/README.md`.

Local runs should override `STATIC_HOME` and `DYNAMIC_HOME` explicitly instead
of relying on project-local fallbacks.
