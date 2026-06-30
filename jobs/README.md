# Speech-to-Speech Jobs

`jobs/env.sh` sources the shared workspace environment from
`../workspace/jobs/env.sh`, then adds only speech-to-speech specific variables.
Each job wrapper sets Hydra overrides, then calls the real entrypoint
`scripts/train.py "$@"`.

Common variables:

| Variable | Meaning |
| --- | --- |
| `S2S_ROOT` | This repository root. Defaults to the parent of `jobs/`. |
| `REPOS_ROOT` | Parent directory containing `workspace/`, `third_party/`, and this repo. |
| `PYTHONPATH` | Prepends this repo's `src` to the workspace-level Python path. |
| `S2S_PYTHON` | Python executable. Defaults to `$WORKSPACE_PYTHON` when set, otherwise `python`. |

Workspace variables such as `LOCATION`, `STATIC_HOME`, `DYNAMIC_HOME`,
`HF_HOME`, `ANYDATASET_HOME`, and `BPE_CACHE_DIR` are documented in
`workspace/jobs/README.md`.

Local runs may set `LOCATION`, `STATIC_HOME`, or `DYNAMIC_HOME` before invoking
a wrapper. The shell env does not initialize those homes, and the default
training root is resolved inside Python as
`zhuyin.env.train_dir("speech-to-speech")`.

Callback settings live under `trainer.callbacks`. For example, append
`trainer.callbacks.generation.every_n_steps=1000` or
`trainer.callbacks.generation.acoustic_sampler=diagonal_bpe` to a wrapper
command.

## Experiment Groups

- `jobs/001/`: staged S0/S1/S2 and first LoRA/full training checks.
- `jobs/003/`: parallel 100k BPE free-running ablations from
  `docs/experiments/schedules/003_parallel_free_running_ablation.md`.
