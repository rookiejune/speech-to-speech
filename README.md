# speech-to-speech

Training and generation components for semantic speech modeling and acoustic
decoding experiments.

The main training path is:

```text
raw sample -> datamodule -> ModelBatch -> model + objective -> Lightning module
```

`runtime` supplies the shared tokenizer, codec, backbone, vocabulary layout,
and flow runtime used along that path. `loss` exposes the explicit
`TokenObjective`, `FlowObjective`, and `RVQObjective` training
compositions. `generation` owns the independent `Request -> Result` inference
interface, batching, text evaluation, and waveform decode instead of treating
an incomplete `ModelBatch` as a request.

The public response service is `generation.generate_responses()`; Lightning
integration is provided by `SpeechToSpeechModule`.

## Entry Points

- `scripts/overfit.py`: fixed-sample TTS/S2ST overfit and callback smoke tests;
  its Hydra root is `configs/overfit.yaml`.
- `scripts/generation_smoke.py`: cached versus full-recompute S2ST generation
  and variable-batch generation checks using the public `generation` package;
  cache probes, benchmarks, and reporting live in separate private script modules.
- `scripts/codec_oracle.py`: Hydra entry point for codec oracle training. With
  no experiment override, `configs/codec_oracle.yaml` uses the full prepared
  dataset through LBA for 1,000,000 steps at `bf16-mixed` precision, logging a
  sample and archiving a checkpoint every 10,000 steps.
- `jobs/`: machine-aware wrappers for formal experiment runs. Each wrapper
  invokes one of the Python entry points directly and forwards extra arguments.

## Experiment Runs

Use the job wrappers as the formal entry points; they load the workspace and
project environments themselves. Hydra-based jobs accept `key=value` overrides,
while the generation smoke accepts normal command-line flags:

```bash
jobs/002/01_tts.sh train.max_steps=2
jobs/002/02_s2st.sh train.max_steps=2 model/acoustic=rvq
jobs/004/01_s2st.sh --batch-sizes 1,2,4
jobs/005/01_longcat.sh codec_oracle.initialization=codec
jobs/005/06_longcat_rvq.sh
jobs/005/02_unicodec.sh
jobs/005/08_longcat_flow_formal.sh codec_oracle.data.root=/path/to/data
jobs/005/09_longcat_flow_ddp_lba_formal.sh codec_oracle.data.root=/path/to/data
jobs/005/10_longcat_rvq_formal.sh codec_oracle.data.root=/path/to/data
jobs/005/11_longcat_rvq_ddp_lba_formal.sh codec_oracle.data.root=/path/to/data
```

The 01-07 wrappers under 005 are full-path validation runs. They select explicit
experiments containing their data, trainer, callback, and step budgets: LongCat
single-GPU and DDP smoke runs use two steps, UniCodec fixed-sample overfit uses
100 steps, and UniCodec DDP smoke uses two steps. The 08-11 LongCat formal
wrappers deliberately omit `experiment=...` and retain the production-oriented
1,000,000-step full-data LBA configuration. Their DDP variants select the
validated static DDP strategy and write to a separate output root from the
single-GPU runs.
The 002 wrappers likewise select `experiment=overfit` explicitly.

For the source-level model/data contract smoke, select
`experiment=toy_smoke`. It uses a random tiny Qwen backbone and deterministic
in-memory codec samples on CPU while retaining the existing `longcat_native`
runtime for the tokenizer, codec, layout, special IDs, and flow sampler. It
therefore avoids the pretrained language-model weights and prepared WMT19
dataset, but it is not an offline fake runtime and does not replace the real
LongCat/UniCodec acceptance runs.

Hydra roots are parsed into strict entry-specific dataclasses before execution.
Both trainer presets use `devices: auto`, so Lightning consumes every device
visible through `CUDA_VISIBLE_DEVICES`. Job wrappers provide machine-facing
single- or two-GPU visibility defaults; override that environment variable at
submission time to change the device set.

`runtime` owns the codec, audio tokenizer, device, dtype, and flow sampling
fields. `model=toy` replaces only the model-owned backbone; it does not select
or construct a runtime. `data=toy` selects deterministic in-memory prepared-code
samples. `model/acoustic=flow|rvq` selects the formal model/objective
composition; unified-token experiments select `runtime=unicodec` without an
acoustic model group.
`pl_module` owns overfit optimizer settings, while `codec_oracle` owns its
decoder, data, initialization, normalization, and optimizer settings. Entry
points reject codec/composition mismatches.

Two-GPU DDP runs use `jobs/005/04_longcat_ddp_lba.sh` for Flow,
`jobs/005/07_longcat_rvq_ddp_lba.sh` for RVQ, and `jobs/005/05_unicodec_ddp.sh`
for unified-token training. Formal LongCat DDP runs use
`jobs/005/09_longcat_flow_ddp_lba_formal.sh` and
`jobs/005/11_longcat_rvq_ddp_lba_formal.sh`. Override machine-facing values such
as `CUDA_VISIBLE_DEVICES`, `SPEECH_TO_SPEECH_PYTHON`,
`SPEECH_TO_SPEECH_UNICODEC_PYTHON`, or `SPEECH_TO_SPEECH_TRAIN_ROOT` only at
submission time. Jobs default `SPEECH_TO_SPEECH_TRAIN_ROOT` to
`$DYNAMIC_HOME/train/speech-to-speech`; training entries write checkpoints and
summary artifacts under `repo_output_root/output_subdir`, while TensorBoard
events are centralized at `repo_output_root/tensorboard/output_subdir/version_*`.
This lets one TensorBoard invocation compare the whole repository. Keep
TensorBoard enabled for long full-model runs and monitor the supervised curves
rather than relying only on the final summary. `generation_smoke.py` writes
`metrics.json` in its own output directory.

```bash
tensorboard --logdir "${SPEECH_TO_SPEECH_TRAIN_ROOT}/tensorboard"
```

UniCodec jobs require a Python environment compatible with `fairseq==0.12.2`;
select it through `SPEECH_TO_SPEECH_UNICODEC_PYTHON` instead of assuming the
main training environment is compatible.

## Documentation

- [`docs/model-design.md`](docs/model-design.md): stable cross-module data,
  ownership, training, and generation contracts.
- [`docs/design/generation.md`](docs/design/generation.md): public request,
  batching, decoding, and text-evaluation contracts.
- [`docs/design/configuration.md`](docs/design/configuration.md): Hydra groups,
  strict entry schemas, and config ownership boundaries.
- [`docs/design/`](docs/design/): public capabilities and boundaries of each
  module.
- [`docs/experiments/conclusion.md`](docs/experiments/conclusion.md): validated
  conclusions with links to their supporting results.
- [`docs/experiments/todo.md`](docs/experiments/todo.md): remaining validation
  work and engineering debt.
- [`docs/experiments/schedules/`](docs/experiments/schedules/): experiment plans.
- [`docs/experiments/results/`](docs/experiments/results/): results corresponding
  to those plans.

Read the contracts before changing a cross-module interface. Treat the Python
entry points and their arguments as the source of truth for execution.

## Local Checks

Python 3.9 is the minimum supported version. Run the checks below in the
workspace's documented `py39` environment; `py312` remains the primary
environment for full-model training and audio experiments. Run from the
repository collection root:

```bash
PYTHONPATH=speech-to-speech/src:workspace/src basedpyright --project speech-to-speech/pyrightconfig.json --pythonpath "$(command -v python)"
PYTHONPATH=speech-to-speech:speech-to-speech/src:workspace/src python -m unittest discover -s speech-to-speech/tests -v
PYTHONPATH=speech-to-speech/src:workspace/src python -m compileall -q speech-to-speech/src speech-to-speech/scripts speech-to-speech/tests
```
