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
- `scripts/train.py`: staged joint training; its Hydra root is
  `configs/train.yaml`, reusable production defaults live in
  `configs/entry/train.yaml`, and concrete runs select `experiment=train/...`.
- `scripts/generation_smoke.py`: cached versus full-recompute S2ST generation
  and variable-batch generation checks using the public `generation` package;
  cache probes, benchmarks, and reporting live in separate private script modules.
- `jobs/`: machine-aware wrappers for formal experiment runs. Each wrapper
  invokes one of the Python entry points directly and forwards extra arguments.

The Stable Codec no-audio-BPE TTS+ASR long run is
`jobs/015/01_stable_codec_stage1.sh`. It selects the Stable Codec full-code
sequence, stage 1 (50% ASR / 50% TTS), a 1M-step budget, 10k-step checkpoints,
and fixed-sample TensorBoard logging for both loaders.
The wrapper requires `SPEECH_TO_SPEECH_STABLE_PYTHON`, because the optional
`stable-codec` dependency has its own compatibility environment.

Acoustic-only codec screening and the former codec-oracle training entry have
moved to `semantic-acoustic-codec`; this repository keeps the joint S2ST
training and generation path.

## Experiment Runs

Use the job wrappers as the formal entry points; they load the workspace and
project environments themselves. Hydra-based jobs accept `key=value` overrides,
while the generation smoke accepts normal command-line flags:

```bash
jobs/002/01_tts.sh train.max_steps=2
jobs/002/02_s2st.sh train.max_steps=2 model/acoustic=rvq
jobs/004/01_s2st.sh --batch-sizes 1,2,4
jobs/005/02_unicodec.sh
jobs/005/05_unicodec_ddp.sh
```

The 005 wrappers are UniCodec full-path validation runs. They select explicit
experiments containing their data, trainer, callback, and step budgets:
UniCodec fixed-sample overfit uses 100 steps, and UniCodec DDP smoke uses two
steps.
The 002 wrappers likewise select `experiment=overfit` explicitly.

For the source-level model/data contract smoke, select
`experiment=toy_smoke`. It uses a random tiny Qwen backbone and deterministic
in-memory codec samples on CPU while retaining the existing `longcat_native`
runtime for the tokenizer, codec, layout, special IDs, and flow sampler. It
therefore avoids the pretrained language-model weights and prepared WMT19
dataset, but it is not an offline fake runtime and does not replace the real
LongCat/UniCodec acceptance runs.

For the flattened-code comparison path, select
`experiment=longcat_full_sequence_smoke`. It uses the LongCat codec with
`runtime=longcat_full_sequence model/acoustic=none`, so the full codec
codebook sequence is trained as audio tokens and the Flow/RVQ acoustic side
channel stays disabled.

Hydra roots are parsed into strict entry-specific dataclasses before execution.
Both trainer presets use `devices: auto`, so Lightning consumes every device
visible through `CUDA_VISIBLE_DEVICES`. Job wrappers provide machine-facing
single- or two-GPU visibility defaults; override that environment variable at
submission time to change the device set.

`runtime` owns the codec, audio tokenizer, device, dtype, and flow sampling
fields. `model=toy` replaces only the model-owned backbone; it does not select
or construct a runtime. `data=toy` selects deterministic in-memory prepared-code
samples. `model/acoustic=none|flow|rvq` selects whether training uses only
semantic audio tokens or also a downstream Flow/RVQ acoustic path; unified-token
experiments select `runtime=unicodec model/acoustic=none`, and flattened
LongCat comparison runs select `runtime=longcat_full_sequence model/acoustic=none`.
`pl_module` owns optimizer settings for the training entries. Entry points
reject codec/composition mismatches.

Two-GPU DDP smoke for unified-token training uses
`jobs/005/05_unicodec_ddp.sh`. Override machine-facing values such as
`CUDA_VISIBLE_DEVICES`, `SPEECH_TO_SPEECH_PYTHON`,
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
export PYTHONPATH=speech-to-speech:speech-to-speech/src:semantic-acoustic-codec/src:workspace/src:third_party/anydataset/src:third_party/anytrain/src
basedpyright --project speech-to-speech/pyrightconfig.json --pythonpath "$(command -v python)"
DYNAMIC_HOME=/private/tmp/speech-to-speech-test PYTHONPYCACHEPREFIX=/private/tmp/speech-to-speech-pycache python -m unittest discover -s speech-to-speech/tests -v
PYTHONPYCACHEPREFIX=/private/tmp/speech-to-speech-pycache python -m compileall -q speech-to-speech/src speech-to-speech/scripts speech-to-speech/tests
```
