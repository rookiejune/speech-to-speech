# speech-to-speech

Training and generation components for semantic speech modeling and acoustic
decoding experiments.

The main training path is:

```text
raw sample -> datamodule -> ModelBatch -> model -> Loss -> Lightning module
```

`runtime` supplies the shared tokenizer, codec, backbone, vocabulary layout,
and flow runtime used along that path. Real inference uses an independent
`Request -> Result` generation interface instead of an incomplete
`ModelBatch`.

## Entry Points

- `scripts/overfit.py`: fixed-sample TTS/S2ST overfit and callback smoke tests.
- `scripts/generation_smoke.py`: cached versus full-recompute S2ST generation
  check.
- `scripts/codec_oracle.py`: Hydra entry point for codec oracle experiments;
  configuration starts at `configs/config.yaml`.
- `jobs/`: machine-aware wrappers for formal experiment runs. Each wrapper
  invokes one of the Python entry points directly and forwards extra arguments.

## Documentation

- [`docs/model-design.md`](docs/model-design.md): stable cross-module data,
  ownership, training, and generation contracts.
- [`docs/design/`](docs/design/): public capabilities and boundaries of each
  module.
- [`docs/experiments/todo.md`](docs/experiments/todo.md): current implementation
  status and remaining validation work.
- [`docs/experiments/schedules/`](docs/experiments/schedules/): experiment plans.
- [`docs/experiments/results/`](docs/experiments/results/): results corresponding
  to those plans.

Read the contracts before changing a cross-module interface. Treat the Python
entry points and their arguments as the source of truth for execution.

## Local Checks

Activate the workspace's documented `py312` Torch environment, then run from
the repository collection root:

```bash
PYTHONPATH=speech-to-speech/src:workspace/src basedpyright --project speech-to-speech/pyrightconfig.json --pythonpath "$(command -v python)"
PYTHONPATH=speech-to-speech:speech-to-speech/src:workspace/src python -m unittest discover -s speech-to-speech/tests -v
PYTHONPATH=speech-to-speech/src:workspace/src python -m compileall -q speech-to-speech/src speech-to-speech/scripts speech-to-speech/tests
```
