# speech-to-speech

Minimal speech-to-speech training components for semantic speech modeling and
acoustic decoding experiments.

## Layout

- `src/speech_to_speech/datamodule/`: raw sample parsing, task sampling, and
  `ModelBatch` construction.
- `src/speech_to_speech/model/`: semantic backbone wrappers, audio embedding,
  and acoustic decoder compositions.
- `src/speech_to_speech/loss/`: semantic CE and acoustic flow matching losses.
- `src/speech_to_speech/pl_module/`: Lightning module plus generation and decode
  helpers.
- `src/speech_to_speech/runtime/`: tokenizer, codec, backbone, layout, and flow
  runtime loading.
- `docs/model-design.md`: cross-module contracts.
- `docs/design/`: module capabilities and boundaries.
- `docs/experiments/todo.md`: implementation stages and validation status.

## Smoke Checks

Use the local Torch environment recorded in the workspace docs:

```bash
PYTHONPATH=speech-to-speech/src /Users/zhuyin/miniconda3/envs/py312/bin/python -m py_compile $(rg --files speech-to-speech/src -g '*.py')
PYTHONPATH=speech-to-speech/src /Users/zhuyin/miniconda3/envs/py312/bin/python -c "import speech_to_speech.pl_module"
```

## Docs

- `docs/model-design.md`: stable cross-module contracts.
- `docs/design/`: datamodule, runtime, model, loss, and Lightning boundaries.
- `docs/experiments/todo.md`: implementation work and validation order.
