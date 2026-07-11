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
- `docs/`: model contracts, roadmap, and detailed design notes.

## Smoke Checks

Use the local Torch environment recorded in the workspace docs:

```bash
PYTHONPATH=speech-to-speech/src /Users/zhuyin/miniconda3/envs/py312/bin/python -m py_compile $(rg --files speech-to-speech/src -g '*.py')
PYTHONPATH=speech-to-speech/src /Users/zhuyin/miniconda3/envs/py312/bin/python -c "import speech_to_speech.pl_module"
```

## Docs

- `docs/contracts.md`: stable data, model, and loss contracts.
- `docs/roadmap.md`: implementation stages and validation order.
- `docs/model-design.md`: detailed design history and rationale.
