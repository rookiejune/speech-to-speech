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
- `src/speech_to_speech/codec_oracle/`: codec oracle models and prepared-code
  data path used by screening experiments.
- `src/speech_to_speech/reporting.py`: shared experiment summary helpers.
- `docs/model-design.md`: cross-module contracts.
- `docs/design/`: module capabilities and boundaries.
- `docs/experiments/todo.md`: implementation stages and validation status.

## Local Checks

Use the workspace's documented `py312` Torch environment and run from the
repository collection root:

```bash
PYTHONPATH=speech-to-speech/src:workspace/src /Users/zhuyin/miniconda3/envs/py312/bin/basedpyright --project speech-to-speech/pyrightconfig.json --pythonpath /Users/zhuyin/miniconda3/envs/py312/bin/python
PYTHONPATH=speech-to-speech:speech-to-speech/src:workspace/src /Users/zhuyin/miniconda3/envs/py312/bin/python -m unittest discover -s speech-to-speech/tests -v
PYTHONPATH=speech-to-speech/src:workspace/src /Users/zhuyin/miniconda3/envs/py312/bin/python -m compileall -q speech-to-speech/src speech-to-speech/scripts speech-to-speech/tests
```

## Docs

- `docs/model-design.md`: stable cross-module contracts.
- `docs/design/`: datamodule, runtime, model, loss, and Lightning boundaries.
- `docs/experiments/todo.md`: implementation work and validation order.
