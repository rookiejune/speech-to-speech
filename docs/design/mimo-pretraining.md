# Kimi-Style MIMO Pretraining

This repository exposes two autoregressive paths:

- `ARFraming.PRETRAINING` keeps the existing single-stream model and builds
  `BOS | text | EOS` or `BOA | audio | EOA` samples without an instruction
  template.
- MIMO keeps text and semantic-audio ids on one aligned time axis.  The body is
  called once on `text_embedding + audio_embedding`; separate readouts and
  local-vocabulary heads compute the two causal losses.

## MIMO sample contract

`MimoSample` and `MimoBatch` contain independent `text_loss_mask` and
`audio_loss_mask`.  Labels are unshifted; the objective shifts both routes by
one position and counts only targets whose predecessor is attended.  Text and
audio ids are local to their own vocabularies.  Do not offset audio ids by the
text vocabulary size.

Continuous audio features are optional.  `audio_feature_mask` must be true
only for observed source audio positions.  Target audio positions are always
masked, so teacher-forcing cannot leak the answer through a continuous side
channel.

## Seven task mixture

`build_mimo_sample` implements the seven Kimi pretraining routes:

`audio_only:text_only:audio_to_text:text_to_audio:audio_to_next_semantic:audio_to_next_text:audio_to_next_semantic_and_text`

with the default weights `1:7:1:1:1:1:2`.  Contextual routes require ordered
adjacent segments from one recording.  Audio targets use configurable delay
blanks (`MimoSpecialTokens.audio_delay_tokens`); keep the delay in the
checkpoint/config rather than hard-coding it because released Kimi code and
the paper use different defaults.

## Runtime and training entry

The standalone entry is deliberately separate from the legacy single-stream
callbacks:

```text
configs/mimo_train.yaml
scripts/mimo_train.py
```

`model.mimo_factory.build_mimo_model` retains the runtime backbone as the only
registered parameter owner while calling the runtime-adapted body.  This keeps
Kimi's `return_dict=True` and activation-checkpoint behavior intact.  The
prepared-data boundary is an importable factory returning `MimoSegment` or
`MimoSample` datasets; `MimoTaskDataset` adds deterministic task sampling and
`MimoDataModule` pads aligned examples.

The Kimi experiment preset reads a prepared JSONL manifest through
`SPEECH_TO_SPEECH_MIMO_SEGMENTS` (or `storage/mimo/segments.jsonl`).  Each
record must include local text/audio ids; contextual tasks additionally require
`recording_id` and consecutive `segment_index` values.  Rows without those
fields remain valid for single-segment tasks but are never joined across
recordings.

For a CPU smoke run:

```bash
env PYTHONPATH=src:../semantic-acoustic-codec/src:../third_party/anydataset/src:../third_party/anytrain/src:../workspace/src \
  /Users/zhuyin/miniconda3/envs/py39/bin/python scripts/mimo_train.py \
  model.toy=true trainer.accelerator=cpu trainer.devices=1 \
  trainer.precision=32-true trainer.enable_checkpointing=false train.max_steps=1
```

Replace the toy segment factory with a prepared workspace factory for a real
run.  The factory must return local text/audio token ids and may include
source-only continuous features.

## Mixture accounting

`LoaderStepMode.TOKEN_WEIGHTED` schedules single-stream loaders by the number
of supervised causal labels, not by the number of microbatches.  It supports
both `ModelBatch` and `MimoBatch`; raw waveform fallback batches are rejected
until codec materialization has produced token labels.  `weighted_window`
remains available for legacy experiments.

## Text corpus input

The single-stream phase-1 path can read local general text through
`TextDatasetName.GENERAL` (JSONL or one-document text files).  Set
`text_datamodule.pack_documents=true` and a `max_tokens` budget for tokenizer-
aware document packing.  This path is separate from the WMT19 translation
dataset, so a translation corpus is not silently used as a general-text
pretraining substitute.
