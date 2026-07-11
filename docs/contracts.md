# Contracts

This document records the stable public contracts. Implementation notes and
open milestones belong in `roadmap.md` or `model-design.md`.

## Data

`Speech.semantic_ids` is frame-level semantic code with shape
`[frames, semantic_codebooks]`.

`Speech.acoustic_ids` is either `None` or frame-level acoustic code with shape
`[frames, acoustic_codebooks]`. When present, it must share the same frame axis
as `semantic_ids`.

`ModelBatch` stores one padded batch:

- `input_ids`: global semantic/text token ids.
- `labels`: causal-LM labels, padded with `-100`.
- `acoustic_input_ids`: source acoustic prompt codes, padded with `-1`.
- `acoustic_input_positions`: source acoustic frame to input-token positions.
- `acoustic_labels`: target acoustic codes, padded with `-1`.
- `acoustic_label_positions`: target acoustic frame to semantic label positions.
- `tasks`: one `Task` per row.

Acoustic fields are batch-wide: every row has them, or no row has them.

## Model

`SpeechToSpeechFlowModel.forward()` returns logits and optional hidden states. It
does not receive labels and does not compute loss.

The semantic backbone consumes global text/audio token ids. Acoustic source
prompt features are injected only at positions provided by the batch.

The acoustic decoder consumes frame-level condition from target semantic hidden
states and predicts continuous acoustic latents.

## Loss

`Loss.forward()` combines semantic CE and optional acoustic flow matching.

Semantic loss follows the Transformers causal-LM shift: position `p - 1`
predicts label position `p`.

Acoustic flow matching uses `acoustic_label_positions - 1` to gather causal
hidden states for each target acoustic frame. Frame padding is masked out.
