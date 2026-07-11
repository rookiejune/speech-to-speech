# Roadmap

## P1: Semantic Closure

- Contract test with fake runtime, fake codec, and tiny backbone.
- TTS/S2ST semantic CE smoke test with forward, backward, and optimizer step.
- Semantic generation smoke test for audio-target tasks.
- `semantic ids -> codec decode -> waveform` validation.

## P2: Acoustic Flow

- Verify target BPE positions expand to acoustic frame positions.
- Verify frame condition, target latent, and frame mask align on `[batch, frame]`.
- Overfit one audio-target batch with semantic CE and acoustic flow loss enabled.
- Add acoustic sampling smoke test.

## P3: Training Integration

- End-to-end Lightning training smoke.
- Sample logging for semantic tokens and audio waveforms.
- Document verified results under `docs/experiments/results/`.

`docs/model-design.md` keeps detailed design rationale. This file only tracks
the execution order.
