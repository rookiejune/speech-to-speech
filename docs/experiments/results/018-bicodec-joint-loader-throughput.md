# BiCodec joint loader throughput

## Scope

- Date: 2026-08-03
- Revision: `7868e3b2ca723f39b08dfbf618c7889fd19b8910`
- Host: Fudan 125, GPUs 3/4/5, 3 x RTX 3090 24GB
- Runtime: BiCodec, Qwen3-0.6B backbone, LoRA, bf16 mixed precision
- Data: Qwen TTS speaker, batch size 8, cost batching enabled with
  `max_batch_frames=4800` and `planning_window=256`
- Tasks: TTS weight 0.9 and MT weight 0.1
- DDP: `ddp_find_unused_parameters_true`
- Output: `/mnt/pami202/zhuyin/dynamic/debug/s2s_probe/20260803-joint-loader-throughput-100step`

The serial and fused runs were executed sequentially on the same three GPUs. Each optimizer step
consumed exactly one TTS batch and one MT batch. Serial used two Lightning microbatches with
gradient accumulation; fused used one `FusedBatch` and one Lightning backward. Both schedules kept
`loader_plan.accumulate_grad_batches=2` because that field defines the two-loader joint window.

Before the A/B, a five-step fused find-unused DDP smoke completed without the static-DDP
unused-parameter reducer failure. Its output is under
`/mnt/pami202/zhuyin/dynamic/debug/s2s_probe/20260803-fused-joint-staged-ddp-5step`.

## Results

| Metric | Serial joint | Fused joint |
| --- | ---: | ---: |
| Completed optimizer steps | 100 | 100 |
| Logged-step wall time, step 9 to 99 | 83.845 s | 84.868 s |
| Optimizer steps/s over that interval | 1.0734 | 1.0605 |
| Real tokens/s over that interval | 15,521 | 15,334 |
| Total real tokens at step 99 | 1,408,391 | 1,408,391 |
| Total padded tokens at step 99 | 1,577,486 | 1,577,486 |
| Cumulative padding ratio | 10.719% | 10.719% |
| Full process wall time including startup | 229 s | 231 s |
| Peak memory GPU 3 | 22,073 MiB | 23,089 MiB |
| Peak memory GPU 4 | 22,365 MiB | 23,287 MiB |
| Peak memory GPU 5 | 22,087 MiB | 23,087 MiB |
| Mean active GPU utilization, GPUs 3/4/5 | 77.6% / 84.2% / 79.1% | 74.5% / 80.3% / 81.5% |

The stable event interval gives serial a 1.22% throughput lead. That difference is small enough to
be measurement noise in a single ordered run, so the defensible performance result is parity, not a
serial speedup. Fused consistently costs about 0.9-1.0 GiB more peak memory per GPU and leaves only
1,289 MiB free on the highest-memory rank.

Per-task loss curves also agree closely. At logged steps 9 and 99, serial TTS loss was
`10.5800 -> 9.3841` and fused was `10.5813 -> 9.3854`; serial MT loss was
`2.3123 -> 1.7730` and fused was `2.3174 -> 1.7756`. The top-level loss summaries are not directly
comparable: serial records 200 alternating task microbatches, while fused records 100 weighted joint
losses.

## Conclusion

For the current stage-0 TTS+MT LoRA policy on 24GB GPUs, keep `serial_joint` as the practical default.
`fused_joint + find-unused DDP` is now correct, but provides no demonstrated throughput gain and has
materially less memory headroom. Keep fused configurable for task sets that cover all trainable
parameters and can use static DDP; that case was not measured here because TTS+MT leaves trainable
adapter/head parameters unused.

This is a single A/B run with serial first. Repeat in reversed order if a sub-2% throughput difference
ever becomes decision-critical.
