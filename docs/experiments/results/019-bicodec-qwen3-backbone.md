# BiCodec Qwen3 backbone comparison

## Scope

- Date: 2026-08-03
- Revision: `7868e3b2ca723f39b08dfbf618c7889fd19b8910`
- Host: Fudan 125, GPUs 3/4/5, 3 x RTX 3090 24GB
- Runtime: BiCodec, bf16 mixed precision, FlashAttention 2, gradient checkpointing, LoRA r16
- Schedule: `serial_joint` with staged DDP; one TTS and one MT microbatch per optimizer step
- Data: speech/text batch size 4, `max_batch_frames=2400`, `planning_window=256`
- Tasks: TTS weight 0.9 and MT weight 0.1
- Budget: 100 optimizer steps per backbone
- Output: `/mnt/pami202/zhuyin/dynamic/debug/s2s_probe/20260803-qwen3-backbone-bs4-100step`

The matched runs compare Qwen3-0.6B snapshot `c1899de` with Qwen3-1.7B snapshot `70d244cc`.
Both completed with status 0 and without a traceback, non-finite loss, OOM, or distributed-process
failure. Before the formal runs, two-step 1.7B serial and fused compatibility gates both completed;
only the selected serial default was used for the formal backbone A/B.

## Data parity

The comparison processed identical data. TensorBoard values for `progress/tokens`, cumulative real
tokens, padded tokens, cumulative padded tokens, and per-task supervised TTS/MT token counts match
exactly at every logged step across the two runs.

| Metric at optimizer step 99 | Both runs |
| --- | ---: |
| Cumulative real compute tokens | 703,692 |
| Cumulative padded compute tokens | 777,144 |
| Cumulative padding ratio | 9.4515% |
| Cumulative supervised TTS tokens | 649,360 |
| Cumulative supervised MT tokens | 34,608 |

The 9.4515% padding ratio is 1.267 percentage points below the earlier bs8 result, but that run used
`max_batch_frames=4800`; this is a batching-policy difference and not a backbone effect.

## Throughput and memory

The cumulative progress tags first appear at optimizer step 19, so the broad matched interval is
step 19 to 99. Both runs contain a similar fixed-cost gap in the next ten-step interval; the later
step 29 to 99 interval is also reported as the steadier long-run estimate. Lightning's displayed
`it/s` is not used because it counts the two serial microbatches rather than optimizer steps.

| Metric | Qwen3-0.6B | Qwen3-1.7B |
| --- | ---: | ---: |
| Total parameters | 782,430,208 | 2,074,500,096 |
| Trainable parameters | 31,071,232 | 43,307,008 |
| Trainable fraction | 3.971% | 2.088% |
| Full process wall time | 211 s | 327 s |
| Step 19 -> 99 logged time | 76.345 s | 106.913 s |
| Step 19 -> 99 optimizer steps/s | 1.0479 | 0.7483 |
| Step 19 -> 99 real tokens/s | 7,831 | 5,592 |
| Step 19 -> 99 padded tokens/s | 8,647 | 6,174 |
| Step 29 -> 99 optimizer steps/s | 1.3904 | 0.8978 |
| Step 29 -> 99 real tokens/s | 9,943 | 6,420 |
| Step 29 -> 99 padded tokens/s | 11,041 | 7,129 |
| GPU-active interval | 60.73 s | 91.81 s |
| Mean active utilization, three-card mean | 72.43% | 83.27% |

On the broader event interval, 1.7B processes identical tokens 28.6% more slowly. On the later
steady interval it is 35.4% slower; the independently sampled GPU-active interval is 51.2% longer.
The higher utilization of 1.7B means it keeps the GPUs busier, not that it is more training-efficient.
The active window is the final continuous three-rank compute segment after DDP initialization, with
single-rank zero-utilization samples retained; it contains 304/459 samples per card at about 200 ms
for 0.6B/1.7B respectively.

| Peak memory | Qwen3-0.6B | Qwen3-1.7B | 1.7B increase |
| --- | ---: | ---: | ---: |
| GPU 3 | 12,765 MiB | 21,881 MiB | 9,116 MiB |
| GPU 4 | 13,257 MiB | 22,395 MiB | 9,138 MiB |
| GPU 5 | 13,563 MiB | 22,627 MiB | 9,064 MiB |

| Active utilization, mean / P50 / P95 | Qwen3-0.6B | Qwen3-1.7B |
| --- | ---: | ---: |
| GPU 3 | 74.9% / 86% / 100% | 84.6% / 97% / 100% |
| GPU 4 | 63.5% / 69% / 100% | 81.5% / 97% / 100% |
| GPU 5 | 78.9% / 92% / 100% | 83.7% / 97% / 100% |

The highest 1.7B rank leaves only 1,949 MiB, or 7.9%, of a 24,576 MiB RTX 3090 free. The two-step
serial gate peaked at only 18,923 MiB, so short gates over variable-length batches are compatibility
checks and cannot establish the formal memory requirement. The measured bs4 run fits, but does not
meet the normal 10%-20% memory-buffer target for a long run on these cards. The GPU CSV has no power
column, so this run does not support an energy-efficiency claim.

## Short-run loss

These are unweighted per-task token losses. Serial's top-level loss alternates weighted TTS and MT
microbatches and is not a valid cross-backbone summary.

| Task | Qwen3-0.6B step 19 -> 99 | Qwen3-1.7B step 19 -> 99 | Logged mean, 0.6B / 1.7B |
| --- | ---: | ---: | ---: |
| TTS | 9.8550 -> 9.3484 | 9.4415 -> 9.1681 | 9.5084 / 9.3106 |
| MT | 2.3311 -> 1.6960 | 1.5890 -> 1.2348 | 1.8937 / 1.4868 |

The 1.7B run has lower short-run CE for both tasks, especially MT. One ordered 100-step training run
does not establish final TTS quality, convergence, text retention, or whether the loss difference is
worth the added compute and memory.

## Conclusion

Keep Qwen3-0.6B as the default when the objective is best training throughput and usable memory
headroom on 24GB GPUs. Qwen3-1.7B is execution-compatible at bs4 and shows a promising short-run
loss advantage, but it is 29%-35% slower on matched token throughput, raises full-process wall time
by 55%, and consumes about 8.9 GiB more memory per rank with less than 8% headroom.

Do not promote 1.7B based on this probe. A quality-driven decision requires a longer matched run with
held-out TTS/MT metrics and fixed-sample audio evaluation; use a 40GB-class GPU or reduce the batch
budget for that run.
