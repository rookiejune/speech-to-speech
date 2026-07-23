# 013 FDU codec oracle and staged smoke

对应计划：[`schedules/013-fdu-codec-oracle-and-stage-smoke.md`](../schedules/013-fdu-codec-oracle-and-stage-smoke.md)。

## Stage 2 joint LBA DDP smoke

2026-07-23 在 FDU `145` 上用 GPU 1、4 运行
`jobs/013/15_stage_2_lba_smoke.sh`。该 job 选择
`fdu_stage_2_lba_smoke`，使用 Qwen3-0.6B、LongCat native token、RVQ acoustic path、
`batches_per_step=10`、`max_steps=2`、`ddp_find_unused_parameters_true`，并同时对
ASR、TTS 和 text MT 子 loader 启用 LBA。

运行命令使用 `NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1 HYDRA_FULL_ERROR=1`，stdout/stderr
写入
`145:/mnt/pami202/zhuyin/dynamic/debug/s2s-joint-lba-ddp-145-exit-20260723-021802/run.log`，
并额外写出同目录 `exit_code`。本次 `exit_code=0`，日志显示两个 rank 均完成 DDP
注册：

- `Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2`
- `Initializing distributed: GLOBAL_RANK: 1, MEMBER: 2/2`
- `All distributed processes registered. Starting with 2 processes`
- `LOCAL_RANK: 0/1 - CUDA_VISIBLE_DEVICES: [1,4]`
- `Trainer.fit stopped: max_steps=2 reached.`

`metrics.json` 写在
`145:/mnt/pami202/zhuyin/dynamic/train/speech-to-speech/013-fdu-stage-lba-smoke/stage_2/stage_2-rvq-8l/metrics.json`。
所有记录值均为 finite：

| Metric | First | Last | Steps |
| --- | ---: | ---: | ---: |
| total loss | `16.43790` | `16.52316` | 2 |
| token | `6.63403` | `6.56360` | 2 |
| rvq | `9.14602` | `9.14847` | 2 |

LBA 在每个子 loader 上都生成了两个 rank 的 jsonl summary：

| Loader | Rank files | Planned batches | Samples after planning | Padding ratio after |
| --- | ---: | ---: | ---: | ---: |
| ASR | 2 | 15 | 56 | `0.07016` |
| TTS | 2 | 12 | 45 | `0.06519` |
| MT toy text | 2 | 1 | 4 | `0.00000` |

三类 loader 的 LBA summary health 均为
`no_ready_calls=0`、`oversized_batches=0`、`planner_oversized_batches=0`、
`spill_events=0`、`spilled_records=0`。

## 修复记录

首次远端 run 已完成 DDP 初始化并创建 ASR/TTS/MT 的 LBA 文件，但在
`OutputsLogger` 的 tuple joint batch 日志阶段失败：logger 用所有子 batch 的 task mask
索引 RVQ loss，导致 mask 长度 `34` 对不上 RVQ loss 行数 `14`。修复后
`OutputsLogger` 按 objective 自己的 loss 行取 task：token 覆盖所有子 batch，RVQ/Flow/REPA
只覆盖带 acoustic target 的子 batch，并在行数不一致时明确报错。

## 结论边界

这次 smoke 验证的是正式 staged joint train entry 在 FDU 两卡 DDP 下可以同时使用 speech/text
LBA，完成 2 个 optimizer steps，并写出 finite metrics 与 per-rank LBA planner summary。
它不验证长跑 distributed sample partition、公平数据覆盖、resume、质量或收敛。当前 text LBA
路径使用 toy map-style samples；真实 WMT19 text preset 仍是 iterable dataset，不能直接交给
LBA。
