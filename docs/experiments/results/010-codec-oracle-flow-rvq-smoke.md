# 010 Codec Oracle Flow/RVQ Smoke

对应计划：[`schedules/010-codec-oracle-flow-rvq-smoke.md`](../schedules/010-codec-oracle-flow-rvq-smoke.md)。

## 环境

2026-07-21 在复旦 `145` 的 RTX 4090D GPU 1、2 上，使用 speech-to-speech `9127e62`、
Python 3.12、PyTorch 2.9.0、Lightning 2.6.1、真实 Qwen3-0.6B、LongCat codec 和 1000 条
WMT19/LongCat prepared samples 验收。单卡使用 GPU 1；DDP 使用 GPU 1、2 和静态 `ddp`。

共享 WMT19 `longcat` store 已是 anydataset schema v2，但合并的 `base` store 仍是 v1。本次将
472 MB base store 迁移到 `145:/tmp/s2s-oracle-010-data/base`，并从同一临时 root 只读引用原
LongCat store。NAS 当时只剩约 126 GB，因此 checkpoint 和日志写在 `145:/tmp`，不是长期产物。

## 结果

四组均使用 codec initialization、8 层 acoustic decoder 和 2 optimizer steps。这里的
initialization 只控制 semantic audio embedding；RVQ acoustic embeddings 仍是 decoder 随机初始化。

| objective / execution | world size | loss step 1 -> 2 | sampled metric step 1 -> 2 | fit wall time, rank 0 |
| --- | ---: | ---: | ---: | ---: |
| Flow / single | 1 | `2.31883 -> 2.27787` | feature MSE `2.46069 -> 2.46954` | `56.16s` |
| RVQ / single | 1 | `9.16811 -> 7.42795` | accuracy `0 -> 0.00926`; feature MSE `2.47581 -> 2.48042` | `59.29s` |
| Flow / DDP + LBA | 2 | `2.51789 -> 2.45364` | feature MSE `2.45649 -> 2.39743` | `86.24s` |
| RVQ / DDP + LBA | 2 | `9.13780 -> 9.14543` | accuracy `0 -> 0`; feature MSE `2.45767 -> 2.46641` | `83.40s` |

DDP `metrics.json` 的 loss 是 callback 在两个 rank 上同步求 mean 后由 global zero 写入。两步 smoke
只验证训练和 callback 闭环；RVQ 的随机 sampled accuracy 以及短程 loss 变化都不支持质量或收敛结论。

## Callback 验收

- 四组都写出 reconstruction、step 1 sample、step 2 sample 三个 `(1, 34560)` waveform，全部 finite。
- Flow TensorBoard 包含 `train/flow_loss`、`train/grad_norm`、`flow/time` histogram、sample feature
  MSE 和 reconstruction/sample audio。
- RVQ TensorBoard 包含总/逐 codebook CE、总/逐 codebook sampled accuracy、feature MSE、三个
  codebook loss histogram、grad norm 和 reconstruction/sample audio。
- 四组都保留 `step-1.ckpt`、`step-2.ckpt` 和 `last.ckpt`。Flow checkpoint 约 2.56 GB，RVQ
  checkpoint 约 2.65 GB；non-finite callback 保持启用且未报告异常。这两个大小来自当时仍持有
  完整 Qwen 的历史 wrapper，不能作为当前轻量 oracle 的大小。Flow 单卡实测见下文；RVQ 和
  DDP 的轻量 checkpoint 大小仍待下一次真实资源验收记录。
- 两组 DDP 的实际 world size 均为 2，并各写出两份独立 rank 的 LBA log/jsonl。
  RVQ 使用静态 `ddp` 完成两个 backward/optimizer step，没有 unused-parameter 错误。

## 现场修复

首次 DDP 在 Lightning 注入 distributed sampler 时丢失 LBA 的 `len_fn`。原因是项目在
`train_dataloader()` 内才 import `LBA`，晚于 Lightning 包装已有 DataLoader 子类的时点。
`9127e62` 将 import 提到模块边界，并增加真实 Lightning reconstruction 回归测试；Flow/RVQ DDP
随后均通过，未修改 third_party LBA 逻辑。

远程 anytrain 初始版本也仍要求 `ANYTRAIN_DEBUG=True` 才能构造 `DebugCallback`。共享副本
fast-forward 到已推送的 anytrain `3e9f4f4` 后使用当前无环境开关契约，正式 job 无需额外 override。

有效产物位于：

- 单卡：`145:/tmp/s2s-oracle-010-final/005-codec-oracle-smoke/longcat/{flow,rvq}-8l/codec/`
- DDP：`145:/tmp/s2s-oracle-010-ddp-fixed/005-codec-oracle-ddp-lba-smoke/longcat/{flow,rvq}-8l/codec/`

## Canonical schema v2 补充验收

2026-07-22 使用 workspace `92bbd05`、anydataset `ffbf946`、anytrain `9f9f07a` 和
speech-to-speech `feee74e`，将复旦 canonical WMT19 TTS base root 从 schema v1 离线迁移为
v2。迁移先发布到同一 NAS 的 sibling staging；切换前后各全量读取 1000 条 base/LongCat 合并样本，source/target
文本、waveform 和 `[frame, codebook]` LongCat tensor 均通过。原 v1 目录保留为
`base-schema-v1-backup-20260722/`，用于短期回滚。

默认 `root=None` 的 codec-oracle DataModule 成功 collate 4 条真实样本，codes shape 为
`[4, 43, 4]`，mask shape 为 `[4, 43]`。随后在 `145` GPU 5 使用默认 canonical 路径完成
Flow 单卡 2-step smoke：loss `2.32828 -> 2.15628`，sample feature MSE
`2.57467 -> 2.51836`，reconstruction 和两个 step sample 均生成；`step-1.ckpt`、
`step-2.ckpt`、`last.ckpt` 各约 998 MiB。`last.ckpt` 包含 97 个 state key，未包含 backbone
key。产物位于
`145:/tmp/wmt19-canonical-oracle-20260722/005-codec-oracle-smoke/longcat/flow-8l/codec/`。
