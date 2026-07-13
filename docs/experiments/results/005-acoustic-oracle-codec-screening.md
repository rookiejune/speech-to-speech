# 005 Codec Oracle Screening

对应计划：[`schedules/005-acoustic-oracle-codec-screening.md`](../schedules/005-acoustic-oracle-codec-screening.md)。

## 121 Codes-Only Smoke

2026-07-13 在 121 的 A100 上，使用 WMT19 TTS prepared codec store 的 train sample 0，
对 LongCat flow 与 UniCodec unified-token 两种 objective 的 codec/random initialization
各运行 2 个训练 step。输入只包含 prepared codes；训练未调用 waveform encoder，也未保存
连续 feature 数据。

| codec / init | prepared codes | objective | codebook | step 1 -> 2 probe metric |
| --- | --- | --- | --- | ---: |
| LongCat / codec | `[17, 4]` | acoustic flow | `[8192, 1280]` | feature MSE `2.06496 -> 2.05809` |
| LongCat / random | `[17, 4]` | acoustic flow | matched `[8192, 1280]` | feature MSE `2.06438 -> 2.05682` |
| UniCodec / codec | `[75, 1]` | causal token | `[16384, 512]` | teacher-forced accuracy `0.0267 -> 0.1467` |
| UniCodec / random | `[75, 1]` | causal token | matched `[16384, 512]` | teacher-forced accuracy `0.0000 -> 0.1200` |

四组均完成 checkpoint 加载、objective forward/backward、TensorBoard、checkpoint、
non-finite callback、oracle reconstruction、训练中 flow sample/token probe decode 和
`metrics.json` 写出。
2-step smoke 只验证路径有效和参数可更新，不支持初始化优劣或 codec 质量结论。

## LongCat 2000-Step DDP + LBA

2026-07-13 在 121 的物理 GPU 2、3 上完成 LongCat codec initialization 的 2000-step
全 prepared train dataset 长跑。训练使用 2-rank NCCL DDP、显式 distributed sampler 和
LBA；`distributed.contract` 验证 `world_size=2`。为避开 121 上多 worker DataLoader 的
共享内存句柄故障，本次每个 rank 使用 `num_workers=0`、`pin_memory=false`。

| metric | value |
| --- | ---: |
| train flow loss, first / last | `2.22379 / 0.65077` |
| 20-step mean, first / last | `2.04266 / 0.63176` |
| last / first 20-step mean | `0.30928` |
| step-200 / step-2000 sample feature MSE | `2.32860 / 1.91538` |
| wall time, rank 0 trainer fit | `130.24s` |

训练按 `max_steps=2000` 正常结束；每 200 step 的 sample callback 共写出 10 个音频，
flow-time histogram 每 20 step 记录。最终 checkpoint 为 `last.ckpt`（约 188 MB），GPU
显存退出后均回落到 10 MiB。监督 loss 明显下降；sample feature MSE 有下降但并非单调，
本次 full-dataset 运行只验证长跑稳定性，不用于 codec/random initialization 对照结论。

产物位于
`/mnt/pami202/zhuyin/dynamic/train/speech-to-speech/005-codec-oracle-ddp-lba/longcat/codec-2000-ddp`，
启动日志位于 121 的 `/tmp/s2s-longcat-2000-ddp.log`。

## 阶段日志

stdout JSON stage 能定位以下边界：

- prepared sample load：LongCat 约 `0.35s`，UniCodec 约 `0.27s`；
- codec checkpoint load：LongCat 约 `6.6-7.3s`，UniCodec 约 `6.6-7.7s`；
- LongCat dequantize probe：约 `0.15-0.18s`，训练首个 batch 的 dequantize 约 `0.002s`；
- UniCodec 16,384 行 codebook extraction：约 `0.05s`；
- callback waveform decode：稳定后约 `0.006-0.007s`，首次 reconstruction 较慢。

日志已分别覆盖 callback dequantize、callback waveform decode 和训练首次 dequantize，
后续远程停住时可以直接判断发生在哪一层。

## 环境与产物

- 运行代码位于 121 隔离副本 `/tmp/s2s-codec-oracle`，没有覆盖共享工作区。
- 产物位于
  `/mnt/pami202/zhuyin/dynamic/debug/speech-to-speech/005-codec-oracle/{longcat,unicodec}/{codec,random}`。
- LongCat 使用 121 主 `py312`；UniCodec 使用项目专用 Python 3.9 / Torch 2.4 环境。
- 隔离运行使用本地当前 `anytrain` wrapper；121 shared `anytrain` 旧版本缺少 LongCat
  time-major 到上游 codebook-major 的转换。

## 数据迁移边界

121 LongCat store 的物理 payload 仍是旧 semantic/acoustic dict。121 当前正式 workspace loader
会在边界转换成 `[frame, codebook]` Tensor，因此本次训练拿到的逻辑数据契约正确；本地最新
严格 workspace loader 已拒绝该旧 payload，并要求重新 materialize。后续切换 shared workspace
前需要迁移 store，不能在训练入口增加静默兼容。

## 已废弃实现基线

本轮调整前曾完成 LongCat、UniCodec、DAC 的 waveform 在线 encode flow smoke。该实现证明三种
codec 都具备 feature-to-waveform 闭环，但违反当前 codes-only 数据契约，并且把 LongCat
semantic condition 固定用于所有 codec；这些数值不再作为 005 当前实验结果，DAC 也不进入
本轮后续 overfit。
