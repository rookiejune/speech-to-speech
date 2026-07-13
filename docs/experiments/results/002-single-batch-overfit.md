# 002 Single Batch Overfit Result

> 状态：待重新验证。原入口的 `sample_index` 只用于日志样本，训练 DataLoader
> 实际遍历了完整数据集。以下数值保留为历史运行记录，不支撑单 batch overfit
> 结论；修正后的入口需重跑 TTS 与 S2ST。

## 环境

- 日期：2026-07-13。
- 机器：121，NVIDIA A100-PCIE-40GB，物理 GPU 0。
- Python 3.12，PyTorch 2.9.1+cu128，Lightning 2.6.1，bf16，
  `flash_attention_2`。
- Backbone：`Qwen/Qwen3-0.6B`；codec：LongCat `16k_4codebooks`；audio
  tokenizer：100k LongCat BPE artifact。
- 数据：`wmt19_tts_codec(codec="longcat", split="train")` 首条样本，batch size 1。
- 代码通过 `/tmp/s2s-overfit` 与 `/tmp/s2s-deps` 临时快照运行，没有修改 121
  的共享 Git 工作树。

## 配置

TTS 和 S2ST 分别运行：

```text
max_steps=100
learning_rate=2e-5
weight_decay=0.01
seed=0
```

正式入口为 `jobs/002/01_tts.sh` 与 `jobs/002/02_s2st.sh`。两个任务均完成
100 次 optimizer step；观察到的 CUDA memory 约 11.1 GiB，任务结束后 GPU memory
恢复为空闲状态。TensorBoard event 位于：

- TTS：`.../002-single-batch-overfit/tts/tensorboard/version_1/`
- S2ST：`.../002-single-batch-overfit/s2st/tensorboard/version_0/`

## 结果

下表比较前 20 与后 20 steps 的均值：

| Task | Objective | First 20 | Last 20 | Last / first |
| --- | --- | ---: | ---: | ---: |
| TTS | total | 7.2862 | 2.2657 | 0.3110 |
| TTS | semantic | 4.5414 | 0.0550 | 0.0121 |
| TTS | flow matching | 2.7448 | 2.2108 | 0.8054 |
| S2ST | total | 6.5335 | 2.2364 | 0.3423 |
| S2ST | semantic | 3.8921 | 0.0367 | 0.0094 |
| S2ST | flow matching | 2.6413 | 2.1997 | 0.8328 |

TensorBoard 同时包含按 task 划分的 semantic text/audio loss、token count、flow loss、
frame count 和 flow time。

## 原结论（已失效）

- TTS 与 S2ST 的 semantic objective 都能在 100 steps 内接近记忆固定样本。
- 加入 source semantic/acoustic condition 后，S2ST 与 TTS 一样保持可优化。
- flow matching 的随机训练均值下降约 17%–20%，证明当前路径能收到有效优化信号；
  本实验不证明随机 flow objective 已充分 overfit，也不用于评价生成音质。

## 暴露项

- 首次前台运行没有传 Hugging Face 共享缓存环境，121 外网不可达；正式远程运行需要
  按 workspace 约定设置 `HF_HOME`、`HF_HUB_CACHE`、`HF_DATASETS_CACHE` 和
  `HF_ENDPOINT`。补齐并启用离线缓存后模型正常加载。
- Lightning 报告训练开始时 427 个 backbone 子模块处于 eval mode。参数仍有梯度且
  两项 objective 均下降，但完整训练前应明确 backbone 的训练/冻结与 module mode 契约。
- Qwen tied-weight 配置提示、LongCat `weight_norm` deprecation warning 和单样本
  DataLoader worker warning 未影响本次 100-step 运行。

## 回归测试

实验完成后的本地工作树测试：`26 passed, 14 subtests passed`。
