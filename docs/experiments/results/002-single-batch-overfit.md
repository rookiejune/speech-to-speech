# 002 Single Batch Overfit Result

> 状态：已重新验证。修正后的入口通过 `Subset(dataset, [sample_index])` 固定同一条
> raw sample；TTS 与 S2ST 均完成 100 个 optimizer steps。

## 环境

- 日期：2026-07-13。
- 机器：144，NVIDIA GeForce RTX 4090 24 GiB，物理 GPU 1。
- Python 3.12，PyTorch 2.9.0+cu129，Lightning 2.6.1，bf16 mixed，
  `flash_attention_2`。
- Backbone：`Qwen/Qwen3-0.6B`；codec：LongCat `16k_4codebooks`；audio
  tokenizer：100k LongCat BPE artifact。
- 数据：`wmt19_tts_codec(codec="longcat", split="train")` 的 sample 0，batch size 1。
- 代码从本地工作树同步到 144 的 `/tmp/s2s-overfit-current` 隔离快照；模型、数据和输出
  使用 `/mnt/pami202/zhuyin` 共享资源，没有修改远程共享 Git 工作树。

计划原指定 121；运行前该机四张 A100 均已有约 29.6 GiB 占用，剩余显存不足以容纳
本实验约 12 GiB 峰值，因此改用空闲的 144 GPU 1。模型、数据、样本和训练配置不变。

## 配置与命令

两项任务均使用：`max_steps=100`、`learning_rate=2e-5`、`weight_decay=0.01`、
`seed=0`、sample 0。正式入口为：

```bash
CUDA_VISIBLE_DEVICES=1 jobs/002/01_tts.sh
CUDA_VISIBLE_DEVICES=1 jobs/002/02_s2st.sh
```

远程运行显式设置 `LOCATION=fudan`、共享 Hugging Face/anytrain cache 和 py312 Python。
TTS 用时约 8 分 11 秒，S2ST 用时约 7 分 31 秒；观察到的 CUDA memory 峰值约
12.0 GiB，结束后 GPU memory 恢复到 4 MiB。

有效 TensorBoard event 位于：

- TTS：`.../002-single-batch-overfit/tts/tensorboard/version_3/`
- S2ST：`.../002-single-batch-overfit/s2st/tensorboard/version_1/`

TTS `version_2` 是一次 SSH 中断后只写入初始化事件的无效运行，不包含训练 step，
不参与下列统计。

## 结果

下表比较前 20 与后 20 steps 的均值：

| Task | Objective | First 20 | Last 20 | Last / first |
| --- | --- | ---: | ---: | ---: |
| TTS | total | 6.0757 | 1.6963 | 0.2792 |
| TTS | semantic | 3.7896 | 0.0469 | 0.0124 |
| TTS | flow matching | 2.2862 | 1.6493 | 0.7214 |
| S2ST | total | 6.2989 | 1.6252 | 0.2580 |
| S2ST | semantic | 4.0174 | 0.0251 | 0.0063 |
| S2ST | flow matching | 2.2815 | 1.6001 | 0.7013 |

两个任务均达到 `max_steps=100`，semantic 与 flow matching 的后 20-step 均值都低于
前 20-step 均值。模型摘要显示 991M 参数可训练、529 个模块处于 train mode、0 个模块
处于 eval mode；loss、梯度和训练过程保持 finite，未发生 OOM 或持续显存增长。

## 结论

- 固定同一条真实样本后，TTS 与 S2ST 的 semantic objective 都能在 100 steps 内接近记忆。
- 加入 source semantic/acoustic condition 后，S2ST 仍保持与 TTS 一致的可优化性。
- 随机 flow matching 的后 20-step 均值相对前 20 steps 下降约 28%（TTS）和 30%
  （S2ST），说明两个路径都能收到有效优化信号。
- 本实验不证明泛化、生成音质或 flow objective 已充分 overfit。

## 回归检查

实验使用的本地工作树通过 basedpyright（0 errors）、68 个本地测试和 compileall。
