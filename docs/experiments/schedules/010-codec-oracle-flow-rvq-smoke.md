# 010 Codec Oracle Flow/RVQ Smoke

## 目标

在复旦真实 Qwen3、LongCat prepared codes 与 codec 资源上，验收 codec oracle 的 Flow/DiT 和
RVQ 两种 objective，以及各自的单卡 fixed-sample、两卡 LBA 配置。

## 配置

- 单卡：`acoustic_oracle_smoke`、`acoustic_oracle_rvq_smoke`。
- 两卡：`acoustic_oracle_ddp_lba_smoke`、`acoustic_oracle_rvq_ddp_lba_smoke`。
- 四组均运行 2 optimizer steps，sample、histogram、grad norm 和 checkpoint 间隔均为 1 step。
- 使用 codec initialization；该选项只初始化 semantic audio embedding，RVQ acoustic embeddings
  仍由 decoder 随机初始化。

## 验收标准

- forward、backward 和 optimizer step 完成，监督 loss finite。
- 单卡与两卡 wrapper 分别默认暴露 1/2 张 GPU；实际 world size 随 `CUDA_VISIBLE_DEVICES`，两卡
  运行使用静态 `ddp` 和 Lightning distributed sampler。
- Flow 产出 feature MSE，RVQ 产出总/逐 codebook accuracy 与 feature MSE。
- reconstruction/sample waveform、TensorBoard histogram、step checkpoint、`last.ckpt` 和
  `metrics.json` 均存在。
