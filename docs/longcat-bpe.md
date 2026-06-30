# LongCat BPE

## 100k 配置

训练侧使用 `configs/bpe/longcat_100k.yaml`：

```yaml
cache_dir_env: BPE_CACHE_DIR
codec_name: longcat
vocab_size: 100000
min_frequency: 0
max_token_length: null
codebook_sizes:
  - 8192
```

对应 artifact 目录为：

```text
/nfs/yin.zhu/bpe/longcat/vocab_100k_minfreq_0_maxlen_none_codes_8192
```

这里要求运行环境中 `BPE_CACHE_DIR=/nfs/yin.zhu/bpe`。代码会通过
`BPEConfig.artifact_name` 推导 `longcat/vocab_100k_minfreq_0_maxlen_none_codes_8192`，
不在配置中直接写绝对 artifact 路径。

## 100k 评估结果

本次 artifact 的实际 vocab size 为 `100000`。评估语料包含 source 和 target 两侧
LongCat semantic ids：

| 指标 | 数值 |
| --- | ---: |
| num_sequences | 1061750 |
| original_tokens | 126876136 |
| encoded_tokens | 47744622 |
| mean_original_length | 119.49718483635507 |
| mean_encoded_length | 44.96785684012244 |
| compression_ratio | 0.37630892227045754 |
| compression_gain | 0.6236910777295425 |
| compression_factor | 2.6573911507771495 |

结论：100k BPE 后的平均序列长度约为原始 LongCat semantic ids 的 `37.63%`，
压缩约 `2.66x`，可作为后续 WMT19 quality 训练的默认大词表候选。

## 训练入口

可直接用 Hydra override 切换 BPE：

```bash
python scripts/train.py experiment=wmt19_quality_muon bpe=longcat_100k
```

也可以使用已经固定 100k BPE 的实验配置：

```bash
python scripts/train.py experiment=wmt19_quality_100k_muon
python scripts/train.py experiment=wmt19_quality_100k_full_adamw
```

这些 quality 实验的模型配置默认启用 `model.acoustic.enabled=true`，并通过
`train.acoustic_loss_weight` 控制 acoustic/FM loss 权重；模型配置分层约定见
[model-config.md](model-config.md)。
