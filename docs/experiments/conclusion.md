# Experiment Conclusions

## 已验证

- 100k LongCat BPE 在 WMT19 source+target semantic ids 上可把平均序列长度压到原始长度的约 37.63%，压缩约 2.66x。
- 纯 translation 或加入 acoustic 的 translation 能学习局部词级映射，但不足以稳定生成连贯句子。
- 加入 acoustic 的 translation 已经能生成可听出的词级翻译，说明生成波形路径和 acoustic-guided FM 约束不是空转。
- 先单独训练 AR 再自由生成的路径仍会受到 AR 推理暴露偏差影响。
- 因此后续正式训练不应长期使用 `acoustic_loss_weight = 0`；acoustic-guided FM 生成损失应以低权重常驻。

## 当前工作假设

- 整体方法仍保持 semantic-first AR + acoustic-guided FM。
- semantic AR loss 负责句子级 speech semantic prior。
- acoustic-guided FM loss 不后置，而是从早期开始作为低权重生成约束。
- 双向训练更适合作为训练约束和 prior 增强，推理主路径先保持单向。
- Qwen3 的文本预训练 prior 不一定能通过 LoRA 充分迁移到 codec token 分布；第一轮需要用 Qwen3-0.6B full backbone 对照判断 LoRA 是否瓶颈。

## 支撑记录

- 100k LongCat BPE 压缩统计见 [002_longcat_bpe_100k.md](results/002_longcat_bpe_100k.md#L12-L24)；配置和训练入口见 [longcat-bpe.md](../longcat-bpe.md#L3-L25).
- Acoustic sampler 性能结论见 [diagonal-profile.md](../diagonal-profile.md).
- 下一轮训练计划见 [001_bidirectional_semantic_ar_acoustic_fm.md](schedules/001_bidirectional_semantic_ar_acoustic_fm.md).
