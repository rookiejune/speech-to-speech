# Bidirectional Semantic AR with Acoustic-Guided FM

## 背景

已有隔离实验表明，纯 translation 或加入 acoustic 的 translation 能学到局部映射，
可以翻译单个词语，但不能稳定生成连贯句子。另一次先训练 AR 再自由生成的实验显示，
主要问题卡在 AR 推理时的暴露偏差上。

因此下一轮不再把 semantic AR、translation 和 acoustic 生成拆成完全串行的阶段。
当前假设是：speech semantic LM prior 仍然应该主导训练，但 acoustic-guided FM 生成损失
需要从训练早期开始常驻，以约束自由生成最终服务于可生成波形。

## 目标

验证以下判断：

1. 双向 AR + 双向 translation 能否比单向 translation 更稳定地形成句子级 speech semantic prior。
2. 已有 acoustic-guided FM 路径能生成可听出的词级翻译后，低权重常驻是否能继续约束句子级 free-running 生成。
3. acoustic/FM loss 是否会在权重较小时提供生成约束，而不是把训练拉回局部词级对齐。
4. 面对 codec token 这种新离散语言时，Qwen3 backbone 只做 LoRA 是否足够，还是需要 full fine-tune 才能学稳 free-running 动力学。

## 非目标

- 不把 teacher-forcing loss 作为是否能生成句子的最终证据。
- 不使用原始 acoustic codes decode free-running semantic 序列来判断句子质量；生成 semantic
  和原 acoustic codes 已经错配，这个评估不可靠。
- 不在第一轮引入复杂推理链路，例如 cycle decoding、双向 rerank 或 back translation。

## 方法

整体路线保持为 semantic-first AR + acoustic-guided FM：

- semantic AR loss 是主损失，负责 speech semantic token 的长程建模和句子级连贯性。
- translation loss 引入 source/target speech semantic 之间的跨语音映射。
- acoustic-guided FM loss 从开始就挂着，但使用低权重，作为生成侧约束。

训练样本按 pair 双向展开：

- source autoregression
- target autoregression
- source-to-target translation
- target-to-source translation

推理主路径先只看目标方向，不在本实验里增加双向推理复杂度。

## 第一轮模型对照

第一轮不要把模型规模和训练配方同时铺成大矩阵。主变量仍然是双向 mixed +
低权重 acoustic/FM 常驻；模型训练方式只做最小对照，用来判断 LoRA 是否成为瓶颈。

建议顺序：

| 优先级 | 模型 | 训练方式 | 目的 |
| --- | --- | --- | --- |
| A | Qwen3-0.6B | audio embedding/head full + backbone LoRA | 快速 baseline，验证训练配方是否有基本信号 |
| B | Qwen3-0.6B | audio embedding/head full + backbone full | 判断 LoRA 是否太弱，codec token 分布是否需要全量改 backbone |
| C | Qwen3-8B | audio embedding/head full + backbone LoRA | 容量和文本预训练 prior 检查，只做短程对照 |
| D | Qwen3-8B | full | 第一轮暂不做，成本高且变量太多 |

优先比较 A 和 B。Qwen 的预训练数据主要是文本，而当前训练目标是 LongCat
semantic BPE / codec token。audio embedding、audio special tokens 和 audio LM head
必须全量训练；如果 backbone 只用 LoRA，可能不足以把文本 LM 的长程建模能力迁移到
codec token 的 free-running 分布上。

如果 A 失败而 B 明显更连贯，说明第一轮瓶颈更可能是 LoRA 更新能力不足。若 A 和 B
都失败，再看 C；如果 C 也不改善，优先回到训练目标、task 比例、acoustic/FM 权重和
暴露偏差处理，而不是直接做 8B full。

建议分组学习率：

```text
audio embedding/head: 1e-4
backbone LoRA/full: 1e-5 到 3e-5
DiT/FM: 1e-4
```

当前训练入口已经支持 AdamW 参数组和 Muon 参数组分别设置 learning rate；更细的
audio embedding/head、backbone、DiT 三路 learning rate 需要先扩展 `anytrain.optim`
的公共分组接口，不在本轮 `speech-to-speech` 内复制实现。

## 建议阶段

### S0: AR-Dominant Warmup

目的：稳住 speech semantic LM prior，同时让 acoustic/FM 路径从一开始接触训练分布。

建议比例：

```text
source_ar: 1
target_ar: 1
source_to_target: 0
target_to_source: 0
```

建议 loss 权重：

```text
semantic_ar: 1.0
translation: 0.0
acoustic_fm: 0.01
```

### S1: Bidirectional Mixed Training

目的：引入双向 translation，但仍保持 AR prior 主导。

建议比例：

```text
source_ar: 1
target_ar: 1
source_to_target: 0.5
target_to_source: 0.5
```

建议 loss 权重：

```text
semantic_ar: 1.0
translation: 1.0
acoustic_fm: 0.03
```

### S2: Translation-Weighted Mixed Training

目的：在不丢失句子级 prior 的前提下，提高 translation 任务权重。

如果主目标是 source-to-target，可使用非对称比例：

```text
source_ar: 1
target_ar: 2
source_to_target: 1
target_to_source: 0.25
```

建议 loss 权重：

```text
semantic_ar: 1.0
translation: 1.0
acoustic_fm: 0.05-0.1
```

## 需要记录的指标

训练日志至少需要拆开：

- total loss
- semantic AR loss
- translation loss
- acoustic/FM loss
- 各 task 的 batch/token 计数
- supervised semantic tokens
- acoustic frames 或有效 acoustic mask 数量

生成评估需要记录：

- 固定样本的 free-running waveform
- 对应的 source/target reference waveform
- 生成长度、EOA 命中情况和是否提前停止
- 人工听感备注：是否成句、是否只翻关键词、是否重复、是否断裂
- 如有 ASR 辅助，记录 ASR 文本和明显错误，但不把 ASR 作为唯一标准

## 判断标准

优先看自由生成 waveform，而不是 teacher-forcing loss。

成功信号：

- 生成音频能形成完整短句，而不是单个词或关键词串。
- source-to-target 主方向比已有 translation+acoustic 实验更连贯。
- acoustic/FM loss 低权重常驻后，没有明显破坏 semantic 生成长度和句子结构。

失败信号：

- teacher-forcing loss 很低，但 free-running 仍然快速退化、重复或只吐关键词。
- acoustic/FM loss 权重稍高就导致 translation 退回局部词级对齐。
- 双向训练提升 target-to-source，但伤害主方向 source-to-target。

## 待确认

- S0 是否需要完全不启用 translation，还是从一开始给很小 translation 比例。
- 第一轮正式训练的 acoustic/FM 初始权重在 `0.01` 还是更低。
- 主任务是否明确固定为 source-to-target；如果是，S2 应采用非对称采样。
- Qwen3-0.6B full 的可用显存和 batch 设置；如果显存压力太大，是否采用 partial full 或更小 LR。
