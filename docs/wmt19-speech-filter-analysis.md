# WMT19 Speech Filter Analysis

## 数据来源

本分析基于 121 上的过滤缓存：

```text
~/repos/speech-to-speech/storage/wmt19-zh-en-tts-longcat-1000/reports/speech_quality_metrics.jsonl
```

语音过滤规则来自 `workspace/scripts/prepare_wmt19_tts.py`：

- `utmos < 3.0` 标记为 `utmos_low`
- `wer > 0.4` 标记为 `wer_high`
- `chrf < 50.0` 标记为 `chrf_low`
- 本次未启用 BLEU 阈值

每条样本的 source 和 target 两侧音频都被检查。1000 条样本均为
`audio_count=2`、`checked_count=2`，没有缺 waveform、缺文本或其他 warning。

## 总体结果

| 阶段 | 样本数 |
| --- | ---: |
| 原始 full-store | 1000 |
| 语音过滤 accept | 475 |
| 语音过滤 reject | 525 |

被语音过滤掉的 525 条里，问题高度集中在 source 中文侧：

| 触发侧 | 样本数 |
| --- | ---: |
| 仅 source | 504 |
| 仅 target | 13 |
| source 和 target 同时触发 | 8 |

## 触发原因

以下 flag 不是互斥的，同一条样本可以同时触发多个原因：

| flag | 次数 |
| --- | ---: |
| `source_wer_high` | 436 |
| `source_utmos_low` | 124 |
| `source_chrf_low` | 63 |
| `target_wer_high` | 20 |
| `target_chrf_low` | 7 |
| `target_utmos_low` | 1 |

主要组合：

| 组合 | 样本数 |
| --- | ---: |
| `source_wer_high` | 333 |
| `source_utmos_low` | 74 |
| `source_wer_high + source_chrf_low` | 50 |
| `source_utmos_low + source_wer_high` | 38 |
| `target_wer_high` | 10 |
| `source_utmos_low + source_wer_high + source_chrf_low` | 9 |

## 指标分布

source 侧 accept 和 reject 的差异主要体现在 WER，其次是 UTMOS：

| 分组 | 指标 | median | mean | 说明 |
| --- | --- | ---: | ---: | --- |
| accept source | WER | 0.0 | 0.001 | 基本完全匹配 |
| reject source | WER | 1.0 | 0.872 | 大量触发 `source_wer_high` |
| accept source | UTMOS | 3.681 | 3.664 | 高于 3.0 阈值 |
| reject source | UTMOS | 3.554 | 3.473 | 有低分样本，但不是全部 reject 的主因 |
| accept target | WER | 0.0 | 0.032 | 目标侧整体稳定 |
| reject target | WER | 0.0 | 0.069 | 目标侧只有少量高 WER 样本 |
| accept target | UTMOS | 4.449 | 4.389 | 目标 TTS 质量高 |
| reject target | UTMOS | 4.466 | 4.424 | 目标侧音质不是主要问题 |

## 关键判断

语音过滤掉样本的第一原因是 source 中文侧 ASR 匹配失败，而不是目标英文语音质量差。

不过 `source_wer_high` 需要谨慎解释。当前 `TextComparisonEvaluator` 会去标点和折叠空白，但
没有中文分词；WER 使用 jiwer 的词级算法。中文句子常常被当成很少几个“词”，轻微差异会把
WER 放大。因此有不少 source 样本虽然 WER 高，但 chrF 仍然很高：

| 切片 | 样本数 |
| --- | ---: |
| `source_wer_high` 总数 | 436 |
| 其中 `source chrf >= 50` | 373 |
| 其中 `source chrf >= 80` | 318 |
| 其中 `source chrf >= 90` | 199 |
| 其中 `source chrf == 100` | 67 |
| 纯 `source_wer_high` 总数 | 333 |
| 纯 `source_wer_high` 且 `source chrf >= 90` | 170 |
| 纯 `source_utmos_low` | 74 |
| 纯 `source_utmos_low` 且 `wer=0/chrf=100` | 74 |

这说明语音过滤 reject 可以拆成两类：

1. 明确音质问题：如 `source_utmos_low`，尤其纯 UTMOS 低分的 74 条，ASR 文本匹配是好的，但主观质量分低。
2. ASR 文本匹配问题：以 `source_wer_high` 为主，其中一部分可能是真错读、漏读或 TTS 不稳，另一部分可能是中文 WER 指标对未分词文本过敏。

## 代表样本

可听样本和 ASR 对照已经放在：

```text
../debug/speech-to-speech/wmt19-speech-reject-samples/
```

每个样本目录包含 `source.wav`、`target.wav`、reference text、重新跑出的
Whisper ASR 文本和 `summary.json`。总览见
`../debug/speech-to-speech/wmt19-speech-reject-samples/README.md` 和 `summary.jsonl`。

`source_wer_high`：

- index 2：source WER 1.0、chrF 93.7、UTMOS 3.19；target 无异常。
- index 4：source WER 1.0、chrF 69.0、UTMOS 4.17；target 无异常。
- index 17：source WER 2.0、chrF 100.0、UTMOS 4.08；target 无异常。

`source_utmos_low`：

- index 10：source UTMOS 2.97、WER 0、chrF 100；target 无异常。
- index 13：source UTMOS 2.80、WER 0、chrF 100；target 无异常。
- index 33：source UTMOS 2.59、WER 0、chrF 100；target 无异常。

target 侧少量异常多出现在英文短标题：

- index 95：`A Comeback Strategy for Europe`，target WER 0.8、chrF 74.1。
- index 133：`The Year That Ended an Epoch?`，target WER 0.67、chrF 37.5。
- index 960：`Making Do With More`，target WER 1.0、chrF 31.3。

补充抽样观察：

- index 2 的 source ASR 几乎正确，主要差异是“这样得/这样的”和标点，说明中文词级 WER 有误杀风险。
- index 23 的 source ASR 是繁体输出，对简体 reference 的 chrF/WER 都不友好，但语义接近。
- index 472 的 source 音频明显异常，77.44 秒音频被 ASR 成 `Thank you. Thank you. Thank you.`，属于真实坏样本。
- index 95 的 target 是短标题，ASR 多了 `The`，导致 WER 高但音频可能仍可用。

## 后续建议

如果要保留更多可用样本，优先不要简单放宽所有阈值，而是分侧调整：

- source 中文侧：用 CER、chrF 或中文分词后的 WER 替代当前词级 WER，至少不要让 `WER>0.4` 单独决定 reject。
- source UTMOS：纯 `source_utmos_low` 的 74 条应抽听，确认 UTMOS 低分是否和真实音质一致。
- target 侧：现有过滤基本合理，异常数量很少，主要关注短标题 ASR 不稳定。
- 后续重跑时建议在 metrics 中额外保存 Whisper transcript，方便区分真实错读和指标误判。
