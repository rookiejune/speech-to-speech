# 023 使用 Qwen3-0.6B 的 UniSS-style live S2ST

## 目标

用 `zhuyin.datasets.s2st:source` 持续消费合成 S2ST catalog，并验证 Qwen3-0.6B 在
GLM4 + BiCodec composed runtime 下的 UniSS-style `target_cot` 训练。数据生成细节对 DataModule
透明：训练始终持有同一个 live dataset，最终 catalog 每次增长时只观察明确的 snapshot 更新。

## 固定组合

```text
source prompt:  source waveform -> online GLM4 semantic + BiCodec global
response:       target text CoT + online BiCodec target sequence
backbone:       Qwen/Qwen3-0.6B，bf16，FlashAttention 2
dataset:        workspace synthetic S2ST live catalog
factories:      translation、tts
```

source declaration 按语言列出有序 slots；物理来源可以重复，也可以在后续 revision 增加语言或 slot。
所有 target text 都由同一个通用 translator 生成。speaker-list 模式用同一个 TTS/speaker 合成一个
family 的 source 与所有 target audio；reference-audio 模式保留原始 source waveform，并用它为所有
target audio 提供同一个 condition。新增 speaker 只影响以后接纳的 family。

主线 trace 为 `target_cot`；`full_cot` 是同预算 ablation，在目标译文前增加 source ASR text。
lexical prompt 只用自然语言描述 response 顺序，不增加额外 task identity token。

## 设备

当前 experiment 的直观配置是：

```yaml
devices:
  translation: [0]
  tts: [1]
```

未列出的可见设备全部用于训练。列表可以增加 id 来提高对应工厂吞吐，但同一 id 不能重复。单设备且
尚无首版 snapshot 时不启动生成，改用 toy 数据测真实模型与在线 codec 性能；已有 sealed catalog 时
不启动工厂，全部可见设备用于训练。

## 阶段与门槛

1. **P0 单卡 toy perf：**在无首版 catalog 的 lineage 上运行有界 warmup/measurement，验证真实
   Qwen、GLM4/BiCodec 在线编码、forward/backward、step time、吞吐和显存；不生成数据、不保存正式
   checkpoint。
2. **P1 首版发布：**用小规模 `initial_sources`（默认 8 families）完成
   `source@r -> translation@r -> tts@r`，训练等待第一个最终 snapshot 后立即开始；验证工厂重启、
   原子发布、live refresh 和 checkpoint cursor。
3. **P2 持续增长：**设置 `interval_sources`，边发布 final snapshot 边训练。增加语言时先回填已有
   families，再处理新增语言来源；纵向增加 source slot 时保持各 slot 的稳定 cursor。
4. **P3 CoT ablation：**在相同 family、token 和 optimizer budget 下比较 `target_cot` 与
   `full_cot`；除非新增 source-ASR step 改善 held-out text/audio consistency，否则保留控制更少的
   `target_cot`。
5. **P4 扩量：**只通过增加 source slots、languages、`interval_sources` 或 factory device list 扩大
   规模；speaker list 的新增值只供未来 family 抽取。

## 发布与恢复契约

每个 revision 固定依赖：

```text
source@r -> translation@r -> tts@r
```

三个 snapshot 在逻辑上属于同一 revision，但发布时间可以不同。translation 必须绑定 exact source
snapshot，TTS 必须绑定 exact translation snapshot；只有最终 TTS catalog 对训练可见。final sample
包含 source/target text、waveform、语言、source slot 和 voice condition，不发布训练 codec view。

旧 family 的 source text/audio 必须从已发布 final catalog 恢复。新增语言回填不能重读旧物理来源，
speaker-list 和 reference-audio 模式跨 revision 都不能重合成历史 source waveform。DataModule 是 live
dataset 的唯一 close owner；cursor 只在 optimizer step 真正前进后提交。

## 日志与验收指标

- toy route：独立 `toy-perf/`、warmup、measurement window、step time、samples/s、valid tokens/s、
  audio frames/s 和 peak memory；
- generation route：`generation/translation.log` 与 `generation/tts.log` 分别记录 revision、exact
  upstream、依赖/反压等待、阶段耗时、完成行数和 snapshot publish；
- training：普通 loss、吞吐和 checkpoint 日志不变；每次 catalog 增长额外记录一次
  `data.snapshot.updated previous=... current=... added_samples=... total_samples=... cursor=...
  wait_seconds=...`；
- 质量：held-out BLEU/chrF、生成语音 ASR WER/CER、text/audio consistency 和 RTF。

## 当前状态与就绪条件

配置解析、workspace source 路由、single-device toy 选择、multi-device factory 切分和 live cursor 契约
已有定向测试。正式状态仍为 `draft`，需要依次完成：

1. 真实 translator/TTS runtime 与各声明语言的短生成验证；
2. P0 真实 Qwen3-0.6B + GLM4/BiCodec forward/backward 与 perf report；
3. P1 小规模首版发布、中断恢复、source waveform 复用和 snapshot refresh；
4. 单 rank、双 rank checkpoint cursor 恢复与工厂提前失败的 fail-fast 验证；
5. P2 短并发 pilot，确认生成阶段等待、训练吞吐和显存后再扩量。

以上门槛全部通过前，toy 或 fake-backend 结果不得写成真实在线训练通过结论。
