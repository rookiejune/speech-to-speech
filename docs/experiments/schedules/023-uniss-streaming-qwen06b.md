# 023 使用 Qwen3-0.6B 的 UniSS-style 流式 S2ST

## 目标

在本地运行时契约上复现 UniSS-style 语音链路：骨干模型换成
`Qwen/Qwen3-0.6B`，任务提示使用自然语言文本表达步骤。任务表示就是 CoT；`trace` 只负责选择
CoT 变体，不生成额外任务身份 token。响应只保留字段边界、语言选择、音频 schema 和 codec grammar
所必需的协议 token。

## 固定组合

```text
source prompt: BiCodec global + GLM4 semantic
response:      target text CoT + 完整 BiCodec target sequence
backbone:      Qwen/Qwen3-0.6B，预训练，bf16，FlashAttention 2
dataset:       workspace 默认过滤的 WMT19 translation seed，展开为 2N
```

主线 trace 为 `target_cot`；`full_cot` 是同预算 ablation，在目标译文前增加 source ASR text。
lexical prompt 只用自然语言描述 response 顺序，不增加 `<s2st>` 或 `<cot>` task token。
`target_cot` 保留 `<mt>`、语言 selector、`</mt>`、BOA/EOA、schema 和 BiCodec private grammar；
`full_cot` 额外保留 `<asr>...</asr>`。这些 token 属于监督 response protocol，不承担任务表示。

## 阶段与门槛

1. **P0 契约：**组合 `runtime=glm4_bicodec_composed`，验证自然语言 prompt、token mask、source
   view 完整性，以及一次真实前向/反向计算。
2. **P1 有界流：**改造已定位的 producer，用 8 或 32 个样本发布两个 immutable snapshot，完成
   单 rank、双 rank、seal 和 resume probe。P1 通过前不得进入正式在线训练。
3. **P2 并发 pilot：**使用默认过滤的 WMT19 seed，边发布 batch 边训练，记录 producer stage、
   wait ratio、吞吐、GPU memory 和 checkpoint cursor。
4. **P3 CoT ablation：**在相同 sample、token 和 optimizer budget 下比较 `target_cot` 与
   `full_cot`；除非新增 source-ASR step 改善 held-out text/audio consistency，否则保留控制更少的
   `target_cot`。
5. **P4 扩量：**仅在 P0-P3 通过后增加 batch size 和 stream size；live stream 不承载 validation，
   另行保留 sealed immutable validation dataset。

## 数据与发布契约

producer 消费 workspace 固定的默认 WMT19 filter，每个 accepted pair 在 translation-seed 层恰好
展开为两个方向；`streaming_s2st` 内不得再次过滤或展开。一次 snapshot 原子发布 `base/`、GLM4
input view 和 BiCodec output view。source sample 必须同时保留 GLM4 semantic 与 BiCodec global；
target sample 必须包含完整 BiCodec sequence。发布前校验 sample count、text alignment、stream
identity 和 filter fingerprint。

## 验收指标

- source/target view 与 index 100% 对齐；固定样本的 codec grammar 完整；loss 和 gradient finite。
- immutable seal 与 checkpoint cursor 精确恢复，不重复提交 sample。
- 每次 pilot 记录 `streaming/wait_ratio`、producer stage latency、samples/s、valid tokens/s、
  audio frames/s 和 peak memory。
- 在相同预算下比较 held-out BLEU/chrF、生成语音 ASR WER/CER、text/audio consistency 和 RTF。

## 当前状态与阻塞

本仓库已经具备 GLM4 input + BiCodec output 的运行时、publisher 和 toy smoke 结构校验；这些校验只
证明 sample/view 合并、CoT 序列和缓存恢复契约，不等于真实 codec 或 Qwen 训练已经通过。
`scripts/streaming_probe_producer.py` 已支持确定性的 GLM4/BiCodec 双 store，可验证 `2N`、两批
immutable snapshot、seal 和中断恢复；它不加载真实模型，也不读取 WMT19，因此只能作为发布/恢复
契约 probe，不能拿它代替本实验的真实 producer。另一个已定位的外部 producer worktree
`/private/tmp/workspace-streaming-producer`（`codex/streaming-s2st-producer`，revision `eed19f9`）
同样把 input/output codec 硬编码为 `longcat`，当前 revision 不能直接运行本 schedule。

本地两步训练 smoke 已实际尝试：联网路径在 Hugging Face `HEAD` 请求超时；使用已有 cache 的离线路径
继续到 BiCodec backend 后，被本机 Spark-TTS runtime 缺少 `einx` 阻塞。因此当前只能声明配置解析、
toy 数据与契约测试通过，不能声明真实 Qwen3-0.6B forward/backward 通过。

正式在线实验的就绪门槛是：

1. producer 接受 workspace 默认过滤的 WMT19 seed，严格按 seed 层展开为 `2N`，并以同一 snapshot
   原子发布 GLM4 input store 与 BiCodec output store；source 同时保留 GLM4 semantic 和 BiCodec
   global，target 保留完整 BiCodec；
2. GLM4 stage 所需的 `transformers==4.44.1` 与 MOSS-TTS/BiCodec runtime 隔离到兼容的独立进程或
   环境，不能在同一依赖环境中混跑；
3. 改造后的 producer 完成 P1 有界流、immutable seal、单/双 rank cursor resume、text/view 对齐
   和损坏或中断后的重试 probe；
4. 在真实 Qwen3-0.6B、bf16 和默认过滤 WMT19 上完成一次短并发 pilot，并记录等待、吞吐、显存和
   checkpoint cursor，确认 lexical prompt 不含非必要 task token。

以上门槛全部通过前，正式在线运行状态保持 `draft`；toy smoke 或 LongCat probe 的结果不得写成
真实 GLM4/BiCodec 训练通过结论。
