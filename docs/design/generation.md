# generation

提供独立于 Lightning 和训练 batch 的真实推理入口。跨模块生成流程见
[总览 §6](../model-design.md)，model 侧原语见 [model](model.md)。

## 对外能力

包级 API 公开以下结构和入口：

- `Request(prompt_ids, task, audio_input_positions, audio_context)`：无 target、无 batch padding 的单条推理输入。
  `prompt_ids` 是一维 layout global token IDs；当固定 `audio_route` 的 decode 需要 prompt stream
  时，`audio_context` 提供同一 reference 的 structured semantic/acoustic codes。可选的
  `audio_input_positions` 只标记 source audio payload 在 prompt 中的位置，供 input tower 使用。
- `Result(response_ids, audio)`：按原请求顺序返回的单条结果。`response_ids` 是不含 EOS/EOA
  的 layout global token IDs；text task 的 `audio=None`。
- `AudioOutput(features, codes, waveform, sample_rate)`：audio task 的 decode 结果。`codes` 保存
  route resolve 后的 structured semantic/acoustic codes；unified-token codec 没有独立 acoustic
  representation，因此 `features=None`。
- `AcousticGeneration(sequence, features, frame_counts)`：acoustic model 与 audio strategy 之间的批量
  返回契约；`features` 是带右侧 padding 的 `[batch, frames, dim]`，`frame_counts` 给出每行
  有效 frame 数。
- `generate_responses()`：校验通用请求外形、按 target modality 分组、padding，并恢复原请求顺序；
  audio generation 与 waveform decode 委托给 `generation.audio` 的 capability strategy。
- `prepare_bicodec_tts_request()`：为 `bicodec_reuse_prompt_global` 构造 reference-conditioned TTS
  request，只接收预编码的 `SemanticAcousticCodes`。
- `prepare_bicodec_global_tts_request()`：为 `bicodec_generate_global` 构造无 audio context 的 TTS
  request。
- `decode_generated_audio()` / `decode_generated_codes()`：分别把 audio token 配合 acoustic
  feature/code 解码为 waveform。semantic/full/structured decode 由 audio strategy 选择。
- `TextProbe` / `TextProbeResult` / `evaluate_text()`：greedy text generation 与 reference NLL
  评估。

`generation.protocol` 定义 service 与 audio strategy 所依赖的窄模型协议：

- `TokenGenerator`：公开 runtime、backbone 和 `generate_tokens()`。
- `FullCodecSequenceGenerator`：在基础 token generation 上增加受 tokenizer grammar 约束的
  `generate_full_codec_sequence()`；frame-aligned flattened codec 与 fixed-length BiCodec
  共用这一 service 能力边界。
- `AcousticFeatureGeneration`：只描述可选的 `generate_audio_features()` 能力。顶层入口的
  `model` 参数仍由 `TokenGenerator` 表达基础契约；需要独立 acoustic codebook 时再检查这个窄
  runtime 协议，避免把 registered `nn.Module` backbone 等无关成员纳入能力识别。
- `AcousticFeatureGenerator`：组合 `TokenGenerator` 与 `AcousticFeatureGeneration`，供训练
  composition 静态表达完整模型契约。
- `TextEvaluationModel`：在 token generation 之外增加 hidden state 与 modality-local logits，
  用于 reference NLL。

## 输入输出

```python
class Request(TypedDict):
    prompt_ids: Tensor
    task: Task
    audio_input_positions: Tensor | None
    audio_context: SemanticAcousticCodes | None

class Result(TypedDict):
    response_ids: Tensor
    audio: AudioOutput | None
```

`prompt_ids` 必须是调用方已经准备好的完整 generation prompt。service 不渲染 chat template、
不插入 instruction；按 task builder 契约构造的 audio-target request 已经以 BOA 结束。
prompt 中若包含 BiCodec reference stream，`prompt_ids` 已经包含 route 规定的 serialized stream，
而 `audio_context` 保存同一份未序列化 codes，供 decode 复用 prompt-owned stream。route 是固定的
experiment/checkpoint contract，不属于 `Request`，请求只能提供 context 数据，不能选择另一条 route。
reference builder 生成的结尾严格为
`[BOA, serialized route.prompt streams, EOA, BOA]`；audio strategy 会重新序列化 `audio_context` 并要求
它与整个后缀逐 token 相等。context 的 shape、dtype、值域和 global 固定长度由
`BiCodecAudioTokenizer.encode_streams()` 在同一条路径校验，不接受“可解码但不是当前 prompt”的
近似匹配。
`FlattenedAudioTokenizer` 的 codec/codebook marker 和各 codebook range 是 codec serialization
grammar。model 侧 full-sequence generation 强制 marker 顺序，首个 codebook 生成至少一个
payload 并决定 frame count，其余 codebook 只能生成相同数量、属于各自 range 的 payload，完整
block 结束后才允许 EOA。marker 与 EOA 都计入 `max_new_tokens`，marker 也保留在
`response_ids` 中供 frame count 与 decode 使用。单码本是同一契约的批量化简化路径。
单码本生成在第一个 payload 位置屏蔽 EOA，并为结尾预留一个 token；如果模型在 payload
预算内没有选择 EOA，grammar 会强制补上 EOA，再解码已经完整生成的 frame。该恢复只适用于
每个 payload 本身就是完整 frame 的单码本序列；多码本或 structured sequence 缺少后续
codebook/stream 时仍显式失败，不把不完整结构交给 codec。
`generation.batch.requests_from_batch()` 会从 teacher-forcing batch 保留 task prefix，直接构造
request 的调用方负责保持相同 task 状态机。
当前 prompt 只由 layout global token IDs 表达；普通 audio-source 内容编码为 semantic audio token，
并可通过 `audio_input_positions` 让 model 在这些 payload 的 embedding 上运行可配置的
`AudioInputTower`。该 tower 只处理 source input，不改变 prompt 长度、generation grammar 或
output head。structured BiCodec route 另外通过 `audio_context` 携带 decode 所需的 reference
global/semantic codes，其中 global 仍使用 `SemanticAcousticCodes.acoustic` 字段；reference
context 不是 `audio_input_positions` 的替代品，当前不会再次经过 source tower。`Request` 不接受
可切换的 acoustic feature side channel。

service 在 padding 前校验每条 request 的通用外形，audio strategy 继续校验 route/context 契约：

- task 必须是 `Task`；prompt 必须是非空一维有符号整数 Tensor，且所有 ID 都属于 runtime
  layout。
- `audio_input_positions` 若存在，必须是一维有符号整数 Tensor，非空位置唯一、位于 prompt 内、
  任务 source modality 为 audio，并指向 runtime codec audio range。service 左 padding 时同步
  偏移这些位置；model 使用 cache 时只在首个完整 prompt 传递它们，后续新 token 不重复运行
  input tower。
- 非空 `audio_context` 必须是 `SemanticAcousticCodes`；text request、没有 route 或 route 没有
  prompt streams 时禁止携带多余 context。reference/prompt-owned route 必须提供 context，且
  structured prompt streams 必须使用 `BiCodecAudioTokenizer`。
## 执行流程

`generate_responses()` 按 target modality 分组。每组 prompt 左 padding，输出仍按原始请求顺序排列。
text 组直接调用 `generate_tokens()`；audio 组只调用统一的 `generate_audio_responses()`，后者按
runtime/model capability 一次选择以下策略：

- semantic-only：semantic token generation + `SemanticCodec.decode()`。
- acoustic side channel：semantic token/condition generation + `AcousticCodec.decode_features()`。
- frame full-sequence：flattened grammar + `FrameCodec.decode()`。
- structured full-sequence：BiCodec grammar + route stream resolve + `StructuredCodec.detokenize()`。

strategy factory 只检查 representation、structured layout 和 model capability，不按 codec 名称分派。
各策略拥有本路径的 generation/decode 配对；共享层只负责 frame count 校验、同 shape 行合批和
`AudioOutput` 构造。

```text
text target
    -> generate_tokens(stop=EOS)
    -> trim EOS
    -> Result(audio=None)

audio target + token-only model
    -> generate_tokens(stop=EOA), or constrained full-codec state machine
    -> FrameCodec full-code decode, route-aware BiCodec detokenize, or SAC SemanticCodecRuntime decode

audio target + runtime acoustic side channel + acoustic feature generator
    -> generate_audio_features()
    -> trim EOA and padded features by frame_counts
    -> codec.decode_features(semantic_codes, features)
```

audio 路径至少要生成一个 codec-decodable token。audio strategy 共享层按
`(generated_token_count, generated_frame_count)` 合并 shape 相同的行执行 codec decode，并要求
codec 保留 batch 轴。`audio_representation=full_codec_sequence` 只对 FrameCodec 通过 `FlattenedAudioTokenizer` 把完整
`[frames, codebooks]` 还原后直接调用 `codec.decode()`，不会因为 LongCat codec 暴露 acoustic
codebooks 而进入 Flow/RVQ acoustic feature generation。flow 与 RVQ 都返回相同的
`AcousticGeneration`；`model/acoustic=none` 即使搭配 LongCat 这类带 acoustic codebook 的
codec，也只走 token-only generation 分支；实际 waveform decoder 仍按 full-code sequence 或
semantic artifact 路径选择。
`FULL_CODEC_SEQUENCE` 对普通 `FrameCodec` 仍调用 `decode(full_codes)`；flattened codec 使用受约束
状态机生成有序 marker、各 codebook 等长 payload 和 EOA，避免 audio head 产生不可解析序列；
多码本每行可独立决定 frame count，返回 batch 以 EOA padding 对齐。BiCodec 使用另一套受约束的
structured state machine，根据固定 route.output 生成 marker、semantic token 和可选的固定数量
slot-major global codebook token，恢复 `SemanticAcousticCodes` 后按 route.decode 合并 prompt
与 output stream，再调用 `detokenize()`。global reuse route 只生成 semantic 并复用 prompt global；
global generate route 生成并使用 output global。legacy acoustic route 在 tokenizer/model 边界规范为
相同 global grammar。配置
`runtime.semantic_codec_artifact` 后，semantic strategy 只处理 structured backend 的 semantic tokens，
并把 waveform decode 交给 `SemanticCodecRuntime`；普通 frame codec 的 `decode()` 不再接收
semantic-only codes。semantic-artifact 与 structured full-sequence 是配置阶段选择的两条解码路径；
structured full-sequence 内部的 prompt/output/decode stream 仍由固定 `audio_route` 决定，不会把
fixed-length global units 伪装成 frame-aligned codes。

自回归 cache、sampling、动态 allowed IDs、逐行 stop 状态和 frame condition 收集属于 model。已有行
生成 stop token 后，后续步骤只对剩余 active rows 执行 backbone 与 sampling；cache 同步收缩，
最终 sequence 仍保持原 batch 顺序。请求分组、padding 与结果顺序属于 service；audio route 校验、
结果裁剪与 decode 属于 audio strategy；ID range、token frame span 与 codec 能力属于 runtime。
各层不重复推导同一约束。

普通 semantic/audio-feature 自回归路径同样在第一步屏蔽 EOA，避免固定样本以零 frame
结束。达到 `max_new_tokens` 但没有 EOA 时，只要已有 token 都是 codec-decodable，audio strategy 将其
作为截断结果解码；零 frame 或不完整 grammar 不会伪造静音 waveform。

## 训练桥接与文本评估

`generation.batch.requests_from_batch()` 仅供 teacher-forcing 日志使用：它直接读取
`ModelBatch.generation_prompt_lengths` 切出每行显式 prompt，并携带对应 `audio_contexts`，再去掉
batch padding；同时保留 `audio_input_positions` 的逐行 source payload 位置。它不从第一个非 `-100`
label 猜 prompt 边界；核心 service 不依赖 `ModelBatch`。

`decode_reference_codes()` 是 raw task sample 的统一重建边界：二维 frame-code tensor 通过
`frame_codec().decode()`，structured mapping 恢复为 `SemanticAcousticCodes` 后通过
`structured_codec().detokenize()`。callback 不按 codec 名称复制 decode 分支，也不把 fixed-length
acoustic units 当作 semantic frame 轴。

`evaluate_text()` 使用 `Task.T2TT` 构造 request，执行 greedy generation；reference NLL 则以
text modality-local logits 计算，并包含 EOS target。`SpeechToSpeechModule.generate()` 与
`evaluate_text()` 只提供 eval-mode/no-grad 的 Lightning 适配，不改变 generation 契约。

`generation.reporting` 提供 generation smoke/probe 复用的比较和摘要 helper；它服务于诊断脚本，
不进入包级 `generation` API，也不参与在线推理流程。
`generation._request` 单点维护 prompt layout、source-audio positions、task 与 audio context 的请求
约束；generation service 在分组和 padding 前调用它，相邻契约测试不再导入 service 的函数级私有实现。
`generation.evaluation` 提供 fixed-sample acoustic evaluation 复用的 waveform/STFT helper，
并负责 overfit 结束后的单样本自回归生成健康度、耗时与 RTF 汇总；训练侧 callback 与脚本只负责
cadence、设备编排、日志和落盘，不维护平行评估实现。

## 边界

- `Request` 表达真实推理，不能用缺 target 的 `ModelBatch` 代替。
- `response_ids` 始终保留 layout global ID 空间且不含 stop token；调用方需要文本时再通过
  runtime layout 与 tokenizer 解码。
- route-aware BiCodec decode 必须同时拥有 route 所声明的 prompt/output stream；缺失
  `audio_context` 或 output stream 时显式失败，不用 target codes 或另一条 route 静默补齐。
- service 与 audio strategy 只依赖 Protocol，不依赖具体 flow/RVQ model 或 LightningModule；
  codec-specific grammar 留在 model，codec-specific decode 配对留在 audio strategy。
- `generate_responses()` / `evaluate_text()` 使用 `no_grad`，但不切换 model 的 train/eval mode；
  直接调用包级入口时由调用方先进入 eval mode，`SpeechToSpeechModule` 才会代为切换并恢复状态。
- 一次请求的 KV cache 不跨调用持久化；cache 与 full-recompute 必须保持相同序列语义。
- generation 或 codec 没有产生每条请求所需的完整结果时显式报错，不返回部分列表。
