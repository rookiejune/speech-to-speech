# generation

提供独立于 Lightning 和训练 batch 的真实推理入口。跨模块生成流程见
[总览 §6](../model-design.md)，model 侧原语见 [model](model.md)。

## 对外能力

公开入口是 OpenAI 风格的 messages API（非 HTTP 服务）：

```text
ChatRequest(messages, task, language?)
    -> HF apply_chat_template (Qwen3)
    -> audio | codec_codes materialize
    -> private Request(prompt_ids, ...)
    -> generate_responses
    -> ChatCompletion(choices[].message)
```

- `create(ChatRequest, model, ...)`：对外 chat 入口。`messages` 使用
  `{role, content}`；`content` 为字符串或 part 列表。
- content parts：
  - `text`：标准文本
  - `audio`：`waveform` + `sample_rate`；入口侧用 runtime codec encode 成 codes
  - `codec_codes`：已 materialize 的 codes，携带 `codec` 名（须等于 `runtime.codec_name`）与
    `codes`（`SemanticAcousticCodes` 或 frame `[frames, codebooks]`）。不同 codec 的轴序/stream
    排列只委托现有 audio tokenizer、内部固定 route 和 `audio_sequence_layout`，不在 messages 层另造布局。
- `task` / `language` 是旁路字段，不伪装成 OpenAI 官方 schema。
- `ChatCompletion`：`choices[].message` 含 `role=assistant`、可选 text `content`、可选
  `audio`（`AudioOutput`）。

私有张量契约（service / strategy 仍使用，不作为包级推荐入口）：

- `Request(prompt_ids, task, audio_input_positions, audio_context, prediction?)`：无 target、无
  batch padding 的单条推理输入，与训练 `ModelSample.request` 共用同一类型。`prompt_ids` 是一维
  layout global token IDs；内部过渡 helper `fixed_audio_route(task)` 只用于判断 decode 是否需要
  prompt stream，此时 `audio_context`
  提供同一 reference 的 structured semantic/acoustic codes。可选的 `audio_input_positions` 只标记
  source audio payload 在 prompt 中的位置。可选的 `prediction` 覆写 task 默认 prediction；未设置
  时与原先一样使用 `task.prediction_modality`。请求不能选择 route，公开配置只选择
  `audio_sequence_layout`。
- `Result(response_ids, audio)`：按原请求顺序返回的单条结果。TEXT / AUDIO 路径的
  `response_ids` 是裁掉 stop token 后的 layout global token IDs；mixed 路径当前保留状态机产生的
  EOS/BOA/EOA，供 audio span 抽取。纯 text prediction 的 `audio=None`；AUDIO 与 token-only mixed
  在成功 decode 后填充 `AudioOutput`。
- `AudioOutput(features, codes, waveform, sample_rate)`：audio task 的 decode 结果。`codes` 保存
  route resolve 后的 structured semantic/acoustic codes；unified-token codec 没有独立 acoustic
  representation，因此 `features=None`。
- `AcousticGeneration(sequence, features, frame_counts)`：acoustic model 与 audio strategy 之间的批量
  返回契约。
- `generate_responses()`：校验私有 request、按有效 `prediction`（request 覆写或 task 默认）分组并生成。
- `generation.bicodec`：BiCodec route 的私有 request 组装 helper；公开路径应走 `create` /
  `to_request`。
- `decode_generated_audio()` / `decode_generated_codes()`：分别把 audio token 配合 acoustic
  feature/code 解码为 waveform。
- `TextProbe` / `TextProbeResult` / `evaluate_text()`：greedy text generation 与 reference NLL
  评估。

`generation.protocol` 定义 service 与 audio strategy 所依赖的窄模型协议：

- `TokenGenerator`：公开 runtime、backbone、`generate_tokens()`，以及供 mixed AR 使用的
  `generation_step()`（按 modality 或候选 `token_ids` 选择输出 head）。ordinary AUDIO
  token generation 统一走 `generate_tokens(generation_modality=AUDIO, stop=EOA)`；codec-specific
  约束只在 decode/parser 阶段使用，不扩展 model protocol。
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
    prediction: NotRequired[PredictionModality | None]

class Result(TypedDict):
    response_ids: Tensor
    audio: AudioOutput | None
```

`prompt_ids` 必须是调用方已经准备好的完整 generation prompt。公开 `create()` 负责 messages →
HF chat template / codec materialize → 私有 `Request`；`generate_responses()` 本身不渲染 chat
template、不插入 instruction。按 task builder 契约构造的 audio-target request 已经以 BOA 结束。
prompt 中若包含 BiCodec reference stream，`prompt_ids` 已经包含 route 规定的 serialized stream，
而 `audio_context` 保存同一份未序列化 codes，供 decode 复用 prompt-owned stream。route 是固定的
experiment/checkpoint contract，不属于 `Request`，请求只能提供 context 数据，不能选择另一条 route。
reference builder 生成的结尾严格为
`[BOA, serialized route.prompt streams, EOA, BOA]`；audio strategy 会重新序列化 `audio_context` 并要求
它与整个后缀逐 token 相等。context 的 shape、dtype、值域和 global 固定长度由
`BiCodecAudioTokenizer.encode_streams()` 在同一条路径校验，不接受“可解码但不是当前 prompt”的
近似匹配。
`FlattenedAudioTokenizer` 的 codebook marker 和各 codebook range 是 codec serialization
parser contract。model 侧 ordinary AUDIO generation 不强制 marker 顺序、codebook range 或
block length；这些规则只在 decode 前解析 `response_ids` 时校验。marker 与 EOA 都计入
`max_new_tokens`，marker 也保留在 `response_ids` 中供 frame count 与 decode 使用。达到
`max_new_tokens` 仍未发出 EOA 时，与其它音频路径一样返回截断 token；若已有 token 可被
codec parser 完整解释则继续 decode，不强制补 EOA。多码本、flattened 或 structured span
缺少后续 codebook/stream、范围非法或结构不完整时，按行 warning、`audio=None`，不把不完整
结构交给 codec，也不让整批结果失败。
`generation.batch.requests_from_batch()` 会从 teacher-forcing batch 保留 task prefix，直接构造
request 的调用方负责保持相同 task prefix 契约。
当前 prompt 只由 layout global token IDs 表达；普通 audio-source 内容编码为 semantic audio token，
并可通过 `audio_input_positions` 让 model 在这些 payload 的 embedding 上运行可配置的
`AudioInputTower`。该 tower 只处理 source input，不改变 prompt 长度、generation token sequence 或
output head。structured BiCodec route 另外通过 `audio_context` 携带 decode 所需的 reference
acoustic/semantic codes（acoustic 字段在 `FIXED_LENGTH` 下为 speaker slots）；reference
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

`generate_responses()` 按有效 `prediction` 分组：优先 `Request.prediction`，否则
`task.prediction_modality`。训练 bridge（`requests_from_batch`）会把
`ModelBatch.predictions` 写入 Request，因此 loader override 可传到真实推理路径；
`create` / `to_request` 不传 `prediction` 时行为与原先一致。每组 prompt 左 padding，输出仍按
原始请求顺序排列。

- `TEXT`：调用 `generate_tokens(generation_modality=TEXT, stop=EOS)`。
- `AUDIO`：调用统一的 `generate_audio_responses()`，按 runtime/model capability 选择策略。
- `PARALLEL` / `INTERLEAVED`：调用 `generate_mixed_responses()`；mixed state machine 属于 generation，
  不进入 `model.generate_tokens(generation_modality=...)`。

AUDIO 策略：

- semantic-only：`generate_tokens(generation_modality=AUDIO, stop=EOA)` + `SemanticCodec.decode()`。
- acoustic side channel：semantic token/condition generation + `AcousticCodec.decode_features()`。
- frame full-sequence：同一套 AUDIO token generation + flattened parser + `FrameCodec.decode()`。
- structured full-sequence：同一套 AUDIO token generation + BiCodec route stream resolve +
  `StructuredCodec.detokenize()`。

strategy factory 只检查 `audio_sequence_layout`、structured layout 和 model capability，不按 codec 名称分派。
各策略拥有本路径的 decode 配对；训练 bridge 与普通 token-only audio generation 在进入
codec-specific parser 前保持统一，只约束 AUDIO vocab 与 EOA。真正产品推理如需更强保证，可以在
audio strategy 层显式选择 codec-specific 策略，但不能作为训练或普通 generation 的隐式结构约束。
共享层只负责 frame count 校验、同 shape 行合批和 `AudioOutput` 构造；parser/decode 失败按行
warning 并返回 `audio=None`。

### Mixed AR

`generation.mixed` 用逐步 `generation_step(..., token_ids=union)` 驱动模型，再按行状态收窄
allowed IDs。状态机：

- `PARALLEL`：`TEXT -> EOS -> force BOA -> AUDIO -> EOA`。
- `INTERLEAVED`：`TEXT <-> AUDIO`，TEXT 可发 BOA 进入 AUDIO，AUDIO 遇 EOA 回到 TEXT；TEXT 遇
  EOS 结束。

model 只提供单步 head 选择与 cache；mixed 不收集 acoustic frame condition，因此启用
`acoustic_side_channel` 且模型实现 `AcousticFeatureGeneration` 时显式失败。token-only 路径在
生成结束后从 `response_ids` 抽出 codec-decodable audio span，复用与 AUDIO 相同的 semantic /
frame / structured decode 配对写入 `Result.audio`；没有 codec audio token 时 `audio=None`。

```text
text prediction
    -> generate_tokens(stop=EOS)
    -> trim EOS
    -> Result(audio=None)

audio prediction + token-only model
    -> generate_tokens(stop=EOA)
    -> FrameCodec full-code decode, route-aware BiCodec detokenize, or SAC SemanticCodecRuntime decode

audio prediction + runtime acoustic side channel + acoustic feature generator
    -> generate_audio_features()
    -> trim EOA and padded features by frame_counts
    -> codec.decode_features(semantic_codes, features)

mixed prediction (PARALLEL | INTERLEAVED)
    -> generation_step loop with per-row TEXT/AUDIO/FORCE_BOA state
    -> extract codec-decodable audio tokens from response_ids
    -> token-only decode into Result.audio (no acoustic feature side channel)
```

token-only audio 路径生成后先尝试抽取 codec-decodable span；零 frame、非法 span 或 decode 失败
按行 warning 并返回 `audio=None`。audio strategy 共享层按
`(generated_token_count, generated_frame_count)` 合并 shape 相同的有效行执行 codec decode，并要求
codec 保留 batch 轴。当 `audio_sequence_layout=flattened` 时，FrameCodec 通过 `FlattenedAudioTokenizer` 把完整
`[frames, codebooks]` 还原后直接调用 `codec.decode()`，不会因为 LongCat codec 暴露 acoustic
codebooks 而进入 Flow/RVQ acoustic feature generation。flow 与 RVQ 都返回相同的
`AcousticGeneration`；`model/acoustic=none` 即使搭配 LongCat 这类带 acoustic codebook 的
codec，也只走 token-only generation 分支；实际 waveform decoder 仍按 full-code sequence 或
semantic artifact 路径选择。
`flattened` layout 对普通 `FrameCodec` 仍调用 `decode(full_codes)`；flattened parser 校验有序
marker、各 codebook 等长 payload 和 EOA，非法序列只影响对应行的 audio decode。BiCodec parser
根据固定 route.output 解析 marker、semantic token 和可选的固定数量 slot-major acoustic codebook
token，恢复 `SemanticAcousticCodes` 后按 route.decode 合并 prompt 与 output stream，再调用
`detokenize()`。`reuse_prompt_acoustic` 只生成 semantic 并复用 prompt
acoustic；`generate_acoustic` 生成并使用 output acoustic。配置
`runtime.semantic_codec_artifact` 后，semantic strategy 只处理 structured backend 的 semantic tokens，
并把 waveform decode 交给 `SemanticCodecRuntime`；普通 frame codec 的 `decode()` 不再接收
semantic-only codes。semantic-artifact 与 structured full-sequence 是配置阶段选择的两条解码路径；
structured full-sequence 内部的 prompt/output/decode ownership 由内部固定 route 决定，generation
请求不能覆盖；
fixed-length 与 frame-aligned 的区别由 `AcousticLayout` 提供，不由第三种 stream 名伪装。

自回归 cache、sampling、逐行 stop 状态和 frame condition 收集属于 model；mixed AR 的动态
allowed IDs 属于 `generation.mixed`。已有行
生成 stop token 后，后续步骤只对剩余 active rows 执行 backbone 与 sampling；cache 同步收缩，
最终 sequence 仍保持原 batch 顺序。请求分组、padding 与结果顺序属于 service；audio route 校验、
结果裁剪与 decode 属于 audio strategy；ID range、token frame span 与 codec 能力属于 runtime。
各层不重复推导同一约束。

普通 semantic/audio-feature 自回归路径同样在第一步屏蔽 EOA，避免固定样本以零 frame
结束。达到 `max_new_tokens` 但没有 EOA 时，只要已有 token 都是 codec-decodable，audio strategy 将其
作为截断结果解码；零 frame、codec parser 不完整或 decode 失败不会伪造静音 waveform。

## 训练桥接与文本评估

`generation.batch.requests_from_batch()` 仅供 teacher-forcing 日志使用：它直接读取
`ModelBatch.generation_prompt_lengths` 切出每行显式 prompt，并携带对应 `audio_contexts` 与
`predictions`，再去掉 batch padding；同时保留 `audio_input_positions` 的逐行 source payload
位置。它不从第一个非 `-100` label 猜 prompt 边界；核心 service 不依赖 `ModelBatch`。

`decode_reference_codes()` 是 raw task sample 的统一重建边界：二维 frame-code tensor 通过
`frame_codec().decode()`，structured mapping 恢复为 `SemanticAcousticCodes` 后通过
`structured_codec().detokenize()`。callback 不按 codec 名称复制 decode 分支，也不把 fixed-length
acoustic units 当作 semantic frame 轴。

`evaluate_text()` 使用 `Task.T2TT` 构造 request，执行 greedy generation；reference NLL 则以
text modality-local logits 计算，并包含 EOS target。`SpeechToSpeechModule.generate()` 与
`evaluate_text()` 只提供 eval-mode/no-grad 的 Lightning 适配，不改变 generation 契约。

`generation.eval.reporting` 提供 generation smoke/probe 复用的比较和摘要 helper；它服务于诊断脚本，
不进入包级 `generation` API，也不参与在线推理流程。
`generation._request` 单点维护 prompt layout、source-audio positions、task 与 audio context 的请求
约束；generation service 在分组和 padding 前调用它，相邻契约测试不再导入 service 的函数级私有实现。
`generation.eval.acoustic` 提供 fixed-sample acoustic evaluation 复用的 waveform/STFT helper，
并负责 overfit 结束后的单样本自回归生成健康度、耗时与 RTF 汇总；训练侧 callback 与脚本只负责
cadence、设备编排、日志和落盘，不维护平行评估实现。

## 诊断入口

`scripts/generation_smoke.py` 通过 Hydra root `configs/generation_smoke.yaml` 组合
`RuntimeConfig`、`DatasetConfig + load_dataset()` 和 `generate_responses()`，验证
cache/full-recompute 及 variable-batch 语义，不直接绑定 workspace 的具体 dataset provider。CPU 模式
不调用 CUDA seed、同步或显存 API；CUDA 模式以模型实际所在 device 同步和统计 peak memory。
cached/full、batched/serial 任一 waveform 非 finite，或 greedy token 不一致，都会在写出
`metrics.json` 后让入口失败。sample index、batch sizes、dataset filter 和 generation token budget
通过普通 Hydra override 配置，并在加载 runtime 前完成边界校验。

## 边界

- `Request` 表达真实推理，不能用缺 target 的 `ModelBatch` 代替。
- `response_ids` 始终保留 layout global ID 空间且不含 stop token；调用方需要文本时再通过
  runtime layout 与 tokenizer 解码。
- route-aware BiCodec decode 必须同时拥有 route 所声明的 prompt/output stream；缺失
  `audio_context` 或 output stream 时显式失败，不用 target codes 或另一条 route 静默补齐。
- service 与 audio strategy 只依赖 Protocol，不依赖具体 flow/RVQ model 或 LightningModule；
  ordinary token generation 只依赖 AUDIO vocab 与 EOA，codec-specific parse/decode 配对留在 audio strategy。
- `generate_responses()` / `evaluate_text()` 使用 `no_grad`，但不切换 model 的 train/eval mode；
  直接调用包级入口时由调用方先进入 eval mode，`SpeechToSpeechModule` 才会代为切换并恢复状态。
- 一次请求的 KV cache 不跨调用持久化；cache 与 full-recompute 必须保持相同序列语义。
- generation 或 codec 没有产生每条请求所需的完整结果时显式报错，不返回部分列表。
