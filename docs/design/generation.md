# generation

提供独立于 Lightning 和训练 batch 的真实推理入口。跨模块所有权见
[设计总览](../model-design.md)，model 侧原语见 [model](model.md)。

## 对外能力

公开入口是 OpenAI 风格的 messages API（非 HTTP 服务）：

```text
ChatRequest(messages, task, language?, trace?)
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
  - `audio`：`waveform` + `sample_rate`；入口按 source role 使用 input tokenizer backend encode 成
    codes。同名 input/output tokenizer 复用同一 backend；不同但可加载的 input tokenizer 使用自己的
    encoder，绝不回退到 output encoder。GLM-4 online backend 是 tokenizer-only；源码 checkout/fork
    不绑定 Git commit，但源码/API/模型契约、weights revision 和 `transformers==4.44.1` 会严格校验。
    当前进程不满足该依赖契约时，改传独立 producer 生成的 `codec_codes`。
  - `codec_codes`：已 materialize 的 codes，携带 `codec` 名；audio-source part 须等于
    `runtime.input_audio_tokenizer_name`（耦合时即 output name），以及
    `codes`（`AudioCodes` 或 frame `[frames, codebooks]`）。BiCodec input 的 `AudioCodes` 保留
    `semantic_codes + global_codes` 完整序列；是否使用独立 BPE 不改变 global stream ownership。
    `SemanticAcousticCodes` 只存在于
    anycodec tokenize/detokenize 边界，不是 chat 输入。不同 codec 的轴序/stream
    排列只委托现有 audio tokenizer、内部固定 route 和 `audio_sequence_layout`，不在 messages 层另造布局。
- `task` / `language` / `trace` 是旁路字段，不伪装成 OpenAI 官方 schema。`trace` 选择 task
  program 中的具体有序 response；未设置时使用 program 默认 response。执行 head 由 resolved
  `ResponseSpec.prediction` 派生，request 不接受独立 prediction override。
- `ChatCompletion`：`choices[].message` 含 `role=assistant`、可选 text `content`、可选
  `audio`（`AudioOutput`）；`content` 是最终 target text，中间 source/target text 阶段写入可选的
  结构化 `trace`。逐行 audio decode 失败时还保留 `decode_error` 的异常类型与消息。

私有张量契约（service / strategy 仍使用，不作为包级推荐入口）：

- `task.contract.Request(prompt_ids, task, audio_input_positions, trace?, target_language?)`：无 target、无
  batch padding 的单条推理输入，与训练 `ModelSample.request` 共用同一类型。`prompt_ids` 是一维
  layout global token IDs；BiCodec reference global stream 也直接序列化在这里，不存在并行的
  context codes 字段。可选的
  `audio_input_positions` 只标记 source audio payload 在 prompt 中的位置。可选 `trace` 唯一确定
  `ResponseSpec`；含 MT step 时还必须携带规范化后的 `target_language`，不能从 prompt 文本猜测。
  trace 未设置时使用 program 默认 response。请求不能直接选择执行 route。
- `generation.result.Result(response_ids, audio, decode_error?)`：按原请求顺序返回的单条结果。普通
  `TEXT_AR` 路径裁掉最终 EOS；结构化 task program 保留模型产生的 `<asr>...</asr>`、
  `<mt><lang_*>...</mt>` 与完整 `BOA/schema/codec/EOA`，供分阶段文本解码与
  audio span 抽取。runtime-only control ID 会在调用 lexical tokenizer 前剥离。纯 text prediction
  的 `audio=None`；AUDIO 与 token-only mixed 在成功恢复 raw codec codes 后填充 `AudioOutput`。
  codec grammar/parser 或 waveform decode 的可恢复逐行失败返回 `audio=None`，并在 `decode_error`
  暴露异常类型与消息。
- `generation.result.AudioOutput(features, codes, waveform, sample_rate)`：audio task 的恢复结果。
  codes-only 路径总会填充 decoder-independent 的 `codes`：semantic/structured 路径使用
  `AudioCodes(semantic_codes, global_codes, acoustic_codes)`，flattened frame codec 使用 raw
  `[frames, codebooks]` tensor。BiCodec 会先解析 prompt/response 的 global stream ownership，再返回
  完整 semantic/global codes。配置 `runtime.audio_output.detokenizer=null` 时不加载 waveform backend，
  `waveform=None`、`sample_rate=None`；配置 decoder 时这两个字段一起填充。unified-token codec 没有
  独立 acoustic feature side channel，因此 `features=None`。
- `model.output.AcousticGeneration(sequence, features, frame_counts)`：acoustic model 与 audio strategy 之间的批量
  返回契约。
- `generate_responses()`：校验私有 request、解析 task/trace 对应的 `ResponseSpec` 后分组并生成。
- `generation.bicodec`：BiCodec 私有 request 组装 helper；公开路径应走 `create` /
  `to_request`。
- `decode_generated_audio()` / `decode_generated_codes()`：分别把 audio token 配合 acoustic
  feature/code 解码为 waveform。
- `TextProbe` / `TextProbeResult` / `evaluate_text()`：greedy text generation 与 reference NLL
  评估。

`generation.contract` 定义 service 与 audio strategy 所依赖的窄模型协议：

- `TokenGenerator`：公开 runtime、backbone、`generate_tokens()`，以及供 program AR 使用的
  `generation_step()`（按 modality 或候选 `token_ids` 选择输出 head）。结构化 response 使用
  candidate-ID grammar：每一步仍读取模型 logits，grammar 只屏蔽非法转移。
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
    trace: NotRequired[str]
    target_language: NotRequired[str]

class Result(TypedDict):
    response_ids: Tensor
    audio: AudioOutput | None
    decode_error: NotRequired[dict[str, str]]
```

`prompt_ids` 必须是调用方已经准备好的完整 generation prompt。公开 `create()` 负责 messages →
HF chat template / codec materialize → 私有 `Request`；`generate_responses()` 本身不渲染 chat
template、不插入 instruction。按 task builder 契约构造的 request 只包含 instruction 与 source/context；
首个 `<asr>`、`<mt>` 或 BOA 必须由模型作为 response 的第一个 token 生成。
公开 Chat 的 audio-source lowering 会先在完整 history 的 rendered template 中保留唯一 source
placeholder，再把 input BOA/schema/payload/EOA 插入该位置；不能把 source audio 追加到 assistant generation
marker 之后，否则会偏离训练 prompt 布局。
prompt 中若包含 BiCodec reference stream，`prompt_ids` 已经包含 serialized global stream，
decode 直接从该 span 恢复 prompt-owned global stream。reference builder 生成的 span 严格为
`[BOA, input schema, serialized global stream, EOA]`；无 reference 时不追加 audio response token。response marker
是序列自己的 generation contract：以 `<begin_of_global>` 开局时 LLM 生成 global，随后生成
semantic；以 `<begin_of_semantic>` 开局时 LLM 只生成 semantic。decode 要求 prompt 与 response
恰好一方包含 global；两边都有或都没有都报错。shape、dtype、值域和 fixed-length global 数量由
`BiCodecAudioTokenizer` 统一校验。
`FlattenedAudioTokenizer` 的 codebook marker 和各 codebook range 是 codec-private grammar。
生成时该 grammar 屏蔽错误 marker/order/range 和过早 EOA，decode 前再做完整性校验。BOA、selector、
marker 与 EOA 都计入
`max_new_tokens`，marker 也保留在 `response_ids` 中供 frame count 与 decode 使用。达到
`max_new_tokens` 仍未发出 EOA 时，与其它音频路径一样返回截断 token；若已有 token 可被
codec parser 完整解释则继续 decode，不强制补 EOA。多码本、flattened 或 structured span
缺少后续 codebook/stream、范围非法或结构不完整时，按行 warning、`audio=None`，不把不完整
结构交给 codec，也不让整批结果失败。
`generation.service.requests_from_batch()` 会从 teacher-forcing batch 保留 task prefix，直接构造
request 的调用方负责保持相同 task prefix 契约。
当前 prompt 只由 layout global token IDs 表达；普通 audio-source 内容编码为 semantic audio token，
并可通过 `audio_input_positions` 让 model 在这些 payload 的 embedding 上运行可配置的
`AudioInputTower`。该 tower 只处理 source input，不改变 prompt 长度、generation token sequence 或
output head。structured BiCodec reference global 已经属于 prompt token sequence，不会通过
request/batch side channel 旁路，也不会再次经过 source tower。
`Request` 不接受可切换的 acoustic feature side channel。

service 在 padding 前校验每条 request 的通用外形，audio strategy 继续校验 codec sequence 契约：

- task 必须是 `Task`；prompt 必须是非空一维有符号整数 Tensor，且所有 ID 都属于 runtime
  layout。
- `audio_input_positions` 若存在，必须是一维有符号整数 Tensor，非空位置唯一、位于 prompt 内、
  任务 source modality 为 audio，并指向 runtime codec audio range。service 左 padding 时同步
  偏移这些位置；model 使用 cache 时只在首个完整 prompt 传递它们，后续新 token 不重复运行
  input tower。
- 旧式 `audio_context` request 字段会被拒绝；BiCodec global ownership 只能由 prompt/response
  marker 表达。
## 执行流程

`generate_responses()` 先按 request 的 `(task, trace, normalized target_language)` 解析
`ResponseSpec` 并分组，再读取 response 的
prediction 选择内部执行路径。训练 bridge（`requests_from_batch`）把 resolved trace 与对应的
normalized `target_language` 写入 Request；
`create` / `to_request` 不传 trace 时使用 program 默认 response。每组 prompt 左 padding，输出仍按
原始请求顺序排列。

- 未受 typed control 约束的单字段 `TEXT_AR`：调用
  `generate_tokens(generation_modality=TEXT, stop=EOS)`。
- `AUDIO`：调用统一的 `generate_audio_responses()`，按 runtime/model capability 选择策略。
- ASR/MT typed text、mixed prediction 或多 step response：调用 program state machine；它按
  `ResponseSpec.steps` 推进，
  属于 generation，不进入 `model.generate_tokens(generation_modality=...)`。

AUDIO 策略：

- codes-only：当 `runtime.audio_output.detokenizer=null` 时，仅用 output audio tokenizer 把模型 token
  还原为 raw frame codes / `AudioCodes`；acoustic side-channel 模型同时保留对齐后的 `features`。
- semantic-only：program grammar 生成完整 envelope，抽出 codec payload 后调用 `SemanticCodec.decode()`。
- acoustic side channel：先从模型 logits 生成 BOA/schema，再生成 semantic token/condition 与 EOA，
  最后调用 `AcousticCodec.decode_features()`。
- frame full-sequence：完整 envelope generation + flattened grammar/parser + `FrameCodec.decode()`。
- structured full-sequence：完整 envelope generation + BiCodec marker-driven stream resolve +
  `StructuredCodec.detokenize()`。

strategy factory 只检查 `audio_sequence_layout`、structured layout 和 model capability，不按 codec 名称分派。
各策略拥有本路径的 decode 配对；训练 bridge 与 token-only audio generation 共享
`BOA -> schema -> selected codec grammar -> EOA`，不会按 tokenizer class 猜 marker 顺序。
共享层只负责 frame count 校验、同 shape 行合批和 `AudioOutput` 构造；codes-only 不进入 waveform
batch decode，parser/decode 失败按行 warning 并返回 `audio=None`。

### Mixed AR

`generation.mixed` 用逐步 `generation_step(..., token_ids=union)` 驱动模型，再按行状态收窄
allowed IDs。sequential/blockwise response 按 step index 推进：ASR/MT 正文只允许 lexical text IDs
和当前 step 的 typed end token；EOS 与其它 control tokens 都被屏蔽。每个 step 的 begin/prefix 也
进入一次正常 logits 选择，只是合法候选通常是 singleton。audio 依次经历 BOA、schema selector、
codec-private grammar 和 EOA；状态机不直接写入任何 token。因此它同时覆盖：

- target CoT：`<mt> <lang_en> target text </mt> BOA schema target audio EOA`。
- full CoT：`<asr> source text </asr> <mt> <lang_en> target text </mt> BOA schema target audio EOA`。
- full text：`<asr> source text </asr> <mt> <lang_en> target text </mt>`，不会在第一个 typed end 提前停止。
- `INTERLEAVED`：`TEXT <-> BOA/schema/AUDIO/EOA`；TEXT 遇
  EOS 结束。

model 只提供单步 head 选择与 cache；mixed 不收集 acoustic frame condition，因此启用
`acoustic_side_channel` 且模型实现 `AcousticFeatureGeneration` 时显式失败。token-only 路径在
生成结束后从 `response_ids` 抽出 codec-decodable audio span，复用与 AUDIO 相同的 semantic /
frame / structured decode 配对写入 `Result.audio`；没有 codec audio token 时 `audio=None`。
structured BiCodec full-output path 只支持 `PARALLEL` 的单个完整 audio span；`INTERLEAVED`
可能产生多个独立 structured span，而当前单一 `Result.audio` 无法保留这些边界，因此在 request
校验时显式拒绝。
公开 Chat 入口进一步显式拒绝 BiCodec mixed trace，避免把 audio-source S2ST 请求误路由为 TTS
speaker-reference helper；direct BiCodec audio response 仍按 prompt/response marker ownership 执行。

```text
ordinary TEXT_AR prediction
    -> generate_tokens(stop=EOS)
    -> trim EOS
    -> Result(audio=None)

typed ASR/MT or multi-step program
    -> generation_step loop with per-step prefix/body/end grammar
    -> every prefix/begin token is selected from model logits
    -> retain typed boundaries in response_ids; strip controls before lexical decode

audio prediction + token-only model
    -> program loop generates BOA -> schema -> codec grammar -> EOA
    -> FrameCodec full-code decode, marker-driven BiCodec detokenize, or generator plugin GeneratorRuntime decode

audio prediction + runtime acoustic side channel + acoustic feature generator
    -> model predicts BOA and schema under singleton masks
    -> generate_audio_features() predicts codec payload and EOA
    -> preserve the full response; pass only codec payload/features to decoder
    -> codec.decode_features(semantic_codes, features)

mixed prediction (PARALLEL | INTERLEAVED)
    -> generation_step loop with per-row PREFIX/TEXT/AUDIO_SCHEMA/AUDIO state
    -> extract codec-decodable audio tokens from response_ids
    -> token-only decode into Result.audio (no acoustic feature side channel)
```

token-only audio 路径生成后先尝试抽取 codec-decodable span；零 frame、非法 span 或恢复失败
按行 warning 并返回 `audio=None`。无 detokenizer 时 mixed 路径直接返回 raw codes；有 detokenizer 时
audio strategy 共享层按
`(generated_token_count, generated_frame_count)` 合并 shape 相同的有效行执行 codec decode，并要求
codec 保留 batch 轴。当 `audio_sequence_layout=flattened` 时，FrameCodec 通过 `FlattenedAudioTokenizer` 把完整
`[frames, codebooks]` 还原后直接调用 `codec.decode()`，不会因为 LongCat codec 暴露 acoustic
codebooks 而进入 Flow/RVQ acoustic feature generation。flow 与 RVQ 都返回相同的
`AcousticGeneration`；`model/acoustic=none` 即使搭配 LongCat 这类带 acoustic codebook 的
codec，也只走 token-only generation 分支；实际 waveform decoder 仍按 full-code sequence 或
semantic artifact 路径选择。
`flattened` layout 对普通 `FrameCodec` 仍调用 `decode(full_codes)`；flattened parser 校验有序
marker、各 codebook 等长 payload 和 EOA，非法序列只影响对应行的 audio decode。BiCodec parser
从 prompt/response 的 marker 解析 semantic token 和固定数量 slot-major global codebook token：
prompt 有 global 时 response 必须以 semantic marker 开局；prompt 没有 global 时 response 必须以
global marker 开局并同时提供 global 与 semantic。parser 不读取 task/request route，只要求两侧恰好
一个 global owner。S2S 内部恢复 `AudioCodes`，只在调用 anycodec `detokenize()` 的边界把
`global_codes` 映射回 `SemanticGlobalCodes.global_codes`。配置
`runtime.audio_output.acoustic_generator_artifact` 后，semantic strategy 只处理 structured backend 的 semantic tokens，
并把 waveform decode 交给 `GeneratorRuntime`；普通 frame codec 的 `decode()` 不再接收
semantic-only codes。semantic-artifact 与 structured full-sequence 是配置阶段选择的两条解码路径；
structured full-sequence 内部的 prompt/output/decode ownership 由 self-describing marker 决定，
generation 请求不能覆盖；
fixed-length 与 frame-aligned 的区别由 `AcousticLayout` 提供，不由第三种 stream 名伪装。

自回归 cache、sampling、逐行 stop 状态和 frame condition 收集属于 model；mixed AR 的动态
allowed IDs 属于 `generation.mixed`。循环的 backbone forward 与 cache 始终保留原 batch 轴；结束行
通过 device mask 屏蔽，避免逐步压缩 cache 和 host 同步。随机 sampling 只接收仍 active 的行；
`DONE` 行不再产生 token。singleton prefix mask 仍读取并选择模型 logits，不能视作状态机插入。
最终 sequence 仍保持原 batch 顺序。请求分组、padding 与结果顺序属于 service；audio route 校验、
结果裁剪与 decode 属于 audio strategy；ID range、token frame span 与 codec 能力属于 runtime。各层不重复
推导同一约束。

普通 semantic/audio-feature 自回归路径同样在第一步屏蔽 EOA，避免固定样本以零 frame
结束。达到 `max_new_tokens` 但没有 EOA 时，只要已有 token 都是 codec-decodable，audio strategy 将其
作为截断结果解码；零 frame、codec parser 不完整或 decode 失败不会伪造静音 waveform。

## 训练桥接与文本评估

`generation.service.requests_from_batch()` 仅供 teacher-forcing 日志使用：它直接读取
`ModelBatch.generation_prompt_lengths` 切出每行显式 prompt，并携带对应 response trace 与
`target_language`，再去掉 batch
padding；BiCodec reference 已经在 prompt 内，不再复制为 generation side channel。bridge 同时保留
`audio_input_positions` 的逐行 source payload
位置。它不从第一个非 `-100` label 猜 prompt 边界；核心 service 不依赖 `ModelBatch`。

`decode_reference_codes()` 是 raw task sample 的统一重建边界：二维 frame-code tensor 通过
`frame_codec().decode()`，structured mapping 先规范化为 `AudioCodes`，再在 codec 边界转换后通过
`structured_codec().detokenize()`。callback 不按 codec 名称复制 decode 分支，也不把 fixed-length
acoustic units 当作 semantic frame 轴。

`evaluate_text()` 是通用 instruction completion probe，因此使用普通 `Task.TEXT_AR` 构造 request，
执行 greedy generation；reference NLL 则以
text modality-local logits 计算，并包含 EOS target。`SpeechToSpeechModule.generate()` 与
`evaluate_text()` 只提供 eval-mode/no-grad 的 Lightning 适配，不改变 generation 契约。

`generation.evaluation` 提供 generation smoke/probe 复用的比较和摘要 helper；它服务于诊断脚本，
不进入包级 `generation` API，也不参与在线推理流程。
`generation.rollout` 的 token logprob 导出当前只接受单 step TEXT response；多阶段 TEXT trace 会在
生成前显式拒绝。单 step ASR/MT rollout 使用对应 typed end token 作为 stop，普通 `TEXT_AR` 才使用 EOS。
`generation.request` 单点维护 prompt layout、source-audio positions 与 task 的请求
约束；generation service 在分组和 padding 前调用它，相邻契约测试不再导入 service 的函数级私有实现。
`generation.evaluation` 同时提供 fixed-sample acoustic evaluation 复用的 waveform/STFT helper，
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
- OpenAI-style Chat adapter 当前要求 task template 含 `{source}`；空 context 的 TEXT_AR/AUDIO_AR 等
  program 可通过私有 `Request` 和训练 bridge 生成，但若要暴露为 Chat continuation，必须先明确 seed
  content 的语义，不能静默丢弃 user message。
- `response_ids` 始终保留 layout global ID 空间。普通 `TEXT_AR` 裁掉最终 EOS；所有结构化 AUDIO
  response 保留 BOA、schema selector、codec-private marker/payload 与 EOA，以便复核 grammar、
  解析 ownership 并抽取 decoder payload。
- BiCodec decode 必须从 prompt/output marker 得到且只得到一个 global owner，并拥有 output
  semantic stream；缺失或重复时显式失败，不用 target codes 或 out-of-band context 静默补齐。
- service 与 audio strategy 只依赖 Protocol，不依赖具体 flow/RVQ model 或 LightningModule；
  audio generation 从 `AudioTokenSpec` 读取 compact marker/range 候选和 prompt continuation variant，
  codec-specific parse/decode 配对仍留在 audio strategy。
- `generate_responses()` / `evaluate_text()` 使用 `no_grad`，但不切换 model 的 train/eval mode；
  直接调用包级入口时由调用方先进入 eval mode，`SpeechToSpeechModule` 才会代为切换并恢复状态。
- 一次请求的 KV cache 不跨调用持久化；cache 与 full-recompute 必须保持相同序列语义。
- generation 或 codec 没有产生每条请求所需的完整结果时显式报错，不返回部分列表。
