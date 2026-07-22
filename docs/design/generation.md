# generation

提供独立于 Lightning 和训练 batch 的真实推理入口。跨模块生成流程见
[总览 §6](../model-design.md)，model 侧原语见 [model](model.md)。

## 对外能力

包级 API 公开以下结构和入口：

- `Request(prompt_ids, task, acoustic_prompt)`：无 target、无 batch padding 的单条推理输入。
  `prompt_ids` 是一维 layout global token IDs；`acoustic_prompt` 保存 source codec-local
  acoustic codes 及其 prompt token 位置。
- `Result(response_ids, audio)`：按原请求顺序返回的单条结果。`response_ids` 是不含 EOS/EOA
  的 layout global token IDs；text task 的 `audio=None`。
- `AudioOutput(features, waveform, sample_rate)`：audio task 的 decode 结果。unified-token codec
  没有独立 acoustic representation，因此 `features=None`。
- `AcousticGeneration(sequence, features, frame_counts)`：acoustic model 与 service 之间的批量
  返回契约；`features` 是带右侧 padding 的 `[batch, frames, dim]`，`frame_counts` 给出每行
  有效 frame 数。
- `generate_responses()`：校验请求、分组 batch generation、逐行截断 stop token、waveform
  decode，并恢复原请求顺序。
- `decode_generated_audio()` / `decode_generated_codes()`：分别把 audio token 配合 acoustic
  feature/code 解码为 waveform。semantic-only decode 由 service 内部处理。
- `TextProbe` / `TextProbeResult` / `evaluate_text()`：greedy text generation 与 reference NLL
  评估。

`generation.protocol` 定义 service 所依赖的窄模型协议：

- `TokenGenerator`：公开 runtime、backbone 和 `generate_tokens()`。
- `AcousticFeatureGeneration`：只描述可选的 `generate_audio_features()` 能力。service 的
  `model` 参数仍由 `TokenGenerator` 表达基础契约；需要独立 acoustic codebook 时再检查这个窄
  runtime 协议，避免把 registered `nn.Module` backbone 等无关成员纳入能力识别。
- `AcousticFeatureGenerator`：组合上述两个协议，供训练 composition 静态表达完整模型契约。
- `TextEvaluationModel`：在 token generation 之外增加 hidden state 与 modality-local logits，
  用于 reference NLL。

## 输入输出

```python
class Request(TypedDict):
    prompt_ids: Tensor
    task: Task
    acoustic_prompt: AcousticPrompt | None

class AcousticPrompt(TypedDict):
    codes: Tensor
    token_positions: Tensor

class Result(TypedDict):
    response_ids: Tensor
    audio: AudioOutput | None
```

`AcousticPrompt.codes` 的形状是 `[frames, acoustic_codebooks]`；`token_positions` 的形状是
`[frames]`，位置基于该 request 的未 padding prompt。它只允许出现在 audio-source task，且只在
codec 确实有独立 acoustic codebooks 时使用。

`prompt_ids` 必须是调用方已经准备好的完整 generation prompt。service 不渲染 chat template、
不插入 instruction，也不追加或校验 response prefix；按 task builder 契约构造的 audio-target
request 已经以 BOA 结束。`generation.batch.requests_from_batch()` 会从 teacher-forcing batch
保留该 prefix，直接构造 request 的调用方负责保持相同状态机。

service 在 padding 前校验每条 request：

- task 必须是 `Task`；prompt 必须是非空一维有符号整数 Tensor，且所有 ID 都属于 runtime
  layout。
- acoustic codes 必须是非空二维有符号整数 Tensor，codebook 数量和每列范围必须匹配 codec。
- frame positions 必须是一维有符号整数 Tensor，与 codes 共用 frame 轴，并指向 prompt 内
  codec-decodable audio token。

`-1` 只由 service 用于 acoustic batch padding，不是合法的 request 输入。非法输入直接报错，
不会删除 frame、裁剪 code 或降级为无 acoustic prompt 的请求。

## 执行流程

`generate_responses()` 按 `(target_modality, has_acoustic_prompt)` 分组。每组 prompt 左 padding，
source frame position 随 padding 宽度平移；输出仍按原始请求顺序排列。

```text
text target
    -> generate_tokens(stop=EOS)
    -> trim EOS
    -> Result(audio=None)

audio target + token-only model
    -> generate_tokens(stop=EOA)
    -> expand token frame spans
    -> codec.decode(semantic_codes)

audio target + acoustic codebooks + acoustic feature generator
    -> generate_audio_features()
    -> trim EOA and padded features by frame_counts
    -> codec.decode_features(semantic_codes, features)
```

audio 路径至少要生成一个 codec-decodable token。service 按
`(generated_token_count, generated_frame_count)` 合并 shape 相同的行执行 codec decode，并要求
codec 保留 batch 轴。flow 与 RVQ 都返回相同的 `AcousticGeneration`；`model/acoustic=none`
即使搭配 LongCat 这类带 acoustic codebook 的 codec，也走 semantic-only decode。

自回归 cache、sampling、allowed IDs、逐行 stop 状态和 frame condition 收集属于 model。已有行
生成 stop token 后，后续步骤只对剩余 active rows 执行 backbone 与 sampling；cache 同步收缩，
最终 sequence 仍保持原 batch 顺序。请求
分组、输入校验、结果裁剪与 decode 属于 service；ID range、token frame span 与 codec 能力属于
runtime。三层不重复推导同一约束。

## 训练桥接与文本评估

`generation.batch.requests_from_batch()` 仅供 teacher-forcing 日志使用：它以每行第一个非
`-100` label 为 prompt 边界，去掉 batch padding，并把可选 source acoustic prompt 恢复为单条
request。核心 service 不依赖 `ModelBatch`。

`evaluate_text()` 使用 `Task.T2TT` 构造 request，执行 greedy generation；reference NLL 则以
text modality-local logits 计算，并包含 EOS target。`SpeechToSpeechModule.generate()` 与
`evaluate_text()` 只提供 eval-mode/no-grad 的 Lightning 适配，不改变 generation 契约。

`generation.reporting` 提供 generation smoke/probe 复用的比较和摘要 helper；它服务于诊断脚本，
不进入包级 `generation` API，也不参与在线推理流程。
`generation.evaluation` 提供 fixed-sample acoustic evaluation 复用的 waveform/STFT helper；训练侧
callback 只调用该诊断函数，不在脚本私有模块中维护平行实现。

## 边界

- `Request` 表达真实推理，不能用缺 target 的 `ModelBatch` 代替。
- `response_ids` 始终保留 layout global ID 空间且不含 stop token；调用方需要文本时再通过
  runtime layout 与 tokenizer 解码。
- service 只依赖 Protocol，不依赖具体 flow/RVQ model 或 LightningModule。
- `generate_responses()` / `evaluate_text()` 使用 `no_grad`，但不切换 model 的 train/eval mode；
  直接调用包级入口时由调用方先进入 eval mode，`SpeechToSpeechModule` 才会代为切换并恢复状态。
- 一次请求的 KV cache 不跨调用持久化；cache 与 full-recompute 必须保持相同序列语义。
- generation 或 codec 没有产生每条请求所需的完整结果时显式报错，不返回部分列表。
