# runtime

加载并聚合一套相互兼容的 tokenizer、codec、backbone、layout 与 flow runtime。

## 对外能力

`Runtime` 是 frozen dataclass，重资源通过 `cached_property` 惰性加载：

- `input_audio_tokenizer_name` / `output_audio_tokenizer_name` / `output_audio_detokenizer_name`：
  source encoder、target encoder 与 waveform decoder 的 backend preset identity；
  `input_audio_view` / `output_audio_view` 及 codebook metadata 由对应 preset 的静态 spec 推导。
- `text_tokenizer` / `input_audio_tokenizer` / `output_audio_tokenizer`：Qwen-compatible text
  tokenizer 与两侧 audio tokenizer；耦合配置中两侧 audio tokenizer 共享同一实例。
- audio backend：按 capability 暴露 `tokenize()` 和/或 `detokenize()`；同一 resolved preset
  同时被 input tokenizer、output tokenizer、output detokenizer 引用时只构造一个实例，不为缺失能力
  提供占位属性。
- `semantic_codec`：semantic token generation 的 waveform decoder。配置
  `runtime.audio_output.acoustic_generator_artifact` 时惰性加载 `semantic-acoustic-generator`
  artifact，并复用同一个
  structured backend；FrameCodec 不把 semantic-only codes 传给自身 `decode()`。
- `acoustic_generator_artifact_sha256`：acoustic generator artifact 的内容身份。单文件按字节计算
  SHA-256；目录按排序后的相对文件路径和内容计算，不包含机器相关的 artifact 根路径或文件元数据。
  该值在单个 Runtime 内缓存；artifact 内容更新后必须新建 Runtime，避免一次训练期间身份漂移。
- `audio_sequence_layout`：公开音频序列格式，当前为 `flattened` 或 `semantic`。`flattened`
  表示完整 codec codes 被序列化为 acoustic-first / semantic-last 的 token 序列；`semantic`
  表示逻辑输入输出仍是 full codes，但 token sequence 只处理 semantic，acoustic 由 side module 或
  semantic-acoustic codec 补齐。Runtime 构造时按该字段派生内部 route，并沿同一运行实例传给
  DataModule、model 与 generation。
- `backbone`：Qwen-compatible HF causal LM。
- `layout`：text/audio global token blocks；text block 由 lexical tokenizer rows 加固定 control rows 组成。
- `input_audio_token_spec` / `output_audio_token_spec`：schema identity、selector 文本、codec tokenizer
  contract 与可执行的 codec-private marker/range/order grammar；当前每侧 registry 只有一个默认 schema，
  但序列仍显式携带 selector。
- `pad/bos/eos_token_id`、`control_token_ids` / `control_token_id()`，以及 audio
  `boa/eoa/mask/schema_token_id`。
- `codec_audio_range`、`audio_generation_allowed_ids` 与 modality generation IDs。
- `flow_matching`：训练 sample 与 generation ODE sampler 的共享 runtime。

`runtime.Config.audio_input` 与 `runtime.Config.audio_output` 是两侧 audio 配置源。
`tokenizer` 选择 waveform→codes capability（例如 `glm4` / `bicodec`），output 的
`detokenizer` 另选 codes→waveform capability；`bpe` 只是 model-facing token sequence 使用的可选
CodecBPE artifact 路径。`AudioView`、vocabulary、frame rate 与 codebook layout 都由 preset spec
唯一推导，不再是 canonical 配置字段；
legacy `view` 只作校验别名，与推导值不同时直接报错。Hydra runtime preset 直接映射
完整 `runtime.Config`，同时选择相互兼容的 output backend、FrameCodec audio sequence
layout、audio tokenization 与 backbone snapshot。`flattened` 直接把完整
FrameCodec codebooks 编入 token 序列，因此 frame-code codec 不能同时配置 BPE audio tokenizer；
BiCodec 是例外，它只把 semantic 子流交给 BPE，再与 global slots 组合。structured 顺序固定为
global-first / semantic-last。ODE method、NFE 与 step 数直接使用 `flow_method`、`flow_nfe` 与
`flow_num_steps`，不再通过独立 sampler 组转换；`Config` 在构造时校验 method、正 NFE 和至少
2 个 steps，因此 token/RVQ composition 也不会静默携带无效 runtime。model composition 由
`model/acoustic` 选择。

### 解耦输入与输出 audio tokenizer

默认 `runtime.audio_input=null`，输入与输出共享 `audio` token block、tokenizer、
BOA/EOA 和 embedding。`runtime.audio_output.tokenizer` 必填，`runtime.audio_input.tokenizer`
一旦配置就显式描述 source audio；`runtime.audio_output.detokenizer=null` 明确表示只产出 codes，
不提供 waveform decode。共享分三层判定：

- input/output tokenizer 与 output detokenizer 解析到同一 preset + revision/artifact 时，复用同一
  backend 实例和同一 prepared-code view；detokenizer 因而参与 resource tying。
- 只有 code schema 与 `bpe` identity 都相同时才共享 model-facing audio tokenizer、token block
  和 embedding；同 backend 但 `bpe` 不同时仍是独立 token space/embedding，但不重复加载
  backend 或 prepared store。
- detokenizer 不直接决定 LLM embedding/head tying；它只消费 output tokenizer 恢复出的 raw
  codes，并要求两侧 code spec 兼容。

`runtime.audio_output` 始终描述 target audio，`audio_sequence_layout` 仍保持在 root。
GLM-4 source 与 BiCodec target 的 canonical 配置为：

```yaml
runtime:
  audio_input:
    tokenizer: glm4
    bpe: null
  audio_output:
    tokenizer: bicodec
    detokenizer: bicodec
    bpe: null             # 可选 semantic CodecBPE artifact 路径
    acoustic_generator_artifact: null
audio_sequence_layout: flattened
```

旧的 `runtime.codec` / `runtime.input_audio` / `runtime.audio_tokenizer` /
`runtime.acoustic_generator_artifact`，以及两侧的 `codec` / `view` / `vocab_size` /
`frame_rate` 仅作为边界迁移或静态 spec 一致性校验字段。
external mapping 入口会迁移到上述
canonical mapping 并发出 deprecation warning。新旧字段同时出现且值冲突时直接报错，不猜测优先级。

解耦 layout 固定为 `text | audio_input | audio`：`audio_input` 拥有独立 BOA/EOA/schema selector 和 embedding，
只用于 source lookup；`audio` 仍是唯一输出 head、loss 和 generation candidate space，并继续拥有
输出 BOA/EOA/MASK/schema selector。dense logits 的 `audio_input` slice 永远为 `-inf`；selected logits 开启
validation 时显式拒绝 input-only ID，generation allowed IDs 从不包含该 block。
`input_modalities={AUDIO}` 只是粗粒度性能提示，实际 embedding 始终按 global
ID block 路由，因此 KV-cache continuation 中新生成的 BiCodec ID 不会误进 GLM-4 embedding。

`audio_input.bpe=null` 时使用 backend native tokenization；包括 GLM-4 在内的 vocabulary、frame
rate 与 codebook metadata 都来自 preset 静态 spec，配置不重复填写。GLM-4 online loader 只暴露
waveform→frame codes，不伪造 decoder；它接受任意 GLM-4-Voice checkout/fork，不绑定源码 Git
commit，但严格校验源码/API/模型契约、固定 tokenizer weights revision 和
`transformers==4.44.1`，source checkout 通过
`GLM4_VOICE_SOURCE_ROOT` 指定。该版本与部分 output codec 的主环境依赖不兼容时，生产
路径应让独立 AnyDataset provider/producer 持续发布 `AudioView.GLM4` store，训练按现有 streaming
边界读取；只有同进程依赖满足严格检查时才启用 waveform fallback。训练 fallback 对具有 tokenize
capability 的 input backend 调用 input tokenizer，绝不用 output backend 伪造 input codes；在线 chat
遵循相同规则。BiCodec input 始终保留 global +
semantic 完整序列；独立 BPE 只改变 semantic 子流的 token space，不丢弃 global units。
checkpoint contract 分别记录 input/output tokenizer backend、output detokenizer、audio schema、selector、
payload range、codec-private grammar、blocks、special IDs 和 embedding ownership；
旧 schema 不做静默迁移。

`backbone_initialization` 显式选择 backbone 权重来源：`pretrained` 使用
`AutoModel.from_pretrained()` 直接加载不含 LM head 的 backbone body；`random` 仍从 `backbone`
snapshot 读取 tokenizer 与完整 HF config，但通过 `AutoModel.from_config()` 随机构造同架构
body，不读取 checkpoint 权重。随机初始化由训练入口的 `train.seed` 控制，并要求
`callback/parameter_policy=full`，避免随机 backbone
被全部或部分冻结。`model.toy` 自己构造 tiny Qwen，不能与 `random` 同时启用。
可训练 body 直接注册在 `model.backbone`，canonical state path 是
`model.backbone.layers.*`、Kimi `model.backbone.mimo_layers.*` 与
`model.backbone.norm.*`。旧的 `model.backbone.model.*` checkpoint 不做 key remap，strict load
会显式失败。

非标准 HF backbone 通过三个 runtime 字段显式声明边界：`backbone_trust_remote_code` 同时传给
tokenizer、pretrained backbone 与 random-init config 加载；`backbone_readout` 选择 model 消费的
hidden tensor，支持 `last_hidden_state` 或单层序列索引形如 `last_hidden_state[1]`；`backbone_supports_cache_position`
决定 token model 调用 backbone body 时是否传入 HF `cache_position` 参数。

## 协议

`runtime/codec_contract.py` 定义 codec capability Protocol 与 metadata helper；
`runtime/tokenizer.py` 定义 `AudioTokenizer` / `TextTokenizer`；
`runtime/backbone/contract.py` 定义 `Backbone` 与 readout contract。
`runtime/protocol.py` 统一定义 `DataRuntime`、`GenerationRuntime` 与
`TokenModelRuntime` capability；消费模块不重复声明相同属性。`DataRuntime` 只公开 parser、
sample builder 和 batch padding 所需资源。

`anytrain.codec` 以 `AudioTokenizer` 与 `AudioDetokenizer` 作为正交 capability；同时实现两者的
backend 也是 `AudioCodec`。现有 `FrameCodec`、`SemanticAcousticCodec` 与
`SemanticGlobalCodec` 继续作为 code layout adapter/兼容 API，不再要求每个 preset 都同时拥有
tokenize 与 detokenize。`FULL_CODEC_SEQUENCE` 对 frame-code tokenizer 展开全部 codebooks，生成后
交给配置的 detokenizer；配置 `runtime.audio_output.acoustic_generator_artifact` 时，S2S 只处理
`SemanticAcousticCodec` 的 semantic units，waveform 由
`semantic_acoustic_generator.runtime.GeneratorRuntime` 负责。semantic-only decoder 不放回
anytrain，也不通过普通 codec 的 `decode()` 伪装。

能力检查按实际接口而不是 codec 名称分派：`frame_tokenizer()` 只要求 encode、frame rate 和
codebook sizes，`frame_codec()` 另外要求 decode；`structured_codec()` 接受完整的
semantic/acoustic 或 semantic/global contract；
`acoustic_codec()` 只用于 frame-aligned side channel，`global_codec()` 只用于独立 global units。一个 backend 只有在
满足对应完整 Protocol 时才进入该路径，不用单个同名属性推断整组能力。
capability metadata 在这些边界统一校验：sample rate 必须是正整数，frame rate 必须是有限正数，
codebook sizes 必须是非空正整数 tuple，feature dim 必须是正整数；semantic/acoustic 只能使用
`FRAME_ALIGNED` 且不设置 unit length，semantic/global 必须提供正整数 `global_unit_length`；接口存在但 metadata
无效时直接暴露错误。只消费采样率或 frame-code codebook metadata 的调用点分别使用
`codec_sample_rate()` 与 `frame_codebook_sizes()`，不要求无关的完整 encode/decode 能力。

LongCat 的 `DECOUPLED + model/acoustic=none` 必须配置
`runtime.audio_output.acoustic_generator_artifact`；没有 artifact
时应选择 `FULL_CODEC_SEQUENCE`。`DECOUPLED + Flow/RVQ` 是现有 S2S 内部 acoustic-feature
训练路径，仍由 `Codec.decode_features()` 消费生成的 features；它不代表 anytrain 提供
semantic-only decoder。

S2S 使用 `AudioCodes(semantic_codes, global_codes, acoustic_codes)` 作为内部设计语言。
`SemanticGlobalCodes` 只映射到 semantic/global，`SemanticAcousticCodes` 只映射到
semantic/frame-aligned acoustic。BiCodec 只使用一套 `flattened`
self-describing sequence。所有 codec token 序列统一采用两层协议：

```text
BOA schema_selector codec-private-markers-and-payload EOA
```

BOA/selector/EOA 由 Runtime 拥有；中间 grammar 由所选 `AudioTokenSpec` 拥有。四部分都是模型 token，
response 中全部参与监督；grammar 只提供合法候选 mask 和完整性校验，不替模型写 token。
BiCodec 的 codec-private 序列为：
fixed-length global payload 采用 slot-major 布局并排在 semantic payload 之前。response 以
`<begin_of_global>` 开局时 LLM 生成 global 与 semantic；以 `<begin_of_semantic>` 开局时只生成
semantic 并复用 prompt global。decode 从 prompt/response marker 解析唯一的 global owner，并要求
两侧恰好一方拥有 global；不再使用独立 `semantic` route、`audio_context` side channel 或可切换的
`audio_route`。BiCodec 不再自造内部 end marker；外层 EOA 是唯一结束符。global/semantic markers
属于 codec-private grammar 并与 payload 一样由模型生成和监督。LongCat 等非 BiCodec semantic-only 路径仍交给 side
module 或 semantic-acoustic codec。

BiCodec 配置 `runtime.audio_output.bpe` 时，该 artifact 只作为 semantic 子 tokenizer：raw
semantic IDs 先经
`CodecBPE`，然后再进入 `BiCodecAudioTokenizer` 的 structured packing；global IDs 不做 BPE。
codec-private stream order 与 markers 记录为 `bicodec-v3`；contract 只使用 `global_*` keys，
不读取或生成旧 `acoustic_*` BiCodec contract。native/BPE 差异
记录在嵌套的 `semantic_tokenizer` contract（`native-v1` 或 `codec-bpe-v1`）及 checkpoint hash 中。

无 reference 的 generate 路由把 speaker/style latent 交给语言模型从 text 条件中自回归预测；在多
speaker 数据上如果没有额外 speaker/style 条件或 latent sampling，模型可能收敛到主导 speaker。
这是建模条件的限制，不由 BiCodec tokenizer 隐式解决；需要在 experiment/checkpoint 设计中显式
加入条件或采样策略。

BiCodec 的 global 轴独立于 semantic frame 轴，不能广播到 frame 轴，也不能接入当前
frame-aligned Flow/RVQ side channel。

`supports_structured()` 只表示 backend 提供完整 semantic/acoustic 或 semantic/global contract，
不决定 token 序列格式。semantic/global 使用 `BiCodecAudioTokenizer` 的 structured 状态机；
frame-aligned full-code 路径必须同时满足 `frame_codec()`，并用
`FlattenedAudioTokenizer` 展开完整 frame codebooks。因此同时实现 frame 与 structured capability
的 LongCat 不会误入固定长度状态机。

`audio_tokenizer/` 提供：

- `NativeAudioTokenizer`：单 semantic codebook identity tokenizer。
- `FlattenedAudioTokenizer`：完整 codec codebook token 序列，按 codebook block 写 marker 和
  offset 后的 code IDs；公开 `codebook_ranges` 作为 generation grammar metadata。不再写外层
  codec marker（Runtime 已固定单一 codec）。marker 的 frame span 为 0，首个 codebook token 的
  frame span 为 1，用于 generation 统计输出帧数。
- `TorchCodecBPE`：为 CodecBPE 增加 tensor API。
- `BiCodecAudioTokenizer`：组合一个 native 或 CodecBPE semantic tokenizer，并序列化 global-only、
  semantic-only 或 global-first / semantic-last 的 self-describing structured sequence；full
  sequence 的 global payload 使用 slot-major 顺序；公开 stream enum 使用 `GLOBAL/SEMANTIC`。
- `semantic_codes_from_audio_tokens()`：把 audio token IDs 解码为
  `[frames, semantic_codebooks]`。

Native tokenizer 的 list 入口只接受 vocabulary 范围内的整数 ID，Tensor 入口要求有符号整数
dtype，避免 PyTorch 对扩展 unsigned dtype 的比较与索引行为在下游晚失败；Tensor
encode/decode 保持原 device，并直接使用向量视图，不经过逐标量 Python 转换。

`frame_spans()` 只返回每个 token 覆盖的 frame 数，不重建内容。

## 组装边界

顶层入口直接通过 `Runtime(config)` 创建本次运行的资源聚合对象，并把它显式传给 model、
DataModule 与 generation service。runtime 不保存进程级 singleton；同一进程需要多套配置时，
每套配置各自拥有一个 `Runtime`，其惰性资源缓存互不共享。

`Runtime` 在入口解析时一次确定 `audio_sequence_layout`，并派生内部 route；`DataRuntimeSnapshot`
携带该派生结果。model 把实际消费的 runtime-derived token/audio 结构纳入完整
`checkpoint_contract`，由 `SpeechToSpeechModule` 保存和校验；checkpoint 不再单独保存一个
`audio_sequence_layout` 字符串作为兼容性判断，因为相同 layout 名下的 vocabulary、grammar 或
composition 仍可能不兼容。校验按结构能力而非 preset 身份：`semantic` layout 要求 semantic-only
decode provider 或 acoustic side module；`flattened` layout 要求 codec 能消费完整 full codes；
BiCodec 固定使用 `flattened`，reference global 直接来自
prompt 中的 structured stream，而不是额外配置轴或 side channel。

文件职责保持分离：`runtime/config.py` 拥有配置、layout 校验和 local-rank device 绑定；
`runtime/core.py` 聚合资源；`runtime/factory.py` 选择 sequence-layout runtime；
`runtime/tokenizer_factory.py` 构造 tokenizer 和 special IDs；`runtime/codec.py` 隔离 codec
adapter/加载；`runtime/audio_tokenizer/` 按实现拆分 Native / Flattened / BiCodec / CodecBPE。
DataModule/Collator 接收显式 `DataRuntime`；parser、sample builder、batch padding、objective
与 generation service 不读取全局 runtime 状态。

HF `apply_chat_template` 只负责对话字符串排版；`pad_token_id` / `eos_token_id` /
`bos_token_id` 是训练与生成用的数值 ID，由 Runtime 从 text tokenizer 的同名属性解析（缺失时再
尝试 `special_tokens_map` 中对应字符串并要求映射为单 token），缺属性时显式报错，不再硬编码
整表 Qwen special tokens。Qwen3 stock tokenizer 已提供 `pad=<|endoftext|>`、
`eos=<|im_end|>`，但 `bos` 为 `None`；加载时若 vocab 含 `<|im_start|>` 则绑定为 `bos_token`，
与旧 chat 边界一致。Runtime 不调用 tokenizer `add_special_tokens()`：它在 lexical vocabulary 尾部固定
分配 `<asr>`、`</asr>`、`<mt>`、`</mt>`、`<lang_en>`、`<lang_zh>` 六个 runtime-owned
control IDs，供 task program framing 使用；这些 ID 不得交给 HF/Kimi tokenizer decode。
`boa` / `eoa` / `mask` / schema selector 属于 audio layout，不在 text special tokens 里。checkpoint 同时绑定 lexical
tokenizer 大小、control token→ID 映射和 model 的独立 control embedding。

## 资源边界

- runtime 只加载并暴露资源，不包含 task、objective 或 Trainer 逻辑。
- device、backbone dtype 与 attention backend 来自显式配置，不依赖 Transformers 环境默认值；
  `runtime.Config.dtype` 只控制 backbone 加载精度。在线 waveform fallback 的 codec encode 是独立
  FP32 预处理边界，由 materializer 关闭 Trainer autocast，不把 backbone BF16 传播给 codec。
- backbone snapshot 同时定义 tokenizer 与模型 config；初始化方式只决定是否读取其中的模型权重。
- layout、backbone/tokenizer lexical vocabulary、固定 control vocabulary 与 codec/audio-tokenizer
  vocabulary 属于同一 snapshot。
- Runtime 不是 `nn.Module`；optimizer/checkpoint ownership 只由 model 属性决定。
- LongCat 直接使用 anytrain 的 semantic-acoustic backend；`UnifiedCodec` 只保留给没有独立
  semantic/acoustic capability 的 UniCodec。消费者只依赖所需的最窄 codec capability。
  UniCodec loader 只在边界转换为窄 `UnifiedCodecSource`，adapter 内不使用 `Any`。
- text tokenizer 必须提供与 chat template 一致的 `pad_token_id` / `eos_token_id` /
  `bos_token_id`（或等价 `special_tokens_map` 字符串）；替换 backbone 前需保证这些 ID 与
  chat adapter 对齐。
