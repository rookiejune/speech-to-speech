# runtime

加载并聚合一套相互兼容的 tokenizer、codec、backbone、layout 与 flow runtime。

## 对外能力

`Runtime` 是 frozen dataclass，重资源通过 `cached_property` 惰性加载：

- `codec_name`：dataset 与 runtime 共用的唯一 codec identity；`audio_view` 由它转换。
- `text_tokenizer` / `audio_tokenizer`：Qwen-compatible text tokenizer 与 Native/CodecBPE audio
  tokenizer。
- `codec`：经本地 capability Protocol adapter 暴露当前 backend 实际拥有的 encode/decode、
  codebook table 或 acoustic feature 能力，不为缺失能力提供占位属性。
- `semantic_codec`：semantic token generation 的 waveform decoder。配置
  `semantic_codec_artifact` 时惰性加载 `semantic-acoustic-codec` artifact，并复用同一个
  structured backend；FrameCodec 不把 semantic-only codes 传给自身 `decode()`。
- `audio_representation`：选择 audio token 序列契约，当前为 `decoupled` 或
  `full_codec_sequence`。
- `backbone`：Qwen-compatible HF causal LM。
- `layout`：text/audio global token blocks。
- `pad/bos/eos_token_id` 与 `boa/eoa_token_id`。
- `codec_audio_range`、`audio_generation_allowed_ids` 与 modality generation IDs。
- `flow_matching`：训练 sample 与 generation ODE sampler 的共享 runtime。

`runtime.Config.codec` 是 codec identity 的唯一配置源；`audio_view` 由字符串枚举转换，未知 codec
显式报错。Hydra runtime preset 直接映射完整 `runtime.Config`，同时选择相互兼容的 codec、audio
representation、audio tokenizer 与 backbone snapshot。`full_codec_sequence` 直接把完整 codec
codebooks 编入 token 序列，因此不能同时配置 BPE audio tokenizer。ODE method、NFE 与 step 数直接使用 `flow_method`、`flow_nfe` 与
`flow_num_steps`，不再通过独立 sampler 组转换；`Config` 在构造时校验 method、正 NFE 和至少
2 个 steps，因此 token/RVQ composition 也不会静默携带无效 runtime。model composition 由
`model/acoustic` 选择。

## 协议

`runtime/types.py` 定义资源对象的 `SemanticCodec`、`Codec`、`StructuredCodec`、
`CodebookCodec`、`AcousticCodec`、`AudioTokenizer`、`TextTokenizer` 与 `Backbone` Protocol。
`runtime/protocol.py` 统一定义 `DataRuntime`、`GenerationRuntime` 与
`TokenModelRuntime` capability；消费模块不重复声明相同属性。`DataRuntime` 只公开 parser、
sample builder 和 batch padding 所需资源。

`anytrain.codec` 只提供两种 backend capability：`FrameCodec` 和
`SemanticAcousticCodec`。S2S 的 `Codec` 只表示完整 frame-code 路径，不能再继承
semantic-only decoder。`FULL_CODEC_SEQUENCE` 对 `FrameCodec` 展开全部 codebooks，生成后调用
完整 `decode(codes)`；配置 `semantic_codec_artifact` 时，S2S 只处理
`SemanticAcousticCodec` 的 semantic units，waveform 由
`semantic_acoustic_codec.runtime.SemanticCodecRuntime` 负责。semantic-only decoder 不放回
anytrain，也不通过普通 codec 的 `decode()` 伪装。

能力检查按实际接口而不是 codec 名称分派：`frame_codec()` 要求完整 encode/decode、frame rate 和
codebook sizes；`structured_codec()` 要求 tokenize/detokenize、semantic/acoustic codebook、layout
与 feature decode 接口；`acoustic_codec()` 只用于 frame-aligned side channel。一个 backend 只有在
满足对应完整 Protocol 时才进入该路径，不用单个同名属性推断整组能力。
capability metadata 在这些边界统一校验：sample rate 必须是正整数，frame rate 必须是有限正数，
codebook sizes 必须是非空正整数 tuple，feature dim 必须是正整数，structured layout 必须是
`AcousticLayout`，且 `FIXED_LENGTH` 必须提供正整数 `acoustic_unit_length`；接口存在但 metadata
无效时直接暴露错误。只消费采样率或 frame-code codebook metadata 的调用点分别使用
`codec_sample_rate()` 与 `frame_codebook_sizes()`，不要求无关的完整 encode/decode 能力。

LongCat 的 `DECOUPLED + model/acoustic=none` 必须配置 `semantic_codec_artifact`；没有 artifact
时应选择 `FULL_CODEC_SEQUENCE`。`DECOUPLED + Flow/RVQ` 是现有 S2S 内部 acoustic-feature
训练路径，仍由 `Codec.decode_features()` 消费生成的 features；它不代表 anytrain 提供
semantic-only decoder。

BiCodec 使用同一个 structured backend，但保留两条互斥的 TTS 路线：

- `DECOUPLED + semantic_codec_artifact` 只生成 semantic units，固定长度 acoustic units 由
  `semantic-acoustic-codec` artifact 采样并解码。
- `FULL_CODEC_SEQUENCE` 生成显式的 structured token layout：codec marker、semantic marker、
  semantic tokens、acoustic marker、slot-major acoustic codebooks、end marker。解码时恢复
  `SemanticAcousticCodes`，直接调用 BiCodec `detokenize()`。

BiCodec 的 acoustic 轴是 `FIXED_LENGTH`，不能广播到 semantic frame 轴，也不能接入当前
frame-aligned Flow/RVQ side channel。

`supports_structured()` 只表示 backend 同时提供 semantic/acoustic structured 能力，不决定 token
序列格式。`FULL_CODEC_SEQUENCE` 按 layout 分派：`FIXED_LENGTH` 使用
`BiCodecAudioTokenizer` 的 structured 状态机；`FRAME_ALIGNED` 必须同时满足 `frame_codec()`，并用
`FlattenedAudioTokenizer` 展开完整 frame codebooks。因此同时实现 frame 与 structured capability
的 LongCat 不会误入固定长度状态机。

`audio_tokenizer.py` 提供：

- `NativeAudioTokenizer`：单 semantic codebook identity tokenizer。
- `FlattenedAudioTokenizer`：完整 codec codebook token 序列，先写 codec marker，再按 codebook
  block 写 marker 和 offset 后的 code IDs；marker 的 frame span 为 0，首个 codebook token 的
  frame span 为 1，用于 generation 统计输出帧数。
- `TorchCodecBPE`：为 CodecBPE 增加 tensor API。
- `BiCodecAudioTokenizer`：分别支持 semantic-only token 和 fixed-length structured full
  sequence；full sequence 的 acoustic payload 使用 slot-major 顺序。
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

文件职责保持分离：`runtime/runtime.py` 实现配置与资源聚合，`runtime/codec.py` 隔离 codec
adapter 和加载。DataModule/Collator 接收显式 `DataRuntime`；parser、sample builder、batch
padding、objective 与 generation service 不读取全局 runtime 状态。

## 资源边界

- runtime 只加载并暴露资源，不包含 task、objective 或 Trainer 逻辑。
- device、dtype 与 attention backend 来自显式配置，不依赖 Transformers 环境默认值。
- layout、backbone/tokenizer vocabulary 与 codec/audio-tokenizer vocabulary 属于同一 snapshot。
- Runtime 不是 `nn.Module`；optimizer/checkpoint ownership 只由 model 属性决定。
- LongCat 直接使用 anytrain 的 semantic-acoustic backend；`UnifiedCodec` 只保留给没有独立
  semantic/acoustic capability 的 UniCodec。消费者只依赖所需的最窄 codec capability。
  UniCodec loader 只在边界转换为窄 `UnifiedCodecSource`，adapter 内不使用 `Any`。
- text special tokens 与 chat template 当前属于 Qwen3 contract；替换 backbone 前需提供对应
  tokenizer/chat adapter。
