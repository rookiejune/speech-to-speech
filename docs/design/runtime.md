# runtime

加载并聚合一套相互兼容的 tokenizer、codec、backbone、layout 与 flow runtime。

## 对外能力

`Runtime` 是 frozen dataclass，重资源通过 `cached_property` 惰性加载：

- `codec_name`：dataset 与 runtime 共用的唯一 codec identity；`audio_view` 由它转换。
- `text_tokenizer` / `audio_tokenizer`：Qwen-compatible text tokenizer 与 Native/CodecBPE audio
  tokenizer。
- `codec`：经本地 capability Protocol adapter 暴露当前 backend 实际拥有的 encode/decode、
  codebook table 或 acoustic feature 能力，不为缺失能力提供占位属性。
- `semantic_codec`：semantic token generation 的 waveform decoder。默认与 `codec` 相同；配置
  `semantic_codec_artifact` 时惰性加载 `semantic-acoustic-codec` artifact，并复用同一个 backend。
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

`runtime/types.py` 定义资源对象的 `SemanticCodec`、`Codec`、`CodebookCodec`、
`AcousticCodec`、`AudioTokenizer`、`TextTokenizer` 与 `Backbone` Protocol。
`runtime/protocol.py` 统一定义 `DataRuntime`、`GenerationRuntime` 与
`TokenModelRuntime` capability；消费模块不重复声明相同属性。`DataRuntime` 只公开 parser、
sample builder 和 batch padding 所需资源。

`SemanticCodec` 只要求 sample/frame rate 与 `decode(codes)`；`Codec` 增加 encode、完整
`codebook_sizes` 与随机 audio embedding 所需的 `semantic_feature_dim`。`CodebookCodec` 进一步
提供真实 `semantic_codebook`，供 native/BPE tokenizer 初始化 embedding；`AcousticCodec` 再增加
独立 acoustic codebooks、code-to-feature 与 feature decode。LongCat 实现 `AcousticCodec`，
UniCodec 实现 `CodebookCodec`，BiCodec full-sequence adapter 只实现基础 `Codec`，不伪造全零
semantic codebook 或不可用 acoustic API。artifact 当前只支持 LongCat decoupled representation，
且入口只允许与 `model/acoustic=none` 组合，避免同一次生成同时启用 S2S acoustic decoder 和外部
semantic support。

`audio_tokenizer.py` 提供：

- `NativeAudioTokenizer`：单 semantic codebook identity tokenizer。
- `FlattenedAudioTokenizer`：完整 codec codebook token 序列，先写 codec marker，再按 codebook
  block 写 marker 和 offset 后的 code IDs；marker 的 frame span 为 0，首个 codebook token 的
  frame span 为 1，用于 generation 统计输出帧数。
- `TorchCodecBPE`：为 CodecBPE 增加 tensor API。
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
- `LongCatCodec`、`UnifiedCodec` 与 `BiCodecCodec` 隔离具体第三方类型，消费者只依赖所需的最窄
  codec capability；
  UniCodec loader 只在边界转换为窄 `UnifiedCodecSource`，adapter 内不使用 `Any`。
- text special tokens 与 chat template 当前属于 Qwen3 contract；替换 backbone 前需提供对应
  tokenizer/chat adapter。
