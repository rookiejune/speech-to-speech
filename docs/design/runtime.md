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
- `audio_sequence_layout`：公开音频序列格式，当前为 `flattened` 或 `semantic`。`flattened`
  表示完整 codec codes 被序列化为 acoustic-first / semantic-last 的 token 序列；`semantic`
  表示逻辑输入输出仍是 full codes，但 token sequence 只处理 semantic，acoustic 由 side module 或
  semantic-acoustic codec 补齐。Runtime 构造时按该字段派生内部 route，并沿同一运行实例传给
  DataModule、model 与 generation。
- `backbone`：Qwen-compatible HF causal LM。
- `layout`：text/audio global token blocks。
- `pad/bos/eos_token_id` 与 `boa/eoa_token_id`。
- `codec_audio_range`、`audio_generation_allowed_ids` 与 modality generation IDs。
- `flow_matching`：训练 sample 与 generation ODE sampler 的共享 runtime。

`runtime.Config.codec` 是 codec identity 的唯一配置源；`audio_view` 由字符串枚举转换，未知 codec
显式报错。Hydra runtime preset 直接映射完整 `runtime.Config`，同时选择相互兼容的 codec、FrameCodec
audio sequence layout、audio tokenizer 与 backbone snapshot。`flattened` 直接把完整
FrameCodec codebooks 编入 token 序列，因此不能同时配置 BPE audio tokenizer；其顺序固定为
acoustic-first / semantic-last。ODE method、NFE 与 step 数直接使用 `flow_method`、`flow_nfe` 与
`flow_num_steps`，不再通过独立 sampler 组转换；`Config` 在构造时校验 method、正 NFE 和至少
2 个 steps，因此 token/RVQ composition 也不会静默携带无效 runtime。model composition 由
`model/acoustic` 选择。

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

BiCodec 使用同一个 structured backend。非 semantic 单元存放在 `SemanticAcousticCodes.acoustic`；
`AcousticLayout.FIXED_LENGTH` 表示这些单元是固定长度 speaker/style slots（口语里的 global），
`FRAME_ALIGNED` 则与 semantic 时间对齐。`audio_route` 只声明有没有 `acoustic` stream 以及谁提供它，
不把 layout 再抬成第三种 stream 名。
`flattened` layout 下，完整 codec codes 进入同一 token 序列；BiCodec 的 fixed-length acoustic
payload 使用 slot-major 布局，并固定排在 semantic payload 之前。`semantic` layout 下，输出 token
只含 semantic；如果 decode 需要 acoustic，BiCodec 从输入 full codes/context 复用 reference acoustic，
LongCat 等 semantic-only 路径则交给 side module 或 semantic-acoustic codec。markers 与 end marker
属于内部 route，强制位置不作为可训练 payload。

无 reference 的 generate 路由把 speaker/style latent 交给语言模型从 text 条件中自回归预测；在多
speaker 数据上如果没有额外 speaker/style 条件或 latent sampling，模型可能收敛到主导 speaker。
这是建模条件的限制，不由 BiCodec tokenizer 隐式解决；需要在 experiment/checkpoint 设计中显式
加入条件或采样策略。

BiCodec 的 acoustic 轴是 `FIXED_LENGTH`，不能广播到 semantic frame 轴，也不能接入当前
frame-aligned Flow/RVQ side channel。

`supports_structured()` 只表示 backend 同时提供 semantic/acoustic structured 能力，不决定 token
序列格式。`FULL_CODEC_SEQUENCE` 按 layout 分派：`FIXED_LENGTH` 使用
`BiCodecAudioTokenizer` 的 structured 状态机；`FRAME_ALIGNED` 必须同时满足 `frame_codec()`，并用
`FlattenedAudioTokenizer` 展开完整 frame codebooks。因此同时实现 frame 与 structured capability
的 LongCat 不会误入固定长度状态机。

`audio_tokenizer/` 提供：

- `NativeAudioTokenizer`：单 semantic codebook identity tokenizer。
- `FlattenedAudioTokenizer`：完整 codec codebook token 序列，按 codebook block 写 marker 和
  offset 后的 code IDs；公开 `codebook_ranges` 作为 generation grammar metadata。不再写外层
  codec marker（Runtime 已固定单一 codec）。marker 的 frame span 为 0，首个 codebook token 的
  frame span 为 1，用于 generation 统计输出帧数。
- `TorchCodecBPE`：为 CodecBPE 增加 tensor API。
- `BiCodecAudioTokenizer`：支持 semantic-only token，以及按 `audio_route` 选择 stream 的
  fixed-length structured sequence；full sequence 的 acoustic payload 使用 slot-major 顺序，
  并公开 route grammar 所需的 prediction groups/ranges。
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
携带该派生结果。checkpoint 严格匹配派生 route metadata，由 `SpeechToSpeechModule` 负责。
校验按结构能力而非 preset 身份：`semantic` layout 要求 semantic-only decode provider 或 acoustic
side module；`flattened` layout 要求 codec 能消费完整 full codes；BiCodec 的 fixed-length 语义由
codec layout 提供，reference acoustic 来自输入 full codes/context 而不是额外配置轴。

文件职责保持分离：`runtime/runtime.py` 实现配置与资源聚合，`runtime/codec.py` 隔离 codec
adapter 和加载，`runtime/audio_tokenizer/` 按实现拆分 Native / Flattened / BiCodec / CodecBPE。
DataModule/Collator 接收显式 `DataRuntime`；parser、sample builder、batch padding、objective
与 generation service 不读取全局 runtime 状态。

HF `apply_chat_template` 只负责对话字符串排版；`pad_token_id` / `eos_token_id` /
`bos_token_id` 是训练与生成用的数值 ID，由 Runtime 从 text tokenizer 的同名属性解析（缺失时再
尝试 `special_tokens_map` 中对应字符串并要求映射为单 token），缺属性时显式报错，不再硬编码
整表 Qwen special tokens。Qwen3 stock tokenizer 已提供 `pad=<|endoftext|>`、
`eos=<|im_end|>`，但 `bos` 为 `None`；加载时若 vocab 含 `<|im_start|>` 则绑定为 `bos_token`，
与旧 chat 边界一致。`boa` / `eoa` / `mask` 属于 audio layout，不在 text special tokens 里。

## 资源边界

- runtime 只加载并暴露资源，不包含 task、objective 或 Trainer 逻辑。
- device、backbone dtype 与 attention backend 来自显式配置，不依赖 Transformers 环境默认值；
  `runtime.Config.dtype` 只控制 backbone 加载精度。在线 waveform fallback 的 codec encode 是独立
  FP32 预处理边界，由 materializer 关闭 Trainer autocast，不把 backbone BF16 传播给 codec。
- backbone snapshot 同时定义 tokenizer 与模型 config；初始化方式只决定是否读取其中的模型权重。
- layout、backbone/tokenizer vocabulary 与 codec/audio-tokenizer vocabulary 属于同一 snapshot。
- Runtime 不是 `nn.Module`；optimizer/checkpoint ownership 只由 model 属性决定。
- LongCat 直接使用 anytrain 的 semantic-acoustic backend；`UnifiedCodec` 只保留给没有独立
  semantic/acoustic capability 的 UniCodec。消费者只依赖所需的最窄 codec capability。
  UniCodec loader 只在边界转换为窄 `UnifiedCodecSource`，adapter 内不使用 `Any`。
- text tokenizer 必须提供与 chat template 一致的 `pad_token_id` / `eos_token_id` /
  `bos_token_id`（或等价 `special_tokens_map` 字符串）；替换 backbone 前需保证这些 ID 与
  chat adapter 对齐。
