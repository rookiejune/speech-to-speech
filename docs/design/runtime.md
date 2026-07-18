# runtime

加载并聚合一套相互兼容的 tokenizer、codec、backbone、layout 与 flow runtime。

## 对外能力

`Runtime` 是 frozen dataclass，重资源通过 `cached_property` 惰性加载：

- `codec_name`：dataset 与 runtime 共用的唯一 codec identity；`audio_view` 由它转换。
- `text_tokenizer` / `audio_tokenizer`：Qwen-compatible text tokenizer 与 Native/CodecBPE audio
  tokenizer。
- `codec`：经本地 `Codec` Protocol adapter 暴露 sample/frame rate、codebook、feature 与
  decode 能力。
- `backbone`：Qwen-compatible HF causal LM。
- `layout`：text/audio global token blocks。
- `pad/bos/eos_token_id` 与 `boa/eoa_token_id`。
- `codec_audio_range`、`audio_generation_allowed_ids` 与 modality generation IDs。
- `flow_matching`：训练 sample 与 generation ODE sampler 的共享 runtime。

`RuntimeConfig.codec` 是 codec identity 的唯一配置源；`audio_view` 由字符串枚举转换，未知 codec
显式报错。codec preset 只保留入口实际消费的 `name`，不暴露可独立覆盖的第二身份。

## 协议

`runtime/types.py` 定义资源对象的 `Codec`、`AudioTokenizer`、`TextTokenizer` 与 `Backbone`
Protocol。`runtime/protocol.py` 统一定义 `DataRuntime`、`GenerationRuntime` 与
`TokenModelRuntime` capability；消费模块不重复声明相同属性。`DataRuntime` 只公开 parser、
sample builder 和 batch padding 所需资源。

`audio_tokenizer.py` 提供：

- `NativeAudioTokenizer`：单 semantic codebook identity tokenizer。
- `TorchCodecBPE`：为 CodecBPE 增加 tensor API。
- `semantic_codes_from_audio_tokens()`：把 audio token IDs 解码为
  `[frames, semantic_codebooks]`。

Native tokenizer 的 list 入口只接受 vocabulary 范围内的整数 ID，Tensor 入口要求有符号整数
dtype，避免 PyTorch 对扩展 unsigned dtype 的比较与索引行为在下游晚失败；Tensor
encode/decode 保持原 device，并直接使用向量视图，不经过逐标量 Python 转换。

`frame_spans()` 只返回每个 token 覆盖的 frame 数，不重建内容。

## Singleton 边界

`init_runtime(config)` 只服务于顶层组装并返回当前进程的 runtime。重复用不同 config 初始化会
报错；底层模块不回读 singleton。

文件职责保持分离：`runtime/runtime.py` 实现配置与资源聚合，`runtime/codec.py` 隔离 codec
adapter 和加载，`runtime/singleton.py` 只保存进程级初始化状态。

model 接收显式 runtime；DataModule/Collator 接收显式 `DataRuntime`。parser、sample
builder、batch padding、objective 与 generation service 不调用 singleton，因此 model 与 data
不会各自读取不同的进程状态。

## 资源边界

- runtime 只加载并暴露资源，不包含 task、objective 或 Trainer 逻辑。
- device、dtype 与 attention backend 来自显式配置，不依赖 Transformers 环境默认值。
- layout、backbone/tokenizer vocabulary 与 codec/audio-tokenizer vocabulary 属于同一 snapshot。
- Runtime 不是 `nn.Module`；optimizer/checkpoint ownership 只由 model 属性决定。
- `LongCatCodec` 与 `UnifiedCodec` 隔离具体第三方类型，模型只依赖本地 `Codec` Protocol；
  UniCodec loader 只在边界转换为窄 `UnifiedCodecSource`，adapter 内不使用 `Any`。
- text special tokens 与 chat template 当前属于 Qwen3 contract；替换 backbone 前需提供对应
  tokenizer/chat adapter。
