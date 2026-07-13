# runtime

提供已经加载好的共享资源，是 datamodule、model、loss 获取 tokenizer/codec/backbone 的单一入口。

## 对外能力

`Runtime`（frozen dataclass，资源按 `cached_property` 惰性加载）提供：

- `text_tokenizer`：当前只承诺 Qwen3-compatible 的 HF tokenizer/chat template。
- `audio_tokenizer`：`config.audio_tokenizer` 为 `None` 时为 `NativeAudioTokenizer`（identity，单 codebook）；否则该配置必须是明确的 CodecBPE artifact 路径，Runtime 通过 workspace 的 `codec_bpe(path)` 加载并包装为 `TorchCodecBPE`。
- `codec`：经本地 adapter 统一为 `Codec` 协议；除 waveform sample rate 外，adapter
  还从已加载 codec 推导 prepared-code frame rate，配置不重复声明该资源属性。
- `backbone`：当前只承诺 Qwen3-compatible 的 HF causal LM。
- `layout`：text/audio 两个 global id block；audio block 末尾预留 boa/eoa 两个位置。
- audio ID 能力：完整 audio head block、generation allowed IDs（semantic + eoa）和 codec-decodable IDs（仅 semantic）是三个不同集合。
- special ids：`pad/bos/eos_token_id`（text）与 `boa/eoa_token_id`（audio block 末两位）。
- `flow_matching`：anytrain 的 `ContinuousFlowRuntime`，统一持有训练时间分布和 acoustic generation sampler 配置。
  正式训练入口从公共 `flow/` Hydra config group 显式传入 method、NFE 和 step 数；Runtime
  不另选实验级 sampler preset。

入口：

- `init_runtime(config)` / `runtime()`：进程级 singleton；重复以不同 config 初始化直接报错。
- 正式组合入口初始化 singleton；model、loss 与 datamodule 构造后使用同一个 runtime snapshot 或同一个顶层 codec 身份，不分别隐式选择资源。

## 协议

`types.py` 定义消费方依赖的结构化协议：`Codec`、`AudioTokenizer`、`TextTokenizer`、
`Backbone`。fake 实现只需满足协议即可支撑 contract test；identity 实现直接使用
`audio_tokenizer.py` 的 `NativeAudioTokenizer`。

`audio_tokenizer.py` 另提供 `semantic_ids_from_audio_tokens()`：把单条 BPE 序列展开为 `[frames, semantic_codebooks]`，供 waveform decode 前使用。
`AudioTokenizer.frame_spans()` 返回每个 token 覆盖的 frame 数；Native tokenizer 固定为 1，
CodecBPE 直接读取已保存的 token 结构，不通过逐 token decode 推导。

## 边界

- runtime 只负责"加载并暴露资源"，不包含任务逻辑或训练逻辑。
- 当前 text special IDs 和 `enable_thinking` chat 参数属于 Qwen3 契约；切换其他 backbone
  前必须先提供独立的 tokenizer/chat adapter，不能只替换 checkpoint 字符串。
- Runtime 不从 tokenizer 名称猜测训练参数，也不隐式训练或选择 BPE artifact。
- `config.device`、`config.dtype` 和 `config.attn_implementation` 显式控制 HF backbone 的设备、参数 dtype 和 attention backend；Runtime 不依赖 Transformers 的环境默认值。
- layout、backbone/tokenizer vocabulary 与 codec/audio-tokenizer vocabulary 属于同一个不可替换的 runtime snapshot。
- Runtime 不是 `nn.Module`；是否进入 optimizer/checkpoint 只由 model 的显式模块属性决定。
- 同一可训练模块在 model 中只能注册一条路径；backbone text embedding 不能同时注册到 backbone 与 multimodal embedding。
- 一个 Runtime 对应一个训练模型组合，不承诺从同一 Runtime 构造多个相互独立训练的模型。
- codec 通过 `_CodecContract` 隔离：模型侧只依赖本地 `Codec` 协议，不直接触碰 LongCat 类型。
- codec acoustic representation 固定其 codebook 输入、feature dimension 与 waveform decode；model 不任意切取 codebook。
- `runtime()` 在未初始化时显式报错，不做隐式默认初始化。
