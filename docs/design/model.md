# model

组装 token backbone、multimodal embedding 与 acoustic decoder。position 语义见
[总览 §2.4](../model-design.md)。

## 对外能力

- `base.TokenModel`：接收显式 runtime，提供 text/semantic-audio embedding、token
  logits、route-aware structured token generation 与 frame condition 对齐原语。
- `audio_input.AudioInputTower`：把 source audio payload 的 semantic embedding 按显式位置
  编码为 backbone hidden states；支持 `none`、同长度 `mlp` 与非 causal `transformer`，只属于
  input path，不参与 semantic-audio output head 或 acoustic generation。
- `audio_output.AudioOutputAdapter`：把 backbone hidden states 逐 token 投影到 semantic-audio
  feature space；支持 `none`、`linear` 与 `mlp`，不做序列混合，以兼容 cached autoregressive
  generation。
- `acoustic.FlowModel`：在基础模型上组合 SAC 维护的 `AcousticDiT`，提供
  flow target、sampling 和 `generate_audio_features()`。
- `acoustic.RVQModel`：组合 SAC 维护的 `AcousticRVQDecoder`，提供 teacher-forced
  codebook logits、sampling 和 `generate_audio_features()`。
- `acoustic.HiddenConditionAdapter`：以 `LayerNorm + Linear` 把对齐后的 backbone hidden state 映射到
  SAC generator 的 condition space；训练和 generation 共用该入口。
- `loss.protocol.TokenObjectiveModel` / `FlowObjectiveModel` / `RVQObjectiveModel`：objective
  所依赖的训练能力。
- `generation.protocol.TokenGenerator` / `AcousticFeatureGeneration`：generation service
  所依赖的基础契约与可选 acoustic runtime 能力；`AcousticFeatureGenerator` 组合两者供训练侧静态
  typing，`TextEvaluationModel` 组合 token generation 与 reference scoring。
- `runtime.protocol.TokenModelRuntime` / `model.protocol.FlowModelRuntime`：token 与 flow
  model 各自消费的 runtime 资源边界。
- `AdapterType`：semantic input/output adapter 的 `linear|mlp` 字符串枚举；`None` 表示输入输出
  dimension 相同的 identity adapter。
- `AudioInputAdapterType` / `AudioInputAdapterConfig`：source audio tower 的 `none|mlp|transformer`
  配置；`transformer` 使用同长度、非 causal 的 encoder layer。
- `AudioOutputAdapterType` / `AudioOutputAdapterConfig`：semantic-audio output adapter 的
  `none|linear|mlp` 配置。
- `ToyConfig` / `create_toy_backbone()`：构造随机初始化的一层或少层 Qwen backbone，用于 CPU
  model/data 契约测试；词表大小来自 runtime layout，但不读取 `runtime.backbone`。
- `AcousticType`、`DecoderConfig`、`FlowRepaConfig`：组合入口的严格配置结构。

## Token 接口

```python
def forward(
    input_ids: Tensor,
    *,
    attention_mask: Tensor | None = None,
    audio_input_positions: Tensor | None = None,
    output_hidden_states: bool = False,
    past_key_values: Cache | None = None,
    use_cache: bool = False,
    position_ids: Tensor | None = None,
    cache_position: Tensor | None = None,
) -> CausalLMOutputWithPast: ...

def token_hidden_states(...) -> Tensor: ...
def token_logits(
    hidden_state: Tensor,
    modality: Modality | None = None,
) -> Tensor: ...
def selected_logits(hidden_state: Tensor, token_ids: Tensor) -> Tensor: ...
def generation_step(...) -> CausalLMOutputWithPast: ...
def generate_tokens(...) -> Tensor: ...
```

- `forward()` 返回 global text+audio logits，不接收 labels 或计算 loss。
- `forward()` 支持 HF backbone 的 cache/position 参数；sampling、stop 和 output-head selection
  参数不进入该通用接口。
- `audio_input_positions` 是 `[batch, frames]` 的完整序列位置，`-1` 只用于 batch padding。它只
  指向 source audio payload token；BOA/EOA、target audio token、generated token 和 BiCodec
  reference `audio_context` 都不经过 `AudioInputTower`。
- `generation_step()` 只返回最后位置的目标 modality 或显式 token 子集 logits，并把 cache
  状态传给 backbone。
- 训练先用 `token_hidden_states()` 取得完整表示，再由 objective 只选有效 predictor rows，并用
  task 的 target modality 调用 `token_logits()`；CE 只构造对应 text 或 audio 局部词表 logits，
  不为 prompt、padding 或另一模态构造大词表 logits。未传 modality 的通用 `forward()` 仍返回
  global text+audio logits。
- backbone 直接调用 HF causal LM 的 `base_model`；自带 text LM head 不会先计算再丢弃。
- text/audio output head 分别产生 local logits，layout offset 只负责恢复 global token ID。
- `selected_logits()` 只计算调用方给出的候选 global IDs；BiCodec route 的 grouped CE 用它避免为
  每个 marker/codebook 位置计算完整 audio vocabulary。
- generation 按 modality 只计算最后一个位置的目标 head；text 屏蔽 PAD/BOS，audio 屏蔽 BOA。
- text/audio vocabulary head 位于私有 `_head.py` mixin；参数仍只注册在 `TokenModel` 的原始
  embedding/adapter/backbone ownership path 下。
- `target_frame_condition()` 与 `target_frame_label_condition()` 都接收 token 自身位置 `p`；
  causal shift `p - 1` 只在 model 内部发生。

## 配置边界

`model.Config` 只包含基础模型真正消费的设置：

- `semantic_audio_adapter`
- `audio_output_adapter`
- `audio_input_adapter`
- `toy`
- `lora`

`semantic_audio_adapter` 使用公开 `AdapterType`；`linear` 是默认值，`mlp` 使用 gated SiLU adapter，
`None` 只在输入输出 dimension 相同时合法。`toy=None` 时模型使用 `runtime.backbone`；非空时由
`ToyConfig` 构造随机 tiny Qwen，runtime 仍负责 tokenizer、codec、layout、special IDs 与 flow
sampler。完整 Qwen 架构的随机初始化属于 `runtime.backbone_initialization=random`，不通过 toy
参数近似。Hydra `model` preset 与这些字段一一对应，overfit/train root schema 直接复用
`model.Config`。

`audio_output_adapter` 是 semantic-audio token head 的显式结构化配置，支持 `none`、`linear` 或
`mlp`。它在 teacher forcing、普通 token logits、候选 token
logits 和 cached generation 中使用同一实例。输出 adapter 不使用 input tower 的 Transformer，
因为 output head 每次 generation 只收到最后一个 hidden state；引入序列混合会额外改变 causal
cache 契约。

`audio_input_adapter` 默认 `type=none`。启用 `mlp` 时，source audio payload 的 semantic embedding
逐帧经过 gated MLP 投影到 backbone hidden dimension；启用 `transformer` 时，先做输入投影，再用
同长度、非 causal 的 Transformer encoder 跨 source frames 建立上下文。两种 tower 都保持 frame
数量不变，并在 overlay 到 `inputs_embeds` 前清零 padding。训练和完整 prompt 的首步会传入显式
`audio_input_positions`；启用 KV cache 后后续 token 只走 backbone，不重复运行 source tower。
该配置不会改变生成 grammar，也不会替换 Flow/RVQ
`HiddenConditionAdapter`。

`lora` 直接持有 `peft.LoraConfig | None`，项目不再维护本地 LoRA config、layer 或注入 facade。
选择 `model/lora=qwen parameter_policy=lora` 后，model 把该 config 直接传给 PEFT
`inject_adapter_in_model()`；rank、alpha、dropout、target modules、初始化方法与 PEFT 后续支持的
字段都沿用官方命名和校验。混合精度 backbone 注入后使用 PEFT 的 mixed-precision cast 规则。
PEFT 决定 backbone 内的 trainable adapter/bias/modules-to-save 参数，parameter policy 额外组合现有
speech/acoustic interface，不再通过本地 LoRA 参数名重新推断 PEFT 的训练语义。

checkpoint 保存 `peft-lora-v2` metadata，包括固定 adapter name、规范化后的完整
`LoraConfig.to_dict()` 和同版本 `LoraConfig()` 默认值；只移除不影响 adapter 语义的
`peft_version`，set/enum 等结构递归转成稳定值。共同字段严格比较；跨 PEFT 版本新增或缺失的字段
只有等于对应版本官方默认值时才兼容，非默认新语义仍明确失败。启用 LoRA 时缺少 metadata 也会
失败；未启用时仍允许加载不含该 metadata 的旧 checkpoint。

decoder 使用独立 `DecoderConfig(hidden_dim, layers, heads, ffn_ratio)`。flow 可额外接收
`FlowRepaConfig(feature_dim, student_layer)`；RVQ 可额外接收初始化 decoder 各 acoustic
codebook 的 `codebook_embeddings`，但没有 REPA 参数。Hydra 使用
`model/acoustic=none|flow|rvq`，`none` 只训练 semantic audio token，flow preset 独占 teacher
与 student REPA 配置。ODE sampling 由 `runtime.Config.flow_*` 拥有。没有独立 acoustic
codebooks 的 unified-token codec 必须使用 `model/acoustic=none`；有独立 acoustic codebook 的
codec 也可以显式选择 `none` 作为 token-only baseline。入口不根据 codec 静默覆盖用户选择。
fixed-length structured codec（例如 BiCodec）使用独立的 model-facing token layout。它只支持
按 `audio_route` 固定的 structured sequence 路线，不接入当前 frame-aligned Flow/RVQ acoustic
side channel。新 route 使用 `global` 表示固定长度 speaker/style stream：
`bicodec_reuse_prompt_global` 只输出 semantic，`bicodec_generate_global` 同时输出 global 与
semantic。BiCodec route 不接受 FrameCodec 使用的 `acoustic` stream。两条 route 使用同一套稳定
vocabulary，route 只改变 grammar 的
output groups 与 decode stream ownership，不按 request 动态改变模型 head。
无 reference 的 `bicodec_generate_global` 不自带 speaker ID；多 speaker 训练若没有额外条件或
latent sampling，global 预测可能偏向数据中的主导 speaker，这属于模型条件设计而不是 codec
序列化问题。

底层 acoustic decoder 的所有权在 `semantic-acoustic-codec`：S2S 的 Flow/RVQ model 只负责
从 backbone hidden state 取 frame-aligned condition，经 `HiddenConditionAdapter` 映射后送入 SAC 的
DiT/DiT+REPA/RVQ decoder。`acoustic.init_artifact` 可在 composition 边界加载 SAC generator；model
构造器不执行 artifact I/O。S2S 不维护 acoustic-only codec oracle，也不复制底层 decoder 实现。

## Embedding

```text
text_token_ids
    -> backbone text embedding

semantic-audio token IDs
    -> codec-initialized or representation-defined random audio embedding
    -> semantic audio adapter

source audio payload positions
    -> semantic audio embedding
    -> AudioInputTower (none | mlp | transformer)
    -> overlay selected inputs_embeds only

Native/BPE semantic tokenizers 使用 codec codebook 初始化；完整 codec sequence tokenizer
使用随机初始化，因为它的 vocab 同时包含多 codebook offset tokens、BiCodec semantic/global
ranges 与 codec/stream/end markers。BiCodec 的 semantic payload、各 fixed-length global slot
和 marker 共用这一稳定 global vocabulary，候选范围由 route grammar 在每个位置收窄。
随机初始化只读取 codec 声明的 semantic feature dimension，并使用 backbone embedding 作为
device reference，不要求 backend 暴露虚构的 codebook tensor。新建的 semantic embedding、
input/output adapter 和 acoustic decoder 一律使用 FP32 参数存储；frozen backbone 可以保持
BF16，forward 计算精度由 trainer autocast 控制。semantic head 与 acoustic decoder 的输入在
模块边界显式转成对应参数 dtype，使训练外的 callback/generation 也遵守同一契约。
codec features 在 acoustic decoder 路径转换到 decoder device/dtype。frame mask 在进入 codec
前把 `-1` code padding 替换为安全值，adapter 后再清除无效位置。

## Acoustic decoder

- `HiddenConditionAdapter` 是 backbone 与 decoder 的唯一 condition 适配边界。它把 backbone hidden
  dimension 映射到 artifact `condition_dim`，属于 `ACOUSTIC_DECODER` 参数组；teacher-forcing 与
  autoregressive generation 都不能绕过它。
- flow decoder 沿 frame 轴做 self-attention；condition 与 timestep embedding 产生逐层 FiLM
  scale、shift 和 residual gate。frame mask 同时约束 attention、decoder 输出与最终 sampled
  features，padding frame 固定为零。
- REPA 启用时，`feature_projection` 把 `feature_layer` 的表示映射到 teacher feature 维度；
  未启用时不注册 projector。
- RVQ decoder 在 frame 间并行、在 codebook 轴自回归。训练和 sampling 先打包有效 frame，
  只让有效 frame 进入 Qwen decoder/head，再 scatter 回原 batch 形状；padding logits/code 为零且
  不消耗 sampling RNG，每个 batch row 必须至少有一个有效 frame。各 codebook 有独立
  embedding/head，sampling 在 codebook 轴复用 Qwen
  KV cache。decoder 自身冻结未使用的 token embedding，以及最后一个 codebook 的
  embedding/projection；最后一级只输出 logits，不会再作为下一 codebook 的输入。该结构约束由
  decoder 单独维护，optimizer 和 performance provider 都沿用同一参数边界。flow/RVQ model 都以
  `sample_acoustic_features()` 向评估入口返回 codec acoustic features；RVQ 的离散采样单独由
  `sample_acoustic_codes()` 表达。
- Runtime codec 固定 codebook 输入、feature dimension 与 waveform decode；model 不任意切取
  codebooks。

联合初始化只支持 frame-aligned SAC artifact。Flow 严格迁移 decoder state 与 feature normalization；
RVQ 只接收 `codebook_ar` generator。route、decoder topology、REPA 和 acoustic backend metadata 不匹配时
直接失败；semantic vocab/embedding 不参与联合初始化校验，因为该路径的输入是 hidden state 而不是 codes。
旧 Flow/RVQ checkpoint 没有 `acoustic_condition.*` 参数，strict resume 不做兼容填充。

## Generation 边界

`generate_tokens()` 与 `generate_audio_condition()` 是 `TokenModel` 的公开原语；flow/RVQ 的
`generate_audio_features()` 在其上采样对应 acoustic representation，并以结构化结果返回
sequence、padded features 与每行有效 frame count。通用 cache、stop state、allowed IDs 和
frame condition 的 `generate_sequence()` 循环位于私有
`model/_generation.py`，只通过有类型的 `generation_step()` 驱动模型。
`generate_full_codec_sequence()` 按 audio tokenizer 分派：frame-aligned
`FlattenedAudioTokenizer` 使用 codebook-block 状态机，首码本决定 frame count 并约束后续等长
payload；fixed-length `BiCodecAudioTokenizer` 使用 `audio_route.output.streams` 选择 grammar：
reuse route 生成 `codec, semantic_marker, semantic..., end`，generate route 生成
`codec, global_marker, global..., semantic_marker, semantic..., end`。service 不复制 marker、
range 或 block-length 规则。

route 的 prompt 属于调用前已序列化的 token context，model 只生成固定的 output streams；model 不
从 target labels 推断 prompt 边界，也不允许 request 临时切换 route。`SpeechToSpeechModule` 保存
规范化 route metadata 和 PEFT config metadata 到 checkpoint，并在恢复时严格比较当前 runtime/model
配置；启用相应能力时缺失 metadata 或配置不匹配都会直接失败。

具体模型不跨文件调用 `_generate()` 或 `_acoustic_features()`。KV cache 只属于一次调用；
首步编码完整 semantic-token prompt，并在有 `audio_input_positions` 时只对 source audio payload
运行 input tower；后续只输入新 token，不再次处理 source audio。frame span lookup 是非持久 buffer，
token-only audio decode 和 acoustic feature generation 都复用该 buffer 统计帧数，避免在
generation service 中重复调用 tokenizer。condition 在设备侧累计并一次展开。
