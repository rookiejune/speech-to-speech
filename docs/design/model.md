# model

组装 token backbone、multimodal embedding 与 acoustic decoder。position 语义见
[总览 §2.4](../model-design.md)。

## 对外能力

- `base.Model`：接收显式 runtime，提供 text/semantic-audio embedding、token
  logits、route-aware structured token generation 与 frame condition 对齐原语。
- `audio_input.AudioInputTower`：把 source audio payload 的 semantic embedding 按显式位置
  编码为 backbone hidden states；支持 `none`、同长度 `mlp` 与非 causal `transformer`，只属于
  input path，不参与 semantic-audio output head 或 acoustic generation。
- `audio_output.AudioOutputAdapter`：把 backbone hidden states 投影到 semantic-audio feature
  space；`none` / `linear` / `mlp` 为无序列混合特例，`transformer` 为带独立 KV cache 的因果栈。
- `acoustic.flow.FlowModel`：在基础模型上组合 SAC `FMFeatureGenerator`（`DiTDecoder` core），提供
  flow target、sampling 和 `generate_audio_features()`；S2S 不再平行维护 DiT 实现。
- `acoustic.rvq.RVQModel`：组合 SAC `AcousticRVQDecoder`，提供 teacher-forced
  codebook logits、sampling 和 `generate_audio_features()`；类型从 SAC 导入，不经 S2S 再导出。
- `acoustic.condition.HiddenConditionAdapter`：以 `LayerNorm + Linear` 把对齐后的 backbone hidden state 映射到
  SAC generator 的 condition space；训练和 generation 共用该入口。
- `acoustic.flow.AcousticFlow`：薄包装，持有 `FMFeatureGenerator` 与 S2S `flow_matching` runtime
  做 ODE sampling，并保留 feature mean/std 归一化缓冲。
- `loss.protocol.TokenObjectiveModel` / `FlowObjectiveModel` / `RVQObjectiveModel`：objective
  所依赖的训练能力。
- `generation.protocol.TokenGenerator` / `AcousticFeatureGeneration`：generation service
  所依赖的基础契约与可选 acoustic runtime 能力；`TokenGenerator` 含 `generate_tokens()` 与
  `generation_step()`（mixed AR 用）；`AcousticFeatureGenerator` 组合两者供训练侧静态 typing，
  `TextEvaluationModel` 组合 token generation 与 reference scoring。
- `runtime.protocol.TokenModelRuntime` / `model.protocol.FlowModelRuntime`：token 与 flow
  model 各自消费的 runtime 资源边界。
- `AdapterType`：semantic input/output adapter 的 `linear|mlp` 字符串枚举；`None` 表示输入输出
  dimension 相同的 identity adapter。
- `AudioInputAdapterType` / `AudioInputAdapterConfig`：source audio tower 的 `none|mlp|transformer`
  配置；`transformer` 使用同长度、非 causal 的 encoder layer。
- `AudioOutputAdapterType` / `AudioOutputAdapterConfig`：semantic-audio output adapter 的
  `none|linear|mlp|transformer` 配置；后三者字段 `layers/heads/ffn_ratio/dropout` 仅
  `transformer` 使用。
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
def selected_logits(...) -> tuple[Tensor, object | None]: ...
def generation_step(...) -> GenerationStepResult: ...
def generate_tokens(...) -> Tensor: ...
```

- `forward()` 返回 global text+audio logits，不接收 labels 或计算 loss。
- `forward()` 支持 HF backbone 的 cache/position 参数；sampling、stop 和 output-head selection
  参数不进入该通用接口。
- `audio_input_positions` 是 `[batch, frames]` 的完整序列位置，`-1` 只用于 batch padding。它只
  指向 source audio payload token；BOA/EOA、target audio token、generated token 和 BiCodec
  reference `audio_context` 都不经过 `AudioInputTower`。
- `generation_step()` 返回 `GenerationStepResult`：最后位置目标 modality / 显式 token 子集
  logits，以及 backbone `past_key_values` 与 audio output adapter 的独立 `audio_output_past`。
  PARALLEL / INTERLEAVED 的切换规则不属于本接口，由 `generation.mixed` 持有状态机，并反复
  调用本步进原语。
- 训练先用 `token_hidden_states()` 取得完整表示，再由 objective 按
  `prediction_modality.supervised_modalities()` 选有效 predictor rows，并对每个 `Modality`
  调用 `token_logits()`；CE 只构造对应 text 或 audio 局部词表 logits，不为 prompt、padding
  或未监督模态构造大词表 logits。未传 modality 的通用 `forward()` 仍返回 global text+audio
  logits。model 不接受 `PredictionModality` 作为 head 参数。
- backbone 直接调用 HF causal LM 的 `base_model`；自带 text LM head 不会先计算再丢弃。
- backbone body 由 runtime 的默认 `backbone_readout` 和按模态
  `backbone_readouts` 选择实际 hidden tensor；默认使用 `last_hidden_state`。Kimi-Audio
  使用 text=`last_hidden_state[0]`、audio=`last_hidden_state[1]`，homogeneous task batch
  按 prediction modality 路由。配置了按模态 readout 时不接受 mixed prediction batch；不支持
  `cache_position` 的 remote-code backbone 通过 `backbone_supports_cache_position=false` 省略该参数。
- text/audio output head 分别对对应 block embedding weight 做 tied linear，layout offset 只负责
  恢复 global token ID；不保留 LM head bias。
- `selected_logits()` 只计算调用方给出的候选 global IDs；BiCodec route 的 grouped CE 用它避免为
  每个 marker/codebook 位置计算完整 audio vocabulary。
- generation 按 modality 只计算最后一个位置的目标 head；text 屏蔽 PAD/BOS，audio 屏蔽 BOA/MASK。
- text/audio vocabulary head 位于私有 `_head.py` mixin；参数注册在 `token_embedding.*`、
  `audio_output_adapter.*` 与 backbone ownership path 下。
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
`None` 只在输入输出 dimension 相同时合法。当 audio embedding 输出维已经等于 backbone
`hidden_size`（例如 Stable FSQ rank-1 embedding）时，默认的 `linear` 输入/输出 adapter 自动退化为
identity/`none`，不再额外投影。`toy=None` 时模型使用 `runtime.backbone`；非空时由
`ToyConfig` 构造随机 tiny Qwen，runtime 仍负责 tokenizer、codec、layout、special IDs 与 flow
sampler。完整 Qwen 架构的随机初始化属于 `runtime.backbone_initialization=random`，不通过 toy
参数近似。Hydra `model` preset 与这些共享字段一一对应；overfit/train root schema 继承
`model.Config`，再按 `model/acoustic=none|flow|rvq` 增加对应的精确 acoustic 字段。

`audio_output_adapter` 是因果族 semantic-audio output adapter：`none` / `linear` / `mlp` 是无序列
混合的特例；`transformer` 是带独立 KV cache 的因果 self-attention。teacher-forcing 对完整
backbone hidden 一次前向；generation 增量喂入新 token hidden，audio adapter 与 backbone cache
始终保留相同的完整 batch 轴，结束行由 generation mask 屏蔽。
pointwise 特例忽略 cache。训练 CE 在 adapter 之后对 audio 行做 tied linear；frame condition 仍取
adapter 前的 backbone hidden。

`audio_input_adapter` 默认 `type=mlp`，让 source audio payload 的 semantic embedding
逐帧经过 gated MLP 投影到 backbone hidden dimension，避免无变换地覆盖 LLM embedding space。
显式选择 `transformer` 时，先做输入投影，再用
同长度、causal 的 Transformer encoder 跨 source frames 建立上下文。两种 tower 都保持 frame
数量不变，并在 overlay 到 `inputs_embeds` 前清零 padding。训练和完整 prompt 的首步会传入显式
`audio_input_positions`；启用 KV cache 后后续 token 只走 backbone，不重复运行 source tower。
该配置不会改变 token generation 契约，也不会替换 Flow/RVQ
`HiddenConditionAdapter`。

`lora` 直接持有 `peft.LoraConfig | None`，项目不再维护本地 LoRA config、layer 或注入 facade。
正式 train 默认组合 LoRA preset（`init_lora_weights=pissa`）与
`callback/parameter_policy=lora`。
选择该组合后，model 把该 config 直接传给 PEFT
`inject_adapter_in_model()`；rank、alpha、dropout、target modules、初始化方法与 PEFT 后续支持的
字段都沿用官方命名和校验。PiSSA 保证 A/B 满秩，以便 `optim.name=muon` 时 anytrain
自动走 LoRA-Muon。混合精度 backbone 注入后使用 PEFT 的 mixed-precision cast 规则。
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
side channel。route 只声明 `acoustic` / `semantic`；fixed-length speaker/style 含义来自
`AcousticLayout.FIXED_LENGTH`，不是单独的 stream 枚举。`reuse_prompt_acoustic` 只输出
semantic 并复用 prompt acoustic；`generate_acoustic` 同时输出 acoustic 与 semantic。两条
route 使用同一套稳定 vocabulary，route 只改变训练时的 output groups 与推理解码时的
decode stream ownership，不按 request 动态改变模型 head 或 token generation 规则。
无 reference 的 `generate_acoustic` 不自带 speaker ID；多 speaker 训练若没有额外条件或
latent sampling，acoustic（speaker）预测可能偏向数据中的主导 speaker，这属于模型条件设计而
不是 codec 序列化问题。

底层 acoustic decoder 的所有权在 `semantic-acoustic-codec`：S2S 的 Flow/RVQ model 只负责
从 backbone hidden state 取 frame-aligned condition，经 `HiddenConditionAdapter` 映射后送入 SAC 的
DiT/DiT+REPA/RVQ decoder。`model.acoustic.init_artifact` 可在 composition 边界加载 SAC generator；model
构造器不执行 artifact I/O。S2S 不维护 acoustic-only codec oracle，也不复制底层 decoder 实现。

## Embedding

```text
global input_ids
    -> anytrain.module.idspace.Embedding
         text: owned nn.Embedding（backbone 经 EmbeddingView 引用）
         audio: codec/random semantic audio embedding
         adapters["audio"]: pointwise 投影到 backbone hidden
    -> AudioInputTower（非因果；仅 overlay source audio_input_positions）
    -> backbone.base_model(inputs_embeds=...)
```

`idspace.Embedding` 只做 block 路由与输入侧 adapter；跨 block 输出 head 由 S2S 读取对应
`embeddings[block].weight` 做 tied linear，不再使用 backbone LM head。text block 的真实
`nn.Embedding` 只挂在 `token_embedding` 下；backbone 通过非 Module 的 `EmbeddingView`
（`model/_helper.py`）引用同一张表，避免双重 ownership。audio adapter 经 `CastOutput` 在边界把 FP32 输出
cast 到 backbone embedding dtype。`token_embedding.*` 为唯一参数路径；旧
`semantic_audio_embedding.*` / `semantic_audio_adapter.*` checkpoint key 与现 schema 不兼容，
strict resume 显式失败。

Native/BPE semantic tokenizers 使用 codec codebook 初始化；完整 codec sequence tokenizer
通常使用随机初始化，因为它的 vocab 同时包含多 codebook offset tokens、BiCodec
semantic/acoustic ranges 与 codec/stream/end markers。BiCodec 的 semantic payload、各
fixed-length acoustic slot 和 marker 共用这一稳定 layout vocabulary；训练 objective
根据 route 解释各位置的监督 groups，普通 generation 仍统一建模，不把这些 group 下推为
隐式结构约束。
随机初始化只读取 codec 声明的 semantic feature dimension，并使用 backbone embedding 作为
device reference，不要求 backend 暴露虚构的 codebook tensor。

当 codec `semantic_feature_dim == 1` 且暴露 `fsq_levels`（Stable Codec）时，audio embedding
改走 rank-1 affine：tokenizer 仍使用 packed product id，embedding 侧按 codec levels unpack，
`e = Σ_j (b_j + q̃_j w_j)`，**默认输出维对齐 backbone `hidden_size`**；marker / BOA/EOA/MASK
保留自由行。此时默认 linear input/output adapter 因维已对齐而退化为 identity/`none`，tied
logits 仍读 materialize 后的 `.weight`。不把 FSQ 展开进 tokenizer 序列。codec 上的
`semantic_feature_dim == 1` 只表示 FSQ 内在标量维，不是 LLM 接口维。

新建的 semantic embedding、input/output adapter 和 acoustic decoder 一律使用 FP32 参数存储；
frozen backbone 可以保持 BF16，forward 计算精度由 trainer autocast 控制。semantic head 与
acoustic decoder 的输入在模块边界显式转成对应参数 dtype，使训练外的 callback/generation 也遵守
同一契约。codec features 在 acoustic decoder 路径转换到 decoder device/dtype。frame mask 在进入
codec 前把 `-1` code padding 替换为安全值，adapter 后再清除无效位置。

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

`generate_tokens()` 与 `generate_audio_condition()` 是 `Model` 的公开原语；flow/RVQ 的
`generate_audio_features()` 在其上采样对应 acoustic representation，并以结构化结果返回
sequence、padded features 与每行有效 frame count。通用 cache、sampling、stop state 和
frame condition 的 `generate_sequence()` 循环位于私有
`model/_generation.py`，只通过有类型的 `generation_step()` 驱动模型。训练和 ordinary AUDIO
token generation 统一建模，推理原语只调用
`generate_tokens(generation_modality=AUDIO, stop=EOA)`，不在 token generation 阶段根据
flattened frame codec 或 BiCodec route 强制 marker/range/block-length 结构。
codec-specific 的 marker、range、block-length 与 route stream ownership 只由推理层解析和
decode 使用；非法 generated codec span 按行 warning 并跳过 audio decode，model 不重试或
补齐结构。产品推理可以在 model 外显式使用 codec-specific 策略，但不能改变训练或普通
generation 的模型契约。mixed prediction 的 TEXT/AUDIO 交替与 force-BOA 规则在
`generation.mixed`，不进入该通用循环，也不扩展
`generate_tokens(generation_modality=...)`。

route 的 prompt 属于调用前已序列化的 token context，model 只生成固定的 output streams；model 不
从 target labels 推断 prompt 边界，也不允许 request 临时切换 route。`SpeechToSpeechModule` 保存
规范化 route metadata 和 PEFT config metadata 到 checkpoint，并在恢复时严格比较当前 runtime/model
配置；启用相应能力时缺失 metadata 或配置不匹配都会直接失败。

具体模型不跨文件调用 `_generate()` 或 `_acoustic_features()`。KV cache 只属于一次调用；
首步编码完整 semantic-token prompt，并在有 `audio_input_positions` 时只对 source audio payload
运行 input tower；后续只输入新 token，不再次处理 source audio。frame span lookup 是非持久 buffer，
token-only audio decode 和 acoustic feature generation 都复用该 buffer 统计帧数，避免在
generation service 中重复调用 tokenizer。condition 在设备侧累计并一次展开。
