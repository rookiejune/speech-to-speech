# model

组装 token backbone、multimodal embedding 与 acoustic decoder。position 语义见
[总览 §2.4](../model-design.md)。

## 对外能力

- `base.TokenModel`：接收显式 runtime，提供 text/semantic-audio embedding、token
  logits、acoustic prompt 注入、frame condition 对齐与 token generation 原语。
- `acoustic.FlowModel`：在基础模型上组合 `AcousticFlow`/`AcousticDiT`，提供
  flow target、sampling 和 `generate_audio_features()`。
- `acoustic.RVQModel`：组合 `AcousticRVQDecoder`，提供 teacher-forced
  codebook logits、sampling 和 `generate_audio_features()`。
- `loss.protocol.TokenObjectiveModel` / `FlowObjectiveModel` / `RVQObjectiveModel`：objective
  所依赖的训练能力。
- `generation.protocol.TokenGenerator` / `AcousticFeatureGeneration`：generation service
  所依赖的基础契约与可选 acoustic runtime 能力；`AcousticFeatureGenerator` 组合两者供训练侧静态
  typing，`TextEvaluationModel` 组合 token generation 与 reference scoring。
- `runtime.protocol.TokenModelRuntime` / `model.protocol.FlowModelRuntime`：token 与 flow
  model 各自消费的 runtime 资源边界。
- `AdapterType`：基础 `Config` 三个 adapter 字段的 `linear|mlp` 字符串枚举；`None` 表示输入输出
  dimension 相同的 identity adapter。
- `ToyConfig` / `create_toy_backbone()`：构造随机初始化的一层或少层 Qwen backbone，用于 CPU
  model/data 契约测试；词表大小来自 runtime layout，但不读取 `runtime.backbone`。
- `AcousticType`、`DecoderConfig`、`FlowRepaConfig`：组合入口的严格配置结构。

## Token 接口

```python
def forward(
    input_ids: Tensor,
    *,
    attention_mask: Tensor | None = None,
    acoustic_prompt_codes: Tensor | None = None,
    acoustic_prompt_positions: Tensor | None = None,
    acoustic_prompt_mask: Tensor | None = None,
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
def generation_step(...) -> CausalLMOutputWithPast: ...
def generate_tokens(...) -> Tensor: ...
```

- `forward()` 返回 global text+audio logits，不接收 labels 或计算 loss。
- `forward()` 支持 HF backbone 的 cache/position 参数；sampling、stop 和 output-head selection
  参数不进入该通用接口。
- `generation_step()` 只返回最后位置的目标 modality 或显式 token 子集 logits，并把 cache
  状态传给 backbone。
- 训练先用 `token_hidden_states()` 取得完整表示，再由 objective 只选有效 predictor rows，并用
  task 的 target modality 调用 `token_logits()`；CE 只构造对应 text 或 audio 局部词表 logits，
  不为 prompt、padding 或另一模态构造大词表 logits。未传 modality 的通用 `forward()` 仍返回
  global text+audio logits。
- backbone 直接调用 HF causal LM 的 `base_model`；自带 text LM head 不会先计算再丢弃。
- text/audio output head 分别产生 local logits，layout offset 只负责恢复 global token ID。
- generation 按 modality 只计算最后一个位置的目标 head；text 屏蔽 PAD/BOS，audio 屏蔽 BOA。
- text/audio vocabulary head 位于私有 `_head.py` mixin；参数仍只注册在 `TokenModel` 的原始
  embedding/adapter/backbone ownership path 下。
- `target_frame_condition()` 与 `target_frame_label_condition()` 都接收 token 自身位置 `p`；
  causal shift `p - 1` 只在 model 内部发生。

## 配置边界

`model.Config` 只包含基础模型真正消费的设置：

- `semantic_audio_adapter`
- `semantic_audio_output_adapter`
- `acoustic_prompt_adapter`
- `toy`

三个字段都使用公开 `AdapterType`；`linear` 是默认值，`mlp` 使用 gated SiLU adapter，`None`
只在输入输出 dimension 相同时合法。`toy=None` 时模型使用 `runtime.backbone`；非空时由
`ToyConfig` 构造随机 tiny Qwen，runtime 仍负责 tokenizer、codec、layout、special IDs 与 flow
sampler。Hydra `model` preset 与这些字段一一对应，overfit 与 codec-oracle root schema 直接复用
`model.Config`。

decoder 使用独立 `DecoderConfig(hidden_dim, layers, heads, ffn_ratio)`。flow 可额外接收
`FlowRepaConfig(feature_dim, student_layer)`；RVQ 可额外接收初始化 decoder 各 acoustic
codebook 的 `codebook_embeddings`，但没有 REPA 参数。Hydra 使用
`model/acoustic=none|flow|rvq`，`none` 只训练 semantic audio token，flow preset 独占 teacher
与 student REPA 配置。ODE sampling 由 `runtime.Config.flow_*` 拥有。没有独立 acoustic
codebooks 的 unified-token codec 必须使用 `model/acoustic=none`；有独立 acoustic codebook 的
codec 也可以显式选择 `none` 作为 token-only baseline。入口不根据 codec 静默覆盖用户选择。
固定 flow 的 codec-oracle 使用 `codec_oracle.decoder`，不读取 REPA 配置。

## Embedding

```text
text_token_ids
    -> backbone text embedding

semantic-audio token IDs
    -> semantic codec codebook initialized embedding
    -> semantic audio adapter

acoustic_prompt_codes
    -> codec.acoustic_codes_to_features()
    -> grouped frame-to-token merge at acoustic_prompt_positions
    -> acoustic prompt adapter + zero-initialized gate
```

codec features 在模型边界转换到 backbone embedding 的 device/dtype。frame mask 在进入 codec
前把 `-1` code padding 替换为安全值，adapter 后再清除无效位置。source acoustic prompt 与
flow target 复用 `acoustic_code_features()`，子类不调用基类私有转换函数。

## Acoustic decoder

- flow decoder 沿 frame 轴做 self-attention；condition 与 timestep embedding 产生逐层 FiLM
  scale、shift 和 residual gate。frame mask 同时约束 attention、decoder 输出与最终 sampled
  features，padding frame 固定为零。
- REPA 启用时，`repa_projection` 把 `repa_student_layer` 的表示映射到 teacher feature 维度；
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

## Generation 边界

`generate_tokens()` 与 `generate_audio_condition()` 是 `TokenModel` 的公开原语；flow/RVQ 的
`generate_audio_features()` 在其上采样对应 acoustic representation，并以结构化结果返回
sequence、padded features 与每行有效 frame count。通用 cache、stop state、allowed IDs 和
frame condition 的 `generate_sequence()` 循环位于私有
`model/_generation.py`，只通过有类型的 `generation_step()` 驱动模型。

具体模型不跨文件调用 `_generate()` 或 `_acoustic_features()`。KV cache 只属于一次调用；
首步注入 source acoustic prompt，后续只输入新 token。frame span lookup 是非持久 buffer，
condition 在设备侧累计并一次展开。
