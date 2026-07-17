# model

组装 token backbone、multimodal embedding 与 acoustic decoder。position 语义见
[总览 §2.4](../model-design.md)。

## 对外能力

- `base.TokenModel`：接收显式 runtime，提供 text/semantic-audio embedding、token
  logits、acoustic prompt 注入、frame condition 对齐与 token generation 原语。
- `acoustic.SpeechToSpeechFlowModel`：在基础模型上组合 `AcousticFlow`/`AcousticDiT`，提供
  flow target、sampling 和 `generate_audio_features()`。
- `acoustic.SpeechToSpeechRVQModel`：组合 `AcousticRVQDecoder`，提供 teacher-forced
  codebook logits、sampling 和 `generate_audio_features()`。
- `loss.protocol.TokenObjectiveModel` / `FlowObjectiveModel` / `RVQObjectiveModel`：objective
  所依赖的训练能力。
- `generation.protocol.TokenGenerator` / `AcousticFeatureGenerator`：generation service
  所依赖的推理能力；`TextEvaluationModel` 组合 token generation 与 reference scoring。
- `model.protocol.TokenModelRuntime` / `FlowModelRuntime`：token 与 flow model 各自消费的
  runtime 资源边界。
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
) -> CausalLMOutputWithPast: ...

def token_hidden_states(...) -> Tensor: ...
def token_logits(hidden_state: Tensor) -> Tensor: ...
def generate_tokens(...) -> Tensor: ...
```

- `forward()` 返回 global text+audio logits，不接收 labels 或计算 loss。
- 训练先用 `token_hidden_states()` 取得完整表示，再由 objective 只选有效 predictor rows 调用
  `token_logits()`，避免为 prompt/padding 构造大词表 logits。
- backbone 直接调用 HF causal LM 的 `base_model`；自带 text LM head 不会先计算再丢弃。
- text/audio output head 分别产生 local logits，layout offset 只负责恢复 global token ID。
- generation 按 modality 只计算最后一个位置的目标 head；text 屏蔽 PAD/BOS，audio 屏蔽 BOA。
- `target_frame_condition()` 与 `target_frame_label_condition()` 都接收 token 自身位置 `p`；
  causal shift `p - 1` 只在 model 内部发生。

## 配置边界

`model.Config` 只包含基础模型真正消费的 adapter：

- `semantic_audio_adapter`
- `semantic_audio_output_adapter`
- `acoustic_prompt_adapter`

decoder 使用独立 `DecoderConfig(hidden_dim, layers, heads, ffn_ratio)`。flow 可额外接收
`FlowRepaConfig(feature_dim, student_layer)`；RVQ 构造函数没有 REPA 参数。Hydra 使用
`acoustic.type=flow|rvq`，flow preset 独占 `teacher_checkpoint`、`teacher_layer` 与
`student_layer`。

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
  scale、shift 和 residual gate。frame mask 同时约束 attention 与输出。
- REPA 启用时，`repa_projection` 把 `repa_student_layer` 的表示映射到 teacher feature 维度；
  未启用时不注册 projector。
- RVQ decoder 在 frame 间并行、在 codebook 轴自回归。各 codebook 有独立 embedding/head，
  sampling 在 codebook 轴复用 Qwen KV cache。flow/RVQ model 都以
  `sample_acoustic_features()` 向评估入口返回 codec acoustic features；RVQ 的离散采样单独由
  `sample_acoustic_codes()` 表达。
- Runtime codec 固定 codebook 输入、feature dimension 与 waveform decode；model 不任意切取
  codebooks。

## Generation 边界

`generate_tokens()` 与 `generate_audio_condition()` 是 `TokenModel` 的公开原语；flow/RVQ 的
`generate_audio_features()` 在其上采样对应 acoustic representation。通用 cache、stop state、
allowed IDs 和 frame condition 循环位于私有 `model/_generation.py`。

具体模型不跨文件调用 `_generate()` 或 `_acoustic_features()`。KV cache 只属于一次调用；
首步注入 source acoustic prompt，后续只输入新 token。frame span lookup 是非持久 buffer，
condition 在设备侧累计并一次展开。
