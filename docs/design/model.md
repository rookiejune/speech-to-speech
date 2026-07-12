# model

semantic backbone、embedding 注入和 acoustic decoder 的组装。position 语义与 condition 契约的权威定义见 [总览 §2.4](../model-design.md)。

## 对外能力

- `base.SemanticModel`：公共层，负责 runtime/backbone/embedding 加载、semantic forward/logits、semantic generation、acoustic prompt 注入和 frame hidden 对齐。
- `acoustic.SpeechToSpeechFlowModel`：flow matching 组合入口，附加 `acoustic_target_latent()`、`sample_acoustic()`、`generate_audio()`。
- `acoustic.SpeechToSpeechRVQModel`：RVQ 组合入口，附加 `acoustic_logits()` 与 codebook 自回归的 `sample_acoustic()`。
- `protocol.BaseModel` / `protocol.FlowMatching` / `protocol.CausalLM`：上层（loss、pl_module）依赖的结构化协议。

## 模型接口

```python
def forward(
    self,
    input_ids: Tensor,
    *,
    attention_mask: Tensor | None = None,
    acoustic_input_ids: Tensor | None = None,
    acoustic_input_positions: Tensor | None = None,
    acoustic_input_mask: Tensor | None = None,
    output_hidden_states: bool = False,
) -> CausalLMOutputWithPast: ...
```

- 模型只返回 logits 和可选 hidden states，不接收 labels，也不计算 loss；objective 由外部 `Loss` 组合。
- `logits` 是拼接后的 global token 分布；内部保留 text/audio 两个 output head，但对外只承诺 global `logits`。
- text 输入和输出分别通过 HF 的 `get_input_embeddings()` / `get_output_embeddings()` 获取；当模型 embedding 行数大于 tokenizer vocabulary 时，text logits 只覆盖 layout 的 text block。
- acoustic decoder 需要 backbone 表示时传 `output_hidden_states=True` 并使用 `output.hidden_states[-1]`。
- condition 接口 `target_frame_condition()` / `target_frame_label_condition()` 统一消费 token 自身位置 `p`；causal shift `p - 1` 只在 model 内部处理（见总览 §2.4）。
- acoustic decoder 的 condition 形态、采样结果和 codec 衔接取决于具体 decoder，不统一其 output 类型。公开接口不暴露具体 Qwen layer、codec adapter 或 decoder layer。

## Embedding 和条件注入

输入分成三路：

```text
text ids
    └── backbone text embedding

semantic audio ids
    ├── codec semantic codebook 初始化的 embedding（RoPE + mean-pool merge）
    └── semantic adapter

acoustic prompt
    └── codec.acoustic_codes_to_features()
        └── acoustic adapter
```

注入形式：

```python
semantic_feature = semantic_base + semantic_gate * semantic_shift
acoustic_feature = acoustic_gate * acoustic_feature
inputs_embeds = semantic_feature + acoustic_feature
```

gate 初始化为 0，避免在训练初期破坏原始 backbone。实现状态：acoustic 路径的 gate 已实现（`base.py` 的 `acoustic_gate`）；semantic 路径的 `semantic_gate * semantic_shift` 在 anytrain 的 `Embedding` 中没有实现，目前 semantic embedding 直接输出 `semantic_base`，该项作为待实现的注入形式保留。

acoustic ids 先经过 codec 得到 frame-level feature，再按 `acoustic_input_positions` 使用 `embedding/audio.py` 的 merge 规则合成为 BPE-level feature，加到对应 input embedding 上。
codec feature 进入模型时统一转换到 backbone embedding 的 device/dtype；codec wrapper
不绑定训练模型的精度或设备策略。source acoustic prompt 与 flow target 共用这条边界。

## Acoustic decoder

- flow decoder 遵循 velocity-model 协议，模型只冻结其输入输出：

```python
class AcousticDecoder(Protocol):
    latent_dim: int

    def __call__(self, x_t: Tensor, t: Tensor, *, condition: Tensor) -> Tensor: ...
```

- acoustic representation 由 Runtime codec 固定，不属于 batch 或 model 的任意切片配置。batch 保存完整离散 acoustic codes，`acoustic_codes_to_features()` 的输入、latent feature dimension 和 `decode_features()` 必须属于同一 codec contract。
- target 对齐：target BPE hidden 按 `bpe_spans` repeat_interleave 得到 frame condition `[B, F, H]`；target 完整、有序 acoustic codebooks 经 codec 得到 latent `[B, F, D]`。
- RVQ decoder 是 frame 并行、codebook 自回归的离散预测器，teacher forcing 返回每个 codebook 的 logits。

## 边界

- 模型构造接收 runtime snapshot（`runtime_snapshot` 参数），内部不反复依赖全局 singleton，以便用 fake codec、fake tokenizer 和 tiny backbone 做 contract test。
- semantic generation 使用 backbone KV cache：首步编码完整多模态 prompt，后续只输入新 token。cache 不保存在 model 实例上，也不通过公开 API 暴露具体 backbone 类型。
- audio generation 在每个 semantic token 被采样时在线收集预测该 token 的 hidden，并按 BPE span 展开 frame condition；不在生成完成后追加一次全序列 forward。
- source acoustic prompt 在首步进入 KV cache，因此在整个生成过程中持续有效。
- batch acoustic generation 的目标契约见总览 §6；当前 model generation 原语只接收单个无 padding prompt，由上层 service 逐 request 调用，属于实现欠账。
