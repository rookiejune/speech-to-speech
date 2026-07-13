# model

semantic backbone、embedding 注入和 acoustic decoder 的组装。position 语义与 condition 契约的权威定义见 [总览 §2.4](../model-design.md)。

## 对外能力

- `AcousticFlow`：组合 `AcousticFlowDecoder` 与 flow runtime，统一正常训练和 codec oracle
  的 acoustic sampling；condition 与 target 的来源由上层组合决定。

- `base.SemanticModel`：公共层，负责 runtime/backbone/embedding 加载、semantic forward/logits、semantic generation、acoustic prompt 注入和 frame hidden 对齐。
- `acoustic.SpeechToSpeechFlowModel`：flow matching 组合入口，附加 `acoustic_target_latent()`、`sample_acoustic()`、`generate_audio()`。
- `acoustic.SpeechToSpeechRVQModel`：RVQ 组合入口，附加 `acoustic_logits()`、codebook 自回归的 `sample_acoustic()` 与 `generate_audio()`。
- `protocol.FlowMatching` / `protocol.RVQMatching`：对应 loss 依赖的训练能力；
  `protocol.AcousticGeneration`：两个 acoustic decoder 共享的推理能力；`protocol.FlowModel` /
  `protocol.RVQModel` 组合 Lightning module 所需的训练与推理契约。

## 模型接口

`AdapterType` 是 embedding、output 和 acoustic prompt adapter 配置的字符串枚举；当前
支持 `LINEAR` 和 `MLP`，`None` 表示 identity 且要求输入输出维度一致。

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
- `logits` 是拼接后的 global token 分布；内部保留 text/semantic-audio 两个 output head，但对外只承诺 global `logits`。
- semantic forward 直接调用 HF causal LM 的 `base_model` 取得 hidden states 与 cache，
  不执行并丢弃 backbone 自带的 text LM head；text/audio logits 只由本模块计算一次。
- generation 已知 allowed token IDs 时，内部只计算对应 modality 的 output head，并从
  output weight 中选择该 ID 子集，只为最后一个 token 返回子集 logits；这条私有优化不改变
  公开 `forward()` 的 global logits 契约。
- text 输入和输出分别通过 HF 的 `get_input_embeddings()` / `get_output_embeddings()` 获取；当模型 embedding 行数大于 tokenizer vocabulary 时，text logits 只覆盖 layout 的 text block。
- acoustic decoder 需要 backbone 表示时传 `output_hidden_states=True` 并使用 `output.hidden_states[-1]`。
- condition 接口 `target_frame_condition()` / `target_frame_label_condition()` 统一消费 token
  自身位置 `p`；causal shift `p - 1` 只在 model 内部处理（见总览 §2.4）。前者属于
  acoustic objective 训练协议，后者是具体模型提供的 teacher-forced/oracle 能力。
- acoustic decoder 的 condition 形态、采样结果和 codec 衔接取决于具体 decoder，不统一其 output 类型。公开接口不暴露具体 Qwen layer、codec adapter 或 decoder layer。

## Embedding 和条件注入

输入分成三路：

```text
text ids
    └── backbone text embedding

semantic audio ids
    ├── codec semantic codebook 初始化的 embedding（RoPE + mean-pool merge）
    └── semantic audio adapter

acoustic prompt
    └── codec.acoustic_codes_to_features()
        └── acoustic prompt adapter
```

注入形式：

```python
semantic_feature = semantic_base + semantic_gate * semantic_shift
acoustic_feature = acoustic_prompt_gate * acoustic_feature
inputs_embeds = semantic_feature + acoustic_feature
```

gate 初始化为 0，避免在训练初期破坏原始 backbone。实现状态：acoustic 路径的 gate
已实现（`base.py` 的 `acoustic_prompt_gate`）；semantic 路径的
`semantic_gate * semantic_shift` 在 anytrain 的 `Embedding` 中没有实现，目前 semantic
embedding 直接输出 `semantic_base`，该项作为待实现的注入形式保留。

acoustic ids 先经过 codec 得到 frame-level feature，再按 `acoustic_input_positions` 使用
`embedding/audio.py` 的张量化 RoPE + grouped mean 规则合成为 BPE-level feature，同时返回
occupied token mask；adapter 后使用该 mask 清除未占用位置的 bias，再加到对应 input
embedding 上。
codec feature 进入模型时统一转换到 backbone embedding 的 device/dtype；codec wrapper
不绑定训练模型的精度或设备策略。source acoustic prompt 与 flow target 共用这条边界。

## Acoustic decoder

- flow decoder 是沿 acoustic frame 轴做 self-attention 的 DiT。Qwen frame condition 不作为
  cross-attention 序列，而是在每个 block 中与 timestep embedding 相加后生成逐帧 FiLM
  scale、shift 和 residual gate。正弦 frame position embedding 提供顺序信息；frame mask
  同时屏蔽 attention key 和 padding 输出。
- decoder 遵循 velocity-model 协议，模型只冻结其输入输出：

```python
class AcousticDecoder(Protocol):
    latent_dim: int

    def __call__(
        self,
        x_t: Tensor,
        t: Tensor,
        *,
        condition: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor: ...
```

- `forward_with_features()` 在一次 DiT 前向中同时返回 velocity 与指定 block 的 frame
  representation；8 层默认取第 4 个 block，经 student projector 映射到 WavLM-base 的
  768 维 hidden。该入口只供 flow + REPA 训练组合使用；sampling 继续消费普通 velocity
  接口且不执行 projector。

- acoustic representation 由 Runtime codec 固定，不属于 batch 或 model 的任意切片配置。batch 保存完整离散 acoustic codes，`acoustic_codes_to_features()` 的输入、latent feature dimension 和 `decode_features()` 必须属于同一 codec contract。
- target 对齐：target BPE hidden 按 `bpe_spans` repeat_interleave 得到 frame condition `[B, F, H]`；target 完整、有序 acoustic codebooks 经 codec 得到 latent `[B, F, D]`。
- RVQ decoder 是独立、随机初始化的 8 层 Qwen3 decoder。frame 之间并行，每个 frame
  内以 `[condition + BOS_q, condition + embedding(code_{q-1})]` 构造 codebook causal
  序列；Qwen 自带 token embedding 不参与计算，各 codebook 保持独立 embedding 和 output
  head。teacher forcing 返回每个 codebook 的 logits，sampling 按 codebook 顺序生成。
- flow DiT 与 RVQ Qwen decoder 共享 `acoustic_decoder_dim/layers/heads/ffn_ratio` 配置，默认
  都是 8 层；两者共享 semantic backbone 和 frame condition，但 acoustic objective 与
  codec output representation 不同。

## 边界

- 模型构造接收 runtime snapshot（`runtime_snapshot` 参数），内部不反复依赖全局 singleton，以便用 fake codec、fake tokenizer 和 tiny backbone 做 contract test。
- semantic generation 使用 backbone KV cache：首步编码完整多模态 prompt，后续只输入新 token。cache 不保存在 model 实例上，也不通过公开 API 暴露具体 backbone 类型。
- audio generation 在每个 semantic token 被采样时在线收集预测该 token 的 hidden，并按 BPE span 展开 frame condition；不在生成完成后追加一次全序列 forward。
- source acoustic prompt 在首步进入 KV cache，因此在整个生成过程中持续有效。
- batch acoustic generation 的目标契约见总览 §6；当前 model generation 原语只接收单个无 padding prompt，由上层 service 逐 request 调用，属于实现欠账。
