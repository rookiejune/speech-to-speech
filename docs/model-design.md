# Speech-to-Speech 模型设计

本文定义 `speech_to_speech` 的模型边界、数据契约和实现路线。目标是先冻结接口，使 semantic 模型和 acoustic decoder 可以独立演进。

## 1. 总体结构

```text
Raw Sample
    │
    ▼
Task / Prompt Builder
    │
    ▼
ModelBatch
    ├── semantic input / labels
    ├── acoustic prompt
    └── acoustic target
         │
         ▼
SpeechToSpeechFlowModel
    ├── Input Embedding
    ├── Semantic Backbone
    ├── Text / Audio Logits
    └── Acoustic Decoder
         ├── Causal RVQ Decoder
         └── Flow Matching Decoder
```

设计原则：

1. semantic stream 是主序列。
2. acoustic stream 只为已经可见的 speech semantic span 提供 side-channel。
3. response 的 acoustic target 不作为 semantic backbone 的输入。
4. semantic backbone 和 acoustic decoder 通过稳定的 hidden-state contract 连接。
5. `SpeechToSpeechFlowModel` 和 `SpeechToSpeechRVQModel` 是显式的模型组装入口。

## 2. 数据契约

### 2.1 Speech

```python
@dataclass
class Speech:
    semantic_ids: Tensor                    # [L, M_semantic]
    acoustic_ids: Tensor | None             # [L, M_acoustic]
    text_ids: Tensor
    language: Language

    @cached_property
    def bpe_ids(self) -> Tensor: ...

    @cached_property
    def bpe_spans(self) -> Tensor: ...
```

`semantic_ids` 和 `acoustic_ids` 共享同一个 frame 轴，但可以有不同的 codebook 数量：`semantic_ids` 为 `[L, M_semantic]`，`acoustic_ids` 为 `[L, M_acoustic]`。

`bpe_ids` 是 audio tokenizer 产生的本地 BPE ids。`bpe_spans` 与 `bpe_ids` 等长，记录每个 BPE token 覆盖的 semantic frame 数量，例如 `[2, 1, 2]` 表示三个 BPE token 分别覆盖 2、1、2 个 frame。

`audio_tokenizer.expand(bpe_ids)` 返回 frame-level semantic units，形状为 `[T, K_semantic]`；每个 frame unit 包含全部 semantic codebooks，不能只取第一个 codebook。生成或解码 batch 时，调用方负责检查各行的 `T` 是否一致并进行 `stack` 或 padding；tokenizer 只负责单条序列的 token 到 frame 展开。

`global_ids` 不属于 `Speech` 的缓存属性。datamodule 在把 prompt 拼接进 backbone 输入时，使用 layout 将 `bpe_ids` 转成 global ids。

single-codebook codec 仍使用 `[L, 1]` 的 `acoustic_ids`；`None` 只表示当前任务不需要 acoustic side-channel，不表示 codec 没有 acoustic code。

### 2.2 ModelBatch

当前 `CausalBatch` 的 acoustic 字段无法表达 acoustic frame 与 semantic token 的对齐关系。建议逐步替换为：

```python
ACOUSTIC_PAD_ID = -1


@dataclass
class ModelBatch:
    input_ids: Tensor
    labels: Tensor
    acoustic_input_ids: Tensor | None
    acoustic_input_positions: Tensor | None
    acoustic_labels: Tensor | None
    acoustic_label_positions: Tensor | None
    tasks: list[Task]

    @cached_property
    def attention_mask(self) -> Tensor:
        return self.input_ids != runtime().pad_token_id

    @cached_property
    def acoustic_input_mask(self) -> Tensor | None:
        if self.acoustic_input_ids is None:
            return None
        return (self.acoustic_input_ids != ACOUSTIC_PAD_ID).all(dim=-1)

    @cached_property
    def acoustic_label_mask(self) -> Tensor | None:
        if self.acoustic_labels is None:
            return None
        return (self.acoustic_labels != ACOUSTIC_PAD_ID).all(dim=-1)
```

acoustic input 和 labels 只需要在字段语义上明确区分，不额外包装成简单 dataclass：

- `acoustic_input_*`：输入条件，只覆盖 prompt 中已经可见的 audio semantic span；`spans` 描述 acoustic frame 与 semantic BPE token 的对齐。
- `acoustic_labels`：输出监督，只用于训练 acoustic decoder，不能注入 semantic backbone。
- `acoustic_label_positions`：输出 frame 对应的 semantic label position。它和 `acoustic_input_positions` 语义不同：前者供 acoustic decoder 从 target label 或对应的 response hidden 读取 condition，后者供 semantic backbone 注入 source acoustic prompt。
- backbone hidden 的 condition position 是 `acoustic_label_positions - 1`，因为 causal LM 的 position `p - 1` 预测 `labels[p]`。

mask 不作为独立字段存储，而是由 padding 值派生并缓存。`input_ids` 使用 `runtime().pad_token_id` padding；semantic `labels` 按 Transformers causal LM 契约使用 `-100` padding，使其不参与 loss。acoustic IDs 是未加 global offset 的 codec 局部 ID，使用不会与合法 code 冲突的固定负数 `ACOUSTIC_PAD_ID = -1` padding；`acoustic_input_positions` 对每个 acoustic frame 给出其对应的完整输入序列位置，padding 使用 `-1`。

acoustic IDs 传给 codec 或 embedding 前，必须先通过派生 mask 把 `-1` 替换成合法的占位 ID，并在得到 feature 后重新应用 mask。一个 batch 的 acoustic 字段必须整体存在或整体不存在：不支持 batch 内部分 row 为 `None`。由 task sampler/bucketing 保证同一 batch 的 source/target modality 一致，避免 DDP 中不同 rank 走不同参数路径。position tensor 的有效值必须小于对应 `input_ids` 的序列长度。

这些 mask 使用 `cached_property`，因此 `ModelBatch` 在完成 padding 和 device transfer 后视为不可变；不能在首次访问 mask 后原地替换其中的 tensor。

## 3. 模型接口

```python
class SpeechToSpeechFlowModel(nn.Module):
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

模型只负责返回 logits 和可选 hidden states，不接收 labels，也不计算 loss。semantic、flow matching 等 objective 由外部 `Loss` 组合；objective 通过 `layout`、`target_frame_condition()` 和 `acoustic_decoder` 这些公开接口读取监督所需的表示。`OutputsLogger` 只消费 `LossItem` 中的 batch-level loss/detail，不依赖模型内部 head。

semantic 模型沿用 Transformers causal LM 的 logits 和 generation 契约，但不接收 `labels`，也不计算 loss。`logits` 表示拼接后的 global token 分布；外部 objective 根据 `layout` 对 text/audio token 分别统计 CE，`-100` padding 不参与计算。acoustic decoder 需要 backbone 表示时传入 `output_hidden_states=True` 并使用 `output.hidden_states[-1]`。KV cache 和 semantic generation 继续复用标准 causal LM 能力。

模型内部可以保留 text/audio 两个 output head，但对外只承诺拼接后的 global `logits`；loss 不依赖具体 head 或 backbone 类型。

acoustic decoder 的 hidden 对齐支持 teacher-forcing 和 generation 两条路径。训练时使用数据层提供的 `acoustic_label_positions`；使用 backbone hidden 时减一得到 causal prediction position。推理时没有 labels，则使用 `logits[:, :-1].argmax(-1)` 得到预测的 next-token，再根据预测 audio BPE 的 tokenizer spans 展开 hidden。两条路径最终都得到 frame-level condition，不在 generation 代码中重复实现对齐逻辑。

acoustic decoder 的 condition 形态、采样结果和 codec 衔接取决于具体 decoder，P0 不统一其 output 类型。模型的公开接口不应暴露具体 Qwen layer、codec adapter 或 decoder layer，具体实现放在 model 内部 helper 中。

batch generation 的目标契约是标准批量自回归：变长响应通过 padding 和 attention mask 处理，每行独立跟踪 eoa；frame 轴同样以 padding + frame mask 贯穿 flow sampling 和 `decode_features()`，不要求 batch 内各行 frame 数相等。当前实现是过渡状态：`pl_module` 的 generation 按行循环调用单样本路径，`generate_audio` 对 batch 内不等长的 frame 展开直接报错；这些限制属于实现欠账，不属于契约。

## 4. Embedding 和条件注入

输入分成三路：

```text
text ids
    └── backbone text embedding

semantic audio ids
    ├── codec semantic codebook 初始化的 embedding
    └── semantic adapter

acoustic prompt
    └── codec.acoustic_codes_to_features()
        └── acoustic adapter
```

建议注入形式：

```python
semantic_feature = semantic_base + semantic_gate * semantic_shift
acoustic_feature = acoustic_gate * acoustic_feature
inputs_embeds = semantic_feature + acoustic_feature
```

其中 gate 初始化为 0，避免在训练初期破坏原始 backbone。

实现状态：acoustic 路径的 gate 已在 `model/base.py` 中实现（`acoustic_gate`，零初始化）。semantic 路径的 `semantic_gate * semantic_shift` 在 anytrain 的 `Embedding` 中没有实现，目前 semantic embedding 直接输出 `semantic_base`；该项作为待实现的注入形式保留在此。两路 embedding 分开计算：`input_ids` 先得到 BPE-level semantic embedding；acoustic ids 先经过 codec 得到 frame-level feature，再根据数据层提供的 `acoustic_input_positions` 使用 `model/embedding/audio.py` 的 merge 规则合成为 BPE-level feature，最后加到对应的 input embedding 上。model 不搜索 source audio token，也不判断 acoustic prompt 边界。

acoustic target 的 codebook 数量是 model/runtime 的固定配置，不属于 batch。batch 保存完整的离散 acoustic codes，decoder 在统一配置下选择前 `k` 个 codebook 转为连续 latent。该配置必须同时决定 `acoustic_codes_to_features()` 的输入、latent feature dimension 和 `decode_features()` 使用的 codec decoder；如果底层 codec 只提供固定 codebook decoder，应配置 decoder 名称而不是在模型中假设任意 `k` 都可用。

当前 `model/embedding/audio.py` 已经提供 codec codebook 初始化和 BPE merge 的雏形，可以沿用；但 tokenizer 的 `expand_with_counts()` 需要先修正实现错误。

## 5. 推荐的第一版模型

第一版采用：

```text
semantic backbone: causal language model
acoustic decoder: flow matching
```

理由：semantic token 数量较少，适合 autoregressive；acoustic frame 数量较多，使用 flow matching 可避免 RVQ 多码本的长序列自回归开销。

训练路径：

```text
semantic prompt + acoustic prompt
        │
        ▼
semantic backbone
        │
        ├── semantic cross entropy
        │
        └── hidden states
                │
                ▼
        BPE expand / frame alignment
                │
                ▼
        flow matching acoustic loss
```

保留可替换 decoder 的 velocity-model 协议；training objective 放在 `loss/flow_matching.py`：

```python
class AcousticDecoder(Protocol):
    latent_dim: int

    def __call__(self, x_t: Tensor, t: Tensor, *, condition: Tensor) -> Tensor: ...
```

模型只冻结 velocity model 的输入输出；不同 decoder 的 objective 和采样策略由外部 loss/generation 模块组合，不让 model 承担 loss 聚合。

acoustic target 的具体对齐为：

```text
target BPE hidden [B, BPE, H]
        │ repeat_interleave(target.bpe_spans)
        ▼
target frame condition [B, F, H]

target acoustic codes [B, F, N]
        │ 选择配置中的前 k 个 codebook，再经 codec
        ▼
target acoustic latent [B, F, D]
```

## 6. 任务定义

任务与输出类型必须保持一致：

| Task | 输入 | semantic target | acoustic target |
| --- | --- | --- | --- |
| ASR | audio | text | no |
| S2TT | audio | text | no |
| S2ST | audio | audio semantic | yes |
| TTS | text | audio semantic | yes |
| T2ST | text | audio semantic | yes |
| TEXT_AR | none | text | no |
| AUDIO_AR | none/audio context | audio semantic | optional |

当前 `S2ST` 的 `target` 写成了 `"text"`，会导致它实际走 S2TT 路径；实现模型前必须修正为 `target = "audio"`。

## 7. Loss 组织

## 8. 模型代码拆分

公共层位于 `model/base.py` 的 `SemanticModel`，只负责 runtime/backbone/embedding 加载、semantic forward/logits、semantic generation、acoustic prompt 注入，以及 frame hidden 对齐等公共逻辑。

具体 acoustic decoder 分开实现：

- `model/acoustic/flow.py`：Flow Matching decoder 和 `SpeechToSpeechFlowModel`；
- `model/acoustic/rvq.py`：RVQ decoder 的独立实现入口，待 RVQ 的 codebook 生成契约冻结后补充。

通过 `SpeechToSpeechFlowModel` 或 `SpeechToSpeechRVQModel` 显式选择 acoustic 组合类，不把两种 decoder 的训练、采样和 decode 逻辑重新塞回公共层。

`Loss` 负责组合 objective，不负责实现模型内部逻辑：

```python
def forward(self, batch: ModelBatch, model) -> Outputs:
    need_acoustic = batch.acoustic_labels is not None
    output = model(
        batch.input_ids,
        attention_mask=batch.attention_mask,
        acoustic_input_ids=batch.acoustic_input_ids,
        acoustic_input_positions=batch.acoustic_input_positions,
        acoustic_input_mask=batch.acoustic_input_mask,
        output_hidden_states=need_acoustic,
    )
    semantic = semantic_loss(output.logits, batch.labels, model.layout)

    acoustic = None
    if batch.acoustic_labels is not None:
        acoustic = acoustic_flow_loss(
            model.acoustic_decoder,
            model.target_frame_condition(
                output.hidden_states[-1],
                batch.acoustic_label_positions - 1,
            ),
            model.acoustic_target_latent(batch.acoustic_labels),
            batch.acoustic_target_mask,
            model.runtime.flow_matching,
        )

    return combine(semantic, acoustic)
```

P1 只实现 semantic CE；P2 再加入 acoustic flow matching loss。REPA 等表示学习目标后续作为独立 objective 加入。

## 9. Runtime 边界

runtime 负责提供已经加载好的资源：

- text tokenizer；
- audio tokenizer；
- codec；
- backbone；
- layout；
- special tokens；
- flow matching runtime。

模型构造时建议接收 runtime snapshot：

```python
model = SpeechToSpeechFlowModel(
    runtime=runtime(),
    config=config,
)
```

模型内部不应反复依赖全局 singleton。这样可以使用 fake codec、fake tokenizer 和 tiny backbone 做 contract test。

## 10. 实施阶段

### P0：冻结契约

已完成

### P1：最小闭环

P1 的核心模块已经具备，但尚未完成可运行闭环和验收。当前状态为：`P1-core` 基本完成，`P1-closure` 未完成。

- Native audio tokenizer。
- semantic embedding 和 acoustic prompt 聚合。
- Qwen backbone。
- semantic CE。
- TTS/S2ST semantic generation。
- codec waveform decode。

P1 closure 仍需完成：

- 完善 Lightning training step。
- fake runtime、tiny backbone 和 fake codec 的 contract test。
- TTS/S2ST semantic CE training smoke test：完成至少一次 forward、backward 和 optimizer step，loss 有限且参数发生更新。
- S2ST semantic generation 的可运行测试。
- `semantic ids → codec decode → waveform` 端到端验证。
- 将上述结果记录到实验结果或 sample logging 中。

P1 closure 不要求单 batch overfit。单 batch overfit 作为独立的 P1 diagnostic，在训练路径跑通后单独验证模型是否具备基本的拟合能力；它不阻塞最小代码闭环验收。

### P2：加入 acoustic decoder

P2 的正式训练依赖 P1 closure；本轮先跳过 P1 closure，完成 P2.0–P2.4 的实现和 fake contract 验证，不执行正式数据训练。

P2 分为以下几个可独立验收的闭环：

#### P2.0：target 对齐契约

- `TaskBase.sample()` 直接传递 target 的缓存 `bpe_spans`。
- 为 target acoustic frame 生成 `acoustic_label_positions`。
- `acoustic_input_positions` 只负责 source acoustic prompt；`acoustic_label_positions` 负责 target semantic labels 及其对应的 response hidden。
- 用 fake tokenizer/backbone 验证 BPE 到 frame 的位置和 mask 完全对齐。

对于 response `[<boa>, bpe_0, ..., bpe_n, <eoa>]`，第一个 target BPE 对应的 hidden position 是 `len(input_ids)`；target frame position 是 BPE position 按 `bpe_spans` 展开后的结果。

#### P2.1：frame-level acoustic condition

- 从 hidden states 和 `acoustic_label_positions - 1` gather frame condition。
- 将 target acoustic codes 按固定 decoder/model 配置选择 codebook 并转换为连续 latent。
- 验证 condition、latent、frame mask 的 `[B, F]` 对齐。

#### P2.2：flow matching training loss

- 实现 frame-level 条件 velocity model，第一版使用 time embedding + FiLM/MLP。
- 使用 `ContinuousVelocityObjective` 生成 noisy latent 和目标 velocity。
- loss 只计算有效 acoustic frame。
- 先在单一 audio-target task batch 上 overfit，确认 semantic CE 和 acoustic flow loss 都能下降。

#### P2.3：acoustic sampling

- 对 semantic response 做 generation。
- 有 labels 时使用真实 target BPE；无 labels 时使用 `logits[:, :-1].argmax(-1)`。
- 根据预测 audio BPE 得到 spans，展开 hidden 到 frame condition。
- 使用 ODE sampler 生成 acoustic latent。

#### P2.4：waveform 和端到端路径

- 将预测 audio BPE 展开为 frame-level semantic ids。
- 调用 `codec.decode_features(semantic_ids, acoustic_features)`。
- `semantic_ids` 保持 `[B, T, K_semantic]`，`acoustic_features` 保持 `[B, T, D]`，只要求 frame 轴 `T` 对齐。
- 覆盖 TTS、S2ST 的 teacher-forcing、semantic generation、acoustic generation 和 waveform decode。

### P3：统一训练和推理

- Lightning training step。
- semantic generation。
- acoustic generation。
- batch generation。
- sample logging。
- 端到端 waveform 测试。

当前实现状态：

- `Loss.forward()` 返回含标量 `loss` 的 mapping，直接满足 Lightning 的训练契约。
- semantic batch generation 按样本裁剪 prompt，支持 batch 内不同 prompt 长度。
- acoustic generation 支持有 target labels 时的 teacher forcing，以及无 labels 时的 semantic generation。
- sample logger 在 logger 支持 `add_audio` 时记录 waveform，否则记录生成 token。
- fake runtime contract tests 覆盖 semantic batch generation、acoustic hidden position 对齐和 waveform decode 前的 token/frame 对齐。

## 11. 验收标准

1. tokenizer 的 `encode → expand → merge` 长度和内容符合预期。
2. `bpe_spans` 能完整覆盖 semantic frame，且与 `bpe_ids` 等长。
3. acoustic prompt 不会注入 response target。
4. 没有 acoustic target 的 ASR/S2TT batch 可以正常计算 loss。
5. 有 acoustic target 的 TTS/S2ST batch 可以正常计算 semantic 和 acoustic loss。
6. `acoustic_input_positions` 和 `acoustic_label_positions` 的职责、shape 和 mask 始终对齐。
7. batch 内 acoustic 字段整体存在或整体不存在，不出现部分 row 为 `None`。
8. 无 labels 时可以从 logits argmax 得到 audio BPE，并完成 hidden 到 frame 的展开。
9. acoustic decoder 可以替换为 fake decoder 完成上层测试。
10. semantic ids 和 acoustic features 可以被 codec 正常 decode 为 waveform。
11. S2ST 的 target 确实是 audio，而不是 text。
