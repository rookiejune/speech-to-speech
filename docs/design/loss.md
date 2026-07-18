# loss

组合训练 objective，消费 `ModelBatch` 和模型公开接口，产出含标量 `loss` 的 mapping。
position 语义见 [总览 §2.4](../model-design.md)。

## 对外能力

- `Objective[ModelT]`：统一描述 model/objective 配对的泛型 `nn.Module` 契约，确保 objective
  的参数和子模块参与设备迁移、checkpoint 与 DDP。
- `TokenObjective`：只组合 text/audio token CE；`FlowObjective`：组合 token CE、
  acoustic flow matching 和可选 REPA；`RVQObjective`：组合 token CE 与 acoustic RVQ CE。
  三者的 `forward(batch, model)` 都返回含标量总损失的 `Outputs`，直接满足 Lightning
  训练契约。
- `TokenLoss`：按 batch task 的 target modality 在对应局部词表上计算 CE，每行必须至少包含一个
  非 `-100` target；causal shift 在此完成，只把有效 predictor hidden states 交给
  `model.token_logits(hidden, modality)`，text/audio head 不做跨模态 softmax 竞争。
- `AcousticFlowLoss`：frame-mask 的 velocity objective，只在有效 acoustic frame 上计算；
  启用 REPA 时通过 `forward_with_features()` 复用同一次 DiT 前向。
- `CausalAcousticLoss`：对每个 RVQ codebook 计算 masked CE，再在 codebook 维等权平均；
  acoustic padding ID 不进入 decoder embedding 或 loss。
- `WavLMTeacher`：按 boolean frame mask 在线解码 target semantic/acoustic codes，以 16 kHz
  waveform 运行冻结 WavLM，取得配置层的 hidden states 并插值、写回原有效 frame 位置。
- `RepaLoss`：把选定 DiT block 的逐帧表示投影到 WavLM hidden dimension，与
  stop-gradient teacher features 计算 masked cosine distance。
- `types.LossItem` / `types.Outputs`：上层日志与训练消费的稳定结构。
- `types.loss_items()`：按 token、flow matching、REPA、RVQ 的稳定顺序遍历实际存在的
  分项，供 callback 和实验 summary 复用。

## Objective 组合

三个组合入口共享 token forward：

```python
prompt = batch.acoustic_prompt
hidden_states = model.token_hidden_states(
    batch.input_ids,
    attention_mask=batch.attention_mask,
    acoustic_prompt_codes=None if prompt is None else prompt["codes"],
    acoustic_prompt_positions=None if prompt is None else prompt["token_positions"],
    acoustic_prompt_mask=batch.acoustic_prompt_mask,
)
token = self.token(
    hidden_states,
    batch.token_labels,
    batch.tasks[0].target_modality,
    model.token_logits,
)
result = {"loss": token.loss.mean(), "token": token}
```

存在独立 acoustic target codes 时，flow 与 RVQ 入口再执行各自分支：

```python
target_data = batch.acoustic_target

# FlowObjective
condition = model.target_frame_condition(
    hidden_states,
    target_data["token_positions"],
)
target = model.acoustic_target_latent(target_data["codes"])
acoustic = self.flow_matching(
    model.acoustic_decoder,
    condition,
    target,
    batch.acoustic_target_mask,
    self.flow_runtime,
)

# RVQObjective
teacher_forced_codes = target_data["codes"].masked_fill(
    ~batch.acoustic_target_mask[..., None],
    0,
)
logits = model.acoustic_logits(
    hidden_states,
    target_data["token_positions"],
    teacher_forced_codes,
)
rvq = self.rvq(logits, target_data["codes"], batch.acoustic_target_mask)
```

所有 batch 都计算 token CE。是否增加 acoustic objective 只由
`batch.acoustic_target is not None` 决定，不通过 task modality 猜测 codec
representation，也不通过模式布尔开关表达组合。结构化 target fields 不完整时直接报错。

`TokenObjective` 不要求 model 提供 acoustic 能力。`FlowObjective` 固定组合 token CE 与
flow matching；传入包含正数 `weight` 和 `teacher` 的 `RepaConfig` 时显式加入 REPA。
`RVQObjective` 固定组合 token CE 与 codebook causal CE。训练入口显式选择 model/objective
配对，不在 objective 内按具体模型类型猜组合。

REPA 只属于 flow 组合。teacher 显式接收 `acoustic_target["semantic_codes"]`、
`acoustic_target["codes"]` 和 `acoustic_target_mask`；dataset 不绑定 WavLM 型号、层号或
teacher features。oracle 等 diagnostic 使用独立 objective/入口。

## 边界

- `TokenObjective`、`FlowObjective` 和 `RVQObjective` 只依赖结构化 Protocol 的
  `layout`、`token_hidden_states()`、`token_logits(hidden, modality)`、`target_frame_condition()`、
  `acoustic_decoder` 等公开能力，不依赖具体模型类。
- target position 表示 token 自身位置 `p`；causal predictor shift `p - 1` 由 model 的
  `target_frame_condition()` 统一处理，objective 不重复偏移。
- `SpeechToSpeechModule` 通过泛型 `Objective` 保留 model/objective 的配对类型，不在训练循环中
  cast。
- flow runtime 等 objective 资源在 `FlowObjective` 构造时显式传入，不通过
  `model.runtime` 向下读取。
- 子 objective 在 `__init__` 中构造完毕，forward 不挂载新 submodule。
- flow matching、RVQ CE 和 REPA 在非线性 loss 计算前把无效 frame 替换为安全值，并只归约
  boolean mask 选中的 frame；padding 位置的 NaN/Inf 不参与 forward，也不产生梯度。
- `causal_lm.py` 只实现离散 acoustic RVQ objective，不读取 model/runtime 或重复 condition
  对齐；其稳定输出键是 `rvq`。
- REPA teacher 始终保持 eval/frozen；teacher features detach，梯度只进入 DiT 与 student
  projector。
