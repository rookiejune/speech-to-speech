# loss

组合训练 objective，消费 `ModelBatch` 和模型公开接口，产出含标量 `loss` 的 mapping。position 语义见 [总览 §2.4](../model-design.md)。

## 对外能力

- `Objective[ModelT]`：统一描述 model/objective 配对的泛型 `nn.Module` 契约，确保 objective
  的参数和子模块参与设备迁移、checkpoint 与 DDP。
- `SemanticObjective`：semantic-only 组合入口；`Loss`：flow objective 组合入口；
  `RVQLoss`：离散 acoustic objective 组合入口。三者的 `forward(batch, model)` 都返回含标量
  总损失的 `Outputs`，直接满足 Lightning 训练契约。
- `SemanticLoss`：按 layout 对 text/audio token 分别统计 CE，`-100` 不参与计算；shift 在此完成（`logits[:, :-1]` 对 `labels[:, 1:]`）。
- `AcousticFlowLoss`：frame-mask 的 velocity objective，只在有效 acoustic frame 上计算；
  启用 REPA 时通过 `forward_with_features()` 复用同一次 DiT 前向。
- `CausalAcousticLoss`：对每个 RVQ codebook 计算 masked CE，再在 codebook 维等权平均；
  acoustic padding ID 不进入 decoder embedding 或 loss。
- `WavLMTeacher`：在线解码完整 target codec codes，以 16 kHz waveform 运行冻结的
  `microsoft/wavlm-base`，取 `hidden_states[9]` 并插值到有效 acoustic frame 轴。
- `RepaLoss`：把 DiT 第 4 个 block 的逐帧表示投影到 WavLM hidden dimension，与
  stop-gradient teacher feature 计算 masked cosine distance。
- `types.LossItem` / `types.Outputs`：上层日志与训练消费的稳定结构。
- `types.loss_items()`：按 semantic、flow matching、REPA、causal LM 的稳定顺序遍历实际
  存在的分项，供 callback 和实验 summary 复用。

## Objective 组合

```python
def forward(self, batch: ModelBatch, model) -> Outputs:
    output = model(
        batch.input_ids,
        attention_mask=batch.attention_mask,
        acoustic_input_ids=batch.acoustic_input_ids,
        acoustic_input_positions=batch.acoustic_input_positions,
        acoustic_input_mask=batch.acoustic_input_mask,
        output_hidden_states=need_acoustic,
    )
    semantic = self.semantic(output.logits, batch.labels)
    # 存在 acoustic target fields 的 batch：
    #   condition = model.target_frame_condition(
    #       output.hidden_states[-1], batch.acoustic_label_positions)
    #   model 内部把 token 位置 p 转为 predictor 位置 p - 1。
```

正常训练的 objective 组合固定为：所有 batch 计算 semantic CE；存在 acoustic target fields
时额外计算模型对应的 acoustic objective。Loss 根据结构化 target fields 选择路径，不通过
task modality 猜测 codec representation，也不通过布尔开关表达组合。

`SemanticObjective` 只计算 semantic CE，不要求 model 提供 acoustic objective 接口。
`Loss` 固定组合 semantic CE 与 flow matching；传入包含正数 `weight` 和 `teacher` 的
`RepaConfig` 时显式加入 REPA。`RVQLoss` 固定组合 semantic CE 与 codebook causal CE。
训练入口显式选择 model/objective 对，不在 objective 内按具体模型类型猜组合。REPA 只属于
flow 组合；teacher 通过结构化接口注入，dataset 不绑定 WavLM 型号或层号。

oracle 等 diagnostic 使用独立 objective/入口。REPA 通过数值权重表达目标组合，不新增
模式布尔开关；未传权重时不计算。启用 acoustic objective 的 batch 必须携带完整 target 字段；
缺失直接报错，不静默跳过。

## 边界

- `Loss` 不实现模型内部逻辑，只通过结构化 Protocol 的 `layout`、`target_frame_condition()`、`acoustic_decoder` 等公开能力读取监督所需表示，不依赖具体模型类。
- `SpeechToSpeech` 通过泛型 `Objective` 保留 model/objective 的配对类型，不在训练循环中 cast。
- flow runtime 等 objective 资源在 Loss 构造时显式传入，不通过 `model.runtime` 向下读取。
- 子 objective 在 `__init__` 中构造完毕，forward 不挂载新 submodule。
- `causal_lm.py` 只实现离散 acoustic objective，不读取 model/runtime 或重复 condition 对齐。
- REPA teacher 始终保持 eval/frozen；teacher feature detach，梯度只进入 DiT 与 student
  projector。
