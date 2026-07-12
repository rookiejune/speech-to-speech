# loss

组合训练 objective，消费 `ModelBatch` 和模型公开接口，产出含标量 `loss` 的 mapping。position 语义见 [总览 §2.4](../model-design.md)。

## 对外能力

- `Loss`：objective 组合入口。`forward(batch, model)` 返回 `Outputs`，其中 `loss` 是标量总损失，直接满足 Lightning 训练契约；分项以 `LossItem` 形式携带 per-sample loss 和 details，供 `OutputsLogger` 按 task 聚合。
- `SemanticLoss`：按 layout 对 text/audio token 分别统计 CE，`-100` 不参与计算；shift 在此完成（`logits[:, :-1]` 对 `labels[:, 1:]`）。
- `AcousticFlowLoss`：frame-mask 的 velocity objective，只在有效 acoustic frame 上计算。
- `types.LossItem` / `types.Outputs`：上层日志与训练消费的稳定结构。

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
    # audio-target batch：
    #   condition = model.target_frame_condition(
    #       output.hidden_states[-1], batch.acoustic_label_positions)
    #   model 内部把 token 位置 p 转为 predictor 位置 p - 1。
```

正常训练的 objective 组合固定为：所有 batch 计算 semantic CE；audio-target batch 额外计算模型对应的 acoustic objective。Loss 根据已验证的 task target modality 选择路径，不通过布尔开关表达非法组合。

oracle、REPA 等 diagnostic 或消融使用独立 objective/入口，不进入正式 `Loss` 的模式矩阵。audio-target batch 必须携带完整 acoustic target 字段；缺失直接报错，不静默跳过。

## 边界

- `Loss` 不实现模型内部逻辑，只通过结构化 Protocol 的 `layout`、`target_frame_condition()`、`acoustic_decoder` 等公开能力读取监督所需表示，不依赖具体模型类。
- flow runtime 等 objective 资源在 Loss 构造时显式传入，不通过 `model.runtime` 向下读取。
- 子 objective 在 `__init__` 中构造完毕，forward 不挂载新 submodule。
- `causal_lm.py` 是 RVQ 离散 acoustic objective 的占位入口，未实现前不暴露正式配置开关。
- REPA 等表示学习目标后续作为独立 objective 加入。
