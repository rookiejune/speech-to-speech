# DiT

## 对外职责

`DiT` 子模块提供 acoustic decoder 的 transformer 实现。它是 `model` 内部的子模块，外部调用方优先通过 `Orchestrator` 使用，不直接依赖 layer 级实现。

对外能力：

- 接收 noised acoustic state `x_t`。
- 接收 Qwen3 semantic hidden states 作为条件。
- 接收 diffusion/flow timestep。
- 接收 acoustic condition，支持后续 CFG 设计。
- 通过上层 `model.acoustic.attention_mode` 显式选择 causal 或 bidirectional self-attention。
- 在 wrapper 内部对 time、Qwen hidden 和 acoustic condition 三路条件做可配置归一化并融合。
- 返回 `BaseModelOutputWithPast`，其中 `last_hidden_state` 是 acoustic decoder 输出。

## 模块边界

- `model.py` 定义 DiT wrapper，负责条件融合、mask/cache 处理和层堆叠。
- `module.py` 定义 DiT layer、AdaLN 和局部计算单元。

## 开发边界

- Do: 让 `model.py` 承担对外 forward 契约；Don't: 让外部模块直接调用 `DiTLayer` 或 `AdaLN`。
- Do: 保持 transformer cache、mask、position ids 的接口和 Qwen3 兼容；Don't: 在 data module 或 pl_module 里复制 DiT mask 逻辑。
- Do: offline full-sequence acoustic flow 用 `model.acoustic.attention_mode=bidirectional` 做对照；streaming、diagonal 或 causal-window 实验保留 `causal` 对照。
- Do: 把 CFG、time embedding、condition 融合策略收敛在 DiT wrapper；Don't: 把这些声学生成规则散落到训练循环。
