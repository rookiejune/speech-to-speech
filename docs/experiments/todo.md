# Experiment TODO

## 下一步

- 用 `scripts/evaluate_free_running.py` 对 003 checkpoints 跑同一组 sample 评估，记录 generated waveform、EOA 命中、生成长度和人工听感备注。
- 视第一组结果决定是否追加 Qwen3-8B LoRA 短程容量检查。
- 明确第一轮正式训练是否以 source-to-target 为主方向。

## 后续可选

- 如果低权重 acoustic/FM 常驻只能维持词级翻译、仍不能带来句子级连贯性，再考虑 scheduled sampling、prefix dropout 或短段 free-running 训练。
- 如果双向训练伤害主方向，尝试降低 target-to-source 比例而不是移除双向任务。
