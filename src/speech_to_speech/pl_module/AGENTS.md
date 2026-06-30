# pl_module

## 对外职责

`pl_module` 负责训练框架适配。它把 data module 的 batch、model 的 forward/generate、优化器、日志和 checkpoint 串起来，但不拥有数据格式和模型结构规则。

对外能力：

- 提供 `SpeechToSpeechModule`，把 `Orchestrator` 包成 LightningModule。
- 定义训练 step、验证 step 和日志指标。
- 配置 optimizer、scheduler 和 precision 相关策略。
- 记录长时间训练需要的 loss、task weight、学习率等曲线。
- 保存和恢复 checkpoint。
- 提供 sample/generation logging callback，callback 是否启用由 trainer 配置决定。

## 开发边界

- Do: 在这里编排训练流程；Don't: 在这里解析 Anydataset 样本字段。
- Do: 调用 model 的公开 forward/generate 接口；Don't: 直接操作 Qwen3、LM head 或 DiT 内部层。
- Do: 从配置读取本次训练的任务权重；Don't: 在训练 step 里写死 autoregression/translation 比例。
- Do: optimizer/scheduler 优先复用 `anytrain.optim` 的 LLM Lightning 配置入口；Don't: 在本模块重复实现 Muon/AdamW 分组规则。
