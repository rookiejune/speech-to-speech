# speech_to_speech Package

## 对外职责

这个包提供 speech-to-speech 训练的顶层 Python 接口。跨模块开发时先看本文件和目标子模块的 `AGENTS.md`，再进入具体代码。

模块边界：

- `config.py` 提供顶层配置结构，顶层配置使用 `datamodule.dataset_factory`、`datamodule.dataloader` 和 `datamodule.lba` 分层表达数据入口和加载策略；模型配置使用 `model.backbone`、`model.token_space` 和 `model.acoustic` 分层表达 Qwen 加载/适配、新增 audio token 训练策略和 acoustic decoder 边界；callback 参数通过 `trainer.callbacks` 分层表达。
- `dataset.py` 提供训练数据集加载入口，当前只支持从 `zhuyin.datasets.wmt19_tts.wmt19_tts_longcat()` 取处理好的对象。
- `types/` 提供跨模块共享的轻量类型，按 `bpe`、`datamodule`、`model` 分层放置，并由 `speech_to_speech.types` 统一 re-export 常用公共契约。
- `runtime.py` 提供运行期资源加载入口，例如 Qwen3 tokenizer、LongCat BPE tokenizer、Qwen3/LongCat 共享 `IdSpace`、LongCat codec。
- `datamodule/` 负责定义任务 schema、task example 转换函数和模型可消费的 batch。
- `model/` 负责 Qwen3、LongCat token space、LM head、DiT 等模型结构和 acoustic condition 生成接口。
- `pl_module/` 负责训练框架适配、日志、优化器、checkpoint 等训练编排。

## 开发边界

- Do: 跨模块只依赖本模块明确暴露的类型、函数和 batch/model 契约；Don't: 读取其他模块的私有 helper 或中间状态。
- Do: 新增公共接口时先更新对应模块文档，说明调用方应该传什么、得到什么；Don't: 让调用方通过读实现猜用法。
- Do: 把业务规则放到所属模块的服务函数或清晰 helper；Don't: 在调用方复制另一个模块的内部规则。
