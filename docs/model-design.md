# Speech-to-Speech 设计总览

本文只维护跨模块结构、所有权和依赖方向。字段级契约、算法细节与配置规则由
[`docs/design/`](design/) 下的模块文档负责，避免同一设计在多处展开后漂移。

## 总体数据流

```text
anydataset raw sample
    -> datamodule.sample / parse / build
    -> datamodule.batch.ModelBatch
    -> pl_module.composition
    -> model + loss
    -> callback / validation

task.io.Request
    -> generation service
    -> model generation protocol
    -> generation.result.Result
```

MIMO 预训练是并列路径：领域契约和任务组合属于 `speech_to_speech.mimo`，dataset、collate
与 loader 属于 `speech_to_speech.datamodule.mimo`，模型和 Lightning module 分别属于
`speech_to_speech.model` 与 `speech_to_speech.pl_module`。

## 依赖方向

依赖只朝实际执行能力收敛：

```text
task / mimo contracts
        ↑
runtime protocols ← datamodule / model / generation
        ↑                 ↑
        └──── loss ← pl_module
                       ↑
                    callback
                       ↑
                    scripts
```

- `task` 拥有任务、prediction、source layout 和真实推理 `Request`，不依赖 generation。
- `mimo` 拥有双流 sample/batch/step 契约与任务语义，不依赖 datamodule。
- `runtime` 加载资源并暴露窄 capability；不包含 task、objective 或 Trainer 逻辑。
- `datamodule` 负责 raw/domain sample、训练 target、batch、collation 和 loader schedule。
- `model` 负责 token/acoustic topology、generation step 和 checkpoint architecture contract。
- `loss` 只依赖 batch 与 model protocol，不拥有模型组装。
- `pl_module` 组合 model、objective、optimizer 和 Lightning 生命周期。
- `callback` 观察训练状态或执行显式物化，不成为业务契约的所有者。
- `scripts` 只保留 Hydra 入口、入口特有校验和运行编排；可复用构造能力放在 `src`。

## 公共契约所有权

| 契约 | 权威模块 | 主要消费者 |
| --- | --- | --- |
| `Request` | `speech_to_speech.task.io` | datamodule bridge、generation、model protocol |
| `Result` | `speech_to_speech.generation.result` | generation service、evaluation、reporting |
| `AcousticGeneration` | `speech_to_speech.model.output` | acoustic model、generation decode |
| raw/domain sample | `speech_to_speech.datamodule.sample` | parser、builder、collator、materializer |
| targets | `speech_to_speech.datamodule.target` | builder、batch、loss |
| model/training batch | `speech_to_speech.datamodule.batch` | loader、pl_module、callback、loss |
| MIMO contract | `speech_to_speech.mimo.contract` | MIMO dataset、model、loss、pl_module |
| runtime config/resources | `speech_to_speech.runtime.config` / `core` | executable entries and all runtime consumers |
| codec capability | `speech_to_speech.runtime.codec_contract` | runtime、datamodule、model、generation |
| tokenizer capability | `speech_to_speech.runtime.tokenizer` | runtime、layout consumers |
| backbone capability | `speech_to_speech.runtime.backbone.contract` | runtime、model、MIMO |
| checkpoint contract | `speech_to_speech.model.contract` | model composition、Lightning checkpoint gate |
| shared training composition | `speech_to_speech.training.composition` | train / overfit entries |

包级 `__init__.py` 只提供稳定 facade。内部模块优先导入权威模块；只有 `loss` 为隔离可选音频训练
依赖保留显式懒加载。

## 跨模块不变量

- text/audio tokenizer ID、codec code 和 layout global token ID 是不同空间，不能靠命名或数值范围
  隐式互换。
- `ModelBatch.input_ids` 是 teacher-forcing 完整序列；真实推理只使用没有 target 的 `Request`。
- target acoustic position 记录 token 自身位置，causal condition 在 model 内读取前一位置 hidden；
  source audio input position 只用于 input embedding overlay，两者不能混用。
- codec capability 由 Protocol 和 metadata 校验决定，不由 codec 名称或单个同名属性猜测。
- `Runtime` 不是 `nn.Module`；可训练对象在 model 中只有一条注册所有权路径。
- checkpoint compatibility 由规范化 runtime/model topology 和 state schema 决定，不由 Hydra preset
  名称决定。
- sequence layout、prediction modality、loader step mode 和 parameter policy 都是显式配置轴；非法
  组合在入口解析或 composition 边界失败。

## 模块文档

- [datamodule](design/datamodule.md)：sample、target、batch、collation、loader 与 workspace 补产。
- [model](design/model.md)：token interface、adapter、acoustic composition 与 checkpoint contract。
- [loss](design/loss.md)：token、CTC、Flow、RVQ、preference 与 rollout objective。
- [runtime](design/runtime.md)：config、codec/tokenizer/backbone capability 与资源组装。
- [generation](design/generation.md)：chat adapter、request batching、decode 与 evaluation。
- [pl_module](design/pl_module.md)：Lightning module、optimizer、validation 与 callback 边界。
- [MIMO pretraining](design/mimo-pretraining.md)：双流契约、七任务混合与独立训练入口。
- [configuration](design/configuration.md)：Hydra schema、experiment composition 与生产默认。
- [reporting](design/reporting.md)：训练与实验窗口摘要。

已验证实验结论见 [experiments/conclusion](experiments/conclusion.md)；未完成事项见
[experiments/todo](experiments/todo.md)。实验记录的提交边界遵循仓库 `AGENTS.md`。
