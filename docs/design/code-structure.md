# 代码结构优化方案

## 目标

- 目录表达领域能力，文件表达稳定抽象，不按类、函数或处理步骤拆文件。
- 同一目录下的并列实现文件必须共同实现一个明确的 `Protocol`、ABC 或框架接口。
- 强耦合的数据结构、校验、默认实现和私有 helper 跟随其所有者，不建立 `types.py`、
  `helper.py`、`result.py`、`build.py` 式的无所有权碎片。
- `__init__.py` 只声明稳定公共接口；不能依靠多层重导出掩盖真实实现位置。
- 结构重构一次性更新仓库内导入，不保留只做旧路径转发的兼容文件。

## 当前状态

重构前 `src/speech_to_speech` 有 173 个 Python 文件，其中 72 个少于 100 行、44 个少于
50 行，另有 27 个 `__init__.py`。完成本轮收敛后为 131 个 Python 文件、20 个
`__init__.py`；少于 100 行的实质文件降为 22 个，少于 50 行的实质文件降为 9 个。保留下来的
小文件主要是同协议实现、框架 callback、跨模块兼容边界或独立的公共 contract。

以下结构符合“协议 + 多实现”，应保留其基本形态：

- `runtime/audio_tokenizer/`：Native、BPE、Flattened、BiCodec tokenizer 是同一音频
  tokenizer 协议的不同实现。
- `runtime/backbone/`：HF、Kimi 和 MIMO 是 backbone/tokenizer 适配边界的不同实现。
- `model/acoustic/` 中的 Flow 与 RVQ：二者是 acoustic model 的不同实现。
- `callback/logging/` 中各 Lightning callback：它们实现同一框架 callback 接口，但
  `TaskSampleLogger` 的内部处理步骤不属于并列实现。

## 目标结构

下面只展开需要调整的目录；未列出的较大、职责明确文件保持不动。

```text
speech_to_speech/
├── audio.py
├── task/
│   ├── contract.py
│   └── templates.py
├── runtime/
│   ├── core.py
│   ├── config.py
│   ├── protocol.py
│   ├── codec_contract.py
│   ├── audio_tokenizer/
│   │   ├── contract.py
│   │   ├── native.py
│   │   ├── bpe.py
│   │   ├── flattened.py
│   │   └── bicodec.py
│   └── backbone/
│       ├── contract.py
│       ├── config.py
│       ├── hf.py
│       ├── kimi.py
│       └── mimo.py
├── datamodule/
│   ├── contract.py
│   ├── sample.py
│   ├── batch.py
│   ├── parse.py
│   ├── builder.py
│   ├── single.py
│   ├── collate.py
│   ├── module.py
│   ├── dataset/
│   ├── loader/
│   └── mimo/
├── model/
│   ├── checkpoint_contract.py
│   ├── factory.py
│   ├── base.py
│   ├── token.py
│   ├── generation.py
│   ├── audio_input.py
│   ├── audio_output.py
│   ├── ctc.py
│   ├── tower.py
│   ├── toy.py
│   └── acoustic/
│       ├── contract.py
│       ├── config.py
│       ├── base.py
│       ├── factory.py
│       ├── flow.py
│       └── rvq.py
├── generation/
│   ├── contract.py
│   ├── service.py
│   ├── request.py
│   ├── audio.py
│   ├── decode.py
│   ├── text.py
│   ├── mixed.py
│   ├── chat.py
│   ├── mimo.py
│   ├── rollout.py
│   └── evaluation.py
├── loss/
│   ├── contract.py
│   ├── supervised.py
│   ├── ctc.py
│   ├── policy.py
│   └── mimo.py
├── pl_module/
│   ├── module.py
│   ├── composition.py
│   ├── optim.py
│   └── mimo.py
└── callback/
    └── logging/
        ├── task_sample.py
        ├── sample_report.py
        └── <其他 callback 实现>.py
```

## 具体调整

| 优先级 | 当前结构 | 调整方案 | 抽象依据 |
| --- | --- | --- | --- |
| P0 | `model/contract/` 的 8 个文件 | 合并为 `model/checkpoint_contract.py` | 这些文件共同构建、序列化和校验同一个模型 checkpoint contract，不是多个实现。 |
| P0 | `datamodule/_helper/` | 删除该目录：CTC/tokenization 进入 `builder.py`，duration 进入 `sample.py`/`parse.py`，task allocation 进入 `collate.py`，`TextLoader` 进入 `dataset/text.py` | 当前目录按 helper 类型拆分，没有独立对外契约。 |
| P0 | `datamodule/build/`、`collate/`、`parse/` | `sample.py + ar.py + masked.py` 合并为 `builder.py`；`single.py` 提升到 datamodule；单实现的 `collate/`、`parse/` 展平为文件 | AR/masked 是一次 sample 构建分派的内部路径；Single 是独立原始样本适配边界。 |
| P0 | `generation/protocol.py`、`result.py`、`task/io.py`、`model/output.py` | Result 与 generation capability 合并为 `generation/contract.py`；`AcousticGeneration` 进入 `model/generation.py`；Request 进入 `task/contract.py` | Request 同时被 datamodule 与 generation 消费；把它放进 generation 会经包初始化形成反向依赖，因此由 task contract 拥有。 |
| P1 | `generation/_request.py`、`batch.py`、`text.py` | `batch.py` 并入 `service.py`；`_request.py` 改为公开所有权清晰的 `request.py`；保留 `text.py` | 请求校验会调用 audio request 规则，不适合塞回反向导入 audio/mixed 的 service；文本响应投影则有独立 `ResponseTextRuntime` contract。 |
| P1 | `generation/audio.py` 与 `decode.py` | 复核后保留两个文件 | 二者都已是数百行的稳定边界：`audio.py` 负责策略选择与结果编排，`decode.py` 负责 codec 表示到 waveform 的纯解码，不属于小文件碎片。 |
| P1 | `generation/eval/` | 合并为 `generation/evaluation.py` | text/acoustic/reporting 当前是一次评估能力的处理步骤；如果未来出现多个 evaluator 实现，再恢复协议目录。 |
| P1 | `loss/protocol.py`、`types.py` | 合并为 `loss/contract.py` | 模型能力协议和 loss 输出是 objective 的公共契约。 |
| P1 | `loss/module.py`、`token.py`、`validation.py` | 合并为 `loss/supervised.py` | Token/Flow/RVQ supervised objective 共享同一执行和结果抽象。 |
| P1 | `loss/preference.py`、`rollout.py`、`logprob.py` | 合并为 `loss/policy.py` | DPO/GRPO 共用 policy log-probability 抽象；`logprob.py` 不是独立实现。 |
| P1 | `callback/logging/_sample_*` | metadata、audio、metrics、protocol 收敛为 `sample_report.py`；writer 并入 `task_sample.py` | 报告数据与音频/指标形成一个稳定报告边界；逐行写出与 callback 生命周期共享同一调度上下文。 |
| P2 | `model/acoustic/_codec.py`、`condition.py`、`initialization.py` | 共用组件并入 `base.py`，初始化/构造并入 `factory.py`；保留 Flow/RVQ 两个实现文件 | Flow/RVQ 是有效的多实现结构，碎片只出现在共用内部步骤。 |
| P2 | `model/_helper.py`、`_checkpointing.py` | 按所有权拆回 `generation.py`、`token.py`、`factory.py`；共享 tower 配置、mask 和 activation checkpointing 形成 `tower.py` | 当前 `_helper.py` 同时包含 sampling、buffer、transformer mask、embedding 和 dtype adapter，是无抽象的聚合。 |
| P2 | `model/toy.py`、`mimo_toy.py` | 合并为 `model/toy.py` | 二者共同表达测试用 model factory，不需要两个小文件。 |
| P2 | `pl_module/optim.py`、`protocol.py` | composition protocol 并入 `composition.py`；保留轻量 `optim.py` | `optim.Config` 同时服务单流和 MIMO；若并入主 module，导入 MIMO 会被迫加载完整 supervised/acoustic 依赖，因此属于明确的可选依赖隔离例外。 |
| P2 | `runtime/factory.py`、`runtime/tokenizer.py` | 单函数 factory 并入 `core.py`；AudioTokenizer/TextTokenizer protocol 移到对应实现边界 | tokenizer 实现目录应显式拥有自己的协议，避免协议与实现分居。 |
| P2 | 只有一个实质文件的 `audio/` 包 | 展平为 `audio.py` | 目录没有多实现或子系统层级。 |

`loss/__init__.py` 当前使用惰性导入来隔离 MIMO 与完整 codec 依赖，这属于明确的依赖边界，
可以保留；不能为了减少文件数破坏可选依赖隔离。

## 执行顺序

1. 建立基线：记录公共 import、运行当前测试，禁止在结构重构中混入行为修改。
2. 整理基础契约：`audio`、`task`、runtime tokenizer protocol，并展平单实现包。
3. 整理 datamodule：先删除 `_helper`，再合并 builder/parse/collate；这是调用面最广的一步。
4. 整理 model：先 checkpoint contract，再 factory/tower/acoustic；保持模型参数路径和
   checkpoint 内容不变。
5. 整理 loss 与 generation：先移动纯类型，再合并执行路径，最后更新 pl_module 消费方。
6. 整理 callback 与 scripts：它们位于依赖图上层，应在底层导入路径稳定后处理。
7. 删除旧文件并一次性更新内部和测试导入；不留下转发模块。

本次实施保留了工作树中已有的暂存和未暂存修改，没有 reset、checkout 或改写索引状态；结构移动与
导入更新只作用于当前仓库所有权范围。

## 每阶段验收

- 一个目录内不应出现两个以上少于 100 行的实质文件，除非它们实现同一明确协议，或用于隔离
  可选依赖、平台差异和导入副作用。
- 除测试白盒检查外，不从其他模块导入 `_` 开头的文件或成员。
- `__init__.py` 只导出稳定公共 API；实现代码内部直接从所有者模块导入，不经过重导出层。
- contract 模块不导入具体实现，factory/service 才选择实现，依赖方向保持单向。
- 每阶段显式使用顶层 `py39` 环境运行 ruff、basedpyright 和相关 pytest；全部阶段完成后运行
  全量测试。
- 源码文件数已从 173 个降到 131 个；少于 100 行的非 `__init__.py` 文件从 72 个降到 22 个。
  剩余数字高于原先的 15 个目标，是因为保留了 tokenizer/backbone/acoustic 的同协议实现、独立
  callback 和兼容/平台边界；数字是结果指标，不作为机械合并的理由。
