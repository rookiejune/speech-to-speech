# Speech-to-Speech

## 目标

构建一个基于 LongCat 语音 semantic token 和 Qwen3 的 speech-to-speech 训练脚手架。

第一阶段先做 speech semantic 自回归 LM。第二阶段在同一个训练框架里加入 speech translation 任务，通过配置调整本次训练包含哪些任务以及它们的采样权重。

模型路线：

1. 预训练 Qwen3 + LongCat BPE token 做 semantic token 的 causal LM 微调。
2. 后续接入 DiT，基于 Qwen3 hidden states 生成声学特征。

## Dataset

数据从 Anydataset 读取。样本持有 source 和 target 两侧语音，`AudioView.LONGCAT` 存的是 LongCat 原始 codes，不是 BPE 后的 ids；训练侧读取 `semantic_codes`，样本日志还会读取 `acoustic_codes` 做 decode 监督。

复旦 121 上有一个测试数据集：

```python
from anydataset import AnyDataset

AnyDataset(spec="store://~/repos/anydataset/storage/fleurs-full-longcat")
```

训练契约只要求：

- `(Role.SOURCE, Modality.AUDIO)` 持有 `AudioView.LONGCAT`
- `(Role.TARGET, Modality.AUDIO)` 持有 `AudioView.LONGCAT`
- 两侧 `AudioView.LONGCAT` 是至少包含 `semantic_codes` 和 `acoustic_codes` 的 dict

文本和语言不进入 speech-to-speech 训练样本契约。若需要用文本做过滤、数据准备或评估，放在训练前处理链路里完成。

## Filtering

过滤是可选训练前处理，不放进 batch builder。

可以使用：

- UTMOS 过滤低质量音频。
- 如果持有文本，可用 ASR 结果和文本的匹配分数过滤。

相关能力优先复用 `anytrain.evaluator`。

## BPE

训练使用 LongCat semantic ids 的 BPE token。数据集里的 `AudioView.LONGCAT` 是原始 semantic ids，所以训练开始时需要先解析 BPE 配置并加载或训练 tokenizer。

BPE 语料使用 source 和 target 两侧全部 speech semantic ids。

缓存位置通过环境变量指定：

```text
BPE_CACHE_DIR/
  longcat/
    vocab_100k_piece_32/
      tokenizer.json
      meta.json
    vocab_200k_piece_32/
      tokenizer.json
      meta.json
```

运行逻辑：

1. 从环境变量 `BPE_CACHE_DIR` 读取缓存根目录。
2. 根据 `vocab_size` 和 `max_piece_frames` 拼出目录名，例如 `vocab_100k_piece_32`。
3. 如果缓存中已有 tokenizer，直接加载。
4. 如果没有缓存，用当前数据集的 source+target 原始 semantic ids 训练 BPE。
5. 写入 `meta.json`，记录 vocab size、max piece frames、数据集 spec 等信息。

## Dataloader

至少支持两类任务：

1. `autoregression`
2. `translation`

本次训练包含哪些任务由配置决定。第一版只需要启用/关闭任务族，不需要阶段式 schedule 或采样权重。

示例：

```yaml
tasks:
  enabled:
    - autoregression
    - translation
```

之后可以通过重新提交训练任务切换任务组合：

```yaml
tasks:
  enabled:
    - autoregression
```

连续过渡策略以后再设计，不要先塞进 dataloader 核心。

data module 的结构应保持为：

```text
AnyDataset -> TaskSampleStream -> Task sample batch -> Batch builder -> CausalLMBatch
```

speech-to-speech 假定输入数据集已经是 anydataset，不再定义额外底层 dataset wrapper。
datamodule 让 anydataset 负责 rank/worker 分片；每条 source/target raw sample 在
`TaskSampleStream` 中展开成 source autoregression、target autoregression、
source-to-target translation 和 target-to-source translation。BPE 语料扫描可以复用
source/target pair。

## Autoregression

第一阶段默认先做自回归。

标准目标是 speech semantic BPE token 的 next-token prediction：

- 输入包含任务 prompt、`BOA` 和整段 audio ids。
- loss 计算 `BOA`、audio continuation 和 `EOA`。
- 不对 prompt 计算 loss。
- 不需要固定业务 prefix。
- prompt 必须通过 Qwen3 tokenizer 的 chat template 生成，不手写 Qwen3 对话模板 token。

随机 crop、最大长度截断、prefix continuation 等策略可以后续配置化；第一版先保证完整 speech token 序列的 causal LM 闭环可跑。

## Translation

第二阶段加入翻译任务。

翻译任务不依赖文本或语言字段。batch builder 只表达 source speech semantic BPE tokens 到 target speech semantic BPE tokens 的任务结构。
翻译 prompt 同样通过 Qwen3 chat template 生成；source speech semantic BPE tokens 作为 user content 中带 `BOA`/`EOA` 的 audio segment 接入，loss 只计算 target 侧 `BOA`、speech semantic BPE tokens 和 `EOA`。

翻译任务的核心目标是：

```text
source speech semantic BPE tokens -> target speech semantic BPE tokens
```

文本字段只作为可选过滤和评估辅助信息，不进入训练 batch。

## Model

使用预训练 Qwen3 作为 semantic decoder。

用 `anytrain.idspace` 把 Qwen3 text token 和 LongCat BPE audio token 放到同一个 token space：

- text token 复用 Qwen3 embedding。
- LongCat BPE audio token 使用新增 embedding，audio modality block 只包含真实 BPE vocab。
- `BOA` 和 `EOA` 注册为 `anytrain.idspace` special tokens，不注册进 Qwen tokenizer，也不占用 LongCat BPE vocab。
- LM head 覆盖 audio token、`BOA` 和 `EOA`；生成侧用 `EOA` 停止，不用 Qwen EOS。

优先先跑通 Qwen3 semantic causal LM 微调，再接入 DiT 声学生成。
