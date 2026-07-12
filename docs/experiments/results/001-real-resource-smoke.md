# 001 Real Resource Smoke Result

## 环境

- 日期：2026-07-13。
- 机器：121，NVIDIA A100-PCIE-40GB，使用物理 GPU 1。
- Python 3.12，PyTorch 2.9.1+cu128，Flash Attention 2.8.3.post1。
- Backbone：`Qwen/Qwen3-0.6B`，bf16，`flash_attention_2`。
- Codec：LongCat `16k_4codebooks`；audio tokenizer 使用 100k LongCat BPE artifact。
- 数据：`wmt19_tts_codec(codec="longcat", split="train")` 标准 store 首条样本。
- 代码通过 `/tmp/s2s-contract-smoke` 临时快照运行，没有覆盖远端共享工作树。

## 数据契约

标准 store 的 source LongCat view 是 `torch.int64 Tensor (27, 4)`，即 `[T, K]`；
第 0 个 codebook 是 semantic，其余 3 个是 acoustic。TTS/S2ST target acoustic batch
均为 `(1, 36, 3)`，保持 anytrain codec 的 `[B, T, K]` 契约。

## 训练闭环

TTS 单 batch：

- `input_ids (1, 35)`，acoustic labels `(1, 36, 3)`。
- loss `18.25`；forward `0.63s`，backward `0.14s`，SGD step `0.03s`。
- ground-truth decode `0.65s`；waveform `(1, 1, 34560)`，float32，全部 finite。
- 峰值 CUDA memory `6,572,149,760` bytes。

S2ST 单 batch：

- `input_ids (1, 27)`，acoustic labels `(1, 36, 3)`。
- loss `24.0`；forward `0.65s`，backward `0.14s`，SGD step `0.03s`。
- ground-truth decode `0.67s`；waveform `(1, 1, 34560)`，float32，全部 finite。
- 峰值 CUDA memory `6,575,763,456` bytes。

loss 来自未训练模型和随机 flow time/noise，只用于 finite smoke，不用于模型质量比较。

## S2ST 生成闭环

从 teacher-forcing batch 裁出纯 S2ST prompt，并携带 source acoustic condition：

- prompt `(1, 25)`，生成 2 个 semantic audio BPE tokens。
- BPE spans `(1, 2)`，展开为 acoustic features `(1, 4, 1024)`。
- semantic generation + flow sampling `0.96s`。
- LongCat decode `0.62s`；waveform `(1, 1, 3840)`，float32，全部 finite。
- `flash_attention_2` 生效；峰值 CUDA memory `5,028,033,024` bytes。

该结果只证明当前短单样本过渡路径可运行，不证明 KV cache、逐行 source condition
持久性或变长 batch generation 已完成。

## 暴露并修复的边界

- chat template 必须先渲染字符串再切 placeholder；BPE token IDs 受前导空格影响，
  不能搜索单独编码的 placeholder IDs。
- LongCat wrapper 的扩展接口统一接收 `[B, T, K]`，wrapper 内转换上游
  `[B, K, T]`，semantic codebook 轴同样保留到 wrapper 边界。
- codec features 在 speech model 边界转换到 backbone device/dtype。
- anytrain continuous flow runtime 保持 `x_1` dtype，避免 float32 time 把 bf16 path
  静默提升为 float32。

上游仍会打印 Qwen tied-weight 配置提示和 `weight_norm` deprecation warning；二者未影响
本次 forward、backward、optimizer step、generation 或 decode。

## 回归测试

- speech-to-speech contract/fake closure：`20 passed, 14 subtests passed`。
- workspace 全量测试：`87 passed`。
- anytrain LongCat/flow focused tests：`17 passed, 1 skipped`；skip 是 optional dependency
  条件分支，不是失败。
- 三个仓库的 `py_compile`（改动文件）与 `git diff --check` 通过。
