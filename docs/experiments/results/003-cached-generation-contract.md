# 003 Cached Generation Contract Result

## 结果

- 新增独立 `generation.Request` / `generation.Result`，真实 generation service 不再以
  `ModelBatch` 是否存在 acoustic labels 判断推理路径。
- semantic generation 首步写入完整 prompt cache，后续只输入新 token；非 cache 路径每步
  重新注入 source acoustic condition，作为 deterministic 对照。
- audio generation 在采样 semantic token 时按 BPE span 收集 predictor hidden，删除结束后的
  full-sequence forward。
- audio result 在一次调用中完成 token 裁剪、flow sampling 和 waveform decode；SampleLogger
  直接复用该 result。
- teacher-forcing batch adapter 会裁掉 target semantic/acoustic 字段和 padding，只构造 prompt
  request。

## 验证

- fake cached path 的输入长度为 `prompt, 1, 1`，full-recompute path 为
  `prompt, prompt+1, prompt+2`，两者 greedy token、acoustic features 和 waveform 一致。
- fake cache 的所有后续 step 保留 source condition，且 acoustic sampling 与 waveform decode
  各只调用一次。
- 随机初始化的一层 tiny Qwen3 cache 与非 cache greedy sequence 一致。
- SampleLogger contract test 验证每次日志事件只调用一次 generation。
- 本地完整测试：`26 passed`；`py_compile`、改动文件 Ruff 和 `git diff --check` 通过。

## 未验证

- 真实 Qwen3/LongCat cached S2ST generation/decode。
- padded variable-length batch、每行独立 EOA/EOS 和 acoustic frame mask。

对应的后续真实资源验收分别见
[004 Real Cached Generation](004-real-cached-generation.md) 与
[008 Real Batch Generation Benchmark](008-real-batch-generation-benchmark.md)。
