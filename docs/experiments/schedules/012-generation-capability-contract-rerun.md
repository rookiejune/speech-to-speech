# 012 Generation Capability Contract And 011 P0 Rerun

## 目标

修复 011 P0 中训练后端到端 generation 的能力识别失败，并用已经固化的
`jobs/011/01_rvq_native_p0_fixed_sample.sh` 原样复测真实 Qwen、LongCat native token 与 RVQ
decoder 的 TTS/S2ST fixed-sample 子项。该计划只解除 011 P0 的 generation blocker，不替代
011 的 staged joint training，也不启动 A-D 长跑。

## 背景

011 在复旦 `145` 上已经验证 TTS/S2ST 的 2-step forward/backward/optimizer、S2ST source
acoustic prompt 路径，以及 teacher-forced RVQ sampling 到 LongCat waveform decode。失败点发生在
`Trainer.fit` 到达 `max_steps=2` 之后的 greedy generation：旧版 `generate_responses()` 对真实
`RVQModel` 抛出 `AcousticFeatureGenerator` capability 相关 `TypeError`。

根因不是 RVQ model 缺少 `generate_audio_features()`，而是 Python 3.12 runtime-checkable
Protocol 对 `nn.Module` registered submodule 的 static lookup false negative。真实 Qwen
`backbone` 通过 `nn.Module.__getattr__` 从 `_modules` 取出，普通访问存在，但
`inspect.getattr_static()` 看不到该成员；当前测试替身把 `backbone` 放成普通属性，所以没有覆盖
真实路径。

## 实施计划

1. 为 generation capability 增加最小回归测试：
   - 构造一个带 registered `nn.Module` backbone、且实现 `generate_audio_features()` 的音频模型。
   - 断言 `generate_responses()` 在 codec 有 acoustic codebooks 且 model 实现 acoustic feature
     generation 时调用 audio feature generation，不因 Protocol false negative 报错。
   - 增加 token-only model 搭配 acoustic-codebook codec 的 semantic decode 回归。
2. 修复能力检查：
   - 不再依赖对整个 `TokenGenerator` 的 runtime Protocol `isinstance` 判断可选能力。
   - 只在 model 暴露 `generate_audio_features()` 且 codec 有 acoustic codebooks 时走 acoustic
     feature generation；否则走 semantic token decode。
   - 不改变 `RVQModel.generate_audio_features()`、RVQ sampling 或 decode 逻辑。
3. 本地验证：
   - 运行聚焦的 generation 单测。
   - 运行 011 job launcher 的 `--cfg job` dry-run，确认 TTS/S2ST 仍 compose 到
     `011-qwen-rvq-native-p0-fixed-sample/{tts,s2st}/rvq-8l`。
4. 复旦复测：
   - 同步代码到共享工作区或在远端更新到修复 commit。
   - 在复旦测试机上使用本地临时 Qwen snapshot 和 prepared data root，运行
     `jobs/011/01_rvq_native_p0_fixed_sample.sh`。
   - 保留 launcher 的 `tts.log`、`s2st.log`、`tts.exit`、`s2st.exit`、`overall.exit`。

## 复测命令

```bash
SPEECH_TO_SPEECH_TRAIN_ROOT=/tmp/s2s-012-generation-contract-rerun \
SPEECH_TO_SPEECH_P0_QWEN_ROOT=<local-qwen-snapshot-or-model-id> \
SPEECH_TO_SPEECH_P0_DATA_ROOT=/tmp/s2s-011-data \
SPEECH_TO_SPEECH_P0_TTS_GPU=1 \
SPEECH_TO_SPEECH_P0_S2ST_GPU=2 \
jobs/011/01_rvq_native_p0_fixed_sample.sh
```

若 NAS 仍接近满盘，继续把 Qwen、LongCat cache、prepared data 和训练输出放在机器本地 `/tmp`。
这只是复测运行环境的 I/O 规避，不应写进 Hydra preset。

## 通过条件

- 本地 generation 回归测试通过，且 token-only negative case 仍抛出清晰 `TypeError`。
- TTS 与 S2ST 两个远端子任务 `exit` 均为 `0`，`overall.exit` 为 `0`。
- 两个输出目录均写出 `evaluation.json`、`generation.json` 和 `metrics.json`。
- `generation.json` 中 response、features、waveform 全部 finite，duration 与 RTF 有效。
- TensorBoard 中 total loss、audio token CE、RVQ CE 与 text probe NLL 正常写入；没有 NaN/Inf。
- GPU 1/2 任务结束后释放，无遗留进程。

## 非目标

- 不修 011 的正式 staged joint training entry。
- 不启动 32-sample 100-step、1k pilot、DDP/resume 或 010 checkpoint import。
- 不调整 RVQ objective、loss weight、decoder 结构或 LongCat codec 逻辑。
- 不把 feature MSE 或 2-step loss 变化解释为质量或收敛结论。

## 结果记录

复测完成后新增 `docs/experiments/results/012-generation-capability-contract-rerun.md`，记录：

- 修复 commit、远端 commit、Python/PyTorch/Lightning 版本。
- 本地单测命令与结果。
- 远端命令、输出 root、launcher exit status。
- TTS/S2ST 的 `metrics.json`、`generation.json`、`evaluation.json` 摘要。
- 是否解除 011 P0 的 generation blocker，以及仍未执行的 P0 子项。
