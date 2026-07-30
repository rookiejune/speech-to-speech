# 017 Real-Resource Function Verify (Logging / Sample / Loss Tags)

2026-07-30 在复旦真实 GPU 上验收当日 logging / sample / loss tag 改动，使用真实
Qwen3-0.6B、LongCat / Stable codec 与 WMT19 prepared data。本结果只证明执行与日志契约，
不支持质量或收敛结论。

## 范围

| 路线 | 机器 | 结果 |
| --- | --- | --- |
| LongCat Flow joint overfit | `145` GPU 0 | 通过 |
| LongCat RVQ joint overfit | `144` GPU 1 | 通过 |
| Stable Codec stage 1（含 FSQ embedding） | `121` GPU 0 | 通过 |

隔离代码：`/tmp/s2s-func-verify-20260730/repos`（含未提交的 logging / py39 兼容改动）。
LongCat 数据：`/mnt/pami201/zhuyin/datasets/wmt19_tts_stage1_small_32_20260727`。
Stable 数据：`/tmp/stable-stage1-data-20260728`（含临时 identity selection）。
SAC artifact：`/tmp/s2s-joint-init-20260728/{artifact,rvq-artifact}`（schema 7）。

## Flow（通过）

入口：`scripts/overfit.py`，`max_steps=2`，`callbacks.task_sample.every_n_steps=1`。

- total / token / Flow loss：`13.334 -> 9.861` / `10.538 -> 7.905` / `2.796 -> 1.957`，均 finite。
- TensorBoard tag：
  - `token/loss/tts`、`token/audio_loss/tts`
  - `token/tokens/tts` 累计 `37 -> 74`（两步各 37）
  - `acoustic/flow_matching/{loss,frames,t}/tts`，frames 累计 `36 -> 72`
  - `sample/tts/0/{target,reference_generation,generated}` audio
  - `sample/tts/0/generation/{response_tokens,reached_max_new_tokens,stopped_without_eoa}`
    为 `256 / 1 / 1`（撞上默认 256 budget，截断音频仍写出）
- 输出：`145:/tmp/s2s-func-verify-train-20260730/002-single-batch-overfit/tts/flow-func-verify-s2/`

## RVQ（通过）

入口同上，`parameter_policy=speech_interface`，`max_steps=2`。

- total / token / RVQ loss：`18.731 -> 17.634` / `9.572 -> 8.911` / `9.159 -> 8.723`，均 finite。
- TensorBoard tag：
  - `token/.../tts` 与累计 `token/tokens/tts=74`
  - `acoustic/rvq/{loss,codebook_0,codebook_1,codebook_2,frames}/tts`
  - `sample/tts/0/` audio + truncation flags，与 Flow 同形
- 输出：`144:/tmp/s2s-func-verify-train-20260730/002-single-batch-overfit/tts/rvq-func-verify-s2/`

## Stable Codec / FSQ（通过）

入口：`jobs/015/01_stable_codec_stage1.sh`，`/tmp/stable-codec-py39-env`（Torch 2.4 /
Python 3.9），`max_steps=1`，`callbacks.task_sample.every_n_steps=1`，
`callbacks.task_sample.max_new_tokens=34`。验收 follow-up 额外覆写
`trainer.log_every_n_steps=1` 以写出 `token/` TB tags（正式配置默认 10）。

环境兼容改动（本验收依赖）：

1. OmegaConf structured dataclass 中会经 `get_type_hints` 求值的 `|` 改为
   `Optional` / `Union`（例如 `StageLoaderConfig.prediction`、`FlowRepaConfig.student_layer`）。
   `from __future__ import annotations` 不够：OmegaConf 会再求值注解，py39 仍炸。
2. `anydataset` 的 `AudioLoader = Callable[[Union[str, Path]], ...]`（运行时类型别名）。
3. 为 smoke root 发布 identity selection：
   `/tmp/stable-stage1-data-20260728/selections/stable-1x46656_400bps/speech_translation_v1/train`
   （Fudan 默认 filter 名；32 样本全量索引，再由 `smoke-splits.json` 子集化）。

执行：

- exit `0`，`Trainer.fit` 因 `max_steps=1` 正常停止。
- 总参数 `751376384`，可训练 `17408`（= `12×1024` FSQ bias/slope + `5×1024` free
  marker rows，对应 `FsqAffineEmbedding` + Qwen3-0.6B hidden `1024`）。
- `metrics.json`：token loss first/last `6.345 / 100.660`，均 finite。
- TensorBoard（`.../stage_1-token/version_2/`）：
  - `token/{loss,text_loss,audio_loss,tokens,text_tokens,audio_tokens}/{asr,tts}`
  - `sample/asr/{0,1,2}/` source audio + text + generation flags
  - `sample/tts/{0,1,2}/{target,generated}` audio，`audio/finite=1`
  - **无** `reference_generation`（Stable token-only，不伪造 Flow/RVQ reference）
- `TrainInterval` 去掉 `step > 0` 后，`max_steps=1` 即可在 `global_step=0` 写出 sample。

日志：`121:/tmp/s2s-func-verify-20260730.stable3.log`；
输出根：`121:/tmp/s2s-func-verify-train-20260730/stable-codec-stage1/`。

## 契约修正：`TrainInterval`

原先 `should_run` 要求 `global_step > 0`，导致 `max_steps=1` smoke 永远不触发
`TaskSampleLogger`。已收成 `step % every_n_steps == 0` + 同 step 去重；Stable 单步
smoke 已验证 sample 可写。

## 结论

- 新 loss tag 分层（`token/...` 与 `acoustic/{flow_matching|rvq}/...`）在真实 Flow/RVQ
  上正确；Stable token-only 写出 `token/.../{asr,tts}`，无 acoustic side channel。
- 累计 `tokens` / `frames` 跨 step 累加正确（Flow/RVQ `max_steps=2`）。
- `sample/{task}/{index}/...`、截断标志与可播放 audio 在真实生成路径上正确。
- Stable/FSQ 真实联跑在 py39 环境下通过，依赖 Optional/Union 注解与 selection 发布。
