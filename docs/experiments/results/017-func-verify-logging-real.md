# 017 Real-Resource Function Verify (Logging / Sample / Loss Tags)

2026-07-30 在复旦真实 GPU 上验收当日 logging / sample / loss tag 改动，使用真实
Qwen3-0.6B、LongCat codec 与 WMT19 prepared data。本结果只证明执行与日志契约，
不支持质量或收敛结论。

## 范围

| 路线 | 机器 | 结果 |
| --- | --- | --- |
| LongCat Flow joint overfit | `145` GPU 0 | 通过 |
| LongCat RVQ joint overfit | `144` GPU 1 | 通过 |
| Stable Codec stage 1（含 FSQ embedding） | `121` | 受阻，见下 |

隔离代码：`/tmp/s2s-func-verify-20260730/repos`（含未提交的 logging 改动）。
数据：`/mnt/pami201/zhuyin/datasets/wmt19_tts_stage1_small_32_20260727`。
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

## 契约问题：`TrainInterval` 与 `max_steps=1`

新 `TrainInterval.should_run` 要求 `global_step > 0`。`TaskSampleLogger` 挂在
`on_train_batch_start`，而第一步开始时 `global_step` 仍为 0，因此 **`max_steps=1`
的 smoke 永远不会触发 sample callback**。Flow 首次 `max_steps=1` 跑通了 loss tag，
但 TensorBoard 无任何 `sample/`；改为 `max_steps=2` 后 sample 在 step 1 正常写出。

正式长跑不受影响；单步 smoke 与 overfit 验收需要至少 2 optimizer steps，或改 interval /
hook 边界。

## Stable Codec / FSQ（受阻）

`jobs/015/01_stable_codec_stage1.sh` 仍依赖 `/tmp/stable-codec-py39-env`（Torch 2.4 /
Python 3.9）。当前 S2S 配置 schema 大量使用 PEP 604 `X | None`；OmegaConf 2.3 在
`get_type_hints` 后于 py39 上失败（`unsupported operand type(s) for |` /
`_UnionGenericAlias`）。本地 unit 级 FSQ affine 构造未在该环境完成真实 codec 联跑。

可选后续：

1. 为 Stable 单独准备 Python ≥3.10 且兼容 `stable-codec` 的环境（推荐）。
2. 把 Hydra structured dataclass 的 `|` 注解脱回 `Optional`/`Union`（改动面大）。

## 结论

- 新 loss tag 分层（`token/...` 与 `acoustic/{flow_matching|rvq}/...`）在真实 Flow/RVQ
  上正确。
- 累计 `tokens` / `frames` 跨 step 累加正确。
- `sample/{task}/{index}/...`、截断标志与可播放 audio 在真实生成路径上正确。
- Stable/FSQ 真实联跑仍被 py39 Stable 环境挡住；`max_steps=1` sample smoke 被新
  interval 契约挡住。
