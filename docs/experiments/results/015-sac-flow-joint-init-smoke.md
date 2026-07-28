# 015 SAC Generator Joint Initialization Smoke

2026-07-28 在 FDU `145` 的 GPU 7（RTX 4090 D, 24 GB）完成真实
`semantic-acoustic-codec` Flow 与 RVQ `codebook_ar` generator 到 S2S hidden-state joint 路线验收。
本结果证明两条 route 的初始化、forward/backward/optimizer 和生成执行契约，并完成 Flow 的正式
checkpoint resume；不支持质量或收敛结论，RVQ checkpoint resume 未单独验证。

## Flow Artifact 与隔离环境

- Python：`/home/zhuyin/anaconda3/envs/py312/bin/python`，Torch `2.9.0+cu128`。
- 隔离代码：`/tmp/s2s-joint-init-20260728/repos`，未修改共享 checkout。
- 原 artifact：`/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/overfit-longcat-fm-8l-sample0/artifact`。
- debug copy：`/tmp/s2s-joint-init-20260728/artifact`；`model.ckpt` 仍以 symlink 指向原始权重。
- 原 schema 5 只在 debug copy 中迁到 schema 7，并写入真实 LongCat backend 元数据：
  `sample_rate=16000`、`frame_rate=semantic_frame_rate=16.666666666666668`、
  `acoustic_feature_dim=1024`、3 个 `8100` acoustic codebook、`frame_aligned`。
- artifact route 为 `fm`，condition dim 为 `1024`，decoder 为 8 层、8 heads、FFN ratio 4；
  generator 共 `162662400` 参数。

旧 checkpoint 的 conditioner state 与当前 schema 不兼容：完整 `load_artifact()` 按预期报告
缺少 `reference_conditioner.null_condition`，并报告两个旧的 unexpected key。修复后的
`load_generator_artifact()` 不构造 conditioner，只严格加载 `generator.*`，同一真实 checkpoint
加载成功且所有参数 finite。回归测试同时确认 generator 自身缺键或多键仍会失败。

## Flow Full Joint Smoke

关键环境为 `CUDA_VISIBLE_DEVICES=7`、`LOCATION=fudan`、`HF_HUB_OFFLINE=1`、
`TRANSFORMERS_OFFLINE=1`、共享 HF/anytrain cache 与 pami201 WMT19 prepared root。Qwen repo id
在离线模式下会被 transformers 的 `model_info()` 阻断，因此 backbone 显式指向同一缓存 snapshot。

```bash
python -u scripts/overfit.py \
  experiment=overfit runtime=longcat_native \
  runtime.backbone=/mnt/pami202/zhuyin/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  acoustic.init_artifact=/tmp/s2s-joint-init-20260728/artifact \
  train.max_steps=1 runtime.device=cuda \
  trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto \
  trainer.precision=bf16-mixed callbacks.task_sample.enabled=false \
  +callbacks.evaluation.enabled=true \
  repo_output_root=/tmp/s2s-joint-init-20260728/output
```

WMT19 TTS sample 0 完成 1 个 optimizer step：total loss `13.177252769470215`，token loss
`10.386824607849121`，Flow loss `2.7904281616210938`。Full policy 模型参数为
`772876032`，全部 trainable；TensorBoard 中 first-backbone parameter 的 token/Flow 梯度 norm
分别为 `23.375` / `0.94921875`，cosine 为 `-0.61328125`，总 grad norm 为
`234.9686279296875`。

训练后 autoregressive generation 产生 `[64, 1024]` acoustic features 和
`[1, 61440]` waveform，duration `3.84s`、RTF `1.8837417189691528`，全部 finite。
独立 acoustic evaluation 的两个 seed 都成功 decode 2.16s waveform；feature MSE 为
`3.050964891910553` / `2.9855934381484985`，采样 RTF 约 `0.121`。

结果位于：

- `/tmp/s2s-joint-init-20260728/output/002-single-batch-overfit/tts/flow-8l/metrics.json`
- 同目录 `generation.json`、`evaluation.json` 与 `hydra/overfit.log`
- TensorBoard：`/tmp/s2s-joint-init-20260728/output/tensorboard/002-single-batch-overfit/tts/flow-8l/version_3`

## Flow Direct Phase B Gradients

第二次运行使用相同 artifact/batch 和 `parameter_policy=speech_interface`，只把临时诊断 callback
放在 `/tmp/s2s-joint-init-20260728/grad_smoke.py`，不进入仓库。176826112 个参数 trainable，
backbone 冻结；loss 与 full run 一致。optimizer step 前直接汇总得到：

| Module | Parameters with grad | L2 norm | Finite |
| --- | ---: | ---: | --- |
| `model.acoustic_condition` | 4 | `0.5894044637680054` | true |
| `model.acoustic_flow.decoder` | 124 | `1.6390935182571411` | true |

这证明真实 hidden state 经 `HiddenConditionAdapter` 驱动了从 SAC 初始化的 decoder，而不是把
semantic codes 继续传给 Phase B generator。

## Flow Formal Checkpoint Resume

同日在同一 GPU、Qwen、LongCat 与 SAC artifact 上，改用正式 `scripts/train.py`、Stage 1、
`speech_interface` policy 和 pami201 的 32-sample 真实 WMT19/LongCat root。batch size 为 1，
BF16 mixed precision，每个 step 都归档 checkpoint；先运行到 step 1，再从同目录
`step-00000001.ckpt` 恢复并把 `max_steps` 提高到 2。日志明确记录 `Restoring states` 和
`Restored all states`，恢复后只执行新增的 step 2。

| Step | Total loss | Token loss | Flow loss |
| ---: | ---: | ---: | ---: |
| 1 | `9.37112808227539` | `6.738102912902832` | `2.663419246673584` |
| 2 after resume | `9.02692985534668` | `6.522792339324951` | `2.5324862003326416` |

step 1 checkpoint 的 `global_step=1`，包含 1 个 optimizer、133 个 optimizer state entry；所有
AdamW step counter 为 1。恢复后的 `global_step=2`，同一批 counter 全部推进到 2。模型 state 中
有 4 个 hidden adapter key 和 124 个 decoder key，没有 runtime acoustic codec key。step 1 到
step 2 的直接 checkpoint delta 为：

| Module | Changed keys | Changed values | L2 delta | Finite |
| --- | ---: | ---: | ---: | --- |
| `model.acoustic_condition` | `4 / 4` | `1051648 / 1051648` | `0.019037006465590204` | true |
| `model.acoustic_flow.decoder` | `124 / 124` | `162656862 / 162662400` | `0.21553355019390313` | true |

首次同目录 resume 还暴露了 Lightning 默认 version counter 会保留旧 `last.ckpt`、把新恢复点写成
`last-v1.ckpt`。正式 S2S checkpoint callback 现显式使用 `enable_version_counter=False`：修复后目录
只包含 `step-00000001.ckpt`、`step-00000002.ckpt` 和 `last.ckpt`，其中固定 `last.ckpt` 的
`global_step=2`，没有 `last-v*.ckpt`。两个 step 归档均为 `3314328933` bytes，`last.ckpt` 为
`3314328997` bytes。最终输出位于
`145:/tmp/s2s-joint-init-20260728/resume-fixed-v2/run`。

## RVQ Codebook-AR Joint Smoke

同日在 GPU 7 继续验收 RVQ route。原 artifact 为
`/mnt/pami202/zhuyin/dynamic/debug/semantic-acoustic-codec/overfit-longcat-rvq-8l-sample0/artifact`：
真实 LongCat、frame-aligned、3 个 `8100` codebook、condition dim 1024、8 层 decoder，predictor
明确为 `codebook_ar`。原 schema 5 只在 `/tmp/s2s-joint-init-20260728/rvq-artifact` 迁到 schema 7，
checkpoint 仍以 symlink 指向原权重。generator-only loader 构造出
`RVQCodeGenerator(AcousticRVQDecoder)`，共 `184031980` 参数，全部 finite；没有使用当前 SAC
默认的 MTP predictor。

```bash
python -u scripts/overfit.py \
  experiment=overfit runtime=longcat_native model/acoustic=rvq \
  runtime.backbone=/mnt/pami202/zhuyin/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  acoustic.init_artifact=/tmp/s2s-joint-init-20260728/rvq-artifact \
  data.root=/mnt/pami201/zhuyin/datasets/wmt19_tts_stage1_small_32_20260727 \
  train.max_steps=1 runtime.device=cuda \
  trainer.accelerator=gpu trainer.devices=1 trainer.strategy=auto \
  trainer.precision=bf16-mixed callbacks.task_sample.enabled=false \
  +callbacks.evaluation.enabled=true \
  repo_output_root=/tmp/s2s-joint-init-20260728/rvq-output
```

使用与 Flow 相同的 Qwen3-0.6B、LongCat、WMT19 sample 0 和 BF16 mixed precision 完成 1 个
optimizer step。Full policy 模型参数为 `794245612`，其中 `785950188` trainable；total/token/RVQ
loss 分别为 `18.904052734375`、`9.745777130126953`、`9.158275604248047`。

训练后 autoregressive generation 产生 `[64, 1024]` finite acoustic features 和
`[1, 61440]` finite waveform，duration `3.84s`、RTF `1.788941324290742`。独立 acoustic
evaluation 的两个 seed 都成功 decode 2.16s waveform，feature MSE 为
`2.3287537693977356` / `2.231229782104492`，采样 RTF 为 `0.01381` / `0.01171`。

第二次运行改用 `parameter_policy=speech_interface`，189900268 个参数 trainable，backbone
冻结；optimizer step 前的直接梯度为：

| Module | Parameters with grad | L2 norm | Finite |
| --- | ---: | ---: | --- |
| `model.acoustic_condition` | 4 | `2.216846466064453` | true |
| `model.acoustic_decoder` | 98 | `5.76848030090332` | true |

这证明 RVQ Phase B 同样由真实 backbone hidden state 经 `HiddenConditionAdapter` 驱动，而不是把
Phase A semantic codes 作为 decoder condition。结果位于：

- `/tmp/s2s-joint-init-20260728/rvq-output/002-single-batch-overfit/tts/rvq-8l`
- 同目录 `metrics.json`、`generation.json`、`evaluation.json` 与 Hydra config
- TensorBoard：`/tmp/s2s-joint-init-20260728/rvq-output/tensorboard/002-single-batch-overfit/tts/rvq-8l/version_0`

## 实测暴露并修复的边界

1. generator-only loader 不再受无关 conditioner schema 阻断，同时保持 generator strict load。
2. S2S model 不再注册/拥有 Runtime LongCat `nn.Module`；codec 参数不会进入模型参数组或 checkpoint。
3. FP32 speech adapter 输出在汇入 BF16 backbone embedding 时显式转换 dtype，adapter storage 仍为 FP32。
4. 无 validation split 的 DataModule 返回合法空 iterable；overfit Trainer 关闭 sanity validation。
5. 正式 checkpoint resume 保持归档 step 文件，并让固定 `last.ckpt` 始终覆盖为最新恢复点。
6. RVQ joint initialization 只接受 frame-aligned `codebook_ar` artifact；默认 MTP 不作为兼容替代。

本地验收：S2S `296 tests OK / 1 CUDA skip`、basedpyright 0 errors，Ruff、compileall 与
`git diff --check` 通过。SAC 全量测试为 `66 passed`，Ruff、compileall 与
`git diff --check` 通过。
