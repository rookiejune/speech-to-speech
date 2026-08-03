# 020 Performance Hot Paths

## 正式结论

2026-08-04 在复旦 `145` 的 RTX 4090 D GPU 5/6/7 上完成预注册的
`A1 -> B1 -> B2 -> A2` 四段 100-step 对照。四段均以状态 0 完成，无 OOM、DDP failure 或
non-finite loss；A/B 的首个 total/token loss 完全相同，可训练参数数目也完全相同。

主训练结果给出稳定但边界性的正向证据：固定 `step 19 -> 99` 窗口内，B 相对 A 的两个顺序方向
分别快 `6.4435%` 和 `3.5740%`，optimizer steps/s 的几何平均增益为 `4.9989%`，高于
`1.6281%` 的重复运行漂移。按实际监督 token 归一化后，两个方向分别提升 `6.4111%` 和
`3.5424%`，几何平均为 `4.9670%`。

这个结果不得写成“已正式证明训练吞吐提升”：预注册门槛要求增益 `>=5%`，两个主口径都略低于
门槛；此外 A/B 的监督 token 工作量存在 `0.030%` 的系统差异，B 的默认 progress counter 又因
`sync_distributed=false` 变为 rank-local，无法完成预注册的全局 non-pad compute-token exact
parity gate。产物也没有预注册要求的 monotonic 端点、trainable/optimizer manifest 和 step-0
parameter fingerprint。可辩护的表述是：**B 在本负载上提供了可重复、约 5% 的训练吞吐改善证据，
但没有通过 020 的严格吞吐晋级门槛。**

以下三项独立结论成立：

- B 的最高 rank 峰值显存从 `22,213 MiB` 降至 `21,589 MiB`，两次配对均减少
  `624 MiB`（`2.81%`），通过预注册的显存收益门槛。
- total parameters 从 `782,430,208` 降至 `626,847,744`，减少 `155,582,464`
  （`19.88%`）；trainable parameters 保持 `31,071,232`。
- anytrain GRPO trusted path 在 exact loss/details/gradient parity 后，CUDA event 中位数从
  `2,131.684 us/call` 降至 `1,173.135 us/call`，局部 forward 为 `1.817x`。这是局部 loss
  microbenchmark，不是完整 GRPO 训练吞吐。

## 代码、资产与环境

主实验只改变 speech-to-speech revision；其余代码和资产在四段之间共享。

| 组件 | A：baseline | B：optimized |
| --- | --- | --- |
| speech-to-speech | `7868e3b2ca723f39b08dfbf618c7889fd19b8910` | `593c16ca0a966b08609d4e578b83cfebd02e2e32` |
| anytrain | `81e5ee00243a9b49c2effca2347a337d6f60c907` | 同 A |
| semantic-acoustic-codec | `795041ff5e8ad3e545597dec9e39b3da6023959b` | 同 A |
| workspace | `4e9ea9483c7fdfff3a2054d5c8c5ea70c891a7fb` | 同 A |
| anydataset | `9b6bceba10315d511413fc67811bbefca094f756` | 同 A |
| length-based-batching-adapter | `771d95c3c0bcdb0c35ccf3ee4a0b4fbf8318918b` | 同 A |
| Spark-TTS | `cf5302b0da797aeffc18af172de4dda0bb6ab718` | 同 A |
| Qwen3-0.6B snapshot | `c1899de289a04d12100db370d81485cdf75e47ca` | 同 A |
| SparkAudio/Spark-TTS-0.5B BiCodec | `642071559bfc6346c2359d19dcb6be3f9dd8a05d` | 同 A |

- Host：`145.pami.group`；GPU 5/6/7，三张 NVIDIA GeForce RTX 4090 D，单卡
  `24,564 MiB`，driver `580.65.06`。
- Python `3.12.0`，PyTorch `2.9.0+cu128` / CUDA `12.8`，Lightning `2.6.1`，
  Transformers `4.57.3`。
- Qwen TTS BiCodec speaker grid：`train_0_10000`，40,000 cells、2 speakers、20,000
  text rows；`speaker_grid_manifest.jsonl` SHA256
  `504a6569a8f107eeb04eb0c874992cebb5fd3366dc180b3fad0fdca72a57dd3e`。
- MT：Hugging Face `wmt/wmt19`、`zh-en`、train split，25,984,574 rows。
- 两个 S2S detached worktree 在运行前后均 clean。125 的 GPU 3/4/5 在 A1 空闲门槛前被其他任务
  占用，整组因而在进入 Python 前统一迁到 145 GPU 5/6/7；125 的 preliminary probe 不计入本结果。

## 固定负载

- BiCodec、Qwen3-0.6B、LoRA r16、`bf16-mixed`、FlashAttention 2、gradient checkpointing。
- 三卡 `ddp_find_unused_parameters_true`；`serial_joint` 每个 optimizer step 顺序消费一个 TTS
  microbatch 和一个 MT microbatch，`accumulate_grad_batches=2`。
- TTS / MT weights `0.9 / 0.1`；speech/text batch size `8`。
- cost batching：`max_batch_frames=4800`、`planning_window=256`；seed `0`。
- 每段 100 optimizer steps；`trainer.log_every_n_steps=10`；关闭 validation、checkpoint、
  task sample、text retention、gradient probe 和 performance callback。
- GPU telemetry 用 `nvidia-smi` 每约 200 ms 记录显存、利用率、温度、功耗和 SM clock。
- 除 output/run name 外，resolved config 的唯一版本差异是 S2S commit 将
  `optim.schedule.sync_cuda/sync_distributed` 从 `true/true` 改为 `false/false`；这是被测提交的一部分。

## 执行与正确性 gate

| Run | Revision | Exit | Optimizer steps | First total / token loss | Total / trainable params |
| --- | --- | ---: | ---: | ---: | ---: |
| A1 | `7868e3b` | 0 | 100 | `13.174126 / 14.637918` | `782,430,208 / 31,071,232` |
| B1 | `593c16c` | 0 | 100 | `13.174126 / 14.637918` | `626,847,744 / 31,071,232` |
| B2 | `593c16c` | 0 | 100 | `13.174126 / 14.637918` | `626,847,744 / 31,071,232` |
| A2 | `7868e3b` | 0 | 100 | `13.174126 / 14.637918` | `782,430,208 / 31,071,232` |

四段日志都明确包含 ``Trainer.fit stopped: max_steps=100 reached``，没有 OOM、distributed error、
NaN/Inf loss 或训练期 traceback。Python 进程在成功写出 metrics 后均出现相同的
`multiprocess.resource_tracker` `RLock._recursion_count` 清理期 traceback；它发生在训练结束之后，
没有改变退出码、metrics、TensorBoard 或 GPU 释放状态，作为环境清理告警保留，不按训练失败处理。
但 schedule 的字面 gate 是“无 traceback”，所以它仍是严格晋级 gate 未通过的一项，而不是被静默
忽略。

`metrics.json` 中四段的首项 loss 完全相同，末窗口也紧密一致且全部 finite。A/B 的
`OutputsLogger` 在本提交中由逐 microbatch point logging 改为 cadence-window mean，因此
TensorBoard cadence 点上的 per-task loss 不是同一聚合语义；本实验不据此声称质量或收敛 parity。

## 数据工作量 gate

A1 与 A2 在所有共同 cadence 点上的 TTS/MT 累计监督 token 完全相同；B1 与 B2 也完全相同，说明
各版本内部可重复。但版本之间不是 exact parity：

| Version | TTS step 19 -> 99 | MT step 19 -> 99 | Interval supervised tokens |
| --- | ---: | ---: | ---: |
| A（A1=A2） | `230,336 -> 1,191,384` | `14,372 -> 67,974` | `1,014,650` |
| B（B1=B2） | `230,208 -> 1,190,948` | `14,376 -> 67,977` | `1,014,341` |
| B - A | `-128 -> -436` | `+4 -> +3` | `-309` (`-0.0304%`) |

step 99 的累计监督 token 总数为 A `1,259,358`、B `1,258,925`，差 `433`（`0.0344%`）。
差异远小于时间差，所以监督 token/s 是有用的归一化敏感性指标；但它仍违反 schedule 要求的
exact token parity。

预注册还要求共同 trainable parameter 的 name/shape/dtype、optimizer manifest 和 step-0
fingerprint 完全相同。当前产物只证明 trainable/optimizer parameter numel 均为 `31,071,232`，并以
相同 seed 得到 bit-identical 的首个 total/token loss；没有保存完整 manifest 或 parameter hash，
因此不能把这两项代理证据升级成 identity gate 通过。

同时，A 的 anytrain progress schedule 使用 distributed reduction，而 B 的默认配置关闭它。
因此 A 的 `progress/tokens_total` 是全局累计值，B 的 TensorBoard 值是 rank-local 且从不同阈值点
开始，不能通过乘 world size 恢复精确全局 work。预注册的 global non-pad compute tokens/s 主指标
无法按同一观测语义计算。这是本次正式吞吐 gate 失败的第二个原因。

## 吞吐

固定窗口使用 TensorBoard `loss` event 的 step 19 与 99 wall time；它排除模型、codec、DDP 和
dataset 初始化。窗口含 80 个 optimizer-step intervals。该时间戳严格递增，但属于 epoch wall
clock，不是 schedule 要求的进程 monotonic clock，所以表中数值是诊断口径而非预注册主指标的完整
实现。监督 token/s 使用上表的实际 interval token 数归一化。

| Run | Step 19 -> 99 time | Optimizer steps/s | Supervised tokens | Supervised tokens/s | Full process wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| A1 | `51.106 s` | `1.5654` | `1,014,650` | `19,854.0` | `237.32 s` |
| B1 | `48.012 s` | `1.6663` | `1,014,341` | `21,126.9` | `189.58 s` |
| B2 | `48.552 s` | `1.6477` | `1,014,341` | `20,892.0` | `187.04 s` |
| A2 | `50.287 s` | `1.5909` | `1,014,650` | `20,177.2` | `188.43 s` |

| Metric | B1 / A1 | B2 / A2 | Geometric mean gain | Repeat drift |
| --- | ---: | ---: | ---: | ---: |
| Optimizer steps/s | `+6.4435%` | `+3.5740%` | `+4.9989%` | `1.6281%` |
| Supervised tokens/s | `+6.4111%` | `+3.5424%` | `+4.9670%` | `1.6281%` |

两个方向的符号一致，几何平均增益也明显高于重复漂移，因此 30-step preliminary probe 的正向结果
不是单纯的运行顺序反转假象。但预注册规则写明 `gain >= 5%` 才能晋级；不对 `4.9989%` 作四舍五入
越线，也不以任意后选窗口替换主窗口。A1 的 full-process wall 包含首轮共享 cache 冷启动；A2 与
B2 的 warm full-process wall 接近，所以 full wall 不作为吞吐主证据。

## 峰值显存与 GPU telemetry

| Physical GPU | A1 / A2 peak | B1 / B2 peak | B reduction |
| --- | ---: | ---: | ---: |
| 5 | `21,897 MiB` | `21,525 MiB` | `372 MiB` (`1.70%`) |
| 6 | `22,213 MiB` | `21,589 MiB` | `624 MiB` (`2.81%`) |
| 7 | `21,619 MiB` | `21,133 MiB` | `486 MiB` (`2.25%`) |
| Highest rank | `22,213 MiB` | `21,589 MiB` | `624 MiB` (`2.81%`) |

最高 rank 的可用余量由约 `2,351 MiB` 增至 `2,975 MiB`。B1/A1 与 B2/A2 的峰值完全复现，
通过“最高 rank 两个方向均至少降低 512 MiB”的独立显存门槛。

step 19 -> 99 窗口内，三卡平均 utilization 为 A1 `55.54%`、B1 `59.69%`、B2 `57.70%`、
A2 `55.32%`；平均 SM clock 约 `2.65-2.73 GHz`，四段最高温度均不超过 `62 C`。B 的平均功耗
略高，但窗口更短；本短采样不发布 energy-efficiency 结论。

## 参数库存

| Metric | A | B | Delta |
| --- | ---: | ---: | ---: |
| Total parameters | `782,430,208` | `626,847,744` | `-155,582,464` (`-19.88%`) |
| Trainable parameters | `31,071,232` | `31,071,232` | `0` |
| Trainable fraction | `3.9711%` | `4.9567%` | inventory effect only |

减少量精确等于 `151,936 x 1,024`，对应 optimized revision 在加载 Qwen backbone 后释放未使用、
冻结的 Hugging Face LM output head；S2S 使用自己的 modality-local heads。该结果证明 B 不再持有
本负载不需要的参数库存，同时保持 LoRA 训练参数数目。它不能把整段约 5%
训练改善全部归因于单一 kernel；被测 commit 还同时包含 logging collective、batch transfer、validation
和 loss/model trusted-path 改动。

## anytrain GRPO CUDA 微基准

微基准使用 anytrain `81e5ee0`、145 GPU 0、PyTorch `2.9.0+cu128`，输入 shape
`[8, 8, 256]`、float32、response density `0.8522`、`sequence_mean`、`clip_range=0.2`、
`kl_beta=0.05`。两条路径复用同一输入；先各 warm up 200 次，再做 5 个 repeat，每个 repeat 按
`validate=True -> False -> False -> True`，每段 1000 calls，使用 CUDA events 计时。

在计时前，`validate=True` 与 `validate=False` 的 total loss、全部 details 和 policy gradient 均
bit-exact；loss 为 `0.00015540761523880064`。

| Metric | validate=True | validate=False | Speedup |
| --- | ---: | ---: | ---: |
| 10 measurements median | `2,131.684 us/call` | `1,173.135 us/call` | `1.817x` |
| Min -> max | `2,049.666 -> 2,151.741 us` | `1,122.854 -> 1,183.187 us` | - |
| Median of 5 per-repeat speedups | - | - | `1.811x` |

该结论只覆盖已由调用方建立 tensor contract 后跳过重复 runtime validation 的 GRPO loss forward。
它不包含 rollout/model forward/backward/optimizer，不能外推为 `1.817x` 的完整 RL 训练加速。

## 决策与边界

- 保留 B：它没有观察到吞吐回退，两个顺序方向均更快，并正式降低了显存和参数库存；完整测试、Ruff
  与 basedpyright 已在提交快照通过。
- 对外或汇总结论写“约 5% 的可重复训练改善证据”，不要写“正式 >=5% 加速已通过”。严格的 020
  吞吐晋级结论是未通过。
- 若吞吐晋级仍是决策关键，下一次应使用新的预注册编号：冻结同一 batch plan，增加无 per-step
  distributed sync 的全局 compute-token 离线计数，并把窗口扩到至少 200 optimizer steps；不得对
  020 事后换窗口或放宽阈值。
- 本实验不支持生成吞吐、不同 codec/backbone/batch policy、长跑质量、收敛或能效结论。

## 原始产物

远端隔离根：

```text
/mnt/pami202/zhuyin/dynamic/debug/s2s_probe/20260804-performance-hot-paths-abba-100step
```

`profile/` 下保留四段 `.log`、`.gpu.csv`、`.time.txt`、`.exit.txt`、`.meta.txt`，以及
`train/<run>/metrics.json`、resolved Hydra config、TensorBoard events；GRPO 原始 JSON 为
`profile/grpo_cuda_formal.json`。这些内部 benchmark 产物与本文只保留在本地或私有仓库，不提交或
push 到公开仓库。
