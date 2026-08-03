# 020 Performance Hot Paths

状态：协议已执行；结果见
[`results/020-performance-hot-paths.md`](../results/020-performance-hot-paths.md)。本文只定义实验
协议和判定门槛；观测值只写入 result，不回填或调整本文门槛来适配结果。

## 研究问题

在真实 BiCodec、Qwen3-0.6B、三卡 DDP 的 TTS + MT `serial_joint` 训练负载上，S2S hot-path
优化是否在保持数据、训练语义和依赖环境不变时，带来可重复、超过运行漂移的有效 token 吞吐提升，
并减少或至少不恶化峰值显存。

主实验只评价 S2S supervised training 路径。anytrain 的 GRPO trusted path 另做 CUDA 微基准，
只作为对应局部改动的次要证据，不能替代或挽救主实验结论。

## 固定代码快照

| 组件 | A：baseline | B：optimized | 四次主运行是否相同 |
| --- | --- | --- | --- |
| speech-to-speech | `7868e3b2ca723f39b08dfbf618c7889fd19b8910` | `593c16ca0a966b08609d4e578b83cfebd02e2e32` | 否，这是唯一处理变量 |
| anytrain | `81e5ee00243a9b49c2effca2347a337d6f60c907` | 同 A | 是 |
| semantic-acoustic-codec | `795041ff5e8ad3e545597dec9e39b3da6023959b` | 同 A | 是 |

A/B 两个 S2S checkout 必须从上述 commit 建立且 `git status --short` 为空。模型、codec、tokenizer、
prepared data、Python environment 和其余 workspace 依赖只解析一次，随后四次运行使用相同的不可变
revision、manifest/fingerprint 和 cache。任何为兼容 A/B 而引入的额外代码差异都会使预注册失效，
应先停止并修订为新的实验编号。

正式结果必须记录所有依赖的完整 commit、Qwen 与 BiCodec snapshot、resolved Hydra config 摘要、
数据 split 与 manifest/fingerprint；不能只记录分支名或浮动模型名。

## 固定训练负载

- 复旦 `145` 的同三张物理 RTX 4090 D 24GB（GPU 5/6/7），并保持
  `CUDA_VISIBLE_DEVICES=5,6,7`。原定 `125` GPU 3/4/5 在 A1 启动门槛前被其他任务占用，
  因而整组在进入 Python 前统一迁移到这组三张空卡；不得把 125 的 preliminary probe 混入
  正式四次统计。
- BiCodec + `Qwen/Qwen3-0.6B`，LoRA，`bf16-mixed`，三卡 DDP。
- `serial_joint`：每个 optimizer step 消耗一个 TTS microbatch 和一个 MT microbatch。
- TTS / MT 权重为 `0.9 / 0.1`；speech/text batch size 均为 `8`。
- cost batching 开启，`max_batch_frames=4800`、`planning_window=256`。
- 四次运行使用同一 seed、split、sample 顺序、worker 数、batch plan 和 optimizer/scheduler 配置。
- 每次恰好完成 `100` 个 optimizer steps。正式分析只使用固定的 `step 19 -> 99` matched
  interval；完整进程 wall time 只作启动成本的次要指标。
- matched interval 内不得触发 sample generation、validation、checkpoint、profiler 或其他只在部分
  run 出现的昂贵 callback。必要的进度与吞吐日志在四次运行中使用完全相同的 cadence。

若正式入口的现状不能满足上述固定项，应在 A1 前停止并更新 schedule；不得在运行后按观测结果选择
不同 batch、窗口或配置。

## 运行顺序与隔离

正式顺序固定为：

```text
A1 -> B1 -> B2 -> A2
```

四次顺序不得交换，每次都从独立、干净的输出目录启动，不从前一次 checkpoint resume。A1 前允许
分别执行不计入结果的最小兼容性 gate，使模型和数据资产进入共享 cache；正式运行之间不清理或替换
cache。若发生 cache miss、即时编译、NAS stall 或其他一次性事件并进入 `step 19 -> 99`，该次运行
作废，整组 A1/B1/B2/A2 重跑，不能只补单次。

每次启动前记录 GPU 进程、利用率、显存、温度和时钟状态。三张卡必须无其他计算进程；四次运行期间
不得并行安排其他负载。GPU 监控使用同一采样频率和字段。若物理卡、主机、环境或共享负载发生变化，
整组比较作废。

## 数据与正确性 Gate

以下条件全部通过后才计算性能结论：

1. A1、B1、B2、A2 均以状态 `0` 完成 100 optimizer steps，无 traceback、OOM、distributed
   failure 或 non-finite 告警。
2. 四次运行在所有共同日志点的累计 non-pad compute tokens、padded compute tokens、TTS/MT
   supervised tokens 和 sample counts 完全一致；至少单独列出 step 19 与 step 99 的值。
3. 四次运行的 padding ratio 与 loader/task schedule 一致。若有效工作量不一致，不得用 step/s
   或 wall time 下 S2S hot-path 因果结论。
4. total 与 per-task loss 均 finite，训练任务和 objective 集合一致。100-step loss 只用于发现执行
   偏差，不用于质量、收敛或最终效果结论。
5. A/B 共同 trainable 参数的名称、shape、dtype、参数策略和 step-0 初始化 fingerprint 一致。
   B 可以按设计不再持有 A 中未参与该负载的冻结或未使用模块；total parameters 的差异必须单列，
   不能伪装成相同模型库存。
6. resolved config 除 checkout revision 和由该 revision 决定的模型库存外完全一致。

任一 gate 失败时，结果只能记录失败原因和复验需求，不能发布正式加速或显存结论。

## 指标与固定计算口径

主指标为 `step 19 -> 99` 的 non-pad compute tokens/s。时间分母使用两个日志事件的单调时钟差，
token 分子使用相同事件间累计 non-pad token 差。因为区间包含 80 个 optimizer-step 间隔，
optimizer steps/s 固定按 `80 / elapsed_seconds` 计算。

每次运行同时报告：

- matched interval elapsed time、non-pad tokens/s 和 optimizer steps/s；
- padded tokens/s、samples/s、target frames/s 或 audio seconds/s（日志存在时）；
- full process wall time，但不把它作为主结论依据；
- 每张 GPU 的 peak memory、active utilization 和可用的温度/时钟/功耗统计；
- total/trainable parameter counts、trainable fraction 和 optimizer parameter counts；
- dataloader wait ratio 与 MFU（现有统一日志能够可靠提供时）。

无功耗记录时不得下 energy-efficiency 结论。Lightning 展示的 microbatch `it/s` 不作为
optimizer-step 吞吐。

对主指标定义：

```text
r_forward = B1 / A1
r_reverse = B2 / A2
gain = sqrt(r_forward * r_reverse) - 1
repeat_drift = max(abs(A2 / A1 - 1), abs(B2 / B1 - 1))
```

同样公式用于 optimizer steps/s 作为一致性复核。结果同时列出四个原始值、两个方向的 ratio、
geometric mean gain 和 repeat drift，不只报告选择性最好的一组。

## 正式结论门槛

只有同时满足数据与正确性 gate，且主指标满足以下全部条件，才写“优化带来正式训练吞吐提升”：

1. `r_forward > 1` 且 `r_reverse > 1`，即 A -> B 和 B -> A 两个方向都支持 B 更快；
2. `gain >= 5%`；
3. `gain > repeat_drift`；
4. optimizer steps/s 的两个方向同样不回退；
5. B 的最高 rank peak memory 相对对应 A 不增加超过 `2%`。

若两个方向符号不一致、`gain < 5%` 或 gain 未超过 repeat drift，结论写“当前 100-step 协议下
未证明有决策意义的吞吐提升”，不得表述为加速。若 B 稳定更慢，则单独记录性能回退。

“峰值显存降低”是独立结论：必须在 B1/A1 与 B2/A2 两个方向都降低，且两组最高 rank peak
memory 均至少降低 `512 MiB`；否则只报告数值，不下显存收益结论。“参数库存降低”也独立验收：
B 的 total parameters 必须降低，同时共同 trainable 参数和 optimizer 参数 gate 保持通过。

最终建议只能覆盖本 schedule 的三卡 4090 D、BiCodec、Qwen3-0.6B、bs8、TTS + MT
`serial_joint` 负载。它不外推到生成吞吐、其他 backbone/codec、不同 batch policy、长跑质量、
收敛或能效。

## GRPO CUDA 微基准（次要证据）

微基准固定使用 anytrain
`81e5ee00243a9b49c2effca2347a337d6f60c907`，比较同一 `GRPOLoss` 有效输入上的默认
`validate=True` 与由调用方已建立 tensor contract 的 `validate=False`。它不比较 S2S A/B commit。

- 在同一张空闲 GPU、同 dtype、同 tensor shape、同 mask、同 loss 配置下复用同一组输入。
- 计时前先验证两条路径的 loss、details 和输入梯度在既定 dtype tolerance 内一致。
- 每种路径先 warm up 至少 100 次；正式计时做 5 个独立 repeat，每个 repeat 至少 1000 次。
- 使用 CUDA event 或显式 synchronize 包围计时，不能用未同步的 host wall clock。
- 每个 repeat 内顺序固定为 `validate=True -> False -> False -> True`，报告每次 us/call、median、
  min/max 或离散度及 speedup。
- 记录 GPU、PyTorch/CUDA、shape、dtype、warmup/iteration 数和精确命令。

只有数值/梯度 parity 先通过，才报告 trusted path 微基准速度。该结果只证明重复 tensor validation
在该合成输入上的局部成本；即使速度显著，也不能补偿主训练 A/B 的 gate 失败，也不能外推为完整
GRPO 训练吞吐。

## 记录产物

- 正式结果：`docs/experiments/results/020-performance-hot-paths.md`。
- 结果中记录四次命令、完整 revision/config identity、输出目录、退出状态、数据 parity 表、四 run
  原始指标、固定公式汇总、GPU 监控摘要、异常和最终判定。
- 原始日志、TensorBoard、GPU CSV 和微基准输出保存在 `debug/` 或远程 dynamic debug 目录，不纳入
  Git。
- 本 schedule 和对应 result 含内部 benchmark 与远程实验信息，只保留在本地或私有仓库，不提交或
  push 到公开仓库。
