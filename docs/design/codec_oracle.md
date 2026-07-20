# codec oracle

为 codec screening 实验提供可测试的 oracle model 和 prepared-code 数据路径。Hydra
配置、codec 资源加载、logger 与 Trainer 组装仍由 `scripts/codec_oracle.py` 负责。

## 运行配置

裸 `scripts/codec_oracle.py` 是生产训练入口：默认遍历完整 prepared dataset、启用 LBA，使用
`bf16-mixed` 训练 1,000,000 steps，并每 10,000 steps 记录 sample 和归档 checkpoint。完整链路
短验收不修改这些默认值，而由 `acoustic_oracle_smoke` /
`acoustic_oracle_ddp_lba_smoke`（flow）和 `acoustic_oracle_rvq_smoke` /
`acoustic_oracle_rvq_ddp_lba_smoke`（RVQ）分别提供默认与显式 DDP smoke 的数据上限、trainer、
callback 和两步预算；对应 job 显式选择 experiment，并通过 `CUDA_VISIBLE_DEVICES` 提供单卡、
两卡默认值。`jobs/005/08-11` 提供 Flow/RVQ 的正式单卡与两卡 wrapper，不选择 experiment，
因此不会覆盖生产数据范围、训练预算和 callback 间隔。

Oracle 的 artifact 目录由 `repo_output_root/output_subdir` 派生；TensorBoard event files 集中写到
`repo_output_root/tensorboard/output_subdir/version_*`。这使 Flow/RVQ、单卡/DDP 和不同初始化可以
在同一个 TensorBoard 根目录下并列比较，同时不改变 checkpoint、waveform 或 `metrics.json` 的位置。

## 对外能力

- `Objective` / `Initialization`：使用字符串枚举选择 `FLOW` 或 `RVQ` objective，以及 codec/random
  embedding initialization 模式。codec/random 权重由 `Initialization` 自身构造。
- `AcousticFlowScreening`：持有完整的 `SpeechToSpeechFlowModel`，prepared semantic codes 通过正式
  global ID、semantic audio embedding/adapter、target latent 与 acoustic flow 路径；wrapper
  只负责 acoustic-only objective、optimizer selection 与实验日志。构造时冻结完整正式模型，
  只解冻 optimizer 持有的 semantic audio embedding、adapter 与 acoustic flow，使静态路径可用
  `find_unused_parameters=False` 的 DDP。
- `AcousticRVQScreening`：持有完整的 RVQ acoustic decoder，以 prepared semantic codes 为条件，
  对每个 acoustic codebook 计算 causal cross-entropy；冻结 decoder token embedding 与未参与
  oracle 的正式模型参数，仅优化条件 embedding、adapter 与 RVQ decoder 输出路径。最后一级
  acoustic embedding/projection 不会用于预测后续 codebook，因此也被冻结，使该路径可用静态 DDP。
- unified-token codec 不使用独立 screening model；其 codes 作为 semantic audio tokens，
  `acoustic_codes=None`，复用正式 DataModule、token model、objective 与 generation/decode。
- `DataModule`：仅供 LongCat acoustic-only screening 加载 prepared codec view，返回可由
  Lightning 注入 distributed sampler 的 LBA/DataLoader；它接收严格 `DataConfig` /
  `LBAConfig`，不依赖 OmegaConf。LBA DataLoader 子类在模块加载时导入，确保 Lightning 进入
  data hook 前能包装其构造方法并在 DDP 重建时保留 `len_fn` 等参数。
  unified-token codec 不使用该入口。
- `codes()` / `collate()` / `single_batch_loader()`：单样本裁剪、变长 padding 与
  single-batch overfit 数据入口。
- `Config`：统一拥有 objective、initialization、flow target normalization、decoder、optimizer
  参数和 `DataConfig`；Hydra `codec_oracle` preset 直接映射该公开契约，`codec_oracle=rvq` 提供
  RVQ 默认值。
- `Initialization` 自身负责 codec/random embedding initialization；flow target normalization
  statistics 仍由运行入口从 probe features 计算。
- `matched_random_weight(reference, seed, rows=None)`：在 reference device/dtype 上按其总体
  mean/std 生成可复现的 matched-normal 权重；默认保持 shape，`rows` 只覆盖首维。公开调用方
  提供二维 floating reference，`Initialization.RANDOM.weight()` 复用该函数。
- `Logger`：按 objective 聚合总 loss；flow 固定样本记录 feature MSE，RVQ 固定样本记录总/逐
  codebook accuracy 和 feature MSE。两者都记录 reconstruction/sample waveform，并在训练结束写
  `metrics.json`；DDP 总 loss 先跨 rank 求均值，histogram 和 artifact 只由 global zero 写。RVQ
  wrapper 另向 Trainer 记录总/逐 codebook cross-entropy。fit start 日志记录实际 strategy 与
  world size，但不约束设备数量。
- `event()` / `timed()`：输出 codec oracle 专用的结构化阶段日志。

## 边界

- 本模块表达 codec oracle 的模型、数据和专用 callback，不选择完整链路测试 experiment 或输出
  目录。
- `scripts/codec_oracle.py` 是真实运行入口，不重复实现 model、collate 或 DataModule。
- 入口按顶层配置构造 `Runtime` 与完整 `SpeechToSpeechFlowModel`；`AcousticFlowScreening` 显式
  接收该 model、flow runtime、optimizer 参数与 target normalization statistics。codec 另行传给
  logger 执行 reconstruction/sample decode，不以零散 callable 复制进 wrapper 契约。
- prepared-code 裁剪和 LBA budget 使用 runtime codec 暴露的 frame rate；codec config
  只选择资源和 dataset view，不重复保存时间尺度。
- oracle runtime 不接受 audio tokenizer artifact；prepared semantic code 直接索引 codec 初始化的
  semantic embedding，CodecBPE token ID 与 raw codec code ID 不能混用。
- codec/random initialization 只作用于 semantic audio embedding。当前 codec 公共契约只暴露
  acoustic codebook size 和 code-to-feature 转换，不暴露 acoustic codebook vectors；因此 RVQ
  decoder 的 acoustic embeddings 使用自身随机初始化，不能声称完成 acoustic codec initialization
  对照。
- `runtime.codec` 同时选择资源和 dataset view；oracle objective 由 `codec_oracle.objective` 选择，
  不暴露无效的 acoustic type。`codec_oracle` preset 提供 decoder、target normalization、optimizer
  和 data；flow sampling 仅由 flow objective 从 `runtime.Config.flow_*` 读取。
- oracle logger、grad norm、non-finite check 与 checkpoint 使用一个成套 callback preset；入口
  显式注入 codec、codes、metadata 和 output directory 等运行时依赖。
- initialization 与 objective 的字符串只在配置边界转换一次；入口据此选择对应 screening model，
  内部不维护无效的字符串分派。
- 中间 metric、waveform 和 checkpoint 写入 `repo_output_root/output_subdir`；默认位于项目根，
  但由 `.gitignore` 排除。
