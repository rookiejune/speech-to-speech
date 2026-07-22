# codec oracle

为 codec screening 实验提供可测试的 oracle model、factory 和 prepared-code 数据路径。Hydra
配置、prepared-code 读取、logger 与 Trainer 组装仍由 `scripts/codec_oracle.py` 负责；
`speech_to_speech.codec_oracle.factory` 负责 runtime、model、screening wrapper 和 metadata 构造。

## 运行配置

裸 `scripts/codec_oracle.py` 是生产训练入口：默认遍历完整 prepared dataset，不设置额外
`max_seconds`；启用 LBA 后，8 秒 batch budget 同时作为单样本硬上限，`overlong=error` 会明确
暴露不满足预算的数据，也可显式选择 `filter` 或 `truncate`。入口使用 8 个持久 worker、pinned
memory 和 4-batch prefetch，以 `bf16-mixed` 训练
1,000,000 steps，并每 10,000 steps 记录 sample 和归档 checkpoint。完整链路短验收不修改这些
默认值，而由
`acoustic_oracle_smoke` /
`acoustic_oracle_ddp_lba_smoke`（flow）和 `acoustic_oracle_rvq_smoke` /
`acoustic_oracle_rvq_ddp_lba_smoke`（RVQ）分别提供默认与显式 DDP smoke 的数据上限、trainer、
callback 和两步预算；对应 job 显式选择 experiment，并通过 `CUDA_VISIBLE_DEVICES` 提供单卡、
两卡默认值。`jobs/005/08-11` 提供 Flow/RVQ 的正式单卡与两卡 wrapper，不选择 experiment，
因此不会覆盖生产数据范围、训练预算和 callback 间隔。

生产入口同时使用 `TrainingFlops` 和 `anytrain.PerformanceCallback` 记录参数量、step time 和 MFU。
FLOPs provider 读取当前 local-rank batch：Flow 按实际执行的 padded frame shape 估算，RVQ 按有效
frame 和 codebook 路径估算；anytrain 在 DDP 下汇总各 rank FLOPs、峰值算力和最慢 rank 时间。
默认每 100 optimizer steps 记录，前 20 steps warmup，窗口为 100 steps；smoke experiment 使用
逐步记录、无 warmup 和 2-step 窗口。设备峰值默认自动推断，特殊机器通过
`callbacks.performance.hardware_peak_flops` 提供单设备 override。

Oracle 的 artifact 目录由 `repo_output_root/output_subdir` 派生；TensorBoard event files 集中写到
`repo_output_root/tensorboard/output_subdir/version_*`。这使 Flow/RVQ、单卡/DDP 和不同初始化可以
在同一个 TensorBoard 根目录下并列比较，同时不改变 checkpoint、waveform 或 `metrics.json` 的位置。

## 对外能力

- `Objective` / `Initialization`：使用字符串枚举选择 `FLOW` 或 `RVQ` objective，以及 codec/random
  embedding initialization 模式。codec/random 权重由 `Initialization` 自身构造。
- `AcousticFlowModel` / `AcousticRVQModel`：只持有 codec semantic embedding、正式 adapter 和对应
  acoustic decoder。semantic embedding 直接复制单个 codec codebook；模型不构造 text/audio
  output head、acoustic prompt adapter 或 Qwen backbone。
- `AcousticFlowScreening`：持有轻量 Flow oracle model；prepared semantic codes 直接通过
  semantic embedding/adapter 形成 condition，wrapper 负责 acoustic-only objective、optimizer
  selection 与实验日志。只训练 semantic embedding、adapter 与 acoustic flow，可使用
  `find_unused_parameters=False` 的静态 DDP。
- `AcousticRVQScreening`：持有轻量 RVQ oracle model，以 prepared semantic codes 为条件，对每个
  acoustic codebook 计算 causal cross-entropy。只训练条件 embedding、adapter 与 decoder 的实际
  输出路径；decoder token embedding 和最后一级不会被消费的 acoustic embedding/projection 保持
  结构性冻结。
- unified-token codec 不使用独立 screening model；其 codes 作为 semantic audio tokens，
  `acoustic_codes=None`，复用正式 DataModule、token model、objective 与 generation/decode。
- `DataModule`：仅供 LongCat acoustic-only screening 加载 prepared codec view，返回可由
  Lightning 注入 distributed sampler 的 LBA/DataLoader；它接收严格 `DataConfig` /
  `LBAConfig`，不依赖 OmegaConf。LBA DataLoader 子类在模块加载时导入，确保 Lightning 进入
  data hook 前能包装其构造方法并在 DDP 重建时保留 `len_fn` 等参数。
  unified-token codec 不使用该入口。
- `codes()` / `collate()` / `single_batch_loader()`：单样本硬上限策略、变长 padding 与
  single-batch overfit 数据入口。启用 LBA 时，有效硬上限是显式 `max_seconds` 与
  `lba.max_batch_seconds` 的较小值；超长样本不会再作为超预算 singleton 静默进入训练。
- `Config`：统一拥有 objective、initialization、flow target normalization、decoder、optimizer
  参数和 `DataConfig`；Hydra `codec_oracle` preset 直接映射该公开契约，`codec_oracle=rvq` 提供
  RVQ 默认值。
- `factory.build_runtime()` / `build_flow()` / `build_rvq()`：从已解析配置和 runtime codec 构造
  oracle runtime、轻量 model、screening wrapper 与运行 metadata；condition dim、dtype、CUDA
  device 选择和 flow target statistics 都在该边界集中处理。
- `Initialization` 自身负责 codec/random embedding initialization；flow target normalization
  statistics 由 factory 从 probe features 计算。
- `matched_random_weight(reference, seed, rows=None)`：在 reference device/dtype 上按其总体
  mean/std 生成可复现的 matched-normal 权重；默认保持 shape，`rows` 只覆盖首维。公开调用方
  提供二维 floating reference，`Initialization.RANDOM.weight()` 复用该函数。
- `Logger`：按 objective 聚合总 loss；flow 固定样本记录 feature MSE，RVQ 固定样本记录总/逐
  codebook accuracy 和 feature MSE。两者都记录 reconstruction/sample waveform，并在训练结束写
  `metrics.json`；训练中只在设备保存前 20/后 20 个 detached loss，结束时一次跨 rank 聚合，避免
  每步 loss `.item()`/reduce；histogram 和 artifact 只由 global zero 写。RVQ
  wrapper 另向 Trainer 记录总/逐 codebook cross-entropy。fit start 日志记录实际 strategy 与
  world size，但不约束设备数量。
- `TrainingFlops`：实现 anytrain 的动态 batch FLOPs provider 契约，按实际 screening module 与
  batch shape 返回当前 local rank 的 forward+backward 训练 FLOPs；不读取 Hydra 中另设的静态
  step FLOPs。估算统一统计 Linear 与 attention matrix multiplication，反向按 forward 的两倍计入；
  lookup、scatter、normalization、activation、loss 和冻结 codec dequantization 不计入模型 FLOPs，
  但它们的耗时仍在实际 step time 中，因此会如实降低 MFU。
- `event()` / `timed()`：输出 codec oracle 专用的结构化阶段日志。

## 边界

- 本模块表达 codec oracle 的模型、数据和专用 callback，不选择完整链路测试 experiment 或输出
  目录。
- `scripts/codec_oracle.py` 是真实运行入口，不重复实现 model factory、collate 或 DataModule。
- factory 按顶层配置构造 `Runtime`，但只触发 codec 与 flow runtime；非 toy 配置通过
  `AutoConfig.from_pretrained()` 读取 backbone hidden size，不访问 `runtime.backbone`，因此不会加载
  或注册完整 Qwen 权重。screening checkpoint 的 trainable state 只含 `model.semantic_audio_*` 与
  `model.acoustic_flow` / `model.acoustic_decoder`，可按同名白名单导入联合模型。
- prepared-code 硬上限和 LBA budget 使用 runtime codec 暴露的 frame rate；codec config
  只选择资源和 dataset view，不重复保存时间尺度。`error`、`filter`、`truncate` 是显式互斥策略，
  不做未记录的兼容回退。
- oracle runtime 不接受 audio tokenizer artifact；prepared semantic code 直接索引 codec 初始化的
  semantic embedding，CodecBPE token ID 与 raw codec code ID 不能混用。
- codec/random initialization 只作用于 semantic audio embedding。当前 codec 公共契约只暴露
  acoustic codebook size 和 code-to-feature 转换，不暴露 acoustic codebook vectors；因此 RVQ
  decoder 的 acoustic embeddings 使用自身随机初始化，不能声称完成 acoustic codec initialization
  对照。
- `runtime.codec` 同时选择资源和 dataset view；oracle objective 由 `codec_oracle.objective` 选择，
  不暴露无效的 acoustic type。`codec_oracle` preset 提供 decoder、target normalization、optimizer
  和 data；flow sampling 仅由 flow objective 从 `runtime.Config.flow_*` 读取。
- oracle logger、grad norm、non-finite check 与 checkpoint 使用一个成套 callback preset；共享
  performance preset 由入口注入 `TrainingFlops`。启用 performance 时省略重复的 grad norm logger，
  但保留 oracle sample/histogram、non-finite check 和 checkpoint。入口同时注入 codec、codes、
  metadata 和 output directory 等运行时依赖。
- initialization 与 objective 的字符串只在配置边界转换一次；入口据此选择对应 screening model，
  内部不维护无效的字符串分派。
- 中间 metric、waveform 和 checkpoint 写入 `repo_output_root/output_subdir`；默认根目录由
  `SPEECH_TO_SPEECH_TRAIN_ROOT` 或 `$DYNAMIC_HOME/train/speech-to-speech` 提供，TensorBoard event
  files 写入 `repo_output_root/tensorboard/...`。
