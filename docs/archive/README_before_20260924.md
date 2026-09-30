# PDD / Wan2.1 1.3B

当前checkpoint策略（2026-09-15最新）：每5个G step保存一份滚动checkpoint，只保留最新完整滚动版本；每50步的checkpoint永久保留。新版本完整保存后才清理旧滚动版本（保留step目录命名，RESUME=auto仍可使用）。取消额外的首步保存，训练最终一步仍保存。固定prompt每25步生成，启动/结束预览设置保留。例：step105保存后保留step50、100、105；checkpoint包含student、fake、两个optimizer及rank状态。已启动的进程需重启后应用，现有文件此刻不会被清理。


2026-09-15 恢复修复：1.3B联合/对照默认在第1次G更新后保存，之后每5步及最终保存；预览仍每25步。`RESUME=auto` 在没有任何完成checkpoint但存在旧metrics时，会将config/metrics/TensorBoard/预览/未完成checkpoint移动到 `OUTPUT/restart_archives/<timestamp>/`，打印 `restart_without_checkpoint` 后从初始化重跑。没有checkpoint的更新无法恢复；已有完成标记但文件缺失或不兼容时仍拒绝重置。设置 `--set auto_restart_without_checkpoint=false` 可恢复严格报错行为。


当前默认（更新）：原始 Wan2.1-T2V-1.3B 初始化，`student_init=null`；PDD/DMD 权重均1，从第一个G更新生效（无ramp）；fake:G=5:1。32卡BS1、GA4、global batch128。新输出目录后缀为 `_gbs128_teacherinit_w1_fake5`，避免恢复旧实验。matched对照同步改为原始Wan初始化，但DMD权重仍为0。以下旧实验说明中的step100初始化、0.1权重与1:1更新比属于历史设置。


## 当前 1.3B 对照实验（2026-09-14）

旧入口 `scripts/train_wan1p3b_480p_450k.sh` 现默认 **Phased DMD + PDD4**；
`EXPERIMENT=pdd_matched` 为同 rollout、同初始化、关闭 DMD 的对照。
两者默认从原 RCM6 run 的 step100 **仅加载 student 权重**，新建 optimizer；各自输出目录内支持 `RESUME=auto`。
下文早期纯 PDD 的算法/测试记录保留为历史参考。联合实验使用三个模型（对照两个），不含 GAN。

```bash
# 本机4卡：每卡BS1、GA32，总batch128
NNODES=1 NPROC_PER_NODE=4 FSDP_SHARD_SIZE=4 bash scripts/train_wan1p3b_480p_450k.sh
# 16卡：每个节点执行同一命令；每卡BS1、GA8，总batch128
MASTER_ADDR=主节点IP NNODES=2 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=8 bash scripts/train_wan1p3b_480p_450k.sh
# 同rollout纯PDD对照（独立输出目录）
EXPERIMENT=pdd_matched NNODES=1 NPROC_PER_NODE=4 FSDP_SHARD_SIZE=4 bash scripts/train_wan1p3b_480p_450k.sh
```

默认480p/81f、450613加权prompt、四个phase、G/Fake LR均1e-6、fake:G=1:1、trajectory权重1、DMD权重0.1（前25个G更新线性增大），250个G更新，25步保存/RCM6固定预览、NFE4。fake:G=1:1是保守算力起点，不是已验证最佳比例；可用 `--set fake_updates_per_g=5` 调整。
BS可覆盖，GA自动匹配128（不可整除时明确报错）；显式GA覆盖会改变总batch。共享wan环境保持硬编码。

联合checkpoint包含 `model.pt`、`fake.pt`、`joint_state.json`、各rank状态及`COMPLETE`；对照不需要`fake.pt`。TensorBoard和JSONL同时记录每phase轨迹loss、DMD surrogate、fake DSM及score time。DMD标量不是KL或视频质量分数。
算法说明见 [设计文档](reports/pdd_phased_dmd_design.md)，本机测试结果见 [联合训练验证报告](reports/wan13b_joint_validation.md)。


基于 [NVlabs/AnyFlow](https://github.com/NVlabs/AnyFlow) 的 Wan/Diffusers backbone，实现 [Parallel Decoding Distillation](https://arxiv.org/html/2607.26004)（Algorithm 3，data-free）。上游完整保存在 `AnyFlow/`，固定 commit `bf9195aa04b708a8cd9a7e742bb03265ec3a6c29`，保留 Apache-2.0 LICENSE。新增 `AnyFlow/pdd/` 使用独立训练入口；不启动 AnyFlow 的 DMD、判别器或 JVP 分支，因此也不依赖其 FSDP2/视频评测全套环境。

本机已完成 **4×A800、832×480、81帧** 的训练、checkpoint恢复、batch边界测试和32个teacher/PDD对比视频。实测结果及画质限制见 [验证报告](reports/validation.md)。当前checkpoint只训练8步，输出仍明显模糊，不是收敛模型。

## 实现

- Native Wan checkpoint 严格转换到 AnyFlow 使用的 `WanTransformer3DModel`；不下载模型，不忽略缺失权重。
- N 个**绝对时间区间**线性 head，从原始 output projection 拷贝初始化；shared backbone 和 heads 都训练。
- 论文 Midpoint 预设：N=128、Lmin=16、Lmax=64，支持 2/4/8 NFE，shift=6、teacher CFG=5、unconditional skip layer index 10、AdamW lr=1e-5、weight decay=0、无 EMA，工程默认 gradient clipping=1.0。负面 embedding 沿用本地 Wan negative cache。
- 采用 Wan 的 sigma: 1→0，网络 velocity 为 noise−data；网格、负 dt 和 Midpoint 均保持该约定。
- 每个样本独立随机选择监督区间；一次 backbone forward 得到被监督 head、到 teacher 查询点的位移、下一在线 block 的位移；线性层先融合再投影，与显式多 head 加权等价。teacher 及在线轨迹均 stop-gradient。
- Euler 默认每个 block 采两个监督区间，Midpoint 采一个区间、两次 solver evaluation。CFG 使每次 teacher evaluation 包含 conditional/unconditional 两次 backbone 调用。
- 支持DDP及student FSDP/HSDP（专用多机入口默认FSDP）；teacher冻结并复制。BF16 autocast + FP32 trainable parameters/Adam states + activation checkpointing。每个梯度累计 microbatch 保留独立在线轨迹，不额外重跑 rollout。
- checkpoint 保存 optimizer、各 rank 的 RNG/在线轨迹/数据游标。`COMPLETE` 仅在各 rank 保存完毕后生成；恢复在线状态要求相同 world size、batch、accumulation 及算法/分辨率配置；不保证混合精度训练逐位确定性。

## 环境与快速启动

早期本机验证使用 PyTorch 2.3.1+cu121、Diffusers 0.34.0；当前启动脚本默认使用共享环境 `/mnt/data/butong/miniconda3/envs/causvid-wan21-final/bin/python`，不依赖节点 PATH 中的 Python。集群依赖入口为 `requirements.txt`（保留镜像的 CUDA PyTorch），本机版本复现见 `requirements-pdd.txt`。原始模型在 `../Wan2.2/checkpoints/Wan2.1-T2V-1.3B`。不要直接安装 AnyFlow 的完整 requirements 来替换已有环境。

```bash
USE_TF=0 PYTHONPATH=AnyFlow python -m unittest discover -s tests -v
bash scripts/smoke_4gpu.sh
```

所有有效测试和默认配置均为 **832×480、81 帧**。`smoke.json` 使用128 heads、每卡 batch 1 和 manifest 中128条真实训练prompt，只用于工程验证。正式 Midpoint 配置为 `wan13b_480p.json`；`wan13b_480p_euler.json` 使用256 heads和每次两个Euler监督区间。

## 450k 数据入口

训练默认读取 Bidirectional `train_sft.sh` 指向的 `youtube_sports_added/training_balanced/wds_weights/wds_weight_index.json`，共 **450,613** 条、带 SFT 采样权重；只读取 tar 中的 `embed.pt`，不读取视频 latent。原 192p/480p cache 都可作为相同的 prompt 来源，不改变生成分辨率。

2026-09-14 本机检查：所有必需 tar 均存在。默认全量数据配置会明确报缺失路径，不自动切换测试 prompts；路径变化可在 config 的 `path_map` 填写源前缀到目标前缀映射。

若无法挂原缓存，可以从同一个 manifest 准备新 embedding：

```bash
USE_TF=0 python scripts/prepare_embeddings.py \
  --manifest ../datasets/taxonomy_pipeline/output/youtube_sports_added/manifest_dedup_plus_youtube_sports.jsonl \
  --limit 128 --output outputs/train_prompts_128
# 全量准备：去掉 --limit，可用 torchrun --nproc_per_node=4 scripts/prepare_embeddings.py ...
EMBEDDING_DIR="$PWD/outputs/train_prompts_128" bash scripts/train_pdd.sh
```

新缓存入口均匀采样所准备的 manifest 行；原 tar 入口保留 SFT 权重。全量 token embedding 很大，请优先复用现有 tar。prepare 输出包含原 manifest ID、行号和 caption SHA256，已完成 shard 可安全跳过。

## Batch 与集群启动

H3风格多机入口：`scripts/train_wan1p3b_480p_450k.sh`。当前默认4×8卡、FSDP分片8、batch1、累计4、global batch128、Phased DMD + PDD；各节点执行相同命令，c10d自动分配rank，详见 [多机启动说明](scripts/README_16gpu_450k.md)。

```bash
# 用正式分辨率、完整 8 个在线 block 探测，包含 backward/Adam 状态；只将 OOM 作为可重试失败。
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 python scripts/tune_batch.py --config configs/wan13b_480p.json \
  --candidates 2 3 4 5 --gpus 4 --target-global-batch 256
source outputs/batch_probe/recommended.env
bash scripts/train_pdd.sh

# 单机8卡；effective batch 自动向上取整至至少256
NPROC_PER_NODE=8 BATCH_SIZE=4 bash scripts/cluster_train.sh

# 每个节点执行相同命令，使用相同rendezvous地址和ID
NNODES=4 MASTER_ADDR=10.0.0.1 MASTER_PORT=29571 RDZV_ID=pdd-run01 \
  NPROC_PER_NODE=8 BATCH_SIZE=4 bash scripts/cluster_train.sh
```

Global batch = NNODES × NPROC_PER_NODE × BATCH_SIZE × GRAD_ACCUM。脚本不提交集群作业；用户在已有资源分配中启动。支持 `PYTHON_BIN/CONFIG/CHECKPOINT/WEIGHT_INDEX/EMBEDDING_DIR/PROMPT_EMBEDDINGS/NEGATIVE_EMBEDDINGS/OUTPUT/STEPS/RESUME` 环境变量；其他参数通过 `--set key=value`。`DRY_RUN=1` 打印完整启动命令。

```bash
RESUME=outputs/wan13b_480p/step_000025 STEPS=250 bash scripts/train_pdd.sh
```

## 性能与生成测试

```bash
CONFIG=configs/wan13b_480p.json bash scripts/eval_pdd.sh \
  --student outputs/wan13b_480p/step_000250 --limit 64 --seeds 42 43 \
  --nfe 2 4 8 --teacher-steps 50 --decode
python scripts/summarize_eval.py outputs/evaluation
MPLCONFIGDIR=/tmp/pdd-matplotlib python scripts/preview_eval.py outputs/evaluation
```

同一 prompt/seed 比较50步 CFG Euler teacher（100次backbone调用）与 PDD；这里的teacher基线不是原生UniPC。评测使用固定的留出prompt，记录预热后的采样时间、真实 backbone 调用数、显存、latent MSE、跨 seed latent 差异，并可导出原生 VAE 解码 MP4。时间不包含模型切换、预热、VAE 和文件写入；测量时仅将当前被测网络驻留 GPU，显存仍包含 latent buffers 及启用解码时的 VAE。latent 指标仅用于排错，**不是 VBench、感知质量或论文 diversity 分数**。少量 smoke 更新也不代表模型收敛。可直接用已有 VBench 环境对这些视频逐方法计算自定义 prompt 的质量维度：

```bash
VBENCH_CACHE_DIR=/path/to/vbench_weights /path/to/vbench/python scripts/score_vbench.py outputs/evaluation
```

该脚本报告各维度，不把自定义 prompt 分数冒充官方 overall。完整 VBench 可复用 `AnyFlow/far/metrics/vbench.py` 的 evaluator，需安装上游 VBench 和准备其模型、标准 prompts/metadata 后评估标准命名的视频。

实测日志在 `reports/`，实际训练/生成结果在 `outputs/`。最终数值和验证范围见 `reports/validation.md`。

本机真实训练 prompt 的完整检查（训练→保存→teacher/PDD 多 seed 生成→MP4→汇总）可用：

```bash
bash scripts/validate_local.sh
```

该入口默认使用 `configs/local128_480p.json` 和已准备的128条训练prompt。输出目录已有训练日志时，必须指定新 `OUTPUT` 或使用 `RESUME`，防止混合两次运行。

本机较大 batch 的入口是 `scripts/train_local_max.sh`：每卡batch4、梯度累计2，global batch32，使用128条训练prompt工程子集。该累计配置实测峰值allocated59.79 / reserved74.57 GiB，约0.260训练样本/s。batch5预留78.30 GiB且吞吐下降，因余量不足未作为推荐。集群脚本默认每卡batch4并计算累计至global batch256；本机尚未验证global batch256。最终实测值见 `reports/validation.md`。集群应重新探测 batch；探测器覆盖全部8个在线block，并额外确认推荐的实际梯度累计配置，默认按95%的预留显存上限筛选（可用 `--memory-fraction` 调整）。

早期192p实验已作废，配置移至 `reports/obsolete_192p_configs/`；历史 outputs/logs 仅保留排错，不作为 Wan1.3B 的性能、质量或 batch 推荐依据。

训练中 TensorBoard loss 和固定 prompt 预览默认已在450k启动配置启用；频率、查看方式和集群依赖安装见 [训练脚本说明](scripts/README_16gpu_450k.md#集群环境和训练预览)。

## Wan2.1 T2V 14B / 480p

新增入口 `scripts/train_wan21_14b_480p_450k.sh`。默认 checkpoint 为共享盘
`/mnt/data/butong/Wan2.2/checkpoints/Wan2.1-T2V-14B`，使用同一共享 Python 环境。
每个节点运行相同命令（MASTER_ADDR 填主节点可达地址，不需要 NODE_RANK）：

```bash
MASTER_ADDR=<主节点地址> NNODES=2 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=8 \
  bash scripts/train_wan21_14b_480p_450k.sh
```

默认 832×480 / 81 帧，450613 条加权文本条件，128 heads，Midpoint 轨迹蒸馏；
每卡 batch=1、GRAD_ACCUM=64、16 卡有效 batch=1024。`GRAD_ACCUM=128` 可扩到 2048，
但每次更新的计算量也翻倍。64 是较大的默认值，不是实测最大值，增大累积并不会减少单个 microbatch 的激活开销。
轨迹缓存默认放 CPU，因此增加累积主要增加 CPU 内存与更新耗时，不保留跨 microbatch 的计算图。

student 和冻结 teacher 均按 Transformer block 做 FSDP/HSDP；student 保留 FP32 master 参数与 Adam 状态，BF16 计算；teacher 为 BF16。
只由 rank0 读取完整权重，其他 rank 在 meta device 初始化，FSDP 广播并分片。
加载器支持原生单文件和索引指定的多分片 safetensors，并检查缺失、重复、额外键及索引一致性。
14B 默认由 rank0 在分片通信前用6线程顺序预读权重到系统页缓存，改善共享盘 mmap 的冷读取；可用 `--set checkpoint_prefetch=false` 关闭。
`--set activation_cpu_offload=true` 可进一步把用于反向传播的保存激活卸载到CPU，适合本机4卡的显存压力；有足够显存时可关闭以减少传输。

默认输出为 `outputs/wan21_14b_480p_450k_rcm6_fsdp16`，250 次 optimizer 更新，
每 25 次保存 checkpoint 并生成 RCM6 固定 prompt 预览，保留 TensorBoard 和普通 loss 日志；14B 默认每8个 microbatch 额外打印本卡 loss 和耗时。
空目录从头训练；同一输出目录自动恢复最新完整 checkpoint。全新实验使用新 `OUTPUT_DIR`，
需要强制禁止恢复时设 `RESUME_PATH=none`（已有训练日志的目录会拒绝覆盖）。

14B 的完整 checkpoint 包含 FP32 student 和两份 Adam 动量，体积约 160 GiB；
保存时 rank0 需要相应 CPU 内存。默认沿用完整 checkpoint 格式，精确恢复需要相同 world/batch/accum。
RCM prompt、450k 文本条件、T5 和 VAE 格式均兼容；模型维度自动从 14B 配置读取。
`optimizer_cpu_offload=auto` 默认在有效分片数小于8时卸载Adam动量到CPU，仅在更新/保存时搬回GPU；
16卡、8分片默认关闭此项，本机4卡自动开启。可通过 `--set optimizer_cpu_offload=true/false` 显式覆盖。
本机4卡只卸载激活仍会在第二个更新OOM，因此不能用第一个更新的显存判断长期运行。
此入口仍是纯 PDD，不包含 DMD、fake-score 网络或 GAN loss。

本机短测与验证结果见 `reports/wan14b_support.md`。
