# Wan1.3B / Phased DMD + PDD / 480p / 450k

当前checkpoint策略（2026-09-15最新）：每5个G step保存一份滚动checkpoint，只保留最新完整滚动版本；每50步的checkpoint永久保留。新版本完整保存后才清理旧滚动版本（保留step目录命名，RESUME=auto仍可使用）。取消额外的首步保存，训练最终一步仍保存。固定prompt每25步生成，启动/结束预览设置保留。例：step105保存后保留step50、100、105；checkpoint包含student、fake、两个optimizer及rank状态。已启动的进程需重启后应用，现有文件此刻不会被清理。


2026-09-15 恢复修复：1.3B联合/对照默认在第1次G更新后保存，之后每5步及最终保存；预览仍每25步。`RESUME=auto` 在没有任何完成checkpoint但存在旧metrics时，会将config/metrics/TensorBoard/预览/未完成checkpoint移动到 `OUTPUT/restart_archives/<timestamp>/`，打印 `restart_without_checkpoint` 后从初始化重跑。没有checkpoint的更新无法恢复；已有完成标记但文件缺失或不兼容时仍拒绝重置。设置 `--set auto_restart_without_checkpoint=false` 可恢复严格报错行为。


当前默认（更新）：原始 Wan2.1-T2V-1.3B 初始化，`student_init=null`；PDD/DMD 权重均1，从第一个G更新生效（无ramp）；fake:G=5:1。32卡BS1、GA4、global batch128。新输出目录后缀为 `_gbs128_teacherinit_w1_fake5`，避免恢复旧实验。matched对照同步改为原始Wan初始化，但DMD权重仍为0。以下旧实验说明中的step100初始化、0.1权重与1:1更新比属于历史设置。


当前默认 **4节点×8卡、BS1、GA4、global batch128**。显式NNODES=2时自动GA8。

2026-09-14 起，旧入口 `train_wan1p3b_480p_450k.sh` 默认启动 Phased DMD + PDD 对照实验；`_16gpu.sh` 和原 fresh 文件名仍是该入口的别名。

## 启动

H3 风格 c10d rendezvous：每个节点执行相同命令，不需要 NODE_RANK，也不会通过 SSH 自动启动其他节点。

```bash
cd /mnt/data/butong/PDD
# 两个8卡节点各运行一次，同一个MASTER_ADDR
MASTER_ADDR=主节点IP NNODES=2 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=8 \
  bash scripts/train_wan1p3b_480p_450k.sh
# 本机四卡
NNODES=1 NPROC_PER_NODE=4 FSDP_SHARD_SIZE=4 \
  bash scripts/train_wan1p3b_480p_450k.sh
# 同rollout纯PDD对照（按实际节点数设NNODES/NPROC_PER_NODE）
EXPERIMENT=pdd_matched NNODES=1 NPROC_PER_NODE=4 FSDP_SHARD_SIZE=4 \
  bash scripts/train_wan1p3b_480p_450k.sh
```

默认每卡BS1，总batch128；4卡GA32，16卡GA8。显式覆盖GRAD_ACCUM时以实际乘积为准；自动计算不能整除128时拒绝启动。16卡shard8使用HSDP的2副本×8分片，本机4卡shard4使用FULL_SHARD。默认student、fake和冻结teacher均分片，gradient checkpointing与CPU activation offload开启。fake不与student共享参数。

同时运行的任务必须使用不同的 rendezvous endpoint/ID 和输出目录。默认 RDZV_ID 按实验名区分；同机器并行多任务仍需指定不同 MASTER_PORT。`DRY_RUN=1` 可打印命令验证。

## 实验配置

- 联合：`configs/wan1p3b_480p_450k_phased_dmd_pdd.json`。
- 对照：`configs/wan1p3b_480p_450k_pdd_matched.json`。
- 默认输出：`outputs/wan1p3b_480p_450k_${EXPERIMENT}_gbs128`。
- 两者从 `outputs/wan1p3b_480p_450k_rcm6_fsdp16_20260912/step_000100` 仅加载student权重，新建优化器；可设置 `STUDENT_INIT=/absolute/checkpoint`。原始Wan作为teacher和fake初始化。
- 832×480、81f、128heads、PDD4的四个32-head phase；新噪声前缀重算、当前phase反传，phase在GA内均衡覆盖。
- G/Fake LR均1e-6；fake:G=1:1；traj权重1，DMD权重0.1，前25个G更新线性ramp。默认250个G更新。
- 训练teacher CFG5、traj skip10；DMD teacher全层CFG5；DMD time shift2、RF gap0.001、residual归一化、fake方差倒数上限10。
- 450613加权prompt，只读embedding。RCM6固定prompt只用于预览。启动前检查所有必需tar；不继承小样本EMBEDDING_DIR/PROMPT_EMBEDDINGS。

尾部 `--set key=value` 可覆盖，如 `--set fake_updates_per_g=5`。默认比例与loss权重是初始实验配置，不是最优值结论。

## 保存、恢复、预览和日志

`MAX_ITER`/`STEPS`、`SAVE_EVERY`、预览频率均按 **generator optimizer updates** 计数；fake另有计数。默认每25步及最终保存，启动/每25步/最终生成RCM6、seed42、NFE4预览。保留原FIXED_PROMPT_*环境变量覆盖。

联合checkpoint：`model.pt`（student+optimizer）、`fake.pt`（fake+optimizer）、`joint_state.json`、各rank的RNG/数据游标、最后写入`COMPLETE`。对照不需要fake.pt。模型格式可交给原评测脚本。`RESUME=auto`只查当前输出目录完整checkpoint，检查联合实验所需额外文件；不能用纯PDD的完整训练状态冒充联合resume。恢复要求相同world/shard/BS/GA与算法配置；改变steps上限可以继续训练。`RESUME=none`须使用新输出目录。

日志文件在 `OUTPUT/logs/`，metrics.jsonl和TensorBoard由rank0写入。记录global loss、每phase的traj/DMD/fake loss、direction RMS、score time、DMD权重和fake更新数。DMD surrogate不是KL数值或质量分数。查看 `OUTPUT/fixed_prompt/index.html` 可预览，MP4为完整480p。

所有节点使用同一共享环境：

```
/mnt/data/butong/miniconda3/envs/causvid-wan21-final/bin/python
```

环境不可执行时直接报错，不回退本地Python。配置与数据及checkpoint必须挂载在所有节点相同绝对路径。配置在启动时读取。

设计见 [算法设计](../reports/pdd_phased_dmd_design.md)，测试范围见 [本机验证](../reports/wan13b_joint_validation.md)。
