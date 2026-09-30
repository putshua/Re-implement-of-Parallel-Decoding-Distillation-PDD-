# 训练与实验操作

## 环境与覆盖顺序

启动脚本依次使用专用包装、通用多机包装、`train_pdd.sh`、`pdd.train`。
JSON 提供基础值，包装脚本通过环境变量转换为 CLI 参数；尾部 `--set key=value`
再次覆盖同名配置。不要只读 JSON 判断实际 LR/BS/GA；用 `DRY_RUN=1` 查看命令，
用实际输出目录的 `config.json` 查看运行配置。

所有节点必须能读取共享 Python、checkpoint、prompt cache 和输出目录。
`scripts/check_runtime.py` 检查依赖；450k 包装启动前检查 tar 是否齐全。

## 纯 PDD 的 450k 启动示例

两个节点各执行一次，`train_pdd.sh` 直接使用纯 PDD 配置，避免进入联合训练默认入口：

```bash
CONFIG=/mnt/data/butong/PDD/configs/wan1p3b_480p_450k_16gpu.json \
RDZV_ENDPOINT=<主节点hostname>:29622 RDZV_ID=<唯一实验ID> \
NNODES=2 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=8 \
BATCH_SIZE=1 GRAD_ACCUM=4 OUTPUT=/mnt/data/butong/PDD/outputs/<新实验名> \
bash scripts/train_pdd.sh
```

该底层入口不执行 450k 包装的完整 tar 预检查；需要时先运行 `scripts/check_data.py --help`。
Global batch = 节点数 × 每节点进程数 × 每卡 batch × 梯度累积。

## 保存与恢复

当前 1.3B 联合及终点 DMD 包装默认每5个 G update 滚动保存、每50个永久保留，最终一步也保存。
其他纯 PDD 和历史验证配置有不同频率，以实际配置为准。
完整 checkpoint 使用 `COMPLETE` 标记，包含模型、optimizer、rank 状态；开启 DMD 时还有 fake 状态。

- `student_init`：只导入 student 权重，optimizer 从头开始，用于从 PDD 转入 DMD。
- `RESUME=auto` / `RESUME_PATH=auto`：对支持该变量的入口，恢复当前输出目录的完整训练状态。
- **终点 DMD 专用包装目前强制 `RESUME_PATH=none`**。要恢复它，应使用底层 `train_pdd.sh`，
  指定原输出目录的 `config.json`、相同 world/BS/GA/shard 和 `RESUME=auto`。
  本轮整理没有改变这一行为。
- 已有训练目录不要用于不同算法或初始化。无完整 checkpoint 的自动重启行为由
  `auto_restart_without_checkpoint` 控制，重启会归档已有状态；不是恢复丢失的更新。

## 日志与预览

- `OUTPUT/logs/`：每节点普通日志，包含 loss 和初始化事件。
- `OUTPUT/metrics.jsonl`：全局聚合标量；`joint/fake_updates` 独立计数。
- `OUTPUT/tensorboard/`：TensorBoard events。
- `OUTPUT/fixed_prompt/index.html`：固定 prompt 对比，视频在各 step 子目录。

默认预览频率及 prompt 数量随入口不同，可用 `FIXED_PROMPT_EVERY/COUNT/EMBEDDINGS` 覆盖。
终点 DMD 默认32个指定 prompt、每25步；无 fake-only 预热配置。
`dmd_weight` 缩放 generator 的 DMD 项，loss 采用 latent mean，反传前除以 GA。
BF16 没有使用 FP16 GradScaler；FP32 optimizer 检查见 `optimizer_precision` 日志。

## 评测

```bash
CONFIG=configs/wan13b_480p.json bash scripts/eval_pdd.sh \
  --student /absolute/checkpoint --limit 32 --seeds 42 43 44 45 46 \
  --nfe 4 --teacher-steps 50 --decode
```

选择与 checkpoint 的 head、shift、分辨率匹配的配置。
普通 evaluate 的 teacher 与原生官方 teacher 是不同评测路径；官方对比入口为
`scripts/eval_rcm6_official_teacher.sh`。延迟、latent MSE 或 DMD loss 不等于感知质量，
应同时检查固定 seed 的外观和运动。历史实验结果见 reports。
