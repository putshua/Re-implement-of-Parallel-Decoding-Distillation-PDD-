# PDD / Wan

Wan2.1 的 PDD 轨迹蒸馏及 DMD 实验代码。自有实现位于 `AnyFlow/pdd/`，上游 AnyFlow 保留在 `AnyFlow/`；训练入口是 `python -m pdd.train`。

## 先选择实验

| 模式 | 启动脚本（位于 scripts/） | 实际目标 |
|---|---|---|
| 纯 PDD | `train_pdd.sh` | 原 PDD 在线轨迹监督；由 CONFIG 指定数据与规模 |
| Phased DMD + PDD | `train_wan1p3b_phased_dmd_pdd.sh` | phase 终点 DMD + 局部轨迹 MSE |
| 同 rollout 的 PDD 对照 | `train_wan1p3b_pdd_matched.sh` | phased rollout，但 DMD 权重为 0 |
| PDD checkpoint → 终点 DMD | `train_wan1p3b_480p_450k_dmd_endpoint_from_pdd.sh` | 四步完整 rollout，只有最终 x0-DMD |
| 14B 纯 PDD | `train_wan21_14b_480p_450k.sh` | 原 PDD 轨迹监督 |
| 14B 联合实验 | `train_wan21_14b_480p_phased_dmd_pdd.sh` | Phased DMD + PDD |

`train_wan1p3b_480p_450k.sh` 是旧的联合/对照入口，默认联合实验。
`_16gpu.sh`、`_rcm6_fresh.sh` 都是它的兼容别名，文件名不代表实际卡数或初始化策略。
新入口只是明确命名的包装，不改变算法和默认参数。

## 两节点启动

两个节点各执行同一条命令，使用相同 rendezvous 和共享盘路径。例如终点 DMD：

```bash
cd /mnt/data/butong/PDD
RDZV_ENDPOINT=<主节点hostname>:29621 RDZV_ID=<唯一实验ID> \
NNODES=2 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=16 \
BATCH_SIZE=1 GRAD_ACCUM=4 MAX_ITER=250 \
OUTPUT_DIR=/mnt/data/butong/PDD/outputs/<新实验名> \
bash scripts/train_wan1p3b_480p_450k_dmd_endpoint_from_pdd.sh
```

上式 global batch=64。先加 `DRY_RUN=1` 可检查完整命令，不加载模型或启动训练。
脚本不会自动向另一个节点提交任务；keeper 两端提交同一 payload 的示例见
[正式实验 payload](reports/keeper_dmd_formal_20260920.json)。使用前重新确认节点空闲。

共享 Python 默认固定为 `/mnt/data/butong/miniconda3/envs/causvid-wan21-final/bin/python`，
不存在时直接报错。依赖入口是 `requirements.txt`；不要用上游完整依赖覆盖该环境。

## 算法与验证范围

- 生成分辨率为 480p；450k 入口读取 450,613 条加权文本 embedding，不读取真实视频 latent。cache 名称中的 192p 不影响生成分辨率。
- 终点 DMD 默认从纯 PDD step100 **只加载 student 权重**，新建 optimizer，trajectory 权重为 0。Phased 联合实验默认从原始 Wan 初始化，PDD/DMD 权重均为 1。
- 当前没有 fake 预热阶段。终点 DMD 使用 boundary 归一化、fake:G=5:1。
- BF16 前向计算、FP32 master 参数和 Adam moments；首次更新打印运行时精度检查。
- 归一化修正后的 DMD 实验仍观察到运动退化，不能把 loss 稳定视为质量收敛。详见 [DMD 检查](reports/dmd_audit_20260920.md) 和 [精度验证](reports/dmd_formal_fp32_20260920.md)。

## 导航

- [训练、恢复、日志和评测](docs/training.md)
- [源码模块与目录职责](docs/codebase.md)
- [配置索引与覆盖优先级](configs/README.md)
- [脚本分类](scripts/README.md)
- [实验记录索引](reports/README.md)
- [历史 README](docs/archive/README_before_20260924.md)：旧参数仅作历史参考

测试入口：

```bash
PYTHONPATH=AnyFlow OMP_NUM_THREADS=1 \
/mnt/data/butong/miniconda3/envs/causvid-wan21-final/bin/python \
  -m unittest discover -s tests -v
```

`outputs/` 保存实际模型、视频和日志；`reports/` 保存检查结果及历史任务 payload。
上游代码与许可证见 `AnyFlow/README.md`、`AnyFlow/LICENSE`。
