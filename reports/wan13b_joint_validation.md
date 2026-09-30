# Wan1.3B Phased DMD + PDD 本机验证

2026-09-14。正在进行实际四卡验证，以下仅列已完成检查；尚未宣称训练收敛或质量提升。

代码：独立phase-aware fake-score（四head），generator128head；phase-balanced GA；fresh on-policy prefix；phase endpoint DMD + local midpoint velocity MSE。

配置：480p=832×480、81f、BS1、4GPU×GA32=128；450613加权prompt；student从原step100加载权重，fresh optimizers；G/Fake LR1e-6；fake:G=1:1；DMD权重0.1前25步ramp；对照仅关闭DMD。

已通过：

- 7项启动脚本测试：共享解释器、4卡GA32、16卡GA8、matched配置、旧入口、14B显式配置兼容。
- 4项resume选择测试：缺失fake.pt的联合checkpoint不能auto resume。
- 5项数学/梯度测试：条件加噪、fake目标、DMD方向、detach与PDD4采样一致性。
- 450613索引、所有需要tar路径均存在。

实际训练输出：

- 联合：`outputs/wan13b_joint_local4_gbs128_20260914`
- 对照：待运行。

完整训练/恢复/预览结果将在测试结束后更新此报告。

## 按用户要求提前交付脚本

用户要求“loss不爆炸就把脚本给我，我来挂”，因此取消后续恢复/对照测试队列，不等待长测。

当前已实际验证：4×A800、BS1、GA32；fake-score完成32/32累积和一次optimizer更新；generator完成24/32联合反传微步，打印loss依次0.10299641、1.18134534、0.59473515、0.61209792，未见NaN/Inf/OOM；显存快照约18.2GiB/卡。fake loss按phase具有较大尺度差异，不能与generator loss直接比较。

40项CPU回归测试全部通过（reports/joint_all_cpu_tests.log）。原始单步本机测试仍在运行，配置steps=1，完成保存/预览后自动退出；此刻尚未验证完整generator更新、联合保存/恢复，也尚未运行matched GPU对照，不能宣称已完成这些检查或证明长期收敛。后续测试队列已取消。

集群命令（每个节点相同）：

```bash
MASTER_ADDR=主节点IP NNODES=2 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=8 \
  bash scripts/train_wan1p3b_480p_450k.sh
```

默认BS1、GA8、全局batch128；EXPERIMENT=pdd_matched切换同rollout控制组。脚本已dry-run验证。
