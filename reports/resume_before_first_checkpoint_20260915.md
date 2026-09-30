# 机器重启后找不到 checkpoint 的修复

实际检查目录：`outputs/wan1p3b_480p_450k_phased_dmd_pdd_gbs128_teacherinit_w1_fake5`。

日志 `logs/train_dlc38i58q1f414kr-master-0_20260914_201056_MiR3Nh.log` 显示完成G step1–4，step5在fake更新期间停止。metrics共4条，最后step4、global batch64（16GPU、BS1、GA4）。该目录没有任何step_* checkpoint，原save_every=25。重启日志明确报 `has training metrics but no complete checkpoint with 16 rank files`。因此不是已有权重无法加载，而是首次保存之前中断，随后严格防覆盖检查拒绝启动。

修复：

- 1.3B联合/对照增加save_first_step=true，默认save_every=5，保存1/5/10/.../final；固定预览仍25。
- auto恢复存在完整checkpoint时照常恢复student/fake/两个optimizer和rank状态。
- 无任何COMPLETE且有metrics时，默认归档旧config、metrics、tensorboard、fixed_prompt和未完成step目录，日志保留原处，打印明确reset事件后从0重跑。归档发生于下次启动rank0，当前目录未被手动移动。
- 已有COMPLETE但缺文件/卡数不兼容时不自动重置；显式RESUME=none仍不覆盖已有日志。
- 重写checkpoint前清除旧COMPLETE，完成全部保存后才写标记；增加checkpoint_start/checkpoint_complete日志。
- 14B wrapper保持原save_every=25默认。

验证：9项run_state测试、7项launcher测试通过；bash语法/ruff/编译检查通过。测试覆盖归档保留、损坏完整checkpoint拒绝重置、有效checkpoint优先恢复、重复启动、保存周期。没有新启动集群训练或进行断电模拟。这4步未保存的参数无法从loss日志恢复。
