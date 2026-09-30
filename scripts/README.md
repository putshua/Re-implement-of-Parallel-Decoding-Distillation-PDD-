# 脚本导航

| 分类 | 入口 |
|---|---|
| 明确命名的联合/对照 | train_wan1p3b_phased_dmd_pdd.sh、train_wan1p3b_pdd_matched.sh |
| 终点 DMD | train_wan1p3b_480p_450k_dmd_endpoint_from_pdd.sh |
| 底层训练 | train_pdd.sh；CONFIG 决定算法 |
| 14B | train_wan21_14b_480p_450k.sh、train_wan21_14b_480p_phased_dmd_pdd.sh |
| 本机验证 | smoke_4gpu.sh、smoke_joint_4gpu.sh、validate_local.sh |
| 历史 batch 入口 | train_local_max.sh、cluster_train.sh、tune_batch.py |
| 环境/数据 | shared_env.sh、check_runtime.py、check_data.py、prepare_embeddings.py |
| 评测 | eval_pdd.sh、eval_rcm6_official_teacher.sh、compare_eval.py、summarize_eval.py |
| 展示 | eval_gallery.py、gallery_loss.py、preview_eval.py、partial_comparison.py |
| 验证工具 | verify_joint_run.py、verify_monitoring.py、check_fsdp_models.py |

`train_wan1p3b_480p_450k.sh` 默认联合训练。
`train_wan1p3b_480p_450k_16gpu.sh` 和 `train_wan1p3b_480p_450k_rcm6_fresh.sh`
是兼容别名，不能按文件名推断节点数、prompt 集合或恢复方式。
通用入口默认4节点×8卡；明确设置 NNODES/NPROC_PER_NODE 才能选择实际规模。

使用说明统一维护在 [训练文档](../docs/training.md) 和 [项目首页](../README.md)。
