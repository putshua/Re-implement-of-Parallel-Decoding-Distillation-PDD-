# 450k 多机训练说明

当前统一入口说明见 [项目首页](../README.md)、[训练文档](../docs/training.md) 和 [脚本索引](README.md)。

`train_wan1p3b_480p_450k_16gpu.sh` 是兼容别名，继承主入口默认4×8卡；要运行16卡，显式设置 `NNODES=2 NPROC_PER_NODE=8`。
旧说明已完整归档至 [历史启动说明](../docs/archive/launch_before_20260924.md)，其中旧 LR、更新比例和初始化参数不再代表当前默认值。

## 集群环境和训练预览

共享 Python、依赖检查、TensorBoard、普通日志、固定 prompt 预览及 checkpoint 恢复，统一见 [训练文档](../docs/training.md)。
