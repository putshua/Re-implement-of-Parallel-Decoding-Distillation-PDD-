# 源码地图

## 目录职责

| 目录 | 内容 |
|---|---|
| AnyFlow/pdd/ | 本项目算法、模型适配、训练与评测 |
| AnyFlow/far/、AnyFlow/options/ | 保留的上游实现；独立 pdd.train 不调用其训练器 |
| assets/ | 本地参考实现和辅助资源，不能与上游 assets 混为一谈 |
| configs/ | 算法配置与历史工程验证配置，见该目录 README |
| scripts/ | 环境、启动、数据检查、评测、gallery 工具 |
| tests/ | 数学、模型、数据、启动和恢复测试 |
| outputs/ | checkpoint、metrics、TensorBoard、生成视频 |
| reports/ | 有日期的实验结果、排查记录、keeper payload |
| docs/archive/ | 归档说明，参数不再作为当前默认值 |

## AnyFlow/pdd 模块

| 文件 | 职责 |
|---|---|
| core.py | sigma 网格、head 系数、PDD loss、采样 |
| model.py | 原生 Wan 权重转换、多 head 投影、teacher CFG |
| phased_dmd.py | 条件加噪、fake DSM、DMD surrogate、局部 phase rollout |
| joint_training.py | phase/final 模式、fake 更新、generator loss、joint checkpoint |
| train.py | 配置解析、数据、模型、主训练循环、日志和保存 |
| distributed.py | FSDP/HSDP/DDP、optimizer state、FP32 检查 |
| run_state.py | 完整 checkpoint 选择、恢复与滚动保留 |
| checkpoint_io.py | 权重文件预读 |
| data.py | prompt embedding 与加权 tar 数据读取 |
| monitoring.py | TensorBoard、固定 prompt 生成、预览页面 |
| evaluate.py、native.py | PDD/teacher 评测与原生 Wan 支持 |

`training_objective=phased_dmd_pdd` 是联合训练器内部入口名，不代表必然混合两项 loss。
还必须看 `dmd_endpoint_mode`、`dmd_weight` 和 `traj_weight`。
`final` 使用完整 rollout 的反向传播；`phase` 使用 stop-gradient 前缀和当前 phase 反传。

保持 `AnyFlow/pdd` 路径是为了兼容现有 PYTHONPATH、集群命令和测试；本轮没有移动包。
顶层 `.git` 当前不是有效 Git 仓库，`AnyFlow/` 是独立上游仓库；整理没有重建 Git 元数据。
