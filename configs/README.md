# 配置索引

## 算法配置

| 配置 | 用途 |
|---|---|
| wan13b_480p.json | 1.3B 纯 PDD Midpoint |
| wan13b_480p_euler.json | 1.3B 纯 PDD Euler |
| wan1p3b_480p_450k_16gpu.json | 1.3B 450k 纯 PDD；与同名脚本的当前默认目标不同 |
| wan1p3b_480p_450k_phased_dmd_pdd.json | phase 联合；原始 Wan 初始化；traj=1、DMD=1，fake:G=5 |
| wan1p3b_480p_450k_pdd_matched.json | 同 rollout 对照；traj=1、DMD=0 |
| wan1p3b_480p_450k_dmd_endpoint_from_pdd.json | PDD step100 初始化；final 模式；traj=0、DMD=1，boundary 归一化 |
| wan21_14b_480p_450k_16gpu.json | 14B 纯 PDD |
| wan21_14b_480p_450k_phased_dmd_pdd.json | 14B phase 联合 |

## 工程验证配置

`smoke`、`local128`、`batch*`、`validation*`、`monitoring*`、`shared_wan*`、
`resume*`、`hsdp*` 用于历史冒烟、batch 探测、日志或恢复验证，可能引用历史输出目录。
保留原路径供脚本和报告复现，不作为新的生产实验默认配置。

## 参数的实际来源

脚本会覆盖 JSON。特别是 matched JSON 的 LR 为1e-6，但通用450k包装默认传入2e-5；
因此直接加载 JSON 与经包装启动并不等价。本轮没有静默调整这些实验参数。
优先检查 `DRY_RUN=1` 的完整命令和输出目录中保存的 `config.json`。
