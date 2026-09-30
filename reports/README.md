# 实验记录索引

报告反映各自生成日期的代码与配置，不是当前默认值。

| 记录 | 内容 |
|---|---|
| [DMD audit](dmd_audit_20260920.md) | 纯色失败分析、归一化修正与短程验证 |
| [正式 DMD / FP32](dmd_formal_fp32_20260920.md) | loss 权重测试、FP32 optimizer 实证、正式任务 |
| [早期 endpoint DMD](dmd_endpoint_20260917.md) | 旧实验及 rendezvous 排查 |
| [Phased 设计](pdd_phased_dmd_design.md) | phase 联合监督设计记录 |
| [1.3B 联合验证](wan13b_joint_validation.md) | 联合训练的工程验证 |
| [14B 联合](wan14b_joint_20260916.md) | 14B 分片与 offload 记录 |

`keeper_*.json`、`keeper_*/` 是历史提交 payload 和响应，不能据此判断当前节点是否空闲。
原始日志和报告保留原位置；训练结果在 `../outputs/`。
2026-09-21 检查正式 DMD step175 时，四个抽查样本运动明显下降：
loss 稳定并不能证明质量改善。该结果尚未说明 fake 预热或联合监督能解决退化。
