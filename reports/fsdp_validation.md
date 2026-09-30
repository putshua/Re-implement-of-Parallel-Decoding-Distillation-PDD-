# H3风格多机启动与FSDP验证

实现：c10d自动分配node rank；各节点相同启动命令，仍需共享MASTER_ADDR或RDZV_ENDPOINT。主入口为 `scripts/train_wan1p3b_480p_450k.sh`，旧16gpu文件名保留为别名。

分片使用本机PyTorch2.3的FSDP1/HSDP，不是H3的FSDP2。student按transformer block分片，teacher冻结复制。16卡/shard8对应2个副本×8卡分片。完整CPU模型与optimizer checkpoint保留，兼容原评测模型权重格式；未迁移至DCP。

已通过：
- 14项CPU单元测试，包含无NODE_RANK启动、rendezvous参数、分片参数和累计计算、路径别名、小样本环境隔离及非法配置检查。
- 四卡tiny Wan通信验证：shard4全分片、shard2两副本HSDP，两者均通过累计2、全局梯度裁剪、全量权重导出、plain PDDWan严格加载，以及恢复后下一步loss/梯度范数对比。tiny模型仅用于数学与通信测试，不是低分辨率Wan1.3B性能测试。
- 真实480p最终版本通过：同机两个独立torchrun agent，各2卡，自动组网，world4/shard2。使用原manifest的128条prompt工程缓存，batch1、累计2、2个优化步。峰值allocated22.01/reserved27.45 GiB。FP32时间和积分系数保留在root输入；仅计算/参数使用BF16。完整checkpoint已保存，825个state tensors已严格加载到普通PDDWan。
- 真实480p checkpoint恢复通过：两个c10d agent从第2步恢复并完成第3步，loss=0.1577368、grad_norm=0.0760112，rank0 head索引接续为57/51。真实模型验证了恢复后能继续更新；逐数值下一步对比由tiny Wan通信测试覆盖。共享盘完整checkpoint读取耗时较长，四worker曾等待页面读取，未发生通信失败。

尚未进行16张物理GPU或真实跨主机网络验证，也未启动全量450k训练。上述本机短测不能代表16卡最大batch或全量质量结果。


关键证据：`fsdp_contract2_final.log`、`fsdp_contract4_final.log`、`hsdp_480p_final.json`、`fsdp_plain_checkpoint_load.json`、`launcher_h3_style_16gpu.txt`、`hsdp_480p_resume_result.json`。初次 `hsdp_480p_validation` 是调整root输入精度之前的中间实验，正式结论使用 `hsdp_480p_final`。
