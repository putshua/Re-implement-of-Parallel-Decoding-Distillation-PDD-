# 480p 验证记录（2026-09-11，四张 A800 80GB）

所有有效性能测试使用 Wan2.1-T2V-1.3B、832×480、81帧。早期192p实验作废，不用于性能和质量结论。

当前版本使用128 heads、Midpoint、逐样本独立采样监督区间。11项CPU单元测试已通过，覆盖逐样本融合数值与梯度、Midpoint逐样本时间/状态、stop-gradient、head初始化、patch顺序和加权数据采样。

## 当前版本训练

4卡DDP、每卡batch4、梯度累计1，完整8个在线block/优化步已通过。稳态62.00秒/步，0.258训练样本/s；峰值allocated59.74 / reserved74.32 GiB（各rank最大）。这些训练吞吐数值不是视频生成吞吐。

最终checkpoint：`outputs/per_example_480p/step_000008`，含 `COMPLETE`。128个head中73个已实际更新，shared backbone也已更新。此次只处理16条在线轨迹的8个block，共128次监督，绝不代表收敛；完整指标见 `training_480p.json` 和 `weight_updates_480p.json`。

第5步checkpoint保留所有rank的n=80轨迹，latent形状均为 `[4,16,21,60,104]`，详见 `checkpoint_480p_streams.json`。恢复第6步通过：rank0 head索引、跨rank最大loss与rank0梯度范数均与原运行完全一致，详见 `resume_comparison_480p.json`；不据单次测试保证任意环境逐位确定性。

全量450,613条索引存在，但部分源tar目录未挂载，详见 `data_check.json`。本机使用从原manifest真实caption重新编码的128条工程子集。尚未完成收敛训练或完整VBench，不以短跑loss或latent诊断代替生成质量结论。

## Batch 边界

- 每卡batch4、累计2：2个完整优化步通过，global batch32；峰值allocated59.79 / reserved74.57 GiB，稳态0.260训练样本/s。
- 每卡batch5、累计1：完成2步后预留78.30 GiB，吞吐降到0.217训练样本/s，因显存余量不足主动停止。没有宣称OOM或完整测试通过。
- 推荐入口 `scripts/train_local_max.sh` 使用batch4、累计2、allocator `max_split_size_mb:512`。机器上其他GPU任务会改变可用显存。
- 集群脚本默认batch4，并累计到global batch256；本机未实测global batch256。探测工具覆盖全部在线block并验证实际累计配置，默认预留5%容量。

## 生成验证（已完成）

使用独立的4条prompt、2个seed，对比50步CFG Euler teacher（100次backbone调用）与PDD 2/4/8 NFE。teacher不是UniPC基线。计时包含实际采样，排除加载/切换/预热/VAE/写盘；显存指标为采样时的峰值allocated，含驻留VAE，不包含解码阶段的峰值。

首组解码视频的视觉检查：teacher场景清晰，当前PDD 2/4/8步输出明显模糊。这里只训练8步，不是可用的收敛少步模型。任何速度比均不是同等画质的加速结论。真实帧对比保存在 `outputs/evaluation_480p/preview_prompt0000_seed42.png`；可用 `scripts/preview_eval.py` 重建。


全部32个生成记录和MP4检查通过：4条prompt × 2个seed × 4种方法，均为832×480、81帧、16FPS。每种方法8个样本，teacher与PDD使用配对prompt/noise。详见 `video_validation_480p.json`、`evaluation_summary_480p.json`。

| 方法 | Backbone调用 | 平均采样秒数 | 相对teacher速度 | 采样峰值allocated GiB |
|---|---:|---:|---:|---:|
| Teacher CFG Euler50 | 100 | 178.41 | 1.00× | 5.05 |
| PDD 2 NFE | 2 | 3.62 | 49.25× | 5.08 |
| PDD 4 NFE | 4 | 7.25 | 24.62× | 5.08 |
| PDD 8 NFE | 8 | 14.50 | 12.31× | 5.08 |

这些速度比仅表示计算耗时差异，当前模型的生成质量明显落后于teacher。没有完成论文质量复现、VBench overall或感知diversity评测。`score_vbench.py` 提供后续自定义prompt维度评测入口，本次未运行该脚本。

## 交付入口

- `scripts/train_local_max.sh`：本机4卡、batch4、累计2，480p工程子集。
- `scripts/cluster_train.sh`：默认8卡/节点、batch4、目标global batch256，完整450k索引；多节点设置NNODES/NODE_RANK/MASTER_ADDR。
- `scripts/tune_batch.py`：实际机器上的batch与累计配置探测；本次采用独立训练命令完成上述边界验证，未完整跑一次自动探测流程。
- `scripts/validate_local.sh`：训练、保存、4-prompt双seed生成和汇总。
- `scripts/eval_pdd.sh` / `scripts/preview_eval.py`：推理对比与实际视频帧预览。

Python核心11项单元测试和Ruff检查通过；shell语法、多机命令展开检查通过。正式大规模训练请使用 `configs/wan13b_480p.json`（Midpoint）或 `configs/wan13b_480p_euler.json`（Euler）；本次GPU短跑验证的是Midpoint，Euler仅完成算法单元测试。
