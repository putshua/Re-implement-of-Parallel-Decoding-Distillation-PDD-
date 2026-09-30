# Wan2.1 14B 支持与 PDD+DMD 方案

## 本次实现

- `configs/wan21_14b_480p_450k_16gpu.json`、`scripts/train_wan21_14b_480p_450k.sh`。
- 原生 6 分片 checkpoint 加载、逐分片转换和严格完整性检查；保留 1.3B 单文件兼容。
- 14B 开启 `low_memory_init=true`，只有全局 rank0 读权重，其他 rank 构造 meta 模型。
- student 在 CPU/meta 上构造后直接 FSDP 分片，避免完整 student 与 teacher 同时挤入一张 GPU。
- `teacher_fsdp=true`：冻结 teacher 也按 block 分片；保留原 PDD teacher 的 CFG=5、skip_layer=10、midpoint 监督设置。
- `offload_streams=true`：每个 microbatch 的后继 latent 与文本 embedding 存 CPU，使用前取回；不保留计算图。恢复 checkpoint 时遵循同一策略。
- 所有 head 从 checkpoint 的原输出投影初始化；14B 输出维度为 5120→64。
- 14B backbone 参数 14,288,491,584，额外 127 个 head 参数 41,623,488，student 总参数 14,330,115,072。
- 16 卡默认 batch1 × accum64 = 1024；accum128 对应2048。64不是实测显存极限，增大累积会同比拉长更新周期。
- 480p/81f 时每条保留轨迹含 FP32 latent 与 BF16 text，约12 MiB；accum64 约0.75 GiB CPU/rank。
- FP32 student + Adam 两个动量的完整 checkpoint 约160 GiB，不含少量元信息和各 rank 轨迹；保存和恢复仍使用现有完整 checkpoint 格式。

## 启动

每个节点使用同一 MASTER_ADDR，运行：

```bash
MASTER_ADDR=<主节点地址> NNODES=2 NPROC_PER_NODE=8 FSDP_SHARD_SIZE=8 \
  GRAD_ACCUM=64 bash scripts/train_wan21_14b_480p_450k.sh
```

默认共享 Python `/mnt/data/butong/miniconda3/envs/causvid-wan21-final/bin/python`；
默认输出 `/mnt/data/butong/PDD/outputs/wan21_14b_480p_450k_rcm6_fsdp16`。
RCM6 默认 prompt、普通日志、TensorBoard loss、每25次更新保存/预览和 auto resume 均沿用现有入口。

## 验证

31项单元测试通过（`reports/wan14b_tests.log`），其中包含6项启动测试，启动测试也单独验证通过（`reports/wan14b_launcher_tests.log`）。
4卡小模型HSDP（2组×2分片）测试通过：meta权重同步、冻结teacher的CFG/跳层与未分片参考一致、累积反传、checkpoint保存后精确恢复（`reports/wan14b_fsdp_regression.log`）。
真实480p短测已完成，退出码0，最终日志 `reports/wan14b_480p_smoke3.log`。
本机真实短测是4张A800、batch1、accum2、两个optimizer update，使用450k文本入口，结束时生成一个RCM预览。真实短测最终以 `reports/wan14b_480p_smoke3.log` 为准。
第一次初始化因共享盘 mmap 页读取很慢而主动停止，当时尚未发生训练更新；并非模型训练OOM。新增6线程顺序预读、在初始化分片通信前完成，由全局rank0执行；进程组初始化等待和多机 rendezvous 等待上限均设为3600秒（可覆盖）。辅助预读实际读完约53.23 GiB，用时约418秒。
预读辅助函数新增2项测试通过，日志 `reports/wan14b_prefetch_tests.log`。
新增 `activation_cpu_offload` 选项，可用 `--set activation_cpu_offload=true` 开启；本机4卡短测启用此项。它通过 PyTorch save_on_cpu 配合梯度检查点，把保存用于反向的激活放CPU，增加CPU内存和PCIe传输开销。4卡HSDP下已验证卸载前后梯度完全一致，保存/恢复通过，见 `reports/wan14b_fsdp_offload_regression.log`。
完整14B checkpoint保存会产生约160 GiB数据，本次短测关闭此项，另用小模型验证新FSDP组合的保存/恢复。
本机短测复现命令（请设置全新的输出目录）：

```bash
NNODES=1 NPROC_PER_NODE=4 FSDP_SHARD_SIZE=4 BATCH_SIZE=1 GRAD_ACCUM=2 \
MAX_ITER=2 RESUME_PATH=none OUTPUT_DIR=/mnt/data/butong/PDD/outputs/wan14b_smoke_new \
FIXED_PROMPT_AT_START=0 FIXED_PROMPT_AT_END=1 FIXED_PROMPT_COUNT=1 \
FIXED_PROMPT_NFE='[4]' bash scripts/train_wan21_14b_480p_450k.sh \
  --set activation_cpu_offload=true --set save_checkpoints=false
```

第二次短测首个更新完成，loss=0.212466、峰值分配61.09 GiB；第二个更新在Adam状态常驻后反向OOM（约76.7 GiB已分配）。日志保留于 `reports/wan14b_480p_smoke2.log`。
为此新增 `optimizer_cpu_offload=auto`：有效FSDP分片数<8时，将Adam动量在microbatch期间保存在CPU，仅在optimizer.step和checkpoint收集期间转回GPU；dtype不变，CPU step计数器不搬动。恢复checkpoint后也应用此策略。默认16卡8分片不启用，本机4卡自动启用。
Adam动量搬移前后逐张量完全一致，搬回GPU后的HSDP checkpoint保存和精确恢复也通过，见 `reports/wan14b_fsdp_adam_offload_regression.log`。第三次短测使用此修复，已从头完成两个更新及NFE4预览。
尚不能把本机短测视为16卡多机或长训稳定性测试。

## PDD 与 DMD 可以结合，但本次尚未加入

建议把这当作两阶段研究方案：先得到纯PDD的14B基线，再从该权重开始加入分布匹配。
保留局部轨迹损失，同时对PDD4最终生成latent加噪，用冻结teacher real-score与可训练fake-score给出分布匹配梯度：
`L_student = lambda_PD * L_PD + lambda_DM * L_DM`。
这里的L_DM指DMD的分布匹配目标/其梯度代理，不能简单替换为两个score输出之间的MSE。

fake-score必须在student生成样本上单独学习去噪，更新时停止对student反传。
对student更新时冻结两个score网络；Wan的flow velocity需要按一致的噪声/时间约定转换为score或等价梯度表达。
现有 `sample` 为 `no_grad` 推理接口，直接往它后面接一个loss不会训练student；需要新增可微rollout，
明确是完整4步反传还是截断近似，并测量显存。对当前 `x_sigma=(1-sigma)*x0+sigma*epsilon`、`v=epsilon-x0` 约定，score 可写为 `-(x_sigma+(1-sigma)*v)/sigma`，需避开 sigma=0，并核对 CFG/时间采样和梯度权重；不能把 flow velocity 当作 score 直接相减后照搬其他参数化的权重。DMD2式更新比可以作为稳定fake-score跟踪的参考，不能直接承诺在Wan14B上有效。

这个组合有可能补充轨迹MSE难以约束的终点质量，但多一个可训练14B fake-score会明显增加参数、Adam状态、激活和计算量；
14B student + teacher + full fake-score不能直接沿用本次纯PDD的显存估计。LoRA fake-score可以作为另一个待验证的降本实验。
纯分布匹配部分可继续只用prompt和生成样本；若加入DMD2的真实数据GAN项，就需要真实视频数据，不能再称为无需真实视频样本。

参考：[DMD](https://arxiv.org/abs/2311.18828)、[DMD2](https://arxiv.org/abs/2405.14867)。
以上是结合方案的推断，尚未实测PDD+DMD效果。

## 最终真实14B短测结果

硬件：本机4×NVIDIA A800 80GB；832×480、81帧；batch1、accum2、有效batch8；
student/teacher FSDP分片4，CPU轨迹和激活卸载，自动启用Adam状态CPU卸载。

| 更新 | 全局平均loss | 更新耗时 | 峰值分配显存（各rank最大值） | 峰值reserved |
|---|---:|---:|---:|---:|
| 1 | 0.21246606 | 198.80 s | 61.09 GiB | 78.13 GiB |
| 2 | 0.16070923 | 190.51 s | 61.09 GiB | 78.13 GiB |

注意allocated不包含所有CUDA/NCCL开销，reserved也不能当作可自由增加batch的余量；这里只证明本机batch1、accum2的两步运行与预览成功，没有测出最大accum或进行16卡长训。

[短测预览](../outputs/wan21_14b_480p_smoke3_20260914/fixed_prompt/index.html)、
[训练指标](../outputs/wan21_14b_480p_smoke3_20260914/metrics.jsonl)、
[校验结果](../outputs/wan21_14b_480p_smoke3_20260914/verification.json)。
NFE4预览正常生成MP4/GIF/JPG；仅训练两个更新，图像仍较模糊，不据此评价14B蒸馏后的质量或收敛情况。

默认16卡配置为batch1、accum64、shard8，Adam状态auto策略关闭CPU搬移；CPU激活/轨迹卸载仍开启。可以显式修改这些内存选项，但必须重新测量对应硬件上的峰值。
