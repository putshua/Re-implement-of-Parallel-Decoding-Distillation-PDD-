# PDD + phase endpoint DMD：实现核对与实验设计

2026-09-14。本文保留最初算法设计。后续已接入1.3B三模型训练和matched对照，并按用户要求切换旧脚本默认实验；实际验证进展见 [联合训练验证](wan13b_joint_validation.md)。以下关于“原型/待实施”的描述属于设计阶段，不代表最新工程状态。

## 推荐实验

先比较同一 PDD4 起点上的三个实验：匹配 rollout 的纯 PDD、PDD + 四个 phase 的 endpoint DMD、PDD + 最终 x0 的普通 DMD。优先实现四 phase 版本作为主方案，普通 DMD作为必要的简单基线。四 phase 方案与 PDD 的局部预测结构吻合；并不能预先保证视频质量胜过普通 DMD。

固定 128 heads、shift=6、480p 81f、450k 加权训练 prompt、RCM6×5 seeds 预览。第一轮先在 Wan1.3B 验证三模型训练和梯度，再扩展 Wan2.1 14B；14B 要额外容纳可训练 fake-score 及其优化器，不能沿用纯 PDD 的显存测试结论。

## 参考代码实际做了什么

`../codebase/Bidirectional/turbodiffusion/rcm/models/t2v_model_distill_rcm.py`：

- `_phase_conditional_noise`：TrigFlow 中的条件高斯加噪，终点是 x_s，不是 x0。
- `_backward_simulation_phase_endpoint`：从噪声出发，按 RF Euler 轨迹生成指定终点；只有最后一次生成前向保留梯度。
- `_student_phased_dmd_step`：固定终点的生成分布与 teacher 做 DMD；teacher/fake 查询 no_grad；支持 residual 自归一化和 boundary 归一化。
- `_training_step_critic_phased`：fake-score 的子区间 DSM 目标，有 inverse variance 上限 10。
- `is_student_phase`：student/fake 更新交替，而非同时对同一个 loss 更新两个网络。

`../codebase/Bidirectional/scripts/train_moe_dmd.sh` 的 high PCM+DMD preset 是 high 整段结束于 RF=0.625，三步路径 `[1, .9375, .8333333333, .625]`，不是每个 PCM 子段都有独立 DMD。该 preset 使用 SFT high teacher、CFG4、PCM scale50、192p real latent 数据。这里仅借鉴数学和 rollout，不能照搬分辨率、数据依赖、权重尺度。

四 phase 同时训练、共享 PDD backbone 是本次提出的扩展；不是原论文逐阶段专家训练的等价复现。参考 Phased DMD 论文（尤其式2、10、11、13）：https://arxiv.org/html/2510.27684v3 。

## Phase 与 rollout

PDD4 真实推理边界：

| phase | head 索引 | RF 路径 |
|---|---|---|
| 0 | [0,32) | 1 → 0.947368 |
| 1 | [32,64) | 0.947368 → 0.857143 |
| 2 | [64,96) | 0.857143 → 0.666667 |
| 3 | [96,128) | 0.666667 → 0 |

一次 generator 更新选 phase j。从新噪声出发，用当前 student 按真实 PDD4 sampler 无梯度推进到该 phase 起点，然后 detach。当前 phase 的端点为：

`x_s = stopgrad(x_start) + sum_{k=start}^{end-1} Δsigma_k * h_k(x_start, sigma_start, prompt)`。

当前 phase 只需一次有梯度的 backbone 前向。与局部 PDD velocity MSE 共用这次前向：输出若干抽样 head、到抽样位置的 displacement、到 phase 终点的 displacement。Teacher target 查询不反传；端点用于 DMD 时必须保留梯度。区间内默认 `k ∈ [start,end)`，midpoint teacher target。MSE 约束局部速度；DMD约束终点分布，不要求与某个 teacher sample 一一对应。

注意当前 PDD 是 advance16、lookahead64；上述主实验改成 advance32、局部抽 head。因此必须加入完全相同新 rollout、但 lambda_dmd=0 的控制组，不能把差异全部归因于 DMD。保留原始 step100 作为额外参考。

每次重新计算前缀，避免旧 stream 混入已更新参数的轨迹。均匀/分层抽 phase，累积步中平衡覆盖四 phase，避免之前的同步 8 step 波动。FSDP 初版每个 microbatch 所有 rank 使用相同 phase，以保证有梯度/无梯度前向和 collective 次序一致；噪声、prompt、目标 head 各 rank 独立。phase 的抽样顺序可以随机打乱，不让全局 optimizer step 固定绑定某个 phase。

不要把每个 phase 的输入直接替换成按 sigma 缩放的高斯：除纯噪声起点外，它必须来自有效 student 前缀。

### 为什么不直接复用旧 stream

旧 advance16 流对应八步分布，PDD4 使用32-head fusion。用旧流当 PDD4 的前缀，会产生采样分布不匹配；保留一条流跨参数更新还会引入 stale policy。可在基线跑通后对缓存/重放单独做消融。

### 为什么不随机截断后直接预测 x0

对早期 phase 也强制跳到 x0 会把监督目标变成“每个阶段都能一次完成全程”。这里早期 phase 只负责自己的带噪终点，最后 phase 才负责 x0。

### 为什么不直接全链反传

PDD4 full BPTT 是合理的普通 DMD 对照，但保留四次模型计算图会增加内存/重算成本。主方案对前缀 stop-gradient，所以是局部更新，不是全局目标对共享参数的完整梯度；共享 backbone 更新仍会影响其他 phase，这是需要用评估检验的相互干扰。

## RF 空间的条件加噪与 fake-score

本仓库直接使用 RF latent：`x_r=(1-r)x0+r*epsilon`。不需要转成参考实现的 TrigFlow 单位方差 latent。

对端点 s 和 score time t，要求 `0 <= s < t < 1`：

```
a = (1-t)/(1-s)
b² = t² - a²*s²
x_t = a*x_s + b*epsilon
```

这定义与 RF 边缘分布一致的高斯 Markov forward kernel；它不等于沿同一条 deterministic RF 轨迹回退。采样 t 应覆盖 `(s,1)` 的嵌套区间，而不只是当前生成段 `(s,start)`。默认留端点间隙（例如 RF 1e-3），并记录实际 t 分布。score-time shift 是独立超参数，不因 generator shift=6 就必须同值。

Fake 输出原生 velocity `v_F(x_t,t,prompt,phase)`。其 scaled DSM 残差：

```
R = b*v_F - [t + (1-t)*s²/(1-s)²]*epsilon + b*x_s/(1-s)
L_F = mean(min(1/b², 10) * R²)
```

`x_s` 必须 detach。s=0 时目标退化为普通 `epsilon-x0` flow matching（保留 clamp 引起的时间权重）。s>0 时用 `epsilon-x_s` 当目标是错的。公式计算使用 FP32，score time 不取精确1，t不等于s。

四个端点生成的 q_s 经条件加噪得到的 q_s^t 通常不同。若 fake 只接收 t、prompt 而不知道 phase，就会拟合混合分布，而不是当前端点分布的 score。推荐单独一个 fake backbone + 四个 phase 输出 head（可复用 PDDWan 的 one-hot head 选择），必要时加入 phase embedding 增强条件表达；fake 与 generator 不共享可训练参数。四个 head 只解决输出的 phase 区分，不保证与四个独立专家等容量。

不要用 generator 的128个轨迹 head 直接充当 fake-score；两者预测任务、输入状态分布和优化目标不同。

## Generator 梯度与联合目标

Wan velocity 到 score 的关系：`score(v)=-(x_t+(1-t)*v)/t`。

在同一 x_t、t、prompt 上查询 frozen real teacher 和 phase-aware fake，均不反传。未自归一化的端点更新方向为：

```
g_s = a*(score_fake-score_real)
    = a*(1-t)/t*(v_real-v_fake)
L_DM = 0.5 * mean((x_s - stopgrad(x_s-g_s))²)
L_G = lambda_traj*L_PDD_MSE + lambda_dmd(step)*L_DM
```

这里已经把条件加噪对 x_s 的 Jacobian a 放入 g_s，不能再对 x_t 套一次同样的 surrogate 导致重复乘 a。gradient descent 沿 -g_s 更新。DMD surrogate 的标量值不是 KL 数值，不应把其下降当成生成质量改善。

保留两种显式选择：`normalization=none` 用上述 score 方向；`normalization=residual` 用每样本 mean(abs(g_s)) 自归一化，接近参考 high preset 的做法，但这是改变时间/样本权重的启发式，不能声称保留原始未加权 KL 梯度。第一轮两者中固定一种进行比较，避免同一实验混用。

所有 loss 都按 latent 元素 mean，以对齐现有 PDD；不能复制参考代码 sum reduction 下的 PCM50/DMD1。先令 lambda_traj=1，校准 lambda_dmd（例如0.01/0.1/1的短跑），并观察分支梯度大小。不同 phase 的 displacement 区间长度差别很大，因此同一个 lambda 不保证各 phase 对参数梯度影响相同，不应未经测量就除以很小的区间宽度。

Teacher guidance：对照保留现有训练 CFG5 和 negative embedding；DMD real-score 建议使用完整网络 CFG（skip_layer=-1），不能默认把 skip-layer 的混合输出当作标准 teacher score。原 PDD traj 分支 skip10 可先保持以减少变量，但需记录两种 teacher 查询策略；后续再对齐为全层 teacher 做消融。CFG下的分布目标本身也属于实践中的 guided-score 近似，不将其表述为严格的无偏真实视频分布 KL。

## Phased vs 普通 DMD 的公平对照

| 实验 | rollout 和梯度 | 优点/需要观察的限制 |
|---|---|---|
| A：matched PDD | 四 phase 前缀重算 + 局部MSE，关闭DMD | 隔离 rollout 改动 |
| B：phased + PDD | 随机phase，监督真实中间端点，仅当前phase反传 | 覆盖各段，低图深；需phase-aware fake |
| C：普通DMD + PDD | 完整PDD4到x0，只有最后步保留梯度 | 简单、直接约束成片；DMD直接梯度偏向最后32heads |
| D：普通DMD full BPTT + PDD | 完整PDD4并四步反传 | 更完整的全局梯度，显存/算力更贵 |

C 的共享 backbone 仍会间接改变前面 phase，但不能说 DMD 已直接监督了所有 head。不要给 C 额外添加随机一步跳 x0 而仍称为相同 PDD4 rollout。主比较 B vs C，D 在1.3B资源允许时做；B不一定更好，普通DMD有局部MSE辅助后可能足够稳定。

同时训练PDD4和PDD8会使同一端点存在不同采样前缀分布；初版固定PDD4，不要无条件混到同一个fake。后续可增加schedule条件或明确研究混合分布目标。

## 训练工程接入要求

三个模型：trainable PDD、frozen teacher、trainable phase-aware fake；两个optimizer。每一轮先用detach生成样本更新fake，再重新查询已更新fake形成student梯度。先做fake warmup，然后逐渐增大DMD权重；fake:G 更新比例5:1可作为候选值，需按fake拟合滞后调整，不把它当成已验证最佳值。

初始 generator 从指定 PDD model weights 加载，作为新实验；不恢复旧纯PDD optimizer/cursor/step。fake 从原Wan初始化，各phase头复制原projection。联合训练自己的resume则必须完整恢复两个optimizer、fake模型、两个更新计数、phase调度、RNG。不要让旧checkpoint的COMPLETE规则误把缺fake的checkpoint判为可恢复。

日志至少记录每phase的 traj MSE、DMD surrogate、g_s RMS/归一化因子、fake DSM、score time分布，以及两类update计数、耗时、峰值显存。预览频率以generator updates为单位，仍用RCM6×5 seeds，480p；除主观质量，还单独看运动、prompt遵循、多seed多样性。当前只有117steps的纯PDD不能视作收敛基线。

本地验证顺序：数学/梯度测试 → tiny FSDP三模型与两个optimizer保存恢复 → 480p batch1至少两个完整G更新及fake更新 → RCM预览 → 集群长跑。14B先校准显存，再定grad_accum；fake增加的optimizer状态与模型常驻不能靠增加grad_accum解决。训练入口应新增，旧纯PDD默认不启用任何DMD。

## 本次已落地的原型

- `AnyFlow/pdd/phased_dmd.py`：条件加噪、fake DSM、带正确符号/Jacobian的endpoint surrogate、真实PDD前缀和局部联合前向。
- `AnyFlow/pdd/core.py`：`pd_loss(..., detach_endpoint=False)` 可保留终点梯度；默认True保持旧行为。
- `tests/test_phased_dmd.py`：边缘分布系数、s=0退化、子区间score目标、梯度方向与隔离、rollout与sampler一致性。

这些是可执行算法原型，不是已完成的三模型训练入口。上述生产接入、480p联合训练、checkpoint恢复和性能比较尚待实施。

验证记录：`reports/phased_dmd_primitives_tests.log` 中5项CPU测试通过（使用共享wan解释器）；它们验证公式和梯度，不代表已验证480p训练或fake-score拟合收敛。旧PDD核心回归另见 `reports/phased_dmd_pdd_regression.log`。
