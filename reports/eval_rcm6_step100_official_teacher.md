# RCM6：PDD4 step100 与官方 Wan1.3B teacher 对比

状态：已完成。60 个视频全部通过 832×480、81 帧、16 FPS 校验；60 个 latent 全部为有限值，30 对样本无缺失或重复。详见输出目录 `verification.json` 与 `COMPLETE`。

## 固定设置

- Student：`outputs/wan1p3b_480p_450k_rcm6_fsdp16_20260912/step_000100`，任务开始时最新完整 checkpoint；整个测试固定使用该 checkpoint。
- 6 个默认 RCM prompt，原文不扩写；每个使用 seeds 42、43、44、45、46，两种方法共 60 个视频。
- 分辨率 832×480，81 帧，16 FPS。同一 prompt/seed 配对使用相同初始噪声和正向文本 embedding。
- PDD：4 次 backbone 前向，128 个 head，使用训练对应的 shift=6 时间网格。
- Teacher：原始 Wan2.1-T2V-1.3B 权重、官方原生模型、UniPC 50 步、shift=8、CFG=6、官方负向 prompt、不跳过 block。CFG 正负分支共 100 次 backbone 前向。
- Teacher FP32 权重、BF16 autocast；PDD Diffusers 后端 BF16 权重。这是实际部署配置的采样耗时对比，并非同后端、同精度或等质量加速比。
- 耗时包含去噪采样，不包含加载、预热、VAE 解码及文件写入。

官方参考：[Wan2.1 1.3B 无 prompt 扩写示例](https://github.com/Wan-Video/Wan2.1#1-without-prompt-extension)、[固定版本 generate.py](https://github.com/Wan-Video/Wan2.1/blob/9737cba9c1c3c4d04b33fcad41c111989865d315/generate.py)。源码快照与 SHA256 位于 `assets/wan21_official/provenance.json`。

## 执行记录

首次使用本机 GPU0–3，GPU0 后来被其他进程占用约 66 GiB，评测在 VAE 解码时 OOM。保留已完成的 14 个视频，剩余使用 GPU1–3 完成；没有改动其他进程或训练 checkpoint。原始日志与重试日志分别为 `reports/eval_rcm6_step100_official_teacher.log`、`reports/eval_rcm6_step100_official_teacher_retry_gpu123.log`。

原始耗时全部保留，同时在 `latency_comparison.json` 单独统计排除首次 GPU0 配对的结果。其余 GPU 仅做离散时刻检查，不宣称全程无外部干扰。

## 画面抽查

以下仅为 seed42 的局部抽帧观察，不能代替 30 对视频的系统质量评测。

- 毛绒怪物与蜡烛：两者均生成主体与红色蜡烛，构图不同。
- 玻璃球禅园：PDD 抽查帧中的小人更清楚，但两者都没有清楚呈现“小人亲自在球内耙沙”的动作。
- 猎豹追逐：teacher 的第 8/40/72 帧可见猎豹和羚羊追逐；PDD 对应帧只有猫科动物，缺少羚羊与追逐交互。
- 狼群：两者都有雪地狼群；teacher 抽帧中的奔跑姿态变化更明显，PDD 细节清楚但动作变化较小。未据此断言完整视频静止。

- 老鹰抓鱼：PDD 在第 72 帧能清楚看到鱼，teacher 对应抽查帧主要呈现入水和水花，没有清楚看到抓鱼。两者的抽查帧均不足以确认完整完成“抓鱼后飞走”的全过程。

- 机器人：teacher 抽帧中的运动、视角与主体位置变化更大；PDD 在抽查帧中主要保持低伏姿态，没有清楚呈现连续跑酷。两者均不能仅凭这三帧确认完成 prompt 中所有动作。

未运行 VBench 或其他标准质量指标。不同生成结果间 latent MSE 仅保留作诊断，不视为质量得分。

## 最终耗时

| 统计范围 | 配对数 | Teacher 平均 | PDD4 平均 | 平均逐对耗时比 |
|---|---:|---:|---:|---:|
| 所有结果 | 30 | 181.30 s | 7.46 s | 24.46× |
| 排除首次 GPU0 的 prompt1/seed42 | 29 | 181.11 s | 7.32 s | 24.74× |

这里的倍数仅表示去噪耗时比，不能解释为“同等质量加速”。官方 teacher 与训练 teacher 的 CFG、shift、负向 prompt 和跳层策略也不同，不能把所有质量差异单独归因于蒸馏。

进一步查看猎豹组五个 seed 的中间帧：teacher 的 42/43/44/46 可见两只动物；PDD 的 43/44 也出现第二只动物，因此 seed42 的缺失不能推广到全部 seed。机器人五个 seed 的中间帧姿态也有明显差异，未做全视频动作成功率统计。

## 结果入口

- [并排播放对比页面](../outputs/eval_rcm6_step100_pdd4_official_teacher_5seeds/comparison.html)：teacher 在左，PDD4 在右，可筛选 prompt/seed，并同步从头播放。
- [耗时统计](../outputs/eval_rcm6_step100_pdd4_official_teacher_5seeds/latency_comparison.json)
- [完整性校验](../outputs/eval_rcm6_step100_pdd4_official_teacher_5seeds/verification.json)
- 每组五个 seed 的中间帧拼图：输出目录 `comparison_prompt0000_5seeds.jpg` 至 `comparison_prompt0005_5seeds.jpg`。
- seed42 的猎豹、狼群、老鹰、机器人三帧对比：输出目录 `preview_prompt0002_seed42.png` 至 `preview_prompt0005_seed42.png`。
- 复现入口：`scripts/eval_rcm6_official_teacher.sh`。默认固定本次 step100、6 prompts、5 seeds 和官方 teacher 设置。重跑到新目录请设置 `OUTPUT`，例如 `CUDA_VISIBLE_DEVICES=1,2,3 NPROC_PER_NODE=3 OUTPUT=/mnt/data/butong/PDD/outputs/eval_rcm6_step100_repeat bash scripts/eval_rcm6_official_teacher.sh`。
