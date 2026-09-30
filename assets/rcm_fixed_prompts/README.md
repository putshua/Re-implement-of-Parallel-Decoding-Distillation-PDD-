RCM 固定预览 prompt

来源：`/mnt/data/butong/codebase/Bidirectional/turbodiffusion/rcm/callbacks/every_n_draw_distill.py` 的 `video_prompts`，使用 AST 提取原始字符串，保持全部6条的原文和顺序。

1. 小怪兽看蜡烛
2. 玻璃球中的禅园
3. 猎豹追羚羊
4. 雪地狼群
5. 白头鹰捕鱼
6. 未来城市机器人跑酷

`prompts.txt` 为可读文本，`manifest.jsonl` 用于原生 Wan UMT5 预编码，`prompts_000000000.pt` 为训练回调读取的 embedding。生成命令：

```bash
/mnt/data/butong/miniconda3/envs/causvid-wan21-final/bin/python scripts/prepare_embeddings.py \
  --manifest assets/rcm_fixed_prompts/manifest.jsonl \
  --output assets/rcm_fixed_prompts --batch-size 2
```

默认使用 Wan1.3B checkpoint 的 UMT5，长度512、维度4096。无需训练时额外加载文本编码器。
