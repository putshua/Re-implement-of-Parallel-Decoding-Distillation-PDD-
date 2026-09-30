# 14B joint training submitted through keeper

Task: `pdd-wan14b-joint-fsdp16-motion32-20260916`.
Submitted 2026-09-16 15:55:47 Asia/Shanghai from the local machine.
Only nodes 10.32.9.182 (dlc1cpem16cf8enc-master-0) and 10.32.4.42
(dlc1enbrpmiva26l-master-0) were selected. Both had no running tasks and
all8 GPUs idle. Each reports8 A800-SXM4-80GB. The four occupied H3 nodes
were not modified or assigned any tasks. No keeper reload/restart was performed.

Both remote preflights completed exit0: shared Python, CUDA allocation,
14B checkpoint shard paths, a real450k prompt-cache sample and32 fixed embeddings.
Training task ID on each host: `t20260916-155547-3`.
Payload is `train.json`; target list is `nodes.txt`; submission results are in
`ledger.jsonl`. API helper uses only those two explicit IPs and reads credentials
from keeper/.token without displaying them.

Configuration: Phased DMD+PDD, original14B teacher initialization, 832x480/81f,
450613 weighted training prompts, BS1, GA8, world16, shard16, global batch128,
PDD:DMD1:1, fake:G5:1, max250 generator updates. CPU activation/Adam offload.
Rolling save5, permanent50. Fixed prompts: all32 from
`/mnt/data/butong/pipeline/test_prompts/wan22_sft_high_motion_proportional_32_extended_umt5.pt`,
NFE4, seed42, step0/every25/final. The32 prompts affect only preview, not training data.

Output:
`/mnt/data/butong/PDD/outputs/wan21_14b_phased_dmd_pdd_fsdp16_motion32_20260916`

Status/log helper (does not resubmit):
```bash
cd /mnt/data/butong/PDD
python3 reports/keeper_wan14b_20260916/api.py status
```

The installed fleet.sh ignores positional node-list arguments. If using fleet.sh
for status/log instead, set FLEET_IPS explicitly to this run's nodes.txt; do not
use its default fleet list for mutations.

The 14B task was stopped on user request; its two sub-tasks are recorded in
`stop.jsonl`. The nodes are eligible for a new job after an explicit idle check.
