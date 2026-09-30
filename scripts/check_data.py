import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--config", required=True)
p.add_argument("--weight-index")
p.add_argument("--checkpoint")
p.add_argument("--check-shards", action="store_true")
p.add_argument("--require-rows", type=int)
a = p.parse_args()
c = json.loads(Path(a.config).read_text())
if a.weight_index:
    c["weight_index"] = a.weight_index
if a.checkpoint:
    c["checkpoint"] = a.checkpoint
result = {"checkpoint": Path(c["checkpoint"]).is_dir()}
if c.get("weight_index"):
    path = Path(c["weight_index"])
    meta = json.loads(path.read_text())
    result["manifest_rows"] = meta["rows"]
    result["index_valid"] = (
        meta.get("version") == "taxonomy_wds_repeat_weight_v1"
        and meta.get("coverage_complete") is True
        and (a.require_rows is None or meta["rows"] == a.require_rows)
    )
    roots = []
    for root in meta["arrays"]:
        mapped = root
        for src, dst in c.get("path_map", {}).items():
            if root.startswith(src):
                mapped = dst + root[len(src) :]
                break
        entry = {"source": root, "resolved": mapped, "exists": Path(mapped).is_dir()}
        if a.check_shards:
            import numpy as np

            weights = np.load(path.parent / meta["arrays"][root], mmap_mode="r")
            expected = [
                i // 256
                for i in range(0, len(weights), 256)
                if np.any(weights[i : i + 256] > 0)
            ]
            available = {
                int(tar.stem.split("_")[-1])
                for tar in Path(mapped).glob("shard_*.tar")
                if tar.is_file() and tar.stat().st_size > 0
            }
            missing = sorted(set(expected) - available)
            entry.update(
                required_shards=len(expected),
                missing_shards=len(missing),
                missing_tar_paths=[
                    str(Path(mapped) / f"shard_{i:06d}.tar") for i in missing
                ],
            )
        roots.append(entry)
    result["cache_roots"] = roots
print(json.dumps(result, indent=2))
raise SystemExit(
    0
    if result["checkpoint"]
    and result.get("index_valid", a.require_rows is None)
    and all(
        x["exists"] and x.get("missing_shards", 0) == 0
        for x in result.get("cache_roots", [])
    )
    else 1
)
