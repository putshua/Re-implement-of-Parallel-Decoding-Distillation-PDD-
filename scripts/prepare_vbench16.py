"""Select a reproducible random VBench subset for Wan fixed-prompt previews."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    prompts = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    indices = random.Random(args.seed).sample(range(len(prompts)), args.count)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for ordinal, index in enumerate(indices):
            handle.write(json.dumps({
                "id": f"vbench-{ordinal:02d}-source-{index:04d}",
                "caption": prompts[index],
                "source": str(source),
                "source_line": index + 1,
                "selection_seed": args.seed,
            }, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output.resolve()), "count": len(indices), "indices": indices}))


if __name__ == "__main__":
    main()
