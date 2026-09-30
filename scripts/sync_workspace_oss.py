#!/usr/bin/env python3
"""Resumable, size-verified copy of this workspace to the configured OSS bucket."""

import argparse
import concurrent.futures
import json
import os
import sys
import time
from pathlib import Path

import oss2
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT.parent / "data_transfer" / "config.yaml"


def file_list(root):
    # Preserve the workspace layout. Checkpoint completion markers are last.
    files = [p for p in root.rglob("*") if p.is_file() and not p.is_symlink()]
    return sorted(files, key=lambda p: (p.name == "COMPLETE", p.relative_to(root).as_posix()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    cfg = yaml.safe_load(args.config.read_text())
    bucket_name = cfg["oss_upload_bucket"]
    endpoint = cfg["oss_upload_endpoint"]
    prefix = cfg.get("oss_backup_prefix", "butong").strip("/") + "/PDD/"
    files = file_list(ROOT)
    total = sum(p.stat().st_size for p in files)
    print(json.dumps({"event": "sync_plan", "source": str(ROOT),
                      "target": f"oss://{bucket_name}/{prefix}",
                      "files": len(files), "bytes": total,
                      "workers": args.workers}), flush=True)
    if args.dry_run:
        return 0

    def transfer(path):
        key = prefix + path.relative_to(ROOT).as_posix()
        before = path.stat()
        auth = oss2.Auth(cfg["oss_upload_access_key_id"],
                         cfg["oss_upload_access_key_secret"])
        bucket = oss2.Bucket(auth, endpoint, bucket_name)
        try:
            if bucket.head_object(key).content_length == before.st_size:
                return "skipped", key, before.st_size
        except oss2.exceptions.NoSuchKey:
            pass
        for attempt in range(5):
            try:
                oss2.resumable_upload(
                    bucket, key, str(path),
                    multipart_threshold=100 * 1024**2,
                    part_size=128 * 1024**2,
                    num_threads=4,
                )
                if path.stat().st_size != before.st_size:
                    raise RuntimeError("Source size changed during upload")
                if bucket.head_object(key).content_length != before.st_size:
                    raise RuntimeError("OSS object size mismatch")
                return "uploaded", key, before.st_size
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 8))

    done = 0
    bytes_done = 0
    failures = []
    # Limit in-flight work; otherwise thousands of futures retain file metadata.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        source = iter(files)

        def submit_next():
            try:
                p = next(source)
            except StopIteration:
                return False
            pending[pool.submit(transfer, p)] = p
            return True

        for _ in range(args.workers * 2):
            submit_next()
        while pending:
            ready, _ = concurrent.futures.wait(
                pending, return_when=concurrent.futures.FIRST_COMPLETED
            )
            for future in ready:
                path = pending.pop(future)
                done += 1
                try:
                    action, key, size = future.result()
                    bytes_done += size
                    print(json.dumps({"event": action, "file": str(path.relative_to(ROOT)),
                                      "bytes": size, "done": done, "total": len(files),
                                      "bytes_done": bytes_done}), flush=True)
                except Exception as exc:
                    failures.append(str(path.relative_to(ROOT)))
                    print(json.dumps({"event": "failed", "file": str(path.relative_to(ROOT)),
                                      "error": str(exc)}), file=sys.stderr, flush=True)
                submit_next()
    print(json.dumps({"event": "complete" if not failures else "partial",
                      "target": f"oss://{bucket_name}/{prefix}",
                      "files_done": done - len(failures), "files_failed": len(failures),
                      "bytes_verified": bytes_done}), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
