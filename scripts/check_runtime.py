"""Check imports once per node before torchrun fans out worker processes."""

import importlib
import json
import sys
from pathlib import Path


def main():
    errors = []
    versions = {}
    for name in (
        "torch",
        "diffusers",
        "transformers",
        "accelerate",
        "safetensors",
        "numpy",
        "einops",
        "ftfy",
        "sentencepiece",
        "imageio",
        "imageio_ffmpeg",
        "PIL",
        "tensorboard",
    ):
        try:
            module = importlib.import_module(name)
            versions[name] = getattr(module, "__version__", "installed")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    if not errors:
        try:
            from pdd.model import PDDWan  # noqa: F401
            from torch.distributed.fsdp import FullyShardedDataParallel  # noqa: F401
            import imageio_ffmpeg

            imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            errors.append(f"Wan/FSDP/FFmpeg: {type(exc).__name__}: {exc}")
    print(
        json.dumps(
            {
                "event": "runtime_check",
                "python": sys.executable,
                "python_version": sys.version.split()[0],
                "packages": versions,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if errors:
        print("\n".join(errors), file=sys.stderr)
        print(
            f"Install dependencies on EVERY node using this interpreter:\n{sys.executable} -m pip install -r {Path(__file__).resolve().parents[1] / 'requirements.txt'}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
