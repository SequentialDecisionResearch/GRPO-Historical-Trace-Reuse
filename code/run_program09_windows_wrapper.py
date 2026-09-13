#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("program09_original_windows_wrapper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def main() -> int:
    ap = argparse.ArgumentParser(description="Windows-safe runtime wrapper for frozen Program 09.")
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    p09_path = root / "09_make_paper_outputs.py"
    if not p09_path.exists():
        raise SystemExit(f"Missing {p09_path}")

    p09 = load_module(p09_path)

    # Runtime-only Windows compatibility patch.
    # The frozen Program 09 source file remains unchanged, so source-hash
    # verification continues to check the original frozen bytes.
    def atomic_save_figure_windows(fig, path: Path, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.stem}.{os.getpid()}.tmp{path.suffix}"
        try:
            fig.savefig(tmp, **kwargs)
            # Windows may reject fsync on a read-only descriptor.
            with tmp.open("rb+") as f:
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            p09.fsync_dir(path.parent)
        finally:
            tmp.unlink(missing_ok=True)

    p09.atomic_save_figure = atomic_save_figure_windows

    print("=" * 82)
    print("PROGRAM 09 WINDOWS RUNTIME WRAPPER")
    print(f"original frozen source : {p09_path}")
    print("source file modified   : NO")
    print("runtime compatibility  : atomic_save_figure uses rb+ for Windows fsync")
    print("=" * 82)

    return int(p09.main([
        "--config", "configs/protocol.yaml",
        "--mode", "paper",
        "--split", "test",
        "--device", "cpu",
        "--output-root", str(root),
    ]))

if __name__ == "__main__":
    raise SystemExit(main())
