#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("program08_original_windows_wrapper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def main() -> int:
    ap = argparse.ArgumentParser(description="Windows-safe runtime wrapper for Program 08.")
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    p08_path = root / "08_run_svamp_robustness.py"
    if not p08_path.exists():
        raise SystemExit(f"Missing {p08_path}")

    p08 = load_module(p08_path)

    # Runtime-only Windows compatibility patch:
    # Program 08's original write_parquet opens the completed parquet file "rb"
    # and calls os.fsync(), which raises EBADF on Windows. Use "rb+" only for
    # the durability flush. The original Program 08 source file remains unchanged.
    def write_parquet_windows(path, rows):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:
            raise p08.Program08Error(f"pyarrow is required: {exc}") from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist([dict(r) for r in rows])
        pq.write_table(table, path, compression="zstd", use_dictionary=True)
        with path.open("rb+") as f:
            f.flush()
            os.fsync(f.fileno())

    p08.write_parquet = write_parquet_windows

    print("=" * 82)
    print("PROGRAM 08 WINDOWS RUNTIME WRAPPER")
    print(f"original source       : {p08_path}")
    print("source file modified  : NO")
    print("runtime compatibility : write_parquet uses rb+ for Windows fsync")
    print("=" * 82)

    return int(p08.run([
        "--config", "configs/protocol.yaml",
        "--mode", "paper",
        "--split", "test",
        "--resume",
        "--device", "cuda",
        "--output-root", str(root),
    ]))

if __name__ == "__main__":
    raise SystemExit(main())
