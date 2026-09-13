#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("program09_original_windows_wrapper_v2", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Windows-safe runtime wrapper for frozen Program 09, including T08 NA validation compatibility."
    )
    ap.add_argument("--root", required=True)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    p09_path = root / "09_make_paper_outputs.py"
    if not p09_path.exists():
        raise SystemExit(f"Missing {p09_path}")

    p09 = load_module(p09_path)

    def atomic_save_figure_windows(fig, path: Path, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / f".{path.stem}.{os.getpid()}.tmp{path.suffix}"
        try:
            fig.savefig(tmp, **kwargs)
            with tmp.open("rb+") as f:
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            p09.fsync_dir(path.parent)
        finally:
            tmp.unlink(missing_ok=True)

    p09.atomic_save_figure = atomic_save_figure_windows

    def _blank(v):
        return v is None or str(v).strip() == ""

    def validate_t08_compat(rows):
        for r in rows:
            if str(r["dataset"]).upper() != "SVAMP":
                raise p09.Program09Error("T08 contains a non-SVAMP dataset row.")

            estimator = str(r["estimator"])
            if estimator not in {"is", p09.PRIMARY_ESTIMATOR}:
                raise p09.Program09Error("T08 contains an unexpected estimator.")

            gate_kind = str(r.get("gate_kind", "")).strip()

            if estimator == "is":
                if gate_kind not in {"", "not_applicable"}:
                    raise p09.Program09Error(
                        f"T08 IS row has unexpected gate_kind={gate_kind!r}."
                    )
                for key in (
                    "gate_accepted",
                    "gate_reliable",
                    "gate_false_accept",
                    "gate_false_reject",
                ):
                    if not _blank(r.get(key)):
                        raise p09.Program09Error(
                            f"T08 IS/not_applicable row unexpectedly contains {key}={r.get(key)!r}."
                        )
                continue

            if gate_kind not in {"primary_ress", "kl_baseline"}:
                raise p09.Program09Error(
                    f"T08 prompt-WIS row has unexpected gate_kind={gate_kind!r}."
                )

            accepted = p09.as_bool(r["gate_accepted"])
            reliable = p09.as_bool(r["gate_reliable"])

            if p09.as_bool(r["gate_false_accept"]) != (accepted and not reliable):
                raise p09.Program09Error("T08 gate_false_accept invariant failed.")

            if p09.as_bool(r["gate_false_reject"]) != ((not accepted) and reliable):
                raise p09.Program09Error("T08 gate_false_reject invariant failed.")

    p09.validate_t08 = validate_t08_compat

    print("=" * 82)
    print("PROGRAM 09 WINDOWS RUNTIME WRAPPER V2")
    print(f"original frozen source : {p09_path}")
    print("source file modified   : NO")
    print("runtime compatibility  : atomic_save_figure uses rb+ for Windows fsync")
    print("T08 compatibility      : IS/not_applicable blank gate booleans treated as NA")
    print("T08 prompt-WIS checks  : ORIGINAL BOOLEAN INVARIANTS PRESERVED")
    print("SVAMP required         : YES")
    print("=" * 82)

    return int(
        p09.main(
            [
                "--config", "configs/protocol.yaml",
                "--mode", "paper",
                "--split", "test",
                "--device", "cpu",
                "--output-root", str(root),
                "--require-svamp",
            ]
        )
    )

if __name__ == "__main__":
    raise SystemExit(main())
