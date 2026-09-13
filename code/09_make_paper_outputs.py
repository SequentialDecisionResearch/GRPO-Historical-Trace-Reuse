#!/usr/bin/env python3
"""
GRPO-OPE Program 09 — paper tables, figures, and provenance.

Research role
-------------
This program is deliberately the final, CPU-only presentation layer. It MUST NOT
call a model, generate completions, rescore trajectories, refit the reuse gate,
or mutate any upstream machine-readable result. It only reads frozen outputs from
Programs 02/05/06/07/08 (and optional model-robustness results), validates their
schemas/invariants/hashes where available, and deterministically produces paper-
ready tables, figures, LaTeX fragments, machine-readable paper values, and a
provenance manifest.

Core evidence chain preserved by the output design:

    GRPO policy drift
        -> sequence-overlap deterioration
        -> OPE reliability deterioration
        -> refresh benefit
        -> frozen Accept-vs-Refresh gate
        -> held-out / official-test / SVAMP validation

Primary outputs
---------------
Raw research tables are never edited. Derived paper assets go under:

    outputs/paper/
      tables/
        Table1_estimator_benchmark.csv/.tex
        Table2_fixed_vs_rolling.csv/.tex
        Table3_frozen_gate_validation.csv/.tex
        Appendix_training_summary.csv/.tex
        Appendix_gate_baseline.csv/.tex
        Appendix_sample_size.csv/.tex              # if T07 exists
        Appendix_svamp_pair_results.csv/.tex       # if T08 exists
        Appendix_model_robustness.csv/.tex         # if T09 exists
      figures/
        Figure1_training_trajectory.pdf/.png
        Figure2_fixed_reuse_frontier.pdf/.png
        Figure3_overlap_vs_error.pdf/.png
        Figure4_rolling_refresh.pdf/.png
        Figure5_frozen_gate_validation.pdf/.png
        Figure6_length_drift_overlap_heatmap.pdf/.png  # opt-in
      numbers/
        paper_values.json
        paper_values.tex
      paper_output_manifest.json

The program is intentionally standalone and imports no torch/transformers/trl/
peft/accelerate/datasets/huggingface_hub modules.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
import os
import platform
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

PROGRAM = "09_make_paper_outputs.py"
PROGRAM_VERSION = "1.0"
OUTPUT_SCHEMA = "grpo_ope_program09_v1"
PROJECT_NAME = "grpo_ope_reuse"

PRIMARY_ESTIMATOR = "prompt_wis"
PRIMARY_GATE_KIND = "primary_ress"
KL_BASELINE_KIND = "kl_baseline"
EXPECTED_TOLERANCES = (0.01, 0.02, 0.05)

FORBIDDEN_MODULE_ROOTS = {
    "torch", "transformers", "trl", "peft", "accelerate", "datasets",
    "huggingface_hub", "vllm",
}

T01_REQUIRED = {
    "training_seed", "step", "mean_reward", "correctness_reward",
    "mean_completion_length", "truncation_rate", "kl_proxy", "mode",
}
T02_REQUIRED = {
    "dataset", "split", "training_seed", "behavior_step", "target_step",
    "purpose", "estimator", "estimate", "estimate_status", "online_reference",
    "signed_error", "absolute_error", "ope_ci_low", "ope_ci_high",
    "online_ci_low", "online_ci_high", "n_prompts", "K",
    "median_prompt_relative_ess", "p10_prompt_relative_ess", "mean_d2",
    "mean_max_normalized_weight", "tokenwise_kl_proxy",
    "mean_completion_length", "truncation_rate",
}
T03_REQUIRED = {
    "dataset", "split", "training_seed", "behavior_step", "target_step",
    "purpose", "prompt_id", "K", "ess", "relative_ess", "d2",
    "max_normalized_weight", "mean_log_weight", "sd_log_weight",
    "mean_log_ratio_per_token", "tokenwise_kl_proxy", "mean_completion_length",
    "truncation_rate", "pair_abs_mean_log_ratio_per_token", "length_quantile",
}
T04_REQUIRED = {
    "dataset", "split", "training_seed", "target_step", "recent_behavior_step",
    "estimator", "online_reference", "old_behavior_step", "old_estimate",
    "recent_estimate", "old_absolute_error", "recent_absolute_error",
    "delta_error_old_minus_recent", "old_median_relative_ess",
    "recent_median_relative_ess", "delta_relative_ess_recent_minus_old",
    "old_mean_d2", "recent_mean_d2", "old_tokenwise_kl_proxy",
    "recent_tokenwise_kl_proxy", "K", "n_prompts",
}
T05_REQUIRED = {
    "dataset", "split", "estimator", "gate_kind", "diagnostic_name",
    "direction", "tolerance", "fitted_threshold", "reject_all",
    "calibration_policy", "calibration_seeds", "n_pairs", "accepted_count",
    "rejected_count", "accept_rate", "false_accept_count",
    "false_accept_rate_among_accepted", "false_accept_fraction_all",
    "accepted_mae", "rejected_mae",
}
T06_REQUIRED = {
    "dataset", "split", "evaluation_role", "training_seed", "behavior_step",
    "target_step", "purpose", "estimator", "gate_kind", "diagnostic_name",
    "direction", "tolerance", "diagnostic_value", "fitted_threshold",
    "reject_all", "decision", "accepted", "true_absolute_error", "reliable",
    "false_accept", "false_reject", "gate_case_id",
}
T07_REQUIRED = {
    "dataset", "split", "training_seed", "behavior_step", "target_step",
    "purpose", "K", "estimator", "estimate", "online_reference",
    "absolute_error", "median_prompt_relative_ess", "bootstrap_ci_width",
    "bootstrap_nonfinite_fraction", "n_prompts",
}
T08_REQUIRED = {
    "dataset", "dataset_revision", "training_seed", "behavior_step",
    "target_step", "pair_purpose", "overlap_band", "representative_rank",
    "estimator", "estimate", "estimate_status", "online_reference",
    "signed_error", "absolute_error", "n_prompts", "K", "L",
    "median_prompt_relative_ess", "p10_prompt_relative_ess", "mean_d2",
    "tokenwise_kl_proxy", "gate_kind", "gate_tolerance", "gate_diagnostic",
    "gate_threshold", "gate_decision", "gate_accepted", "gate_reliable",
    "gate_false_accept", "gate_false_reject",
}


class Program09Error(RuntimeError):
    """Controlled fail-fast error for Program 09."""


@dataclass(frozen=True)
class InputTable:
    key: str
    path: Path
    rows: tuple[dict[str, Any], ...]
    sha256: str


# ---------------------------------------------------------------------------
# Basic deterministic I/O and provenance
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace(os.sep, "/")
    except Exception:
        return str(path.resolve())


def fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Mapping[str, Any]) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def csv_scalar(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return format(v, ".17g")
    return v


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    import io
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for row in rows:
        w.writerow({c: csv_scalar(row.get(c)) for c in columns})
    atomic_write_text(path, buf.getvalue())


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Program09Error(f"Missing required JSON: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise Program09Error(f"Cannot parse JSON {path}: {e}") from e
    if not isinstance(obj, dict):
        raise Program09Error(f"Expected JSON object in {path}")
    return obj


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise Program09Error(f"Missing required CSV: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def table_columns(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        try:
            return set(next(reader))
        except StopIteration:
            return set()


def load_table(key: str, path: Path, required: set[str], optional: bool = False) -> InputTable | None:
    if not path.exists():
        if optional:
            return None
        raise Program09Error(f"Required upstream table is missing: {path}")
    cols = table_columns(path)
    missing = sorted(required - cols)
    if missing:
        raise Program09Error(f"{path.name} is missing required columns: {missing}")
    rows = read_csv_rows(path)
    if not rows:
        raise Program09Error(f"Upstream table is empty: {path}")
    return InputTable(key, path, tuple(rows), sha256_file(path))


def first_present(obj: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for keys in paths:
        cur: Any = obj
        ok = True
        for k in keys:
            if not isinstance(cur, Mapping) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok:
            return cur
    return None


def as_float(v: Any, *, allow_none: bool = True) -> float | None:
    if v is None or v == "":
        if allow_none:
            return None
        raise Program09Error("Required numeric value is missing.")
    try:
        x = float(v)
    except Exception as e:
        raise Program09Error(f"Invalid numeric value: {v!r}") from e
    if math.isnan(x):
        return None if allow_none else x
    return x


def as_int(v: Any) -> int:
    try:
        return int(float(v))
    except Exception as e:
        raise Program09Error(f"Invalid integer value: {v!r}") from e


def as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"true", "1", "yes", "y"}:
        return True
    if s in {"false", "0", "no", "n"}:
        return False
    raise Program09Error(f"Invalid boolean value: {v!r}")


def finite_or_none(v: Any) -> float | None:
    x = as_float(v)
    return x if x is not None and math.isfinite(x) else None


def mean(xs: Iterable[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return math.fsum(vals) / len(vals) if vals else None


def median(xs: Iterable[float | None]) -> float | None:
    vals = sorted(float(x) for x in xs if x is not None and math.isfinite(float(x)))
    n = len(vals)
    if n == 0:
        return None
    m = n // 2
    return vals[m] if n % 2 else (vals[m - 1] + vals[m]) / 2.0


def root_mean_square(xs: Iterable[float | None]) -> float | None:
    vals = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return math.sqrt(math.fsum(x * x for x in vals) / len(vals)) if vals else None


def format_number(v: Any, digits: int = 4) -> str:
    x = finite_or_none(v)
    if x is None:
        return "NA"
    if abs(x) >= 1000 or (0 < abs(x) < 10 ** (-(digits - 1))):
        return f"{x:.3e}"
    return f"{x:.{digits}f}"


def git_commit(root: Path) -> str | None:
    import subprocess
    try:
        p = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=3)
        if p.returncode == 0:
            s = p.stdout.strip()
            return s or None
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Hard guarantee: no model stack in this script
# ---------------------------------------------------------------------------


def audit_source_for_forbidden_imports(script_path: Path) -> None:
    tree = ast.parse(script_path.read_text(encoding="utf-8"))
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in FORBIDDEN_MODULE_ROOTS:
                    bad.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_MODULE_ROOTS:
                bad.append(node.module)
    if bad:
        raise Program09Error(f"Program 09 source imports forbidden model-stack modules: {sorted(set(bad))}")


# ---------------------------------------------------------------------------
# Config / frozen-protocol verification
# ---------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except Exception as e:
        raise Program09Error("PyYAML is required to read protocol.yaml.") from e
    if not path.exists():
        raise Program09Error(f"Missing protocol config: {path}")
    try:
        obj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as e:
        raise Program09Error(f"Cannot parse protocol config {path}: {e}") from e
    if not isinstance(obj, dict):
        raise Program09Error("protocol.yaml must contain a mapping at the top level.")
    return obj


def protocol_version(cfg: Mapping[str, Any]) -> str:
    project = cfg.get("project") or {}
    return str(project.get("protocol_version", "1.0")) if isinstance(project, Mapping) else "1.0"


def configured_seeds(cfg: Mapping[str, Any]) -> tuple[int, ...]:
    tr = cfg.get("training") or {}
    vals = tr.get("seeds", [20260826, 20260827, 20260828]) if isinstance(tr, Mapping) else [20260826, 20260827, 20260828]
    seeds = tuple(int(x) for x in vals)
    if not seeds or len(seeds) != len(set(seeds)):
        raise Program09Error("training.seeds must be nonempty and unique.")
    return seeds


def configured_target_steps(cfg: Mapping[str, Any]) -> tuple[int, ...]:
    ope = cfg.get("ope") or {}
    vals = ope.get("target_steps", list(range(0, 401, 20))) if isinstance(ope, Mapping) else list(range(0, 401, 20))
    steps = tuple(int(x) for x in vals)
    if not steps or len(steps) != len(set(steps)) or sorted(steps) != list(steps):
        raise Program09Error("ope.target_steps must be a unique sorted nonempty list.")
    return steps


def verify_protocol_and_gate(root: Path, config_sha: str, pv: str, paper_mode: bool) -> dict[str, Any]:
    out: dict[str, Any] = {"protocol_config_sha256": config_sha, "protocol_version": pv}
    lock_path = root / "manifests" / "protocol_lock.json"
    gate_path = root / "outputs" / "frozen_gate.json"
    gate_side = root / "manifests" / "frozen_gate_manifest.json"

    if paper_mode and not lock_path.exists():
        raise Program09Error("Paper outputs require manifests/protocol_lock.json.")
    if lock_path.exists():
        lock = read_json(lock_path)
        locked = first_present(lock, (("config_sha256",), ("protocol_config_sha256",), ("protocol", "config_sha256"), ("inputs", "config_sha256")))
        if locked != config_sha:
            raise Program09Error(f"protocol.yaml differs from protocol lock: locked={locked}, observed={config_sha}")
        out["protocol_lock_sha256"] = sha256_file(lock_path)

    if paper_mode and (not gate_path.exists() or not gate_side.exists()):
        raise Program09Error("Paper outputs require the frozen Program 07 gate and its manifest.")
    if gate_path.exists() or gate_side.exists():
        if not (gate_path.exists() and gate_side.exists()):
            raise Program09Error("Frozen gate and frozen gate manifest must either both exist or both be absent.")
        gate = read_json(gate_path)
        side = read_json(gate_side)
        if gate.get("manifest_type") != "frozen_reuse_gate":
            raise Program09Error("Invalid frozen gate manifest_type.")
        if str(gate.get("protocol_version")) != pv or gate.get("protocol_config_sha256") != config_sha:
            raise Program09Error("Frozen gate protocol identity differs from current protocol.")
        observed = sha256_file(gate_path)
        expected = first_present(side, (("gate_file_sha256",), ("file_sha256",), ("frozen_gate_sha256",)))
        if expected != observed:
            raise Program09Error("Frozen gate file hash differs from frozen_gate_manifest.json.")
        out["frozen_gate_sha256"] = observed
        out["frozen_gate_manifest_sha256"] = sha256_file(gate_side)
        out["frozen_gate"] = gate
    return out


def verify_program_manifest_output(root: Path, manifest_path: Path, required_type: str, required_split: str | None = None) -> dict[str, Any]:
    m = read_json(manifest_path)
    if m.get("manifest_type") != required_type:
        raise Program09Error(f"Unexpected manifest_type in {manifest_path}: {m.get('manifest_type')}")
    if required_split is not None and str(m.get("split")) != required_split:
        raise Program09Error(f"{manifest_path.name} split mismatch: expected={required_split}, observed={m.get('split')}")
    for rec in m.get("outputs", []):
        if not isinstance(rec, Mapping):
            continue
        p = rec.get("path")
        expected = rec.get("sha256")
        if isinstance(p, str) and isinstance(expected, str):
            fp = root / p
            if fp.exists() and sha256_file(fp) != expected:
                raise Program09Error(f"Current upstream output differs from manifest: {fp}")
    return m


# ---------------------------------------------------------------------------
# Research-table validation
# ---------------------------------------------------------------------------


def assert_close(observed: float | None, expected: float | None, label: str, atol: float = 1e-10, rtol: float = 1e-9) -> None:
    if observed is None or expected is None:
        return
    if not math.isclose(observed, expected, abs_tol=atol, rel_tol=rtol):
        raise Program09Error(f"Invariant failed for {label}: observed={observed}, expected={expected}")


def validate_t01(rows: Sequence[Mapping[str, Any]], expected_seeds: Sequence[int], paper: bool) -> None:
    seeds = {as_int(r["training_seed"]) for r in rows}
    if paper and not set(expected_seeds).issubset(seeds):
        raise Program09Error(f"T01 lacks one or more frozen training seeds: expected={expected_seeds}, observed={sorted(seeds)}")
    for r in rows:
        step = as_int(r["step"])
        if step < 0:
            raise Program09Error("T01 contains a negative training step.")
        if paper and str(r.get("mode")) != "paper":
            raise Program09Error("Paper T01 contains non-paper training rows.")
        tr = finite_or_none(r.get("truncation_rate"))
        if tr is not None and not (-1e-12 <= tr <= 1 + 1e-12):
            raise Program09Error("T01 truncation_rate outside [0,1].")


def validate_t02(rows: Sequence[Mapping[str, Any]]) -> None:
    for r in rows:
        if as_int(r["behavior_step"]) != 0:
            raise Program09Error("T02 fixed frontier contains behavior_step != 0.")
        k = as_int(r["K"])
        if k <= 0:
            raise Program09Error("T02 K must be positive.")
        ress = finite_or_none(r["median_prompt_relative_ess"])
        if ress is not None and not (1.0 / k - 1e-8 <= ress <= 1 + 1e-8):
            raise Program09Error(f"T02 median rESS outside [1/K,1]: {ress}")
        status = str(r.get("estimate_status", ""))
        est = finite_or_none(r.get("estimate"))
        online = finite_or_none(r.get("online_reference"))
        signed = finite_or_none(r.get("signed_error"))
        absolute = finite_or_none(r.get("absolute_error"))
        if status == "ok" and est is not None and online is not None:
            assert_close(signed, est - online, "T02 signed_error")
            assert_close(absolute, abs(est - online), "T02 absolute_error")


def validate_t03(rows: Sequence[Mapping[str, Any]]) -> None:
    allowed_q = {"Q1_short", "Q2", "Q3", "Q4_long", ""}
    for r in rows:
        k = as_int(r["K"])
        ess = finite_or_none(r["ess"])
        ress = finite_or_none(r["relative_ess"])
        cmax = finite_or_none(r["max_normalized_weight"])
        if ess is not None and not (1 - 1e-7 <= ess <= k + 1e-7):
            raise Program09Error(f"T03 ESS outside [1,K]: {ess}, K={k}")
        if ress is not None and not (1.0 / k - 1e-7 <= ress <= 1 + 1e-7):
            raise Program09Error(f"T03 rESS outside [1/K,1]: {ress}")
        if ess is not None and ress is not None:
            assert_close(ress, ess / k, "T03 relative_ess", atol=1e-8)
        if cmax is not None and not (1.0 / k - 1e-7 <= cmax <= 1 + 1e-7):
            raise Program09Error(f"T03 max normalized weight outside [1/K,1]: {cmax}")
        if str(r.get("length_quantile", "")) not in allowed_q:
            raise Program09Error(f"Unknown T03 length_quantile={r.get('length_quantile')!r}")


def validate_t04(rows: Sequence[Mapping[str, Any]]) -> None:
    for r in rows:
        oe = finite_or_none(r["old_absolute_error"])
        re_ = finite_or_none(r["recent_absolute_error"])
        de = finite_or_none(r["delta_error_old_minus_recent"])
        if oe is not None and re_ is not None:
            assert_close(de, oe - re_, "T04 delta_error")
        oldr = finite_or_none(r["old_median_relative_ess"])
        newr = finite_or_none(r["recent_median_relative_ess"])
        dr = finite_or_none(r["delta_relative_ess_recent_minus_old"])
        if oldr is not None and newr is not None:
            assert_close(dr, newr - oldr, "T04 delta_rESS")
        if as_int(r["recent_behavior_step"]) >= as_int(r["target_step"]):
            raise Program09Error("T04 recent behavior checkpoint must precede target checkpoint.")


def validate_t05(rows: Sequence[Mapping[str, Any]]) -> None:
    for r in rows:
        if str(r["split"]) != "development":
            raise Program09Error("T05 calibration table contains non-development rows.")
        if str(r["estimator"]) != PRIMARY_ESTIMATOR:
            raise Program09Error("T05 gate calibration must use the primary prompt-WIS estimator.")
        ar = finite_or_none(r["accept_rate"])
        if ar is not None and not (-1e-12 <= ar <= 1 + 1e-12):
            raise Program09Error("T05 accept_rate outside [0,1].")


def validate_t06(rows: Sequence[Mapping[str, Any]]) -> None:
    for r in rows:
        eps = finite_or_none(r["tolerance"])
        err = finite_or_none(r["true_absolute_error"])
        accepted = as_bool(r["accepted"])
        reliable = as_bool(r["reliable"])
        fa = as_bool(r["false_accept"])
        fr = as_bool(r["false_reject"])
        if eps is not None and err is not None:
            if reliable != (err <= eps + 1e-12):
                raise Program09Error("T06 reliable flag disagrees with true_absolute_error <= tolerance.")
        if fa != (accepted and not reliable):
            raise Program09Error("T06 false_accept invariant failed.")
        if fr != ((not accepted) and reliable):
            raise Program09Error("T06 false_reject invariant failed.")
        decision = str(r["decision"]).lower()
        if accepted and decision not in {"accept", "reuse", "accepted"}:
            raise Program09Error("T06 accepted=True but decision is not an accept/reuse label.")


def validate_t08(rows: Sequence[Mapping[str, Any]]) -> None:
    for r in rows:
        if str(r["dataset"]).upper() != "SVAMP":
            raise Program09Error("T08 contains a non-SVAMP dataset row.")
        if str(r["estimator"]) not in {"is", PRIMARY_ESTIMATOR}:
            raise Program09Error("T08 contains an unexpected estimator.")
        accepted = as_bool(r["gate_accepted"])
        reliable = as_bool(r["gate_reliable"])
        if as_bool(r["gate_false_accept"]) != (accepted and not reliable):
            raise Program09Error("T08 gate_false_accept invariant failed.")
        if as_bool(r["gate_false_reject"]) != ((not accepted) and reliable):
            raise Program09Error("T08 gate_false_reject invariant failed.")


# ---------------------------------------------------------------------------
# Machine-readable input loading
# ---------------------------------------------------------------------------


def load_inputs(root: Path, expected_seeds: Sequence[int], paper: bool, require_svamp: bool, require_model: bool) -> dict[str, InputTable | None]:
    tdir = root / "outputs" / "tables"
    inputs: dict[str, InputTable | None] = {
        "T01": load_table("T01", tdir / "T01_training_trajectory.csv", T01_REQUIRED),
        "T02": load_table("T02", tdir / "T02_fixed_reuse.csv", T02_REQUIRED),
        "T03": load_table("T03", tdir / "T03_overlap.csv", T03_REQUIRED),
        "T04": load_table("T04", tdir / "T04_rolling_comparison.csv", T04_REQUIRED),
        "T05": load_table("T05", tdir / "T05_gate_calibration.csv", T05_REQUIRED),
        "T06": load_table("T06", tdir / "T06_gate_test.csv", T06_REQUIRED),
        "T07": load_table("T07", tdir / "T07_sample_size.csv", T07_REQUIRED, optional=True),
        "T08": load_table("T08", tdir / "T08_svamp.csv", T08_REQUIRED, optional=not require_svamp),
        "T09": load_table("T09", tdir / "T09_model_robustness.csv", set(), optional=not require_model),
    }
    validate_t01(inputs["T01"].rows, expected_seeds, paper)  # type: ignore[union-attr]
    validate_t02(inputs["T02"].rows)  # type: ignore[union-attr]
    validate_t03(inputs["T03"].rows)  # type: ignore[union-attr]
    validate_t04(inputs["T04"].rows)  # type: ignore[union-attr]
    validate_t05(inputs["T05"].rows)  # type: ignore[union-attr]
    validate_t06(inputs["T06"].rows)  # type: ignore[union-attr]
    if inputs["T08"] is not None:
        validate_t08(inputs["T08"].rows)
    return inputs


def ensure_paper_test_complete(root: Path, inputs: Mapping[str, InputTable | None], expected_seeds: Sequence[int]) -> dict[str, Any]:
    p06_test = root / "outputs" / "diagnostics" / "program06_test_manifest.json"
    p07_validate = root / "outputs" / "diagnostics" / "program07_validate_manifest.json"
    p07_test = root / "outputs" / "diagnostics" / "program07_test-only_manifest.json"
    if not p06_test.exists() or not p07_validate.exists() or not p07_test.exists():
        raise Program09Error(
            "Paper mode requires Program 06 official-test plus Program 07 held-out-seed validation "
            "and test-only manifests."
        )
    m06 = verify_program_manifest_output(root, p06_test, "program06_analysis_manifest", "test")
    m07v = verify_program_manifest_output(root, p07_validate, "program07_gate_manifest")
    m07 = verify_program_manifest_output(root, p07_test, "program07_gate_manifest")

    t02_test_seeds = {as_int(r["training_seed"]) for r in inputs["T02"].rows if str(r["split"]) == "test"}  # type: ignore[union-attr]
    t06_test_seeds = {as_int(r["training_seed"]) for r in inputs["T06"].rows if str(r["evaluation_role"]) == "official_test"}  # type: ignore[union-attr]
    t06_validation_seeds = {as_int(r["training_seed"]) for r in inputs["T06"].rows if str(r["evaluation_role"]) == "heldout_seed_validation"}  # type: ignore[union-attr]
    if set(expected_seeds) - t02_test_seeds:
        raise Program09Error(f"T02 official test lacks seeds {sorted(set(expected_seeds)-t02_test_seeds)}")
    if set(expected_seeds) - t06_test_seeds:
        raise Program09Error(f"T06 official test lacks seeds {sorted(set(expected_seeds)-t06_test_seeds)}")
    if not t06_validation_seeds:
        raise Program09Error("T06 has no held-out training-path validation rows; Program 07 validate must run before final paper outputs.")
    return {
        "program06_test_manifest_sha256": sha256_file(p06_test),
        "program07_validate_manifest_sha256": sha256_file(p07_validate),
        "program07_test_manifest_sha256": sha256_file(p07_test),
        "program06_test_manifest": m06,
        "program07_validate_manifest": m07v,
        "program07_test_manifest": m07,
    }


# ---------------------------------------------------------------------------
# Optional Program 05 online development accuracy for Figure 1
# ---------------------------------------------------------------------------


def load_dev_online_reference(root: Path, expected_seeds: Sequence[int]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = root / "data" / "online_reference" / "gsm8k" / "split=development"
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    if not base.exists():
        return rows, provenance
    for seed in expected_seeds:
        seed_dir = base / f"seed={seed}"
        if not seed_dir.exists():
            continue
        for target_dir in sorted(seed_dir.glob("target_step=*")):
            p = target_dir / "reference_summary.json"
            if not p.exists():
                continue
            obj = read_json(p)
            if obj.get("manifest_type") != "online_reference_target_summary":
                raise Program09Error(f"Invalid Program 05 reference summary: {p}")
            if str(obj.get("split")) != "development" or as_int(obj.get("training_seed")) != seed:
                raise Program09Error(f"Program 05 reference identity mismatch: {p}")
            rows.append({
                "training_seed": seed,
                "target_step": as_int(obj["target_step"]),
                "online_reference": finite_or_none(obj.get("online_reference")),
                "samples_per_prompt": as_int(obj.get("samples_per_prompt")),
            })
            provenance.append({"path": rel(p, root), "sha256": sha256_file(p)})
    return rows, provenance


def verify_dev_online_grid(rows: Sequence[Mapping[str, Any]], seeds: Sequence[int], target_steps: Sequence[int]) -> None:
    observed = {(as_int(r["training_seed"]), as_int(r["target_step"])) for r in rows}
    expected = {(int(s), int(t)) for s in seeds for t in target_steps}
    missing = sorted(expected - observed)
    if missing:
        preview = missing[:10]
        raise Program09Error(
            f"Paper Figure 1 requires Program 05 development on-policy references for the frozen target grid; "
            f"missing {len(missing)} seed-target summaries, first={preview}."
        )


# ---------------------------------------------------------------------------
# Paper tables
# ---------------------------------------------------------------------------


def select_split(rows: Sequence[Mapping[str, Any]], split: str) -> list[Mapping[str, Any]]:
    return [r for r in rows if str(r.get("split")) == split]


def main_estimator_rows(t02: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    rows = [r for r in t02 if str(r["split"]) == split and str(r["estimator"]) in {"is", PRIMARY_ESTIMATOR}]
    out: list[dict[str, Any]] = []
    for est in ("is", PRIMARY_ESTIMATOR):
        g = [r for r in rows if str(r["estimator"]) == est]
        if not g:
            continue
        abs_err = [finite_or_none(r["absolute_error"]) for r in g]
        signed = [finite_or_none(r["signed_error"]) for r in g]
        nonfinite = sum(1 for r in g if str(r.get("estimate_status")) != "ok" or finite_or_none(r.get("estimate")) is None)
        ci_width = []
        for r in g:
            lo, hi = finite_or_none(r.get("ope_ci_low")), finite_or_none(r.get("ope_ci_high"))
            ci_width.append((hi - lo) if lo is not None and hi is not None else None)
        out.append({
            "split": split,
            "estimator": est,
            "n_pair_seed_cases": len(g),
            "mean_absolute_error": mean(abs_err),
            "median_absolute_error": median(abs_err),
            "rmse": root_mean_square(signed),
            "mean_signed_error": mean(signed),
            "mean_ope_ci_width": mean(ci_width),
            "mean_median_prompt_relative_ess": mean(finite_or_none(r["median_prompt_relative_ess"]) for r in g),
            "nonfinite_case_fraction": nonfinite / len(g),
        })
    return out


def fixed_vs_rolling_rows(t04: Sequence[Mapping[str, Any]], split: str) -> list[dict[str, Any]]:
    rows = [r for r in t04 if str(r["split"]) == split and str(r["estimator"]) == PRIMARY_ESTIMATOR]
    groups: dict[int | str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        lag = as_int(r["target_step"]) - as_int(r["recent_behavior_step"])
        groups[lag].append(r)
    groups["overall"] = rows
    out: list[dict[str, Any]] = []
    for lag in sorted([x for x in groups if isinstance(x, int)]) + ["overall"]:
        g = groups[lag]
        if not g:
            continue
        olde = [finite_or_none(r["old_absolute_error"]) for r in g]
        newe = [finite_or_none(r["recent_absolute_error"]) for r in g]
        de = [finite_or_none(r["delta_error_old_minus_recent"]) for r in g]
        dr = [finite_or_none(r["delta_relative_ess_recent_minus_old"]) for r in g]
        out.append({
            "split": split,
            "recent_log_lag_steps": lag,
            "n_pair_seed_cases": len(g),
            "mean_old_absolute_error": mean(olde),
            "mean_recent_absolute_error": mean(newe),
            "mean_error_reduction_old_minus_recent": mean(de),
            "fraction_recent_lower_error": mean(1.0 if (a is not None and b is not None and b < a) else 0.0 for a, b in zip(olde, newe)),
            "mean_old_median_relative_ess": mean(finite_or_none(r["old_median_relative_ess"]) for r in g),
            "mean_recent_median_relative_ess": mean(finite_or_none(r["recent_median_relative_ess"]) for r in g),
            "mean_relative_ess_gain": mean(dr),
        })
    return out


def aggregate_gate_cases(rows: Sequence[Mapping[str, Any]], *, dataset: str, role: str, gate_kind: str = PRIMARY_GATE_KIND) -> list[dict[str, Any]]:
    filtered = [r for r in rows if str(r.get("gate_kind")) == gate_kind]
    by_eps: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for r in filtered:
        eps = finite_or_none(r.get("tolerance", r.get("gate_tolerance")))
        if eps is not None:
            by_eps[eps].append(r)
    out: list[dict[str, Any]] = []
    for eps in sorted(by_eps):
        g = by_eps[eps]
        accepted_key = "accepted" if "accepted" in g[0] else "gate_accepted"
        reliable_key = "reliable" if "reliable" in g[0] else "gate_reliable"
        fa_key = "false_accept" if "false_accept" in g[0] else "gate_false_accept"
        err_key = "true_absolute_error" if "true_absolute_error" in g[0] else "absolute_error"
        accepted = [as_bool(r[accepted_key]) for r in g]
        reliable = [as_bool(r[reliable_key]) for r in g]
        false_accept = [as_bool(r[fa_key]) for r in g]
        errors = [finite_or_none(r[err_key]) for r in g]
        ae = [e for e, a in zip(errors, accepted) if a]
        re = [e for e, a in zip(errors, accepted) if not a]
        na = sum(accepted)
        out.append({
            "dataset": dataset,
            "evaluation_role": role,
            "gate_kind": gate_kind,
            "tolerance": eps,
            "n_pair_seed_cases": len(g),
            "accept_rate": na / len(g),
            "false_accept_count": sum(false_accept),
            "false_accept_rate_among_accepted": (sum(false_accept) / na) if na else 0.0,
            "false_accept_fraction_all": sum(false_accept) / len(g),
            "reliable_fraction": sum(reliable) / len(g),
            "accepted_mae": mean(ae),
            "rejected_mae": mean(re),
            "overall_mae": mean(errors),
        })
    return out


def gate_validation_rows(t06: Sequence[Mapping[str, Any]], t08: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    held = [r for r in t06 if str(r["evaluation_role"]) == "heldout_seed_validation"]
    test = [r for r in t06 if str(r["evaluation_role"]) == "official_test"]
    if held:
        out.extend(aggregate_gate_cases(held, dataset="GSM8K", role="heldout_training_seed"))
    if test:
        out.extend(aggregate_gate_cases(test, dataset="GSM8K", role="official_test"))
    if t08:
        sv = [r for r in t08 if str(r["estimator"]) == PRIMARY_ESTIMATOR and str(r["gate_kind"]) == PRIMARY_GATE_KIND]
        if sv:
            out.extend(aggregate_gate_cases(sv, dataset="SVAMP", role="external_prompt_distribution"))
    return out


def gate_baseline_rows(t06: Sequence[Mapping[str, Any]], t08: Sequence[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for role, label in (("heldout_seed_validation", "heldout_training_seed"), ("official_test", "official_test")):
        g = [r for r in t06 if str(r["evaluation_role"]) == role]
        if g:
            out.extend(aggregate_gate_cases(g, dataset="GSM8K", role=label, gate_kind=KL_BASELINE_KIND))
    if t08:
        sv = [r for r in t08 if str(r["estimator"]) == PRIMARY_ESTIMATOR and str(r["gate_kind"]) == KL_BASELINE_KIND]
        if sv:
            out.extend(aggregate_gate_cases(sv, dataset="SVAMP", role="external_prompt_distribution", gate_kind=KL_BASELINE_KIND))
    return out


def training_summary_rows(t01: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for r in t01:
        by_seed[as_int(r["training_seed"])].append(r)
    out: list[dict[str, Any]] = []
    for seed in sorted(by_seed):
        g = sorted(by_seed[seed], key=lambda r: as_int(r["step"]))
        last = g[-1]
        peak_mem = [finite_or_none(r.get("gpu_memory_peak_bytes")) for r in g]
        out.append({
            "training_seed": seed,
            "final_step": as_int(last["step"]),
            "final_mean_reward": finite_or_none(last.get("mean_reward")),
            "final_correctness_reward": finite_or_none(last.get("correctness_reward")),
            "final_kl_proxy": finite_or_none(last.get("kl_proxy")),
            "final_mean_completion_length": finite_or_none(last.get("mean_completion_length")),
            "final_truncation_rate": finite_or_none(last.get("truncation_rate")),
            "peak_gpu_memory_bytes": max((x for x in peak_mem if x is not None), default=None),
            "mean_tokens_per_second": mean(finite_or_none(r.get("tokens_per_second")) for r in g),
        })
    return out


def sample_size_rows(t07: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    by: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for r in t07:
        by[(as_int(r["K"]), str(r["estimator"]))].append(r)
    out = []
    for (k, est), g in sorted(by.items()):
        out.append({
            "K": k, "estimator": est, "n_pair_seed_cases": len(g),
            "mean_absolute_error": mean(finite_or_none(r["absolute_error"]) for r in g),
            "mean_median_prompt_relative_ess": mean(finite_or_none(r["median_prompt_relative_ess"]) for r in g),
            "mean_bootstrap_ci_width": mean(finite_or_none(r["bootstrap_ci_width"]) for r in g),
            "mean_bootstrap_nonfinite_fraction": mean(finite_or_none(r["bootstrap_nonfinite_fraction"]) for r in g),
        })
    return out


# ---------------------------------------------------------------------------
# LaTeX table and paper-value production
# ---------------------------------------------------------------------------


def latex_escape(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in s)


def table_to_latex(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], caption: str, label: str) -> str:
    align = "l" + "r" * max(0, len(columns) - 1)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        rf"\caption{{{latex_escape(caption)}}}", rf"\label{{{latex_escape(label)}}}",
        rf"\begin{{tabular}}{{{align}}}", r"\toprule",
        " & ".join(latex_escape(c.replace("_", " ")) for c in columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        vals: list[str] = []
        for c in columns:
            v = row.get(c)
            if isinstance(v, float) or (isinstance(v, str) and re.fullmatch(r"[-+]?\d+(\.\d+)?([eE][-+]?\d+)?", v.strip() or "x")):
                vals.append(latex_escape(format_number(v)))
            else:
                vals.append(latex_escape(str(v if v not in (None, "") else "NA")))
        lines.append(" & ".join(vals) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def write_table_pair(base: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str], caption: str, label: str) -> list[Path]:
    csv_path = base.with_suffix(".csv")
    tex_path = base.with_suffix(".tex")
    atomic_write_csv(csv_path, rows, columns)
    atomic_write_text(tex_path, table_to_latex(rows, columns, caption, label))
    return [csv_path, tex_path]


def macro_name(key: str) -> str:
    # TeX control-word names may contain letters only. Convert numeric chunks
    # to letter names rather than silently producing invalid macros such as
    # \GRPOGateTauEps010.
    digit_words = {
        "0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
        "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine",
    }
    parts = re.findall(r"[A-Za-z]+|\d+", key)
    rendered: list[str] = []
    for x in parts:
        if x.isdigit():
            rendered.append("".join(digit_words[d] for d in x))
        else:
            rendered.append(x[0].upper() + x[1:])
    name = "GRPO" + "".join(rendered)
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise Program09Error(f"Internal LaTeX macro-name construction failed for key={key!r}")
    return name


def build_paper_values(
    *, estimator_table: Sequence[Mapping[str, Any]], rolling_table: Sequence[Mapping[str, Any]],
    gate_table: Sequence[Mapping[str, Any]], t05: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    vals: dict[str, Any] = {}
    for r in estimator_table:
        est = str(r["estimator"])
        vals[f"test_{est}_mean_absolute_error"] = r["mean_absolute_error"]
        vals[f"test_{est}_median_absolute_error"] = r["median_absolute_error"]
        vals[f"test_{est}_nonfinite_case_fraction"] = r["nonfinite_case_fraction"]
    overall = next((r for r in rolling_table if str(r["recent_log_lag_steps"]) == "overall"), None)
    if overall:
        vals["rolling_mean_error_reduction"] = overall["mean_error_reduction_old_minus_recent"]
        vals["rolling_mean_relative_ess_gain"] = overall["mean_relative_ess_gain"]
        vals["rolling_fraction_recent_lower_error"] = overall["fraction_recent_lower_error"]
    for r in t05:
        if str(r["gate_kind"]) == PRIMARY_GATE_KIND:
            eps = finite_or_none(r["tolerance"])
            if eps is not None:
                tag = str(int(round(eps * 1000))).zfill(3)
                vals[f"gate_tau_eps_{tag}"] = finite_or_none(r["fitted_threshold"])
                vals[f"gate_calibration_accept_rate_eps_{tag}"] = finite_or_none(r["accept_rate"])
    for r in gate_table:
        eps = finite_or_none(r["tolerance"])
        if eps is None:
            continue
        tag = str(int(round(eps * 1000))).zfill(3)
        ds = str(r["dataset"]).lower()
        role = str(r["evaluation_role"]).lower()
        prefix = f"gate_{ds}_{role}_eps_{tag}"
        vals[prefix + "_accept_rate"] = r["accept_rate"]
        vals[prefix + "_false_accept_fraction_all"] = r["false_accept_fraction_all"]
        vals[prefix + "_accepted_mae"] = r["accepted_mae"]
        vals[prefix + "_rejected_mae"] = r["rejected_mae"]
    return vals


def write_paper_values(base: Path, values: Mapping[str, Any]) -> list[Path]:
    json_path = base / "paper_values.json"
    tex_path = base / "paper_values.tex"
    atomic_write_json(json_path, {"schema_version": "1.0", "values": dict(sorted(values.items()))})
    lines = ["% Auto-generated by 09_make_paper_outputs.py. Do not edit by hand."]
    for key, val in sorted(values.items()):
        name = macro_name(key)
        rendered = format_number(val, digits=4) if isinstance(val, (int, float)) or val is None else latex_escape(str(val))
        lines.append(rf"\newcommand{{\{name}}}{{{rendered}}}")
    lines.append("")
    atomic_write_text(tex_path, "\n".join(lines))
    return [json_path, tex_path]


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------


def import_plotting() -> tuple[Any, Any]:
    try:
        import numpy as np
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as e:
        raise Program09Error("Program 09 requires numpy and matplotlib for paper figures.") from e
    return np, plt


def atomic_save_figure(fig: Any, path: Path, **kwargs: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.stem}.{os.getpid()}.tmp{path.suffix}"
    try:
        fig.savefig(tmp, **kwargs)
        with tmp.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_dir(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def save_figure_pair(fig: Any, base: Path) -> list[Path]:
    pdf = base.with_suffix(".pdf")
    png = base.with_suffix(".png")
    atomic_save_figure(fig, pdf, bbox_inches="tight")
    atomic_save_figure(fig, png, dpi=220, bbox_inches="tight")
    return [pdf, png]


def grouped_xy(rows: Sequence[Mapping[str, Any]], xkey: str, ykey: str, groupkey: str = "training_seed") -> dict[Any, tuple[list[float], list[float]]]:
    by: dict[Any, list[tuple[float, float]]] = defaultdict(list)
    for r in rows:
        x, y = finite_or_none(r.get(xkey)), finite_or_none(r.get(ykey))
        if x is not None and y is not None:
            by[r.get(groupkey)].append((x, y))
    return {g: ([p[0] for p in sorted(v)], [p[1] for p in sorted(v)]) for g, v in by.items()}


def pooled_by_x(rows: Sequence[Mapping[str, Any]], xkey: str, ykey: str) -> tuple[list[float], list[float]]:
    by: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        x, y = finite_or_none(r.get(xkey)), finite_or_none(r.get(ykey))
        if x is not None and y is not None:
            by[x].append(y)
    xs = sorted(by)
    return xs, [math.fsum(by[x]) / len(by[x]) for x in xs]


def figure1_training(t01: Sequence[Mapping[str, Any]], dev_online: Sequence[Mapping[str, Any]], out_base: Path) -> list[Path]:
    np, plt = import_plotting()
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)

    for seed, (x, y) in grouped_xy(t01, "step", "correctness_reward").items():
        axes[0].plot(x, y, linewidth=1.0, alpha=0.45, label=f"train seed {seed}")
    px, py = pooled_by_x(t01, "step", "correctness_reward")
    if px:
        axes[0].plot(px, py, linewidth=2.2, label="mean training correctness")
    if dev_online:
        for seed, (x, y) in grouped_xy(dev_online, "target_step", "online_reference").items():
            axes[0].plot(x, y, linestyle="--", linewidth=1.0, alpha=0.45)
        dx, dy = pooled_by_x(dev_online, "target_step", "online_reference")
        if dx:
            axes[0].plot(dx, dy, linestyle="--", linewidth=2.2, label="mean dev on-policy accuracy")
    axes[0].set_ylabel("Correctness / accuracy")
    axes[0].set_ylim(-0.02, 1.02)
    axes[0].legend(fontsize=8, ncol=2)

    for _, (x, y) in grouped_xy(t01, "step", "kl_proxy").items():
        axes[1].plot(x, y, linewidth=1.2, alpha=0.65)
    kx, ky = pooled_by_x(t01, "step", "kl_proxy")
    if kx:
        axes[1].plot(kx, ky, linewidth=2.2)
    axes[1].set_ylabel("Training KL proxy")

    for _, (x, y) in grouped_xy(t01, "step", "mean_completion_length").items():
        axes[2].plot(x, y, linewidth=1.2, alpha=0.65)
    lx, ly = pooled_by_x(t01, "step", "mean_completion_length")
    if lx:
        axes[2].plot(lx, ly, linewidth=2.2)
    axes[2].set_ylabel("Mean completion length")
    axes[2].set_xlabel("GRPO training step")
    fig.suptitle("Figure 1. GRPO training trajectory")
    fig.tight_layout()
    paths = save_figure_pair(fig, out_base)
    plt.close(fig)
    return paths


def figure2_fixed_frontier(t02: Sequence[Mapping[str, Any]], split: str, out_base: Path) -> list[Path]:
    np, plt = import_plotting()
    rows = [r for r in t02 if str(r["split"]) == split and str(r["estimator"]) == PRIMARY_ESTIMATOR]
    if not rows:
        raise Program09Error(f"No {PRIMARY_ESTIMATOR} T02 rows for split={split}")
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True)
    for seed, g in _by_seed(rows).items():
        g = sorted(g, key=lambda r: as_int(r["target_step"]))
        x = [as_int(r["target_step"]) for r in g]
        y = [finite_or_none(r["absolute_error"]) for r in g]
        rr = [finite_or_none(r["median_prompt_relative_ess"]) for r in g]
        axes[0].plot(x, y, marker="o", markersize=3, linewidth=1.1, alpha=0.7, label=f"seed {seed}")
        axes[1].plot(x, rr, marker="o", markersize=3, linewidth=1.1, alpha=0.7)
    x1, y1 = pooled_by_x(rows, "target_step", "absolute_error")
    x2, y2 = pooled_by_x(rows, "target_step", "median_prompt_relative_ess")
    axes[0].plot(x1, y1, linewidth=2.4, label="seed mean")
    axes[1].plot(x2, y2, linewidth=2.4)
    axes[0].set_ylabel("Absolute pWIS error")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_ylabel("Median prompt rESS")
    axes[1].set_xlabel("Target checkpoint step (behavior = step 0)")
    axes[1].set_ylim(0, 1.02)
    fig.suptitle(f"Figure 2. Fixed-log reuse frontier ({split})")
    fig.tight_layout()
    paths = save_figure_pair(fig, out_base)
    plt.close(fig)
    return paths


def _by_seed(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[Mapping[str, Any]]]:
    out: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        out[as_int(r["training_seed"])].append(r)
    return out


def figure3_overlap_error(t02: Sequence[Mapping[str, Any]], split: str, out_base: Path) -> list[Path]:
    np, plt = import_plotting()
    rows = [r for r in t02 if str(r["split"]) == split and str(r["estimator"]) == PRIMARY_ESTIMATOR]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for seed, g in _by_seed(rows).items():
        x1 = [finite_or_none(r["median_prompt_relative_ess"]) for r in g]
        x2 = [finite_or_none(r["mean_d2"]) for r in g]
        y = [finite_or_none(r["absolute_error"]) for r in g]
        keep1 = [(a, b) for a, b in zip(x1, y) if a is not None and b is not None]
        keep2 = [(a, b) for a, b in zip(x2, y) if a is not None and b is not None]
        if keep1:
            axes[0].scatter([a for a, _ in keep1], [b for _, b in keep1], s=24, alpha=0.72, label=f"seed {seed}")
        if keep2:
            axes[1].scatter([a for a, _ in keep2], [b for _, b in keep2], s=24, alpha=0.72)
    axes[0].set_xlabel("Median prompt rESS")
    axes[0].set_ylabel("Absolute pWIS error")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("Mean D2 proxy")
    axes[1].set_ylabel("Absolute pWIS error")
    fig.suptitle(f"Figure 3. Offline overlap diagnostics vs OPE error ({split})")
    fig.tight_layout()
    paths = save_figure_pair(fig, out_base)
    plt.close(fig)
    return paths


def figure4_rolling(t04: Sequence[Mapping[str, Any]], split: str, out_base: Path) -> list[Path]:
    np, plt = import_plotting()
    rows = [r for r in t04 if str(r["split"]) == split and str(r["estimator"]) == PRIMARY_ESTIMATOR]
    if not rows:
        raise Program09Error(f"No rolling pWIS rows for split={split}")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    x = [finite_or_none(r["old_absolute_error"]) for r in rows]
    y = [finite_or_none(r["recent_absolute_error"]) for r in rows]
    pts = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    axes[0].scatter([a for a, _ in pts], [b for _, b in pts], s=25, alpha=0.75)
    if pts:
        hi = max(max(a for a, _ in pts), max(b for _, b in pts))
        axes[0].plot([0, hi], [0, hi], linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("Old-log absolute error")
    axes[0].set_ylabel("Recent-log absolute error")

    xr = [finite_or_none(r["old_median_relative_ess"]) for r in rows]
    yr = [finite_or_none(r["recent_median_relative_ess"]) for r in rows]
    ptsr = [(a, b) for a, b in zip(xr, yr) if a is not None and b is not None]
    axes[1].scatter([a for a, _ in ptsr], [b for _, b in ptsr], s=25, alpha=0.75)
    axes[1].plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Old-log median rESS")
    axes[1].set_ylabel("Recent-log median rESS")
    fig.suptitle(f"Figure 4. Rolling log refresh ({split})")
    fig.tight_layout()
    paths = save_figure_pair(fig, out_base)
    plt.close(fig)
    return paths


def figure5_gate(gate_table: Sequence[Mapping[str, Any]], out_base: Path) -> list[Path]:
    np, plt = import_plotting()
    if not gate_table:
        raise Program09Error("Cannot make Figure 5: no frozen-gate validation rows.")
    roles = []
    for ds, role in (("GSM8K", "heldout_training_seed"), ("GSM8K", "official_test"), ("SVAMP", "external_prompt_distribution")):
        if any(str(r["dataset"]) == ds and str(r["evaluation_role"]) == role for r in gate_table):
            roles.append((ds, role))
    eps = sorted({float(r["tolerance"]) for r in gate_table})
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.8), sharex=True)
    width = 0.8 / max(1, len(roles))
    centers = np.arange(len(eps), dtype=float)
    for j, (ds, role) in enumerate(roles):
        g = {(float(r["tolerance"])): r for r in gate_table if str(r["dataset"]) == ds and str(r["evaluation_role"]) == role}
        xs = centers + (j - (len(roles)-1)/2) * width
        acc = [float(g[e]["accept_rate"]) if e in g else np.nan for e in eps]
        fa = [float(g[e]["false_accept_fraction_all"]) if e in g else np.nan for e in eps]
        label = f"{ds}: {role.replace('_',' ')}"
        axes[0].bar(xs, acc, width=width, label=label)
        axes[1].bar(xs, fa, width=width, label=label)
    axes[0].set_ylabel("Accepted fraction")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    axes[1].set_ylabel("False-accept fraction (all cases)")
    axes[1].set_ylim(0, 1)
    axes[1].set_xticks(centers)
    axes[1].set_xticklabels([f"ε={e:g}" for e in eps])
    fig.suptitle("Figure 5. Frozen reuse gate validation")
    fig.tight_layout()
    paths = save_figure_pair(fig, out_base)
    plt.close(fig)
    return paths


def figure6_heatmap(t03: Sequence[Mapping[str, Any]], split: str, out_base: Path) -> list[Path]:
    np, plt = import_plotting()
    rows = [r for r in t03 if str(r["split"]) == split and as_int(r["behavior_step"]) == 0]
    vals = [finite_or_none(r["pair_abs_mean_log_ratio_per_token"]) for r in rows]
    finite_vals = np.asarray([x for x in vals if x is not None], dtype=float)
    if len(np.unique(finite_vals)) < 4:
        raise Program09Error("Figure 6 requested but T03 has insufficient distinct drift levels for a 4-bin heatmap.")
    edges = np.quantile(finite_vals, [0, .25, .5, .75, 1.0])
    edges = np.unique(edges)
    if len(edges) < 5:
        raise Program09Error("Figure 6 requested but drift quantiles collapse; do not force a misleading heatmap.")
    qlabels = ["Q1_short", "Q2", "Q3", "Q4_long"]
    matrix = np.full((4, 4), np.nan)
    for i in range(4):
        lo, hi = edges[i], edges[i+1]
        for j, q in enumerate(qlabels):
            xs = []
            for r in rows:
                d = finite_or_none(r["pair_abs_mean_log_ratio_per_token"])
                rr = finite_or_none(r["relative_ess"])
                if d is None or rr is None or str(r["length_quantile"]) != q:
                    continue
                inside = (lo <= d <= hi) if i == 3 else (lo <= d < hi)
                if inside:
                    xs.append(rr)
            if xs:
                matrix[j, i] = float(np.median(xs))
    if np.isnan(matrix).sum() > 4:
        raise Program09Error("Figure 6 requested but too many length×drift cells are empty.")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(matrix, aspect="auto", origin="lower", vmin=0, vmax=1)
    ax.set_yticks(range(4)); ax.set_yticklabels(["short Q1", "Q2", "Q3", "long Q4"])
    ax.set_xticks(range(4)); ax.set_xticklabels(["drift Q1", "drift Q2", "drift Q3", "drift Q4"])
    ax.set_xlabel("Per-token drift quantile")
    ax.set_ylabel("Completion-length quantile")
    ax.set_title(f"Figure 6. Median prompt rESS by length × drift ({split})")
    cb = fig.colorbar(im, ax=ax); cb.set_label("Median prompt rESS")
    fig.tight_layout()
    paths = save_figure_pair(fig, out_base)
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def choose_analysis_split(inputs: Mapping[str, InputTable | None], requested: str, paper: bool) -> str:
    available = {str(r["split"]) for r in inputs["T02"].rows}  # type: ignore[union-attr]
    if requested not in available:
        raise Program09Error(f"Requested analysis split={requested!r} not present in T02; available={sorted(available)}")
    if paper and requested != "test":
        raise Program09Error("Paper mode must use --split test for final inferential tables/figures.")
    return requested


def input_records(inputs: Mapping[str, InputTable | None], root: Path) -> list[dict[str, Any]]:
    out = []
    for key in sorted(inputs):
        t = inputs[key]
        if t is None:
            continue
        out.append({"key": key, "path": rel(t.path, root), "sha256": t.sha256, "bytes": t.path.stat().st_size, "row_count": len(t.rows)})
    return out


def output_records(paths: Sequence[Path], root: Path) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for p in paths:
        rp = rel(p, root)
        if rp in seen or not p.exists():
            continue
        seen.add(rp)
        out.append({"path": rp, "sha256": sha256_file(p), "bytes": p.stat().st_size})
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO-OPE Program 09: make paper tables/figures from machine-readable results only; never call a model.")
    p.add_argument("--config", default="configs/protocol.yaml")
    p.add_argument("--mode", choices=("smoke", "pilot", "paper"), default="paper")
    p.add_argument("--split", choices=("development", "test"), default="test", help="Primary split for Figures 2–4 and main estimator/refresh tables. Paper mode requires test.")
    p.add_argument("--resume", action="store_true", help="Accepted for CLI consistency. Program 09 is cheap and regenerates derived outputs atomically.")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cpu", help="Program 09 is CPU-only; --device cuda is rejected.")
    p.add_argument("--output-root", default=".")
    p.add_argument("--include-figure6", action="store_true", help="Opt-in appendix mechanism heatmap. Fails rather than forcing the figure when the data grid is inadequate.")
    p.add_argument("--require-svamp", action="store_true", help="Require T08/program08 outputs; otherwise SVAMP is included automatically when present.")
    p.add_argument("--require-model-robustness", action="store_true", help="Require optional T09_model_robustness.csv.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device != "cpu":
        raise Program09Error("Program 09 is CPU-only and must never use CUDA or call a model. Use --device cpu.")

    root = Path(args.output_root).expanduser().resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    audit_source_for_forbidden_imports(Path(__file__).resolve())

    cfg = load_yaml(config_path)
    config_sha = sha256_file(config_path)
    pv = protocol_version(cfg)
    seeds = configured_seeds(cfg)
    target_steps = configured_target_steps(cfg)
    paper = args.mode == "paper"
    frozen = verify_protocol_and_gate(root, config_sha, pv, paper)

    inputs = load_inputs(root, seeds, paper, args.require_svamp, args.require_model_robustness)
    primary_split = choose_analysis_split(inputs, args.split, paper)
    final_proof = ensure_paper_test_complete(root, inputs, seeds) if paper else {}

    # Optional robustness manifest: if T08 exists, it must be backed by Program 08 manifest.
    p08_manifest_path = root / "outputs" / "diagnostics" / "program08_svamp_manifest.json"
    p08_manifest: dict[str, Any] | None = None
    if inputs["T08"] is not None:
        if not p08_manifest_path.exists():
            raise Program09Error("T08_svamp.csv exists without program08_svamp_manifest.json.")
        p08_manifest = read_json(p08_manifest_path)
        if p08_manifest.get("manifest_type") != "program08_svamp_robustness_manifest":
            raise Program09Error("Invalid Program 08 robustness manifest.")
        rec = (p08_manifest.get("outputs") or {}).get("T08_svamp") if isinstance(p08_manifest.get("outputs"), Mapping) else None
        if isinstance(rec, Mapping) and rec.get("sha256") != inputs["T08"].sha256:
            raise Program09Error("T08_svamp.csv differs from Program 08 manifest hash.")

    dev_online, dev_online_prov = load_dev_online_reference(root, seeds)
    if paper:
        verify_dev_online_grid(dev_online, seeds, target_steps)

    base = root / "outputs" / "paper"
    if args.mode != "paper":
        base = base / f"_{args.mode}"
    tables_dir = base / "tables"
    figs_dir = base / "figures"
    nums_dir = base / "numbers"
    for d in (tables_dir, figs_dir, nums_dir):
        d.mkdir(parents=True, exist_ok=True)

    t01 = list(inputs["T01"].rows)  # type: ignore[union-attr]
    t02 = list(inputs["T02"].rows)  # type: ignore[union-attr]
    t03 = list(inputs["T03"].rows)  # type: ignore[union-attr]
    t04 = list(inputs["T04"].rows)  # type: ignore[union-attr]
    t05 = list(inputs["T05"].rows)  # type: ignore[union-attr]
    t06 = list(inputs["T06"].rows)  # type: ignore[union-attr]
    t07 = list(inputs["T07"].rows) if inputs["T07"] is not None else []
    t08 = list(inputs["T08"].rows) if inputs["T08"] is not None else []
    t09 = list(inputs["T09"].rows) if inputs["T09"] is not None else []

    estimator_table = main_estimator_rows(t02, primary_split)
    rolling_table = fixed_vs_rolling_rows(t04, primary_split)
    gate_table = gate_validation_rows(t06, t08)
    baseline_table = gate_baseline_rows(t06, t08)
    training_table = training_summary_rows(t01)

    if paper and not estimator_table:
        raise Program09Error("No estimator benchmark rows available for official test.")
    if paper and not rolling_table:
        raise Program09Error("No rolling-refresh rows available for official test.")
    if paper and not any(r["evaluation_role"] == "official_test" for r in gate_table):
        raise Program09Error("No frozen-gate official-test rows available.")

    generated: list[Path] = []
    generated += write_table_pair(
        tables_dir / "Table1_estimator_benchmark", estimator_table,
        ("split", "estimator", "n_pair_seed_cases", "mean_absolute_error", "median_absolute_error", "rmse", "mean_signed_error", "mean_ope_ci_width", "nonfinite_case_fraction"),
        "Estimator benchmark on the fixed-log reuse frontier.", "tab:estimator_benchmark",
    )
    generated += write_table_pair(
        tables_dir / "Table2_fixed_vs_rolling", rolling_table,
        ("split", "recent_log_lag_steps", "n_pair_seed_cases", "mean_old_absolute_error", "mean_recent_absolute_error", "mean_error_reduction_old_minus_recent", "fraction_recent_lower_error", "mean_old_median_relative_ess", "mean_recent_median_relative_ess", "mean_relative_ess_gain"),
        "Matched comparison of old fixed logs and recent rolling logs.", "tab:fixed_vs_rolling",
    )
    generated += write_table_pair(
        tables_dir / "Table3_frozen_gate_validation", gate_table,
        ("dataset", "evaluation_role", "tolerance", "n_pair_seed_cases", "accept_rate", "false_accept_rate_among_accepted", "false_accept_fraction_all", "accepted_mae", "rejected_mae", "overall_mae"),
        "Validation of the frozen rESS reuse gate.", "tab:frozen_gate_validation",
    )
    generated += write_table_pair(
        tables_dir / "Appendix_training_summary", training_table,
        ("training_seed", "final_step", "final_mean_reward", "final_correctness_reward", "final_kl_proxy", "final_mean_completion_length", "final_truncation_rate", "peak_gpu_memory_bytes", "mean_tokens_per_second"),
        "GRPO training-path summary by seed.", "tab:training_summary",
    )
    generated += write_table_pair(
        tables_dir / "Appendix_gate_baseline", baseline_table,
        ("dataset", "evaluation_role", "gate_kind", "tolerance", "n_pair_seed_cases", "accept_rate", "false_accept_rate_among_accepted", "false_accept_fraction_all", "accepted_mae", "rejected_mae", "overall_mae"),
        "Single KL-threshold baseline for the frozen reuse gate.", "tab:gate_baseline",
    )
    if t07:
        ss = sample_size_rows(t07)
        generated += write_table_pair(
            tables_dir / "Appendix_sample_size", ss,
            ("K", "estimator", "n_pair_seed_cases", "mean_absolute_error", "mean_median_prompt_relative_ess", "mean_bootstrap_ci_width", "mean_bootstrap_nonfinite_fraction"),
            "Development-only behavior-sample-size sensitivity.", "tab:sample_size",
        )
    if t08:
        # Preserve pair-level external-validation values without re-aggregation loss.
        svamp_rows = [r for r in t08 if str(r["estimator"]) == PRIMARY_ESTIMATOR and str(r["gate_kind"]) == PRIMARY_GATE_KIND]
        cols = ("training_seed", "behavior_step", "target_step", "overlap_band", "gate_tolerance", "estimate", "online_reference", "absolute_error", "median_prompt_relative_ess", "tokenwise_kl_proxy", "gate_decision", "gate_false_accept")
        generated += write_table_pair(tables_dir / "Appendix_svamp_pair_results", svamp_rows, cols, "SVAMP external-validation pair results under the frozen primary gate.", "tab:svamp_pairs")
    if t09:
        cols = tuple(t09[0].keys())
        generated += write_table_pair(tables_dir / "Appendix_model_robustness", t09, cols, "Optional second-model robustness results.", "tab:model_robustness")

    values = build_paper_values(estimator_table=estimator_table, rolling_table=rolling_table, gate_table=gate_table, t05=t05)
    generated += write_paper_values(nums_dir, values)

    generated += figure1_training(t01, dev_online, figs_dir / "Figure1_training_trajectory")
    generated += figure2_fixed_frontier(t02, primary_split, figs_dir / "Figure2_fixed_reuse_frontier")
    generated += figure3_overlap_error(t02, primary_split, figs_dir / "Figure3_overlap_vs_error")
    generated += figure4_rolling(t04, primary_split, figs_dir / "Figure4_rolling_refresh")
    generated += figure5_gate(gate_table, figs_dir / "Figure5_frozen_gate_validation")
    if args.include_figure6:
        generated += figure6_heatmap(t03, primary_split, figs_dir / "Figure6_length_drift_overlap_heatmap")

    # Human-readable index contains no hand-entered result values: it points to generated assets.
    index_lines = [
        "# GRPO–OPE paper outputs",
        "",
        "Auto-generated by `09_make_paper_outputs.py` from machine-readable upstream results.",
        "Do not edit numerical values by hand; regenerate this directory instead.",
        "",
        f"Primary analysis split: `{primary_split}`",
        f"SVAMP included: `{bool(t08)}`",
        f"Second-model robustness included: `{bool(t09)}`",
        "",
        "## Main tables",
        "- Table1_estimator_benchmark",
        "- Table2_fixed_vs_rolling",
        "- Table3_frozen_gate_validation",
        "",
        "## Main figures",
        "- Figure1_training_trajectory",
        "- Figure2_fixed_reuse_frontier",
        "- Figure3_overlap_vs_error",
        "- Figure4_rolling_refresh",
        "- Figure5_frozen_gate_validation",
    ]
    if args.include_figure6:
        index_lines.append("- Figure6_length_drift_overlap_heatmap (optional mechanism figure)")
    index_path = base / "README.md"
    atomic_write_text(index_path, "\n".join(index_lines) + "\n")
    generated.append(index_path)

    manifest_path = base / "paper_output_manifest.json"
    manifest = {
        "schema_version": OUTPUT_SCHEMA,
        "manifest_type": "program09_paper_output_manifest",
        "created_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "project_name": PROJECT_NAME,
        "mode": args.mode,
        "primary_analysis_split": primary_split,
        "protocol_version": pv,
        "protocol_config_sha256": config_sha,
        "source_code_sha256": sha256_file(Path(__file__).resolve()),
        "git_commit": git_commit(root),
        "hard_guarantees": {
            "calls_model": False,
            "imports_model_stack": False,
            "refits_gate": False,
            "mutates_upstream_results": False,
            "paper_numbers_hand_copied": False,
            "all_generated_outputs_derived_from_machine_readable_inputs": True,
        },
        "frozen_protocol": {k: v for k, v in frozen.items() if k != "frozen_gate"},
        "final_test_proof": {k: v for k, v in final_proof.items() if not k.endswith("_manifest")},
        "inputs": input_records(inputs, root),
        "program05_dev_online_references": dev_online_prov,
        "program08_manifest_sha256": sha256_file(p08_manifest_path) if p08_manifest_path.exists() else None,
        "optional_assets": {
            "T07_sample_size_present": bool(t07),
            "T08_svamp_present": bool(t08),
            "T09_model_robustness_present": bool(t09),
            "Figure6_generated": bool(args.include_figure6),
        },
        "paper_output_contract": {
            "main_tables": ["Table1_estimator_benchmark", "Table2_fixed_vs_rolling", "Table3_frozen_gate_validation"],
            "main_figures": ["Figure1_training_trajectory", "Figure2_fixed_reuse_frontier", "Figure3_overlap_vs_error", "Figure4_rolling_refresh", "Figure5_frozen_gate_validation"],
            "primary_estimator": PRIMARY_ESTIMATOR,
            "primary_gate": PRIMARY_GATE_KIND,
            "figure6_role": "optional mechanism figure only; generated solely on explicit request and sufficient data support",
        },
        "paper_values_sha256": sha256_file(nums_dir / "paper_values.json"),
        "outputs": output_records(generated, root),
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": __import__("numpy").__version__,
            "matplotlib": __import__("matplotlib").__version__,
        },
    }
    atomic_write_json(manifest_path, manifest)

    print("=" * 78)
    print("GRPO-OPE Program 09 — paper output production")
    print(f"mode                    : {args.mode}")
    print(f"primary analysis split  : {primary_split}")
    print(f"protocol version        : {pv}")
    print(f"config SHA-256          : {config_sha}")
    print(f"training seeds          : {list(seeds)}")
    print(f"SVAMP included          : {bool(t08)}")
    print(f"model robustness        : {bool(t09)}")
    print(f"Figure 6                : {bool(args.include_figure6)}")
    print(f"paper output directory  : {base}")
    print("=" * 78)
    print("PROGRAM 09 PASSED")
    print(f"main tables             : {tables_dir}")
    print(f"main figures            : {figs_dir}")
    print(f"paper values            : {nums_dir / 'paper_values.json'}")
    print(f"provenance manifest     : {manifest_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Program09Error as e:
        print(f"PROGRAM 09 FAILED: {e}", file=sys.stderr)
        raise SystemExit(2)
