#!/usr/bin/env python3
"""
Program 07 — Calibrate, freeze, validate, and test the OPE reuse gate
=====================================================================

This is the decision layer of the GRPO–OPE reuse study.  It consumes the
machine-readable pair-level results produced by Program 06 and turns the
observed development relationship between overlap and OPE error into a frozen,
minimal reuse rule:

    Accept OPE  <=>  median_prompt_relative_ess >= tau_epsilon

for each predeclared absolute-error tolerance epsilon.

Research firewall
-----------------
* Calibration uses DEVELOPMENT only and only the predeclared calibration
  training seeds (default: seeds 1–2 from training.seeds).
* The remaining training seed(s) are held out from gate fitting and may only be
  used for training-path validation.
* The official GSM8K test is evaluated only in --mode test-only, after
  outputs/frozen_gate.json already exists and its immutable hash verifies.
* validate and test-only never call the calibration routines and never modify
  the frozen gate.
* The gate decision uses ONLY an offline diagnostic.  Online-reference error is
  read only after the decision to score false acceptance / MAE.

Primary gate
------------
For epsilon in reuse_gate.tolerances (default 0.01, 0.02, 0.05):

    reliable(pair; epsilon) := absolute OPE error <= epsilon
    accept(pair; epsilon)   := median prompt relative ESS >= tau_epsilon

The memo requires false acceptance to be prioritized but does not specify an
additional target false-acceptance hyperparameter.  To avoid silently inventing
one, the implementation freezes the following simple calibration policy:

    among thresholds with ZERO observed false acceptances on the calibration
    cases, select the threshold with maximum accepted coverage.

If no non-empty accepted set satisfies this rule, the frozen threshold rejects
all calibration cases.  This policy is deliberately simple, deterministic, and
fully auditable; it is not claimed to provide a formal population guarantee.

Single baseline
---------------
The only baseline is a simple behavior-sampled tokenwise-KL-proxy threshold:

    Accept baseline <=> tokenwise_kl_proxy <= kappa_epsilon

It is calibrated with the same zero-observed-false-accept / maximum-coverage
rule.  The baseline is stored for comparison but never replaces the primary
rESS gate.

Inputs
------
  outputs/tables/T02_fixed_reuse.csv
  outputs/tables/T04_rolling_comparison.csv
  outputs/diagnostics/program06_development_manifest.json  (calibrate/validate)
  outputs/diagnostics/program06_test_manifest.json         (test-only)
  manifests/protocol_lock.json
  manifests/data_manifest.json
  manifests/model_manifest.json
  manifests/split_registry_manifest.json

Outputs
-------
Calibration:
  outputs/tables/T05_gate_calibration.csv
  outputs/frozen_gate.json
  manifests/frozen_gate_manifest.json
  outputs/diagnostics/gate_evaluation_summary.csv
  outputs/diagnostics/program07_calibrate_manifest.json

Held-out training-path validation:
  outputs/tables/T06_gate_test.csv             # development validation rows
  outputs/diagnostics/gate_evaluation_summary.csv
  outputs/diagnostics/program07_validate_manifest.json

Official test:
  outputs/tables/T06_gate_test.csv             # appends/replaces split=test rows
  outputs/diagnostics/gate_evaluation_summary.csv
  outputs/diagnostics/program07_test-only_manifest.json

Typical commands
----------------
  python scripts/07_calibrate_and_test_gate.py \
      --config configs/protocol.yaml --mode calibrate \
      --device cpu --output-root .

  python scripts/07_calibrate_and_test_gate.py \
      --config configs/protocol.yaml --mode validate \
      --device cpu --output-root .

  python scripts/07_calibrate_and_test_gate.py \
      --config configs/protocol.yaml --mode test-only \
      --device cpu --output-root .
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROGRAM = "07_calibrate_and_test_gate.py"
PROGRAM_VERSION = "1.1"
PROJECT_NAME = "grpo_ope_reuse"
GATE_SCHEMA = "grpo_ope_reuse_gate_v1"
PROGRAM07_SCHEMA = "grpo_ope_program07_v1"
PRIMARY_ESTIMATOR = "prompt_wis"
PRIMARY_DIAGNOSTIC = "median_prompt_relative_ess"
BASELINE_DIAGNOSTIC = "tokenwise_kl_proxy"
CALIBRATION_POLICY = "zero_observed_false_accept_max_coverage"
DEFAULT_SEEDS = (20260826, 20260827, 20260828)
DEFAULT_TOLERANCES = (0.01, 0.02, 0.05)

T05_COLUMNS = (
    "dataset", "split", "estimator", "gate_kind", "diagnostic_name", "direction",
    "tolerance", "fitted_threshold", "reject_all", "calibration_policy",
    "calibration_seeds", "n_pairs", "accepted_count", "rejected_count",
    "accept_rate", "false_accept_count", "false_accept_rate_among_accepted",
    "false_accept_fraction_all", "accepted_mae", "rejected_mae",
    "reliable_count", "unreliable_count", "threshold_candidate_count",
)

T06_COLUMNS = (
    "dataset", "split", "evaluation_role", "training_seed", "behavior_step",
    "target_step", "purpose", "estimator", "gate_kind", "diagnostic_name",
    "direction", "tolerance", "diagnostic_value", "fitted_threshold",
    "reject_all", "decision", "accepted", "true_absolute_error", "reliable",
    "false_accept", "false_reject", "gate_case_id",
)

SUMMARY_COLUMNS = (
    "dataset", "split", "evaluation_role", "gate_kind", "diagnostic_name",
    "tolerance", "seed_scope", "n_pairs", "accepted_count", "rejected_count",
    "accept_rate", "reliable_count", "unreliable_count", "false_accept_count",
    "false_accept_rate_among_accepted", "false_accept_fraction_all",
    "false_reject_count", "false_reject_rate_among_reliable", "accepted_mae",
    "rejected_mae", "overall_mae",
)


class Program07Error(RuntimeError):
    """Controlled, fail-fast Program 07 error."""


@dataclass(frozen=True)
class GateSpec:
    seeds: tuple[int, ...]
    calibration_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    tolerances: tuple[float, ...]
    estimator: str
    primary_diagnostic: str
    fit_split: str


@dataclass(frozen=True)
class GateCase:
    dataset: str
    split: str
    training_seed: int
    behavior_step: int
    target_step: int
    purpose: str
    estimator: str
    absolute_error: float
    median_prompt_relative_ess: float
    tokenwise_kl_proxy: float

    @property
    def case_id(self) -> str:
        return sha256_bytes(canonical_bytes({
            "dataset": self.dataset,
            "split": self.split,
            "training_seed": self.training_seed,
            "behavior_step": self.behavior_step,
            "target_step": self.target_step,
            "purpose": self.purpose,
            "estimator": self.estimator,
        }))


@dataclass(frozen=True)
class ThresholdFit:
    gate_kind: str
    diagnostic_name: str
    direction: str
    tolerance: float
    threshold: float
    reject_all: bool
    n_pairs: int
    accepted_count: int
    false_accept_count: int
    threshold_candidate_count: int


# ---------------------------------------------------------------------------
# Deterministic utilities
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=str,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def semantic_csv_scope_sha256(path: Path, column: str, value: str) -> str:
    rows = [dict(r) for r in read_csv_rows(path) if str(r.get(column)) == value]
    if not rows:
        raise Program07Error(f"No rows for semantic scope {column}={value!r} in {path}")
    rows.sort(key=lambda r: canonical_bytes(r))
    return sha256_bytes(canonical_bytes(rows))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Program07Error(f"Required JSON file does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        raise Program07Error(f"Could not parse JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise Program07Error(f"Expected JSON object in {path}")
    return obj


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
        try:
            dfd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dfd)
            finally:
                os.close(dfd)
        except Exception:
            pass
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, obj: Mapping[str, Any]) -> None:
    data = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"
    atomic_write_bytes(path, data)


def make_read_only(path: Path) -> None:
    try:
        path.chmod(0o444)
    except Exception:
        pass


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path)


def first_present(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for p in paths:
        cur: Any = mapping
        ok = True
        for key in p:
            if not isinstance(cur, Mapping) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return None


def finite_float(value: Any, *, field: str) -> float:
    try:
        x = float(value)
    except Exception as exc:
        raise Program07Error(f"Field {field} is not numeric: {value!r}") from exc
    if not math.isfinite(x):
        raise Program07Error(f"Field {field} must be finite, observed {x}")
    return x


def int_value(value: Any, *, field: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise Program07Error(f"Field {field} is not integer-like: {value!r}") from exc


def mean_or_none(xs: Iterable[float]) -> float | None:
    vals = [float(x) for x in xs if math.isfinite(float(x))]
    return math.fsum(vals) / len(vals) if vals else None


def csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise Program07Error(f"Required CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise Program07Error(f"CSV has no header: {path}")
        return [dict(r) for r in reader]


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({c: csv_scalar(row.get(c)) for c in columns})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def merge_csv_by_predicate(
    path: Path,
    new_rows: Sequence[Mapping[str, Any]],
    columns: Sequence[str],
    replace_predicate,
) -> None:
    old: list[dict[str, Any]] = []
    if path.exists():
        old = read_csv_rows(path)
        old = [r for r in old if not replace_predicate(r)]
    combined: list[Mapping[str, Any]] = [*old, *new_rows]
    atomic_write_csv(path, combined, columns)


# ---------------------------------------------------------------------------
# Protocol and immutable input verification
# ---------------------------------------------------------------------------


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise Program07Error(f"Protocol config does not exist: {path}")
    raw = path.read_bytes()
    sha = sha256_bytes(raw)
    try:
        import yaml  # type: ignore
        obj = yaml.safe_load(raw.decode("utf-8")) or {}
    except Exception as exc:
        raise Program07Error(f"Could not parse protocol YAML {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise Program07Error("protocol.yaml must contain a mapping at the top level.")
    return obj, sha


def protocol_version(cfg: Mapping[str, Any]) -> str:
    p = cfg.get("project") or {}
    if isinstance(p, Mapping):
        return str(p.get("protocol_version", "1.0"))
    return "1.0"


def resolve_gate_spec(cfg: Mapping[str, Any]) -> GateSpec:
    training = cfg.get("training") or {}
    rg = cfg.get("reuse_gate") or {}
    if not isinstance(training, Mapping) or not isinstance(rg, Mapping):
        raise Program07Error("training and reuse_gate must be mappings in protocol.yaml")

    seeds_raw = training.get("seeds", list(DEFAULT_SEEDS))
    seeds = tuple(int(x) for x in seeds_raw)
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise Program07Error("Program 07 requires at least three unique training seeds.")

    cal_raw = rg.get("calibration_seeds")
    val_raw = rg.get("validation_seeds")
    if cal_raw is None:
        calibration = tuple(seeds[:2])
    else:
        calibration = tuple(int(x) for x in cal_raw)
    if val_raw is None:
        validation = tuple(s for s in seeds if s not in calibration)
    else:
        validation = tuple(int(x) for x in val_raw)

    if not calibration or not validation:
        raise Program07Error("Need nonempty calibration seeds and at least one held-out validation seed.")
    if set(calibration) & set(validation):
        raise Program07Error("Calibration and validation seed sets must be disjoint.")
    if not set(calibration).issubset(seeds) or not set(validation).issubset(seeds):
        raise Program07Error("Calibration/validation seeds must be members of training.seeds.")

    tol_raw = rg.get("tolerances", list(DEFAULT_TOLERANCES))
    tolerances = tuple(sorted(float(x) for x in tol_raw))
    if not tolerances or any((not math.isfinite(x) or x <= 0.0 or x >= 1.0) for x in tolerances):
        raise Program07Error("reuse_gate.tolerances must be finite values strictly between 0 and 1.")
    if len(set(tolerances)) != len(tolerances):
        raise Program07Error("reuse_gate.tolerances must be unique.")

    primary = str(rg.get("primary_diagnostic", PRIMARY_DIAGNOSTIC))
    if primary != PRIMARY_DIAGNOSTIC:
        raise Program07Error(
            f"Main paper Program 07 is frozen to primary_diagnostic={PRIMARY_DIAGNOSTIC!r}; observed {primary!r}."
        )
    fit_split = str(rg.get("fit_split", "development_only"))
    if fit_split not in ("development_only", "development"):
        raise Program07Error("reuse_gate.fit_split must be development_only for the main study.")

    estimator = str(rg.get("estimator", PRIMARY_ESTIMATOR))
    if estimator != PRIMARY_ESTIMATOR:
        raise Program07Error(
            f"Primary frozen gate is defined for estimator={PRIMARY_ESTIMATOR!r}; observed {estimator!r}."
        )
    policy = str(rg.get("calibration_policy", CALIBRATION_POLICY))
    if policy != CALIBRATION_POLICY:
        raise Program07Error(
            f"Unsupported calibration_policy={policy!r}. Main Program 07 uses {CALIBRATION_POLICY!r}."
        )
    return GateSpec(seeds, calibration, validation, tolerances, estimator, primary, fit_split)


def verify_protocol_lock(root: Path, config_sha256: str) -> dict[str, Any]:
    path = root / "manifests" / "protocol_lock.json"
    if not path.exists():
        raise Program07Error(f"Program 07 requires frozen {path}")
    lock = read_json(path)
    locked = first_present(lock, (
        ("config_sha256",), ("protocol_config_sha256",),
        ("protocol", "config_sha256"), ("inputs", "config_sha256"),
    ))
    if not isinstance(locked, str) or locked != config_sha256:
        raise Program07Error(
            f"protocol.yaml hash differs from protocol lock: locked={locked}, observed={config_sha256}"
        )
    for key in ("training_config_sha256", "data_manifest_sha256", "model_revision", "split_registry_sha256"):
        if not isinstance(lock.get(key), str) or not lock.get(key):
            raise Program07Error(f"protocol_lock.json must freeze {key}.")
    source_hashes = lock.get("source_code_sha256") or {}
    expected_source = source_hashes.get(PROGRAM) if isinstance(source_hashes, Mapping) else None
    if not isinstance(expected_source, str) or expected_source != sha256_file(Path(__file__).resolve()):
        raise Program07Error("Program 07 source code differs from the frozen protocol lock.")
    return lock


def verify_upstream_static_manifests(root: Path) -> dict[str, Any]:
    dp = root / "manifests" / "data_manifest.json"
    mp = root / "manifests" / "model_manifest.json"
    sp = root / "manifests" / "split_registry_manifest.json"
    data = read_json(dp)
    model = read_json(mp)
    split = read_json(sp)
    if data.get("manifest_type") != "data":
        raise Program07Error("Invalid Program 00 data manifest header.")
    if model.get("manifest_type") != "model":
        raise Program07Error("Invalid Program 00 model manifest header.")
    if split.get("manifest_type") != "split_registry":
        raise Program07Error("Invalid Program 01 split manifest header.")
    gsm = ((data.get("datasets") or {}).get("gsm8k") or {})
    primary = ((model.get("models") or {}).get("primary") or {})
    data_rev = gsm.get("resolved_revision")
    model_rev = primary.get("resolved_revision")
    split_source = split.get("source") or {}
    split_rev = first_present(split_source, (("dataset_revision",), ("resolved_revision",)))
    if data_rev is not None and split_rev is not None and data_rev != split_rev:
        raise Program07Error("Program 01 dataset revision differs from Program 00.")
    return {
        "data_manifest_path": rel(dp, root),
        "data_manifest_sha256": sha256_file(dp),
        "model_manifest_path": rel(mp, root),
        "model_manifest_sha256": sha256_file(mp),
        "split_manifest_path": rel(sp, root),
        "split_manifest_sha256": sha256_file(sp),
        "dataset_revision": data_rev,
        "model_revision": model_rev,
        "split_content_fingerprint_sha256": split.get("content_fingerprint_sha256"),
    }


def verify_program06_manifest(root: Path, split: str, config_sha: str, pv: str) -> dict[str, Any]:
    path = root / "outputs" / "diagnostics" / f"program06_{split}_manifest.json"
    m = read_json(path)
    if m.get("manifest_type") != "program06_analysis_manifest":
        raise Program07Error(f"Invalid Program 06 manifest header: {path}")
    if str(m.get("split")) != split:
        raise Program07Error(f"Program 06 manifest split mismatch: {path}")
    if str(m.get("protocol_version")) != pv:
        raise Program07Error("Program 06 protocol version differs from current frozen protocol.")
    if str(m.get("protocol_config_sha256")) != config_sha:
        raise Program07Error("Program 06 config hash differs from current frozen protocol.")
    if str(m.get("mode")) != "paper":
        raise Program07Error("Program 07 paper gate requires Program 06 paper-mode outputs.")

    # Program 06 tables are appendable across development/test. Verify the
    # immutable split-specific semantic digest, not a stale whole-file hash.
    outputs = m.get("outputs") or []
    if not isinstance(outputs, list):
        raise Program07Error("Program 06 manifest outputs must be a list.")
    required = {"T02_fixed_reuse.csv", "T04_rolling_comparison.csv"}
    seen: set[str] = set()
    for rec in outputs:
        if not isinstance(rec, Mapping):
            continue
        rp = Path(str(rec.get("path", "")))
        if rp.name not in required:
            continue
        full = root / rp if not rp.is_absolute() else rp
        expected = rec.get("semantic_sha256")
        scope = rec.get("semantic_scope") or {}
        if not full.exists() or not isinstance(expected, str):
            raise Program07Error(f"Program 06 semantic provenance is missing for {full}.")
        if not isinstance(scope, Mapping) or scope.get("column") != "split" or str(scope.get("value")) != split:
            raise Program07Error(f"Program 06 semantic scope mismatch for {full}.")
        if semantic_csv_scope_sha256(full, "split", split) != expected:
            raise Program07Error(f"Program 06 {split} rows changed after their manifest was frozen: {full}")
        seen.add(rp.name)
    if seen != required:
        raise Program07Error(f"Program 06 manifest does not hash both required gate tables: missing {required - seen}")
    return {"path": rel(path, root), "sha256": sha256_file(path), "manifest": m}


# ---------------------------------------------------------------------------
# Program 06 case extraction
# ---------------------------------------------------------------------------


def require_columns(row: Mapping[str, Any], columns: Sequence[str], source: str) -> None:
    missing = [c for c in columns if c not in row]
    if missing:
        raise Program07Error(f"{source} is missing required columns: {missing}")


def load_gate_cases(root: Path, split: str, estimator: str) -> tuple[list[GateCase], dict[str, str]]:
    t02 = root / "outputs" / "tables" / "T02_fixed_reuse.csv"
    t04 = root / "outputs" / "tables" / "T04_rolling_comparison.csv"
    fixed = read_csv_rows(t02)
    rolling = read_csv_rows(t04)

    cases: list[GateCase] = []
    for r in fixed:
        if str(r.get("split")) != split or str(r.get("estimator")) != estimator:
            continue
        require_columns(r, (
            "dataset", "training_seed", "behavior_step", "target_step", "purpose",
            "absolute_error", "median_prompt_relative_ess", "tokenwise_kl_proxy",
        ), "T02_fixed_reuse.csv")
        err = finite_float(r["absolute_error"], field="T02.absolute_error")
        ress = finite_float(r["median_prompt_relative_ess"], field="T02.median_prompt_relative_ess")
        kl = finite_float(r["tokenwise_kl_proxy"], field="T02.tokenwise_kl_proxy")
        if ress < -1e-12 or ress > 1.0 + 1e-12:
            raise Program07Error(f"relative ESS must lie in [0,1], observed {ress}")
        cases.append(GateCase(
            dataset=str(r["dataset"]), split=split,
            training_seed=int_value(r["training_seed"], field="T02.training_seed"),
            behavior_step=int_value(r["behavior_step"], field="T02.behavior_step"),
            target_step=int_value(r["target_step"], field="T02.target_step"),
            purpose="fixed", estimator=estimator, absolute_error=err,
            median_prompt_relative_ess=min(1.0, max(0.0, ress)), tokenwise_kl_proxy=kl,
        ))

    # T04 stores matched old/recent rows.  The old b=0 row is already represented
    # in T02, so only the RECENT rolling case is added here.
    for r in rolling:
        if str(r.get("split")) != split or str(r.get("estimator")) != estimator:
            continue
        require_columns(r, (
            "dataset", "training_seed", "target_step", "recent_behavior_step",
            "recent_absolute_error", "recent_median_relative_ess", "recent_tokenwise_kl_proxy",
        ), "T04_rolling_comparison.csv")
        err = finite_float(r["recent_absolute_error"], field="T04.recent_absolute_error")
        ress = finite_float(r["recent_median_relative_ess"], field="T04.recent_median_relative_ess")
        kl = finite_float(r["recent_tokenwise_kl_proxy"], field="T04.recent_tokenwise_kl_proxy")
        if ress < -1e-12 or ress > 1.0 + 1e-12:
            raise Program07Error(f"relative ESS must lie in [0,1], observed {ress}")
        cases.append(GateCase(
            dataset=str(r["dataset"]), split=split,
            training_seed=int_value(r["training_seed"], field="T04.training_seed"),
            behavior_step=int_value(r["recent_behavior_step"], field="T04.recent_behavior_step"),
            target_step=int_value(r["target_step"], field="T04.target_step"),
            purpose="rolling", estimator=estimator, absolute_error=err,
            median_prompt_relative_ess=min(1.0, max(0.0, ress)), tokenwise_kl_proxy=kl,
        ))

    if not cases:
        raise Program07Error(f"No {estimator} gate cases found for split={split} in Program 06 tables.")
    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        raise Program07Error("Duplicate gate cases detected after combining T02 fixed and T04 recent rolling rows.")
    cases.sort(key=lambda c: (c.training_seed, c.behavior_step, c.target_step, c.purpose))
    return cases, {
        "T02": semantic_csv_scope_sha256(t02, "split", split),
        "T04": semantic_csv_scope_sha256(t04, "split", split),
    }


# ---------------------------------------------------------------------------
# Deterministic threshold calibration
# ---------------------------------------------------------------------------


def diagnostic_value(case: GateCase, name: str) -> float:
    if name == PRIMARY_DIAGNOSTIC:
        return case.median_prompt_relative_ess
    if name == BASELINE_DIAGNOSTIC:
        return case.tokenwise_kl_proxy
    raise Program07Error(f"Unsupported gate diagnostic: {name}")


def apply_threshold(value: float, threshold: float, direction: str) -> bool:
    if direction == "ge":
        return value >= threshold
    if direction == "le":
        return value <= threshold
    raise Program07Error(f"Unsupported threshold direction: {direction}")


def fit_zero_false_accept_threshold(
    cases: Sequence[GateCase], *, tolerance: float, diagnostic_name: str,
    direction: str, gate_kind: str,
) -> ThresholdFit:
    if not cases:
        raise Program07Error("Cannot calibrate gate with zero cases.")
    vals = [diagnostic_value(c, diagnostic_name) for c in cases]
    if any(not math.isfinite(x) for x in vals):
        raise Program07Error(f"Nonfinite values in calibration diagnostic {diagnostic_name}")

    unique = sorted(set(vals))
    if direction == "ge":
        # A threshold immediately above the maximum is a finite reject-all sentinel.
        reject_all_threshold = math.nextafter(max(unique), math.inf)
        candidates = [*unique, reject_all_threshold]
    elif direction == "le":
        reject_all_threshold = math.nextafter(min(unique), -math.inf)
        candidates = [reject_all_threshold, *unique]
    else:
        raise Program07Error(f"Unknown threshold direction {direction}")

    feasible: list[tuple[int, float, int]] = []  # accepted_count, threshold, false_accept_count
    for threshold in candidates:
        accepted = [c for c in cases if apply_threshold(diagnostic_value(c, diagnostic_name), threshold, direction)]
        false_accept = sum(1 for c in accepted if c.absolute_error > tolerance)
        if false_accept == 0:
            feasible.append((len(accepted), threshold, false_accept))
    if not feasible:
        raise Program07Error("Internal calibration error: reject-all threshold should always be feasible.")

    max_accepted = max(x[0] for x in feasible)
    best = [x for x in feasible if x[0] == max_accepted]
    # Deterministic tie break: choose the least restrictive threshold among
    # thresholds yielding the same accepted set size.
    if direction == "ge":
        accepted_count, threshold, false_accept_count = min(best, key=lambda x: x[1])
    else:
        accepted_count, threshold, false_accept_count = max(best, key=lambda x: x[1])
    reject_all = accepted_count == 0
    return ThresholdFit(
        gate_kind=gate_kind, diagnostic_name=diagnostic_name, direction=direction,
        tolerance=tolerance, threshold=float(threshold), reject_all=reject_all,
        n_pairs=len(cases), accepted_count=accepted_count,
        false_accept_count=false_accept_count, threshold_candidate_count=len(candidates),
    )


def evaluate_fit(cases: Sequence[GateCase], fit: ThresholdFit, role: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for c in cases:
        x = diagnostic_value(c, fit.diagnostic_name)
        accepted = apply_threshold(x, fit.threshold, fit.direction)
        reliable = c.absolute_error <= fit.tolerance
        rows.append({
            "dataset": c.dataset,
            "split": c.split,
            "evaluation_role": role,
            "training_seed": c.training_seed,
            "behavior_step": c.behavior_step,
            "target_step": c.target_step,
            "purpose": c.purpose,
            "estimator": c.estimator,
            "gate_kind": fit.gate_kind,
            "diagnostic_name": fit.diagnostic_name,
            "direction": fit.direction,
            "tolerance": fit.tolerance,
            "diagnostic_value": x,
            "fitted_threshold": fit.threshold,
            "reject_all": fit.reject_all,
            "decision": "accept" if accepted else "refresh",
            "accepted": accepted,
            "true_absolute_error": c.absolute_error,
            "reliable": reliable,
            "false_accept": bool(accepted and not reliable),
            "false_reject": bool((not accepted) and reliable),
            "gate_case_id": c.case_id,
        })
    return rows


def summarize_decisions(rows: Sequence[Mapping[str, Any]], seed_scope: str) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        raise Program07Error("Cannot summarize zero gate decisions.")
    accepted = [r for r in rows if bool(r["accepted"])]
    rejected = [r for r in rows if not bool(r["accepted"])]
    reliable = [r for r in rows if bool(r["reliable"])]
    false_accept = [r for r in rows if bool(r["false_accept"])]
    false_reject = [r for r in rows if bool(r["false_reject"])]
    first = rows[0]
    return {
        "dataset": first["dataset"], "split": first["split"],
        "evaluation_role": first["evaluation_role"], "gate_kind": first["gate_kind"],
        "diagnostic_name": first["diagnostic_name"], "tolerance": first["tolerance"],
        "seed_scope": seed_scope, "n_pairs": n, "accepted_count": len(accepted),
        "rejected_count": len(rejected), "accept_rate": len(accepted) / n,
        "reliable_count": len(reliable), "unreliable_count": n - len(reliable),
        "false_accept_count": len(false_accept),
        "false_accept_rate_among_accepted": (len(false_accept) / len(accepted)) if accepted else 0.0,
        "false_accept_fraction_all": len(false_accept) / n,
        "false_reject_count": len(false_reject),
        "false_reject_rate_among_reliable": (len(false_reject) / len(reliable)) if reliable else 0.0,
        "accepted_mae": mean_or_none(float(r["true_absolute_error"]) for r in accepted),
        "rejected_mae": mean_or_none(float(r["true_absolute_error"]) for r in rejected),
        "overall_mae": mean_or_none(float(r["true_absolute_error"]) for r in rows),
    }


def summaries_for_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # group first by role/gate/tolerance/dataset/split
    groups: dict[tuple[str, str, float, str, str], list[Mapping[str, Any]]] = {}
    for r in rows:
        key = (
            str(r["evaluation_role"]), str(r["gate_kind"]), float(r["tolerance"]),
            str(r["dataset"]), str(r["split"]),
        )
        groups.setdefault(key, []).append(r)
    for _, group in sorted(groups.items(), key=lambda kv: kv[0]):
        out.append(summarize_decisions(group, "all"))
        by_seed: dict[int, list[Mapping[str, Any]]] = {}
        for r in group:
            by_seed.setdefault(int(r["training_seed"]), []).append(r)
        for seed, rows_seed in sorted(by_seed.items()):
            out.append(summarize_decisions(rows_seed, str(seed)))
    return out


def fit_to_calibration_row(fit: ThresholdFit, decisions: Sequence[Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    s = summarize_decisions(decisions, "all")
    return {
        "dataset": s["dataset"], "split": s["split"], "estimator": PRIMARY_ESTIMATOR,
        "gate_kind": fit.gate_kind, "diagnostic_name": fit.diagnostic_name,
        "direction": fit.direction, "tolerance": fit.tolerance,
        "fitted_threshold": fit.threshold, "reject_all": fit.reject_all,
        "calibration_policy": CALIBRATION_POLICY,
        "calibration_seeds": ",".join(str(x) for x in seeds),
        "n_pairs": s["n_pairs"], "accepted_count": s["accepted_count"],
        "rejected_count": s["rejected_count"], "accept_rate": s["accept_rate"],
        "false_accept_count": s["false_accept_count"],
        "false_accept_rate_among_accepted": s["false_accept_rate_among_accepted"],
        "false_accept_fraction_all": s["false_accept_fraction_all"],
        "accepted_mae": s["accepted_mae"], "rejected_mae": s["rejected_mae"],
        "reliable_count": s["reliable_count"], "unreliable_count": s["unreliable_count"],
        "threshold_candidate_count": fit.threshold_candidate_count,
    }


def assert_threshold_monotonicity(fits: Sequence[ThresholdFit]) -> None:
    if not fits:
        return
    ordered = sorted(fits, key=lambda f: f.tolerance)
    # For rESS >= tau: looser error tolerance should not require a higher tau.
    if ordered[0].direction == "ge":
        for a, b in zip(ordered, ordered[1:]):
            if a.threshold + 1e-15 < b.threshold:
                raise Program07Error(
                    "Primary gate thresholds violate nested tolerance monotonicity; this indicates a calibration bug."
                )
    # For KL <= kappa: looser error tolerance should not require a lower kappa.
    elif ordered[0].direction == "le":
        for a, b in zip(ordered, ordered[1:]):
            if a.threshold > b.threshold + 1e-15:
                raise Program07Error(
                    "KL baseline thresholds violate nested tolerance monotonicity; this indicates a calibration bug."
                )


# ---------------------------------------------------------------------------
# Frozen gate serialization / verification
# ---------------------------------------------------------------------------


def gate_paths(root: Path) -> tuple[Path, Path]:
    return root / "outputs" / "frozen_gate.json", root / "manifests" / "frozen_gate_manifest.json"


def threshold_record(f: ThresholdFit) -> dict[str, Any]:
    return {
        "tolerance": f.tolerance,
        "threshold": f.threshold,
        "reject_all": f.reject_all,
        "direction": f.direction,
        "diagnostic_name": f.diagnostic_name,
        "gate_kind": f.gate_kind,
    }


def build_frozen_gate(
    *, root: Path, cfg_sha: str, pv: str, spec: GateSpec,
    upstream: Mapping[str, Any], program06: Mapping[str, Any], table_hashes: Mapping[str, str],
    primary_fits: Sequence[ThresholdFit], baseline_fits: Sequence[ThresholdFit],
    calibration_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    primary_summaries = summaries_for_rows([r for r in calibration_rows if r["gate_kind"] == "primary_ress"])
    baseline_summaries = summaries_for_rows([r for r in calibration_rows if r["gate_kind"] == "kl_baseline"])
    gate = {
        "schema_version": GATE_SCHEMA,
        "manifest_type": "frozen_reuse_gate",
        "created_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "project_name": PROJECT_NAME,
        "protocol_version": pv,
        "protocol_config_sha256": cfg_sha,
        "estimator": spec.estimator,
        "fit_split": "development",
        "calibration_seeds": list(spec.calibration_seeds),
        "validation_seeds": list(spec.validation_seeds),
        "all_training_seeds": list(spec.seeds),
        "calibration_policy": {
            "name": CALIBRATION_POLICY,
            "description": (
                "Choose the maximum-coverage threshold among thresholds with zero observed false "
                "acceptances on predeclared development calibration seeds; reject-all remains feasible."
            ),
            "false_accept_definition": "gate accepts AND absolute OPE error > tolerance",
            "important_note": "Empirical calibration rule; not a formal population-level error guarantee.",
        },
        "primary_gate": {
            "gate_kind": "primary_ress",
            "diagnostic_name": PRIMARY_DIAGNOSTIC,
            "decision_rule": "accept iff median_prompt_relative_ess >= threshold",
            "thresholds": [threshold_record(f) for f in sorted(primary_fits, key=lambda x: x.tolerance)],
            "calibration_summary": primary_summaries,
        },
        "baseline": {
            "gate_kind": "kl_baseline",
            "diagnostic_name": BASELINE_DIAGNOSTIC,
            "decision_rule": "accept iff tokenwise_kl_proxy <= threshold",
            "role": "single simple baseline only; never replaces the primary rESS gate",
            "thresholds": [threshold_record(f) for f in sorted(baseline_fits, key=lambda x: x.tolerance)],
            "calibration_summary": baseline_summaries,
        },
        "inputs": {
            "upstream_static_manifests": dict(upstream),
            "program06_development_manifest_path": program06["path"],
            "program06_development_manifest_sha256": program06["sha256"],
            "T02_fixed_reuse_sha256": table_hashes["T02"],
            "T04_rolling_comparison_sha256": table_hashes["T04"],
        },
        "test_firewall": {
            "thresholds_frozen_before_official_test": True,
            "test_mode_refit_allowed": False,
            "test_mode_threshold_override_allowed": False,
            "test_decision_uses_outcome": False,
        },
    }
    # A content fingerprint excludes the timestamp, allowing deterministic
    # semantic comparison without pretending the literal file bytes are stable.
    semantic = dict(gate)
    semantic.pop("created_at_utc", None)
    gate["content_fingerprint_sha256"] = sha256_bytes(canonical_bytes(semantic))
    return gate


def write_frozen_gate_once(root: Path, gate: Mapping[str, Any], calibration_path: Path) -> tuple[str, str]:
    gate_path, manifest_path = gate_paths(root)
    if gate_path.exists() or manifest_path.exists():
        raise Program07Error(
            "Frozen gate already exists. Calibration is write-once. Use --mode validate or --mode test-only; "
            "do not refit thresholds after freeze."
        )
    atomic_write_json(gate_path, gate)
    gate_sha = sha256_file(gate_path)
    manifest = {
        "schema_version": "1.0",
        "manifest_type": "frozen_gate_manifest",
        "created_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "gate_path": rel(gate_path, root),
        "gate_file_sha256": gate_sha,
        "file_sha256": gate_sha,  # compatibility with Programs 03–06 firewall readers
        "frozen_gate_sha256": gate_sha,
        "gate_content_fingerprint_sha256": gate.get("content_fingerprint_sha256"),
        "calibration_table_path": rel(calibration_path, root),
        "calibration_table_sha256": sha256_file(calibration_path),
        "immutable": True,
    }
    atomic_write_json(manifest_path, manifest)
    make_read_only(gate_path)
    make_read_only(manifest_path)
    return gate_sha, sha256_file(manifest_path)


def load_and_verify_frozen_gate(root: Path, cfg_sha: str, pv: str, spec: GateSpec) -> tuple[dict[str, Any], str, str]:
    gate_path, manifest_path = gate_paths(root)
    gate = read_json(gate_path)
    manifest = read_json(manifest_path)
    if gate.get("manifest_type") != "frozen_reuse_gate" or gate.get("schema_version") != GATE_SCHEMA:
        raise Program07Error("Invalid frozen gate header/schema.")
    observed = sha256_file(gate_path)
    expected = first_present(manifest, (("gate_file_sha256",), ("file_sha256",), ("frozen_gate_sha256",)))
    if not isinstance(expected, str) or observed != expected:
        raise Program07Error(f"Frozen gate file hash mismatch: expected={expected}, observed={observed}")
    if str(gate.get("protocol_config_sha256")) != cfg_sha or str(gate.get("protocol_version")) != pv:
        raise Program07Error("Frozen gate belongs to a different protocol/config.")
    if str(gate.get("estimator")) != spec.estimator:
        raise Program07Error("Frozen gate estimator differs from current protocol.")
    if tuple(int(x) for x in gate.get("calibration_seeds", [])) != spec.calibration_seeds:
        raise Program07Error("Frozen gate calibration seed set differs from current protocol.")
    if tuple(int(x) for x in gate.get("validation_seeds", [])) != spec.validation_seeds:
        raise Program07Error("Frozen gate validation seed set differs from current protocol.")
    frozen_tols = tuple(float(x["tolerance"]) for x in gate["primary_gate"]["thresholds"])
    if frozen_tols != spec.tolerances:
        raise Program07Error("Frozen gate tolerance set differs from current protocol.")
    return gate, observed, sha256_file(manifest_path)


def fits_from_frozen_gate(gate: Mapping[str, Any], section: str) -> list[ThresholdFit]:
    sec = gate.get(section)
    if not isinstance(sec, Mapping):
        raise Program07Error(f"Frozen gate missing section {section}")
    rows = sec.get("thresholds")
    if not isinstance(rows, list) or not rows:
        raise Program07Error(f"Frozen gate section {section} has no thresholds")
    out: list[ThresholdFit] = []
    for r in rows:
        if not isinstance(r, Mapping):
            raise Program07Error("Malformed frozen threshold record")
        out.append(ThresholdFit(
            gate_kind=str(r["gate_kind"]), diagnostic_name=str(r["diagnostic_name"]),
            direction=str(r["direction"]), tolerance=float(r["tolerance"]),
            threshold=float(r["threshold"]), reject_all=bool(r["reject_all"]),
            n_pairs=0, accepted_count=0, false_accept_count=0, threshold_candidate_count=0,
        ))
    assert_threshold_monotonicity(out)
    return out


# ---------------------------------------------------------------------------
# Output merge / provenance
# ---------------------------------------------------------------------------


def output_paths(root: Path) -> tuple[Path, Path, Path]:
    return (
        root / "outputs" / "tables" / "T05_gate_calibration.csv",
        root / "outputs" / "tables" / "T06_gate_test.csv",
        root / "outputs" / "diagnostics" / "gate_evaluation_summary.csv",
    )


def write_summary_rows(path: Path, new_rows: Sequence[Mapping[str, Any]], role: str) -> None:
    merge_csv_by_predicate(
        path, new_rows, SUMMARY_COLUMNS,
        lambda r: str(r.get("evaluation_role")) == role,
    )


def write_t06_rows(path: Path, new_rows: Sequence[Mapping[str, Any]], role: str) -> None:
    merge_csv_by_predicate(
        path, new_rows, T06_COLUMNS,
        lambda r: str(r.get("evaluation_role")) == role,
    )


def write_program07_manifest(
    *, root: Path, mode: str, cfg_sha: str, pv: str, gate_sha: str | None,
    gate_manifest_sha: str | None, program06: Mapping[str, Any], upstream: Mapping[str, Any],
    outputs: Sequence[Path], row_counts: Mapping[str, int], spec: GateSpec,
) -> Path:
    path = root / "outputs" / "diagnostics" / f"program07_{mode}_manifest.json"
    manifest = {
        "schema_version": PROGRAM07_SCHEMA,
        "manifest_type": "program07_gate_manifest",
        "created_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "mode": mode,
        "protocol_version": pv,
        "protocol_config_sha256": cfg_sha,
        "gate_file_sha256": gate_sha,
        "gate_manifest_sha256": gate_manifest_sha,
        "gate_spec": {
            "estimator": spec.estimator,
            "primary_diagnostic": spec.primary_diagnostic,
            "tolerances": list(spec.tolerances),
            "calibration_seeds": list(spec.calibration_seeds),
            "validation_seeds": list(spec.validation_seeds),
            "calibration_policy": CALIBRATION_POLICY,
        },
        "inputs": {
            "upstream_static_manifests": dict(upstream),
            "program06_manifest_path": program06["path"],
            "program06_manifest_sha256": program06["sha256"],
        },
        "outputs": [
            (
                {
                    "path": rel(p, root),
                    "sha256_at_write_time": sha256_file(p),
                    "bytes_at_write_time": p.stat().st_size,
                    "semantic_scope": {
                        "column": "evaluation_role",
                        "value": "heldout_seed_validation" if mode == "validate" else "official_test",
                    },
                    "semantic_sha256": semantic_csv_scope_sha256(
                        p,
                        "evaluation_role",
                        "heldout_seed_validation" if mode == "validate" else "official_test",
                    ),
                }
                if mode in {"validate", "test-only"} and p.name in {"T06_gate_test.csv", "gate_evaluation_summary.csv"}
                else {"path": rel(p, root), "sha256": sha256_file(p), "bytes": p.stat().st_size}
            )
            for p in outputs if p.exists()
        ],
        "row_counts": dict(row_counts),
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform()},
    }
    atomic_write_json(path, manifest)
    return path


# ---------------------------------------------------------------------------
# Mode orchestration
# ---------------------------------------------------------------------------


def calibrate(
    *, root: Path, cfg_sha: str, pv: str, spec: GateSpec,
    upstream: Mapping[str, Any], program06: Mapping[str, Any], cases: Sequence[GateCase],
    table_hashes: Mapping[str, str],
) -> None:
    calibration_cases = [c for c in cases if c.training_seed in set(spec.calibration_seeds)]
    validation_leak = [c for c in calibration_cases if c.training_seed in set(spec.validation_seeds)]
    if validation_leak:
        raise Program07Error("Held-out validation seed leaked into gate calibration.")
    if not calibration_cases:
        raise Program07Error("No development cases for calibration seeds.")
    seen_cal_seeds = set(c.training_seed for c in calibration_cases)
    if seen_cal_seeds != set(spec.calibration_seeds):
        raise Program07Error(
            f"Calibration table does not contain all frozen calibration seeds: expected={spec.calibration_seeds}, observed={sorted(seen_cal_seeds)}"
        )

    primary_fits: list[ThresholdFit] = []
    baseline_fits: list[ThresholdFit] = []
    all_decisions: list[dict[str, Any]] = []
    t05_rows: list[dict[str, Any]] = []
    for eps in spec.tolerances:
        pf = fit_zero_false_accept_threshold(
            calibration_cases, tolerance=eps, diagnostic_name=PRIMARY_DIAGNOSTIC,
            direction="ge", gate_kind="primary_ress",
        )
        bf = fit_zero_false_accept_threshold(
            calibration_cases, tolerance=eps, diagnostic_name=BASELINE_DIAGNOSTIC,
            direction="le", gate_kind="kl_baseline",
        )
        primary_fits.append(pf); baseline_fits.append(bf)
        for fit in (pf, bf):
            decisions = evaluate_fit(calibration_cases, fit, "calibration")
            all_decisions.extend(decisions)
            t05_rows.append(fit_to_calibration_row(fit, decisions, spec.calibration_seeds))

    assert_threshold_monotonicity(primary_fits)
    assert_threshold_monotonicity(baseline_fits)

    t05, _, summary_path = output_paths(root)
    atomic_write_csv(t05, t05_rows, T05_COLUMNS)
    summary_rows = summaries_for_rows(all_decisions)
    write_summary_rows(summary_path, summary_rows, "calibration")

    gate = build_frozen_gate(
        root=root, cfg_sha=cfg_sha, pv=pv, spec=spec, upstream=upstream,
        program06=program06, table_hashes=table_hashes,
        primary_fits=primary_fits, baseline_fits=baseline_fits,
        calibration_rows=all_decisions,
    )
    gate_sha, gate_manifest_sha = write_frozen_gate_once(root, gate, t05)
    gate_path, gate_manifest_path = gate_paths(root)
    manifest_path = write_program07_manifest(
        root=root, mode="calibrate", cfg_sha=cfg_sha, pv=pv, gate_sha=gate_sha,
        gate_manifest_sha=gate_manifest_sha, program06=program06, upstream=upstream,
        outputs=(t05, summary_path, gate_path, gate_manifest_path),
        row_counts={"T05": len(t05_rows), "calibration_decisions": len(all_decisions), "summary": len(summary_rows)},
        spec=spec,
    )

    print("\nPROGRAM 07 CALIBRATION PASSED")
    for f in primary_fits:
        print(
            f"epsilon={f.tolerance:.3f}  primary tau={f.threshold:.12g}  "
            f"accepted={f.accepted_count}/{f.n_pairs}  false_accept=0"
        )
    print(f"frozen gate      : {gate_path}")
    print(f"frozen gate SHA  : {gate_sha}")
    print(f"gate manifest    : {gate_manifest_path}")
    print(f"Program07 manifest: {manifest_path}")


def evaluate_frozen(
    *, root: Path, mode: str, spec: GateSpec, gate: Mapping[str, Any], gate_sha: str,
    gate_manifest_sha: str, cfg_sha: str, pv: str, upstream: Mapping[str, Any],
    program06: Mapping[str, Any], cases: Sequence[GateCase],
) -> None:
    if mode == "validate":
        role = "heldout_seed_validation"
        selected = [c for c in cases if c.training_seed in set(spec.validation_seeds)]
        expected_seeds = set(spec.validation_seeds)
    elif mode == "test-only":
        role = "official_test"
        selected = list(cases)
        expected_seeds = set(spec.seeds)
    else:
        raise Program07Error(f"Unsupported frozen evaluation mode {mode}")

    if not selected:
        raise Program07Error(f"No cases selected for {mode}")
    observed = set(c.training_seed for c in selected)
    if observed != expected_seeds:
        raise Program07Error(f"{mode} seed coverage mismatch: expected={sorted(expected_seeds)}, observed={sorted(observed)}")
    if mode == "validate" and any(c.training_seed in set(spec.calibration_seeds) for c in selected):
        raise Program07Error("Calibration seeds leaked into held-out validation evaluation.")

    primary_fits = fits_from_frozen_gate(gate, "primary_gate")
    baseline_fits = fits_from_frozen_gate(gate, "baseline")
    decisions: list[dict[str, Any]] = []
    for fit in [*primary_fits, *baseline_fits]:
        decisions.extend(evaluate_fit(selected, fit, role))

    _, t06, summary_path = output_paths(root)
    write_t06_rows(t06, decisions, role)
    summary_rows = summaries_for_rows(decisions)
    write_summary_rows(summary_path, summary_rows, role)
    manifest_path = write_program07_manifest(
        root=root, mode=mode, cfg_sha=cfg_sha, pv=pv, gate_sha=gate_sha,
        gate_manifest_sha=gate_manifest_sha, program06=program06, upstream=upstream,
        outputs=(t06, summary_path),
        row_counts={"T06_current_role": len(decisions), "summary_current_role": len(summary_rows)},
        spec=spec,
    )

    print(f"\nPROGRAM 07 {mode.upper()} PASSED")
    # Compact primary-gate summaries only.
    for s in summary_rows:
        if s["gate_kind"] != "primary_ress" or s["seed_scope"] != "all":
            continue
        print(
            f"epsilon={float(s['tolerance']):.3f}  accept_rate={float(s['accept_rate']):.3f}  "
            f"false_accept_rate={float(s['false_accept_rate_among_accepted']):.3f}  "
            f"accepted_MAE={s['accepted_mae']}  rejected_MAE={s['rejected_mae']}"
        )
    print(f"T06 decisions     : {t06}")
    print(f"summary           : {summary_path}")
    print(f"Program07 manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Program 07: calibrate/freeze the median-rESS OPE reuse gate, validate on held-out seed, or apply test-only."
    )
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--mode", choices=("calibrate", "validate", "test-only"), required=True)
    ap.add_argument("--split", choices=("development", "test"), default=None,
                    help="Optional explicit split. calibrate/validate require development; test-only requires test.")
    ap.add_argument("--resume", action="store_true",
                    help="Accepted for unified CLI. Frozen outputs are hash-verified/write-once rather than refit.")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--output-root", default=".")
    ap.add_argument("--verify-only", action="store_true",
                    help="Verify the frozen gate and immutable inputs without writing validation/test outputs.")
    return ap.parse_args(argv)


def validate_args(args: argparse.Namespace) -> str:
    if args.device != "cpu":
        raise Program07Error("Program 07 is CPU-only; gate calibration/evaluation must not consume GPU.")
    required_split = "test" if args.mode == "test-only" else "development"
    if args.split is not None and args.split != required_split:
        raise Program07Error(f"--mode {args.mode} requires --split {required_split}")
    return required_split


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    split = validate_args(args)
    root = Path(args.output_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (root / config_path).resolve()
    cfg, cfg_sha = load_protocol(config_path)
    pv = protocol_version(cfg)
    spec = resolve_gate_spec(cfg)
    verify_protocol_lock(root, cfg_sha)
    upstream = verify_upstream_static_manifests(root)
    program06 = verify_program06_manifest(root, split, cfg_sha, pv)
    cases, table_hashes = load_gate_cases(root, split, spec.estimator)

    print("=" * 82)
    print("PROGRAM 07 — CALIBRATE / FREEZE / TEST REUSE GATE")
    print(f"protocol version   : {pv}")
    print(f"config SHA-256     : {cfg_sha}")
    print(f"mode / split       : {args.mode} / {split}")
    print(f"device             : cpu")
    print(f"estimator          : {spec.estimator}")
    print(f"primary diagnostic : {spec.primary_diagnostic}")
    print(f"tolerances         : {list(spec.tolerances)}")
    print(f"calibration seeds  : {list(spec.calibration_seeds)}")
    print(f"validation seeds   : {list(spec.validation_seeds)}")
    print(f"available cases    : {len(cases)}")
    print(f"calibration policy : {CALIBRATION_POLICY}")
    print("=" * 82)

    if args.mode == "calibrate":
        if args.verify_only:
            gate, gate_sha, gate_manifest_sha = load_and_verify_frozen_gate(root, cfg_sha, pv, spec)
            print("FROZEN GATE VERIFIED")
            print(f"gate SHA           : {gate_sha}")
            print(f"gate manifest SHA  : {gate_manifest_sha}")
            print(f"content fingerprint: {gate.get('content_fingerprint_sha256')}")
            return 0
        gate_path, gate_manifest_path = gate_paths(root)
        if gate_path.exists() or gate_manifest_path.exists():
            raise Program07Error(
                "Frozen gate already exists. Calibration is write-once and refuses BEFORE modifying T05/summary outputs. "
                "Use --mode validate or --mode test-only."
            )
        calibrate(
            root=root, cfg_sha=cfg_sha, pv=pv, spec=spec, upstream=upstream,
            program06=program06, cases=cases, table_hashes=table_hashes,
        )
        return 0

    gate, gate_sha, gate_manifest_sha = load_and_verify_frozen_gate(root, cfg_sha, pv, spec)
    if args.verify_only:
        print("FROZEN GATE + INPUTS VERIFIED")
        print(f"gate SHA          : {gate_sha}")
        return 0

    evaluate_frozen(
        root=root, mode=args.mode, spec=spec, gate=gate, gate_sha=gate_sha,
        gate_manifest_sha=gate_manifest_sha, cfg_sha=cfg_sha, pv=pv,
        upstream=upstream, program06=program06, cases=cases,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Program07Error as exc:
        print(f"\nPROGRAM 07 FAILED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
