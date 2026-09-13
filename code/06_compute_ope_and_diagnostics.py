#!/usr/bin/env python3
"""
Program 06 — OPE, overlap diagnostics, and Experiments 1–3
=================================================================

This is a CPU-only statistical program for the GRPO–OPE reuse study.  It reads
immutable artifacts produced by Programs 03–05 and NEVER loads or calls an LLM.

Primary research chain supported here:
    policy drift -> sequence overlap deterioration -> OPE reliability loss
    -> reuse old logs or refresh behavior logs

Implemented estimators / diagnostics
------------------------------------
For prompt j with K behavior trajectories, rewards R_jk in {0,1}, and
log-weights z_jk = log pi_e(Y_jk|x_j) - log mu_b(Y_jk|x_j):

  ordinary IS:
      v_IS,j = (1/K) sum_k exp(z_jk) R_jk
      V_IS   = (1/M) sum_j v_IS,j

  prompt-normalized WIS (primary WIS):
      v_pWIS,j = sum_k exp(z_jk) R_jk / sum_k exp(z_jk)
      V_pWIS   = (1/M) sum_j v_pWIS,j

  prompt ESS and relative ESS:
      ESS_j  = (sum w)^2 / sum w^2
      rESS_j = ESS_j / K

  second-moment / Renyi-2 proxy:
      D2_j = log[(1/K) sum_k w_jk^2]

  max normalized weight:
      C_j = max_k w_jk / sum_l w_jl

  behavior-sampled tokenwise KL proxy:
      KL_proxy = sum_{j,k} [log mu(Y_jk|x_j)-log pi_e(Y_jk|x_j)]
                 / sum_{j,k} completion_length_jk
               = -sum logW / total completion tokens.

All weight algebra starts in log-space.  Ordinary IS is NOT silently clipped or
converted to WIS when it overflows.  A non-representable ordinary-IS estimate is
recorded as estimator_status="nonfinite_estimator".

Outputs (paper mode)
--------------------
  outputs/tables/T02_fixed_reuse.csv
  outputs/tables/T03_overlap.csv
  outputs/tables/T04_rolling_comparison.csv
  outputs/tables/T07_sample_size.csv        # development sensitivity if possible
  outputs/diagnostics/bootstrap_<...>.parquet
  outputs/diagnostics/program06_<split>_manifest.json

The table files are split-aware: running official test later preserves existing
development rows and replaces only rows for the current split.

Typical command
---------------
  python scripts/06_compute_ope_and_diagnostics.py \
      --config configs/protocol.yaml --mode paper --split development \
      --device cpu --output-root .

Official test is permitted only after protocol_lock.json and the frozen gate
exist and verify, matching the Program 03–05 firewall.
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
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROGRAM = "06_compute_ope_and_diagnostics.py"
PROGRAM_VERSION = "1.2"
PROJECT_NAME = "grpo_ope_reuse"
RESCORE_SCHEMA = "1.0"
ONLINE_SCHEMA = "1.0"
PAIR_SCHEMA = "1.0"
OUTPUT_SCHEMA = "grpo_ope_program06_v1"

DEFAULT_SEEDS = (20260826, 20260827, 20260828)
DEFAULT_TARGET_STEPS = tuple(range(0, 401, 20))
DEFAULT_BOOTSTRAP_REPS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260826
DEFAULT_K_MAIN = 16
FLOAT64_LOG_MAX = math.log(sys.float_info.max)
FLOAT64_LOG_MIN_SUBNORMAL = math.log(float.fromhex("0x0.0000000000001p-1022"))

FIXED_COLUMNS = (
    "dataset", "split", "training_seed", "behavior_step", "target_step", "purpose",
    "estimator", "estimate", "estimate_status", "online_reference", "signed_error",
    "absolute_error", "ope_ci_low", "ope_ci_high", "online_ci_low", "online_ci_high",
    "signed_error_ci_low", "signed_error_ci_high", "absolute_error_ci_low",
    "absolute_error_ci_high", "bootstrap_reps", "bootstrap_nonfinite_fraction",
    "n_prompts", "K", "median_prompt_ess", "median_prompt_relative_ess",
    "p10_prompt_relative_ess", "mean_d2", "median_d2", "mean_max_normalized_weight",
    "p90_max_normalized_weight", "tokenwise_kl_proxy", "mean_abs_log_weight",
    "mean_log_weight", "sd_log_weight", "mean_completion_length", "truncation_rate",
)

OVERLAP_COLUMNS = (
    "dataset", "split", "training_seed", "behavior_step", "target_step", "purpose",
    "prompt_id", "K", "ess", "relative_ess", "d2", "max_normalized_weight",
    "mean_log_weight", "sd_log_weight", "mean_abs_log_weight",
    "mean_log_ratio_per_token", "tokenwise_kl_proxy", "mean_completion_length",
    "median_completion_length", "truncation_rate", "reward_rate",
    "pair_abs_mean_log_ratio_per_token", "pair_tokenwise_kl_proxy", "length_quantile",
)

ROLLING_COLUMNS = (
    "dataset", "split", "training_seed", "target_step", "recent_behavior_step",
    "estimator", "online_reference", "old_behavior_step", "old_estimate",
    "recent_estimate", "old_absolute_error", "recent_absolute_error",
    "delta_error_old_minus_recent", "old_median_relative_ess", "recent_median_relative_ess",
    "delta_relative_ess_recent_minus_old", "old_mean_d2", "recent_mean_d2",
    "old_mean_max_normalized_weight", "recent_mean_max_normalized_weight",
    "old_tokenwise_kl_proxy", "recent_tokenwise_kl_proxy", "K", "n_prompts",
)

SAMPLE_SIZE_COLUMNS = (
    "dataset", "split", "training_seed", "behavior_step", "target_step", "purpose",
    "K", "estimator", "estimate", "online_reference", "absolute_error",
    "median_prompt_relative_ess", "p10_prompt_relative_ess", "bootstrap_ci_width",
    "bootstrap_nonfinite_fraction", "n_prompts",
)

BOOTSTRAP_COLUMNS = (
    "replicate", "dataset", "split", "training_seed", "behavior_step", "target_step",
    "purpose", "estimator", "ope_estimate", "online_reference", "signed_error",
    "absolute_error", "estimator_status",
)


class Program06Error(RuntimeError):
    """Controlled, fail-fast Program 06 error."""


@dataclass(frozen=True)
class AnalysisSpec:
    seeds: tuple[int, ...]
    target_steps: tuple[int, ...]
    bootstrap_reps: int
    bootstrap_seed: int
    k_main: int


@dataclass(frozen=True)
class Pair:
    pair_id: str
    training_seed: int
    dataset: str
    split: str
    behavior_step: int
    target_step: int
    purpose: str
    protocol_version: str


@dataclass
class PromptStats:
    prompt_id: str
    K: int
    reward_rate: float
    log_is: float
    pwis: float
    ess: float
    relative_ess: float
    d2: float
    max_norm_weight: float
    mean_logw: float
    sd_logw: float
    mean_abs_logw: float
    mean_ratio_per_token: float
    tokenwise_kl_proxy: float
    mean_length: float
    median_length: float
    truncation_rate: float


# ---------------------------------------------------------------------------
# Generic deterministic utilities
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str).encode("utf-8")


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


def semantic_csv_scope_sha256(
    path: Path,
    column: str,
    value: str,
    *,
    allow_empty: bool = False,
) -> str:
    """Hash CSV rows in one immutable logical scope, independent of later appends.

    Empty semantic scopes remain fail-fast by default.  The sole intended
    exception is an engineering smoke/pilot table whose experiment is
    structurally unavailable in the reduced pair registry (currently T04
    rolling comparison in smoke).  In that case the empty scope is hashed as
    the canonical empty list, so provenance remains deterministic without
    pretending that rolling evidence exists.
    """
    rows = [dict(r) for r in read_csv_rows(path) if str(r.get(column)) == value]
    if not rows:
        if allow_empty:
            return sha256_bytes(canonical_bytes([]))
        raise Program06Error(f"No rows for semantic scope {column}={value!r} in {path}")
    rows.sort(key=lambda r: canonical_bytes(r))
    return sha256_bytes(canonical_bytes(rows))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Program06Error(f"Missing required JSON file: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Program06Error(f"Cannot parse JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise Program06Error(f"Expected JSON object in {path}")
    return obj


def fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except (OSError, AttributeError):
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=str)
            f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path); fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("x", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c) for c in columns})
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path); fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def merge_split_csv(path: Path, new_rows: Sequence[Mapping[str, Any]], columns: Sequence[str], split: str) -> None:
    old = read_csv_rows(path)
    kept = [r for r in old if str(r.get("split", "")) != split]
    merged = kept + [dict(r) for r in new_rows]
    atomic_write_csv(path, merged, columns)


def atomic_write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise Program06Error(f"pyarrow is required for Program 06: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        table = pa.Table.from_pylist([dict(r) for r in rows])
        pq.write_table(table, tmp, compression="zstd", use_dictionary=True)
        with tmp.open("rb+") as f:
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path); fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
        return [dict(x) for x in pq.read_table(path).to_pylist()]
    except Exception as exc:
        raise Program06Error(f"Cannot read Parquet {path}: {exc}") from exc


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.resolve().as_posix()


def _dig(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def first_present(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for p in paths:
        v = _dig(mapping, p)
        if v is not None:
            return v
    return None


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    xs = sorted(float(x) for x in values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * q
    lo = int(math.floor(pos)); hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    a = pos - lo
    return xs[lo] * (1.0 - a) + xs[hi] * a


def median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    return float(statistics.median(values))


def sample_sd(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 0.0
    m = math.fsum(values) / len(values)
    return math.sqrt(math.fsum((x - m) ** 2 for x in values) / (len(values) - 1))


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    if any(math.isnan(x) for x in values):
        raise Program06Error("NaN encountered in log-weight input.")
    if any(x == math.inf for x in values):
        return math.inf
    finite = [x for x in values if x != -math.inf]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(math.fsum(math.exp(x - m) for x in finite))


def safe_exp_float64(logx: float) -> tuple[float | None, str]:
    if math.isnan(logx):
        return None, "nonfinite_estimator"
    if logx == -math.inf:
        return 0.0, "ok"
    if logx == math.inf or logx > FLOAT64_LOG_MAX:
        return None, "nonfinite_estimator"
    if logx < FLOAT64_LOG_MIN_SUBNORMAL:
        return 0.0, "ok_underflow_to_zero"
    x = math.exp(logx)
    if not math.isfinite(x):
        return None, "nonfinite_estimator"
    return x, "ok"


def stable_hash_int(payload: Any, bits: int = 63) -> int:
    h = hashlib.sha256(canonical_bytes(payload)).digest()
    return int.from_bytes(h[:8], "big") & ((1 << bits) - 1)


# ---------------------------------------------------------------------------
# Config / firewall / source roots
# ---------------------------------------------------------------------------


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise Program06Error(f"Missing protocol config: {path}")
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        raise Program06Error(f"Cannot parse protocol config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise Program06Error("protocol.yaml must contain a YAML mapping.")
    project = cfg.get("project") or {}
    if isinstance(project, Mapping) and project.get("name") not in (None, PROJECT_NAME):
        raise Program06Error(f"Unexpected project.name={project.get('name')!r}")
    return cfg, sha256_file(path)


def resolve_spec(cfg: Mapping[str, Any], mode: str, analysis_k: int | None) -> AnalysisSpec:
    tr = cfg.get("training") or {}
    ope = cfg.get("ope") or {}
    beh = cfg.get("behavior") or {}
    if not isinstance(tr, Mapping) or not isinstance(ope, Mapping) or not isinstance(beh, Mapping):
        raise Program06Error("training/behavior/ope config sections must be mappings.")
    seeds = tuple(int(x) for x in tr.get("seeds", DEFAULT_SEEDS))
    target_steps = tuple(int(x) for x in ope.get("target_steps", DEFAULT_TARGET_STEPS))
    bootstrap_reps = int(ope.get("bootstrap_reps", DEFAULT_BOOTSTRAP_REPS))
    bootstrap_seed = int(ope.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED))
    k_main = int(beh.get("K_main", beh.get("K", DEFAULT_K_MAIN)))
    if analysis_k is not None:
        if mode == "paper":
            raise Program06Error("--analysis-k is forbidden in paper mode; freeze K in protocol.yaml before paper analysis.")
        k_main = int(analysis_k)
    if mode == "smoke":
        bootstrap_reps = min(bootstrap_reps, 50)
        k_main = min(k_main, 4)
    elif mode == "pilot":
        bootstrap_reps = min(bootstrap_reps, 200)
        k_main = min(k_main, 8)
    if not seeds or len(set(seeds)) != len(seeds):
        raise Program06Error("training.seeds must be unique and nonempty.")
    if bootstrap_reps <= 0 or k_main <= 0:
        raise Program06Error("bootstrap_reps and K must be positive.")
    return AnalysisSpec(seeds, target_steps, bootstrap_reps, bootstrap_seed, k_main)


def protocol_version(cfg: Mapping[str, Any]) -> str:
    p = cfg.get("project") or {}
    return str(p.get("protocol_version", "1.0")) if isinstance(p, Mapping) else "1.0"


def verify_protocol_lock(root: Path, config_sha256: str) -> dict[str, Any]:
    path = root / "manifests" / "protocol_lock.json"
    if not path.exists():
        raise Program06Error(f"Paper mode requires frozen {path}")
    lock = read_json(path)
    locked = first_present(lock, (("config_sha256",), ("protocol_config_sha256",), ("protocol", "config_sha256"), ("inputs", "config_sha256")))
    if not isinstance(locked, str) or locked != config_sha256:
        raise Program06Error(f"protocol.yaml hash differs from protocol lock: locked={locked}, observed={config_sha256}")
    for key in ("training_config_sha256", "data_manifest_sha256", "model_revision", "split_registry_sha256"):
        if not isinstance(lock.get(key), str) or not lock.get(key):
            raise Program06Error(f"protocol_lock.json must freeze {key}.")
    source_hashes = lock.get("source_code_sha256") or {}
    expected_source = source_hashes.get(PROGRAM) if isinstance(source_hashes, Mapping) else None
    if not isinstance(expected_source, str) or expected_source != sha256_file(Path(__file__).resolve()):
        raise Program06Error("Program 06 source code differs from the frozen protocol lock.")
    return lock


def verify_frozen_gate_for_test(root: Path, lock: Mapping[str, Any]) -> str:
    gate = root / "outputs" / "frozen_gate.json"
    if not gate.exists():
        raise Program06Error("Official test Program 06 is forbidden before outputs/frozen_gate.json exists.")
    observed = sha256_file(gate)
    expected = first_present(lock, (("frozen_gate_sha256",), ("gate", "file_sha256"), ("inputs", "frozen_gate_sha256"), ("frozen", "gate_sha256")))
    sidecar = root / "manifests" / "frozen_gate_manifest.json"
    if expected is None and sidecar.exists():
        sm = read_json(sidecar)
        expected = first_present(sm, (("gate_file_sha256",), ("file_sha256",), ("gate", "file_sha256")))
    if not isinstance(expected, str) or observed != expected:
        raise Program06Error(f"Frozen gate hash cannot be verified: expected={expected}, observed={observed}")
    return observed


def verify_upstream_static_manifests(root: Path) -> dict[str, Any]:
    """Verify the immutable Program 00/01 manifest chain needed for provenance.

    Program 06 does not re-read raw GSM8K or model weights, but it refuses to
    analyze derived artifacts whose pinned dataset/model/split identities no
    longer match the frozen upstream manifests.
    """
    dp = root / "manifests" / "data_manifest.json"
    mp = root / "manifests" / "model_manifest.json"
    sp = root / "manifests" / "split_registry_manifest.json"
    data = read_json(dp); model = read_json(mp); split = read_json(sp)
    if data.get("manifest_type") != "data" or model.get("manifest_type") != "model" or split.get("manifest_type") != "split_registry":
        raise Program06Error("Invalid Program 00/01 manifest headers.")
    gsm = ((data.get("datasets") or {}).get("gsm8k") or {})
    primary = ((model.get("models") or {}).get("primary") or {})
    data_rev = gsm.get("resolved_revision")
    model_rev = primary.get("resolved_revision")
    split_source = split.get("source") or {}
    split_data_rev = first_present(split_source, (("dataset_revision",), ("resolved_revision",)))
    if split_data_rev is not None and data_rev is not None and split_data_rev != data_rev:
        raise Program06Error("Program 01 split registry dataset revision differs from Program 00 data manifest.")
    return {
        "data_manifest_path": str(dp), "data_manifest_sha256": sha256_file(dp),
        "model_manifest_path": str(mp), "model_manifest_sha256": sha256_file(mp),
        "split_manifest_path": str(sp), "split_manifest_sha256": sha256_file(sp),
        "dataset_revision": data_rev, "model_revision": model_rev,
        "split_content_fingerprint_sha256": split.get("content_fingerprint_sha256"),
    }


def pair_registry_paths(root: Path, mode: str) -> tuple[Path, Path]:
    if mode == "paper":
        return root / "manifests" / "pair_registry.parquet", root / "manifests" / "pair_registry_manifest.json"
    base = root / "manifests" / f"_{mode}"
    return base / "pair_registry.parquet", base / "pair_registry_manifest.json"


def rescore_data_root(root: Path, mode: str) -> Path:
    return root / "data" / "target_rescores" if mode == "paper" else root / "data" / "target_rescores" / f"_{mode}"


def online_data_root(root: Path, mode: str) -> Path:
    return root / "data" / "online_reference" if mode == "paper" else root / "data" / "online_reference" / f"_{mode}"


def output_dirs(root: Path, mode: str) -> tuple[Path, Path]:
    if mode == "paper":
        return root / "outputs" / "tables", root / "outputs" / "diagnostics"
    return root / "outputs" / f"_{mode}" / "tables", root / "outputs" / f"_{mode}" / "diagnostics"


def load_pairs(root: Path, mode: str, split: str, spec: AnalysisSpec, pv: str, config_sha256: str, upstream: Mapping[str, Any]) -> tuple[list[Pair], dict[str, Any]]:
    pp, mp = pair_registry_paths(root, mode)
    if not pp.exists() or not mp.exists():
        raise Program06Error("Program 04 pair registry is missing; run Program 04 first.")
    m = read_json(mp)
    if m.get("manifest_type") != "behavior_target_pair_registry" or str(m.get("schema_version")) != PAIR_SCHEMA:
        raise Program06Error("Invalid Program 04 pair registry manifest/schema.")
    if (m.get("parquet") or {}).get("file_sha256") not in (None, sha256_file(pp)):
        raise Program06Error("Pair registry Parquet hash mismatch.")
    if m.get("protocol_config_sha256") not in (None, config_sha256):
        raise Program06Error("Pair registry was created under a different protocol.yaml hash.")
    if upstream.get("dataset_revision") is not None and m.get("dataset_revision") != upstream.get("dataset_revision"):
        raise Program06Error("Pair registry dataset revision differs from Program 00.")
    if upstream.get("model_revision") is not None and m.get("model_revision") != upstream.get("model_revision"):
        raise Program06Error("Pair registry model revision differs from Program 00.")
    rows = read_parquet_rows(pp)
    out: list[Pair] = []
    for r in rows:
        if str(r.get("split")) != split:
            continue
        seed = int(r["training_seed"])
        if seed not in spec.seeds:
            continue
        p = Pair(str(r["pair_id"]), seed, str(r["dataset"]), split, int(r["behavior_step"]), int(r["target_step"]), str(r["purpose"]), str(r["protocol_version"]))
        if p.protocol_version != pv:
            raise Program06Error(f"Pair registry protocol version mismatch for {p.pair_id}")
        out.append(p)
    out.sort(key=lambda x: (x.training_seed, x.behavior_step, x.target_step, x.purpose))
    if not out:
        raise Program06Error(f"No pair-registry rows found for split={split} and selected seeds.")
    return out, m


# ---------------------------------------------------------------------------
# Immutable Program 04 / 05 readers
# ---------------------------------------------------------------------------


def pair_rescore_prefix(data_root: Path, p: Pair) -> Path:
    return data_root / "gsm8k" / f"split={p.split}" / f"seed={p.training_seed}" / f"behavior_step={p.behavior_step:04d}" / f"target_step={p.target_step:04d}"


def load_rescore_rows(data_root: Path, p: Pair, k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prefix = pair_rescore_prefix(data_root, p)
    if not prefix.exists():
        raise Program06Error(f"Missing Program 04 rescore prefix for pair {p.pair_id}: {prefix}")
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for mp in sorted(prefix.rglob("manifest.json")):
        if not mp.parent.name.startswith("shard="):
            continue
        m = read_json(mp)
        if m.get("manifest_type") != "target_rescore_shard":
            continue
        if str(m.get("schema_version")) != RESCORE_SCHEMA:
            raise Program06Error(f"Unsupported Program 04 rescore schema in {mp}: {m.get('schema_version')!r}")
        if m.get("pair_id") != p.pair_id:
            raise Program06Error(f"Rescore shard pair_id mismatch: {mp}")
        pp = mp.parent / "rescored.parquet"
        if not pp.exists():
            raise Program06Error(f"Missing rescored.parquet beside {mp}")
        expected = (m.get("parquet") or {}).get("file_sha256")
        observed = sha256_file(pp)
        if expected is not None and expected != observed:
            raise Program06Error(f"Rescore Parquet hash mismatch: {pp}")
        shard_rows = read_parquet_rows(pp)
        expected_content = (m.get("parquet") or {}).get("content_sha256")
        # Program 04 already validated its semantic hash.  Program 06 preserves the
        # manifest as provenance and validates all stable keys/values below.
        for r in shard_rows:
            if int(r["training_seed"]) != p.training_seed or str(r["split"]) != p.split:
                raise Program06Error(f"Rescore row seed/split mismatch in {pp}")
            if int(r["behavior_step"]) != p.behavior_step or int(r["target_step"]) != p.target_step:
                raise Program06Error(f"Rescore row pair-step mismatch in {pp}")
            if str(r.get("purpose")) != p.purpose:
                raise Program06Error(f"Rescore row purpose mismatch in {pp}")
            idx = int(r["sample_index"])
            if idx < k:
                z = float(r["log_weight"])
                if not math.isfinite(z):
                    raise Program06Error(f"Non-finite log_weight in Program 04 output: {r.get('rescore_id')}")
                lb = float(r["behavior_sequence_logprob"]); le = float(r["target_sequence_logprob"])
                if not (math.isfinite(lb) and math.isfinite(le)) or not math.isclose(z, le - lb, rel_tol=0.0, abs_tol=1e-9):
                    raise Program06Error(f"Program 04 log-weight identity failed for {r.get('rescore_id')}: logW != target - behavior")
                L = int(r["completion_length"])
                mlr = float(r["mean_log_ratio_per_token"])
                if not math.isclose(mlr, z / L, rel_tol=0.0, abs_tol=1e-10):
                    raise Program06Error(f"Program 04 mean_log_ratio_per_token mismatch for {r.get('rescore_id')}")
                src_expected = (m.get("source_behavior") or {}).get("content_sha256")
                if src_expected is not None and r.get("source_behavior_content_sha256") != src_expected:
                    raise Program06Error(f"Program 04 row/source behavior content hash mismatch in {pp}")
                rew = float(r["correctness_reward"])
                if rew not in (0.0, 1.0):
                    raise Program06Error("Program 04 correctness_reward must be binary 0/1.")
                if int(r["completion_length"]) <= 0:
                    raise Program06Error("Program 04 completion_length must be positive.")
                rows.append(dict(r))
        manifests.append({"path": str(mp), "file_sha256": sha256_file(mp), "parquet_file_sha256": observed, "content_sha256": expected_content})
    if not rows:
        raise Program06Error(f"No Program 04 rows with sample_index < K={k} for pair {p.pair_id}")
    # Completeness: every prompt must have exactly indices 0..K-1.
    by: dict[str, set[int]] = {}
    seen_traj: set[str] = set()
    for r in rows:
        pid = str(r["prompt_id"]); idx = int(r["sample_index"])
        by.setdefault(pid, set()).add(idx)
        tid = str(r["trajectory_id"])
        if tid in seen_traj:
            raise Program06Error(f"Duplicate trajectory_id across rescore shards: {tid}")
        seen_traj.add(tid)
    expected_idx = set(range(k))
    bad = [pid for pid, xs in by.items() if xs != expected_idx]
    if bad:
        raise Program06Error(f"Incomplete Program 04 sample indices 0..{k-1} for {len(bad)} prompts; first={bad[0]}")
    rows.sort(key=lambda r: (str(r["prompt_id"]), int(r["sample_index"])))
    return rows, manifests


def online_target_prefix(data_root: Path, split: str, seed: int, target_step: int) -> Path:
    return data_root / "gsm8k" / f"split={split}" / f"seed={seed}" / f"target_step={target_step:04d}"


def load_online_prompt_summary(data_root: Path, split: str, seed: int, target_step: int) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    prefix = online_target_prefix(data_root, split, seed, target_step)
    rp = prefix / "reference_summary.json"
    pp = prefix / "prompt_summary.parquet"
    if not rp.exists() or not pp.exists():
        raise Program06Error(f"Missing Program 05 target summary for seed={seed}, target={target_step}: {prefix}")
    ref = read_json(rp)
    if ref.get("manifest_type") != "online_reference_target_summary" or str(ref.get("schema_version")) != ONLINE_SCHEMA:
        raise Program06Error(f"Invalid Program 05 reference summary/schema: {rp}")
    if int(ref.get("training_seed", -1)) != seed or int(ref.get("target_step", -1)) != target_step or str(ref.get("split")) != split:
        raise Program06Error(f"Program 05 reference summary key mismatch: {rp}")
    expected_sha = (ref.get("prompt_summary") or {}).get("file_sha256")
    observed_sha = sha256_file(pp)
    if expected_sha is not None and expected_sha != observed_sha:
        raise Program06Error(f"Program 05 prompt_summary hash mismatch: {pp}")
    rows = read_parquet_rows(pp)
    out: dict[str, float] = {}
    n_samples_seen: set[int] = set()
    for r in rows:
        pid = str(r["prompt_id"])
        if pid in out:
            raise Program06Error(f"Duplicate prompt_id in Program 05 prompt summary: {pid}")
        n = int(r["n_samples"]); s = int(r["success_count"]); a = float(r["prompt_accuracy"])
        if n <= 0 or s < 0 or s > n or not math.isclose(a, s / n, abs_tol=1e-15, rel_tol=0.0):
            raise Program06Error(f"Invalid Program 05 prompt summary row for {pid}")
        out[pid] = a; n_samples_seen.add(n)
    if len(n_samples_seen) != 1:
        raise Program06Error("Program 05 prompt summary must have uniform L within target checkpoint.")
    computed = math.fsum(out.values()) / len(out)
    if not math.isclose(computed, float(ref["online_reference"]), rel_tol=0.0, abs_tol=1e-12):
        raise Program06Error(f"Program 05 online reference does not equal prompt-summary mean: {rp}")
    provenance = {"reference_path": str(rp), "reference_sha256": sha256_file(rp), "prompt_summary_path": str(pp), "prompt_summary_sha256": observed_sha}
    return out, ref, provenance


# ---------------------------------------------------------------------------
# Core OPE mathematics
# ---------------------------------------------------------------------------


def prompt_stats(prompt_id: str, rows: Sequence[Mapping[str, Any]]) -> PromptStats:
    if not rows:
        raise Program06Error("prompt_stats received an empty prompt cluster.")
    rows = sorted(rows, key=lambda r: int(r["sample_index"]))
    K = len(rows)
    if [int(r["sample_index"]) for r in rows] != list(range(K)):
        raise Program06Error(f"Prompt {prompt_id} sample indices are not contiguous 0..K-1.")
    z = [float(r["log_weight"]) for r in rows]
    rewards = [float(r["correctness_reward"]) for r in rows]
    lengths = [int(r["completion_length"]) for r in rows]
    trunc = [int(bool(r["was_truncated"])) for r in rows]
    if any(not math.isfinite(x) for x in z):
        raise Program06Error(f"Non-finite log weight for prompt {prompt_id}")
    lse_w = logsumexp(z)
    lse_w2 = logsumexp([2.0 * x for x in z])
    log_ess = 2.0 * lse_w - lse_w2
    ess = math.exp(log_ess) if log_ess <= FLOAT64_LOG_MAX else float("inf")
    # Numerical guard only for tiny rounding noise; never weight clipping.
    if not math.isfinite(ess) or ess < 1.0 - 1e-9 or ess > K + 1e-8:
        raise Program06Error(f"ESS invariant failed for prompt {prompt_id}: ESS={ess}, K={K}")
    ess = min(float(K), max(1.0, ess))
    r_ess = ess / K
    d2 = lse_w2 - math.log(K)
    cmax = math.exp(max(z) - lse_w)
    reward_logs = [x for x, r in zip(z, rewards) if r == 1.0]
    log_is = logsumexp(reward_logs) - math.log(K) if reward_logs else -math.inf
    pwis = math.exp(logsumexp(reward_logs) - lse_w) if reward_logs else 0.0
    if pwis < -1e-15 or pwis > 1.0 + 1e-12:
        raise Program06Error(f"pWIS outside [0,1] for prompt {prompt_id}: {pwis}")
    pwis = min(1.0, max(0.0, pwis))
    total_tokens = sum(lengths)
    sum_logw = math.fsum(z)
    token_kl = -sum_logw / total_tokens
    mean_ratio_token = sum_logw / total_tokens
    return PromptStats(
        prompt_id=prompt_id,
        K=K,
        reward_rate=math.fsum(rewards) / K,
        log_is=log_is,
        pwis=pwis,
        ess=ess,
        relative_ess=r_ess,
        d2=d2,
        max_norm_weight=cmax,
        mean_logw=math.fsum(z) / K,
        sd_logw=sample_sd(z),
        mean_abs_logw=math.fsum(abs(x) for x in z) / K,
        mean_ratio_per_token=mean_ratio_token,
        tokenwise_kl_proxy=token_kl,
        mean_length=math.fsum(lengths) / K,
        median_length=median(lengths),
        truncation_rate=math.fsum(trunc) / K,
    )


def pair_prompt_stats(rows: Sequence[Mapping[str, Any]]) -> list[PromptStats]:
    by: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r["prompt_id"]), []).append(r)
    return [prompt_stats(pid, by[pid]) for pid in sorted(by)]


def estimate_is(stats: Sequence[PromptStats], indices: Sequence[int] | None = None) -> tuple[float | None, str, float]:
    if indices is None:
        vals = [s.log_is for s in stats]
    else:
        vals = [stats[i].log_is for i in indices]
    if not vals:
        raise Program06Error("Cannot estimate IS on zero prompts.")
    log_v = logsumexp(vals) - math.log(len(vals))
    value, status = safe_exp_float64(log_v)
    return value, status, log_v


def estimate_pwis(stats: Sequence[PromptStats], indices: Sequence[int] | None = None) -> float:
    vals = [s.pwis for s in stats] if indices is None else [stats[i].pwis for i in indices]
    return math.fsum(vals) / len(vals)


def estimate_global_wis(rows: Sequence[Mapping[str, Any]]) -> float:
    z = [float(r["log_weight"]) for r in rows]
    rz = [float(r["log_weight"]) for r in rows if float(r["correctness_reward"]) == 1.0]
    return math.exp(logsumexp(rz) - logsumexp(z)) if rz else 0.0


def aggregate_pair_diagnostics(stats: Sequence[PromptStats], rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    z = [float(r["log_weight"]) for r in rows]
    lengths = [int(r["completion_length"]) for r in rows]
    trunc = [int(bool(r["was_truncated"])) for r in rows]
    total_tokens = sum(lengths)
    return {
        "median_prompt_ess": median([s.ess for s in stats]),
        "median_prompt_relative_ess": median([s.relative_ess for s in stats]),
        "p10_prompt_relative_ess": percentile([s.relative_ess for s in stats], 0.10),
        "mean_d2": math.fsum(s.d2 for s in stats) / len(stats),
        "median_d2": median([s.d2 for s in stats]),
        "mean_max_normalized_weight": math.fsum(s.max_norm_weight for s in stats) / len(stats),
        "p90_max_normalized_weight": percentile([s.max_norm_weight for s in stats], 0.90),
        "tokenwise_kl_proxy": -math.fsum(z) / total_tokens,
        "mean_abs_log_weight": math.fsum(abs(x) for x in z) / len(z),
        "mean_log_weight": math.fsum(z) / len(z),
        "sd_log_weight": sample_sd(z),
        "mean_completion_length": math.fsum(lengths) / len(lengths),
        "truncation_rate": math.fsum(trunc) / len(trunc),
        "pair_abs_mean_log_ratio_per_token": abs(math.fsum(z) / total_tokens),
    }


def length_quantile_labels(stats: Sequence[PromptStats]) -> dict[str, str]:
    ordered = sorted(stats, key=lambda s: (s.mean_length, s.prompt_id))
    n = len(ordered)
    out: dict[str, str] = {}
    labels = ("Q1_short", "Q2", "Q3", "Q4_long")
    for rank, s in enumerate(ordered):
        q = min(3, (4 * rank) // max(1, n))
        out[s.prompt_id] = labels[q]
    return out


# ---------------------------------------------------------------------------
# Prompt-cluster paired bootstrap
# ---------------------------------------------------------------------------


def bootstrap_pair(
    *, p: Pair, stats: Sequence[PromptStats], online: Mapping[str, float], estimator: str,
    reps: int, base_seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Paired prompt-cluster bootstrap.

    The bootstrap RNG seed is PAIR-specific, not estimator-specific, so IS and
    prompt-WIS use exactly the same resampled prompt clusters.  NumPy is used in
    bounded chunks for speed and memory control; no token/completion-level
    resampling occurs.
    """
    pids = [s.prompt_id for s in stats]
    if set(pids) != set(online):
        miss_online = sorted(set(pids) - set(online))[:3]
        miss_ope = sorted(set(online) - set(pids))[:3]
        raise Program06Error(f"Prompt mismatch between OPE and online reference for pair {p.pair_id}: missing_online={miss_online}, missing_ope={miss_ope}")
    n = len(stats)
    seed = stable_hash_int({"program": PROGRAM, "base_seed": base_seed, "pair_id": p.pair_id, "reps": reps})
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise Program06Error(f"NumPy is required for efficient prompt-cluster bootstrap: {exc}") from exc

    online_vec = np.asarray([float(online[x]) for x in pids], dtype=np.float64)
    if estimator == "prompt_wis":
        stat_vec = np.asarray([s.pwis for s in stats], dtype=np.float64)
    elif estimator == "is":
        stat_vec = np.asarray([s.log_is for s in stats], dtype=np.float64)
    else:
        raise Program06Error(f"Unsupported bootstrap estimator: {estimator}")

    rng = np.random.default_rng(seed)
    out_rows: list[dict[str, Any]] = []
    finite_ope: list[float] = []
    all_online: list[float] = []
    finite_signed: list[float] = []
    finite_abs: list[float] = []
    nonfinite = 0
    chunk_reps = min(256, reps)
    done = 0
    log_n = math.log(n)
    while done < reps:
        bsz = min(chunk_reps, reps - done)
        idx = rng.integers(0, n, size=(bsz, n), dtype=np.int32)
        online_b = online_vec[idx].mean(axis=1)
        if estimator == "prompt_wis":
            ope_b_arr = stat_vec[idx].mean(axis=1)
            finite_mask = np.isfinite(ope_b_arr)
            status_arr = np.where(finite_mask, "ok", "nonfinite_estimator")
        else:
            selected = stat_vec[idx]
            row_max = np.max(selected, axis=1)
            logv = np.full(bsz, -np.inf, dtype=np.float64)
            finite_max = np.isfinite(row_max)
            if np.any(finite_max):
                centered = selected[finite_max] - row_max[finite_max, None]
                # exp(-inf) is exactly zero; centered finite maxima prevent inf-inf.
                sums = np.exp(centered).sum(axis=1)
                logv[finite_max] = row_max[finite_max] + np.log(sums) - log_n
            ope_b_arr = np.zeros(bsz, dtype=np.float64)
            finite_mask = np.isfinite(logv) & (logv <= FLOAT64_LOG_MAX)
            zero_mask = np.isneginf(logv)
            ope_b_arr[zero_mask] = 0.0
            exp_mask = finite_mask & ~zero_mask
            ope_b_arr[exp_mask] = np.exp(logv[exp_mask])
            finite_mask = finite_mask | zero_mask
            status_arr = np.where(finite_mask, "ok", "nonfinite_estimator")

        for j in range(bsz):
            ob = float(online_b[j]); all_online.append(ob)
            if bool(finite_mask[j]) and math.isfinite(float(ope_b_arr[j])):
                ov = float(ope_b_arr[j]); signed = ov - ob; ae = abs(signed)
                finite_ope.append(ov); finite_signed.append(signed); finite_abs.append(ae)
            else:
                ov = None; signed = None; ae = None; nonfinite += 1
            out_rows.append({
                "replicate": done + j, "dataset": p.dataset, "split": p.split,
                "training_seed": p.training_seed, "behavior_step": p.behavior_step,
                "target_step": p.target_step, "purpose": p.purpose, "estimator": estimator,
                "ope_estimate": ov, "online_reference": ob, "signed_error": signed,
                "absolute_error": ae, "estimator_status": str(status_arr[j]),
            })
        done += bsz

    def ci(xs: Sequence[float]) -> tuple[float | None, float | None]:
        return (percentile(xs, 0.025), percentile(xs, 0.975)) if xs else (None, None)
    o_lo, o_hi = ci(finite_ope); on_lo, on_hi = ci(all_online)
    s_lo, s_hi = ci(finite_signed); a_lo, a_hi = ci(finite_abs)
    return out_rows, {
        "ope_ci_low": o_lo, "ope_ci_high": o_hi,
        "online_ci_low": on_lo, "online_ci_high": on_hi,
        "signed_error_ci_low": s_lo, "signed_error_ci_high": s_hi,
        "absolute_error_ci_low": a_lo, "absolute_error_ci_high": a_hi,
        "bootstrap_nonfinite_fraction": nonfinite / reps,
        "bootstrap_seed": seed,
    }


# ---------------------------------------------------------------------------
# Per-pair analysis and Experiments 1–3
# ---------------------------------------------------------------------------


def analyze_pair(
    *, p: Pair, rescore_rows: Sequence[Mapping[str, Any]], online: Mapping[str, float],
    spec: AnalysisSpec, diagnostics_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    stats = pair_prompt_stats(rescore_rows)
    if not stats or any(s.K != spec.k_main for s in stats):
        raise Program06Error(f"Pair {p.pair_id} does not have exactly K={spec.k_main} per prompt.")
    if set(s.prompt_id for s in stats) != set(online):
        raise Program06Error(f"Pair {p.pair_id} prompt set differs from Program 05 online reference.")
    online_value = math.fsum(float(online[s.prompt_id]) for s in stats) / len(stats)
    pairdiag = aggregate_pair_diagnostics(stats, rescore_rows)
    qlabels = length_quantile_labels(stats)
    overlap_rows: list[dict[str, Any]] = []
    for s in stats:
        overlap_rows.append({
            "dataset": p.dataset, "split": p.split, "training_seed": p.training_seed,
            "behavior_step": p.behavior_step, "target_step": p.target_step, "purpose": p.purpose,
            "prompt_id": s.prompt_id, "K": s.K, "ess": s.ess, "relative_ess": s.relative_ess,
            "d2": s.d2, "max_normalized_weight": s.max_norm_weight,
            "mean_log_weight": s.mean_logw, "sd_log_weight": s.sd_logw,
            "mean_abs_log_weight": s.mean_abs_logw,
            "mean_log_ratio_per_token": s.mean_ratio_per_token,
            "tokenwise_kl_proxy": s.tokenwise_kl_proxy,
            "mean_completion_length": s.mean_length, "median_completion_length": s.median_length,
            "truncation_rate": s.truncation_rate, "reward_rate": s.reward_rate,
            "pair_abs_mean_log_ratio_per_token": pairdiag["pair_abs_mean_log_ratio_per_token"],
            "pair_tokenwise_kl_proxy": pairdiag["tokenwise_kl_proxy"],
            "length_quantile": qlabels[s.prompt_id],
        })
    estimator_results: dict[str, dict[str, Any]] = {}
    fixed_rows: list[dict[str, Any]] = []
    bootstrap_outputs: list[dict[str, Any]] = []
    for estimator in ("is", "prompt_wis"):
        if estimator == "is":
            estimate, status, log_estimate = estimate_is(stats)
        else:
            estimate = estimate_pwis(stats); status = "ok"; log_estimate = math.log(estimate) if estimate > 0 else -math.inf
        boot_rows, bsum = bootstrap_pair(p=p, stats=stats, online=online, estimator=estimator, reps=spec.bootstrap_reps, base_seed=spec.bootstrap_seed)
        bpath = diagnostics_dir / f"bootstrap_{p.split}_seed{p.training_seed}_b{p.behavior_step:04d}_e{p.target_step:04d}_{estimator}.parquet"
        atomic_write_parquet(bpath, boot_rows)
        bootstrap_outputs.append({"path": str(bpath), "sha256": sha256_file(bpath), "row_count": len(boot_rows), "pair_id": p.pair_id, "estimator": estimator})
        if estimate is None:
            signed = None; abserr = None
        else:
            signed = estimate - online_value; abserr = abs(signed)
        result = {
            "dataset": p.dataset, "split": p.split, "training_seed": p.training_seed,
            "behavior_step": p.behavior_step, "target_step": p.target_step, "purpose": p.purpose,
            "estimator": estimator, "estimate": estimate, "estimate_status": status,
            "online_reference": online_value, "signed_error": signed, "absolute_error": abserr,
            **{k: bsum.get(k) for k in (
                "ope_ci_low", "ope_ci_high", "online_ci_low", "online_ci_high",
                "signed_error_ci_low", "signed_error_ci_high", "absolute_error_ci_low",
                "absolute_error_ci_high", "bootstrap_nonfinite_fraction")},
            "bootstrap_reps": spec.bootstrap_reps, "n_prompts": len(stats), "K": spec.k_main,
            **{k: pairdiag[k] for k in (
                "median_prompt_ess", "median_prompt_relative_ess", "p10_prompt_relative_ess",
                "mean_d2", "median_d2", "mean_max_normalized_weight", "p90_max_normalized_weight",
                "tokenwise_kl_proxy", "mean_abs_log_weight", "mean_log_weight", "sd_log_weight",
                "mean_completion_length", "truncation_rate")},
            "log_estimate": log_estimate,
        }
        estimator_results[estimator] = result
        if p.purpose == "fixed" and p.behavior_step == 0 and p.target_step > 0:
            fixed_rows.append({k: result.get(k) for k in FIXED_COLUMNS})
    # Appendix-only global WIS is retained in diagnostics/provenance, not T02 main table.
    estimator_results["global_wis_appendix"] = {
        "estimate": estimate_global_wis(rescore_rows), "online_reference": online_value,
        "note": "Appendix diagnostic only; primary WIS is prompt-normalized WIS."
    }
    return fixed_rows, overlap_rows, estimator_results, bootstrap_outputs


def build_rolling_rows(pair_results: Mapping[tuple[int, int, int], dict[str, dict[str, Any]]], split: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    # key=(seed,b,e)
    for (seed, b, e), results in sorted(pair_results.items()):
        if b == 0:
            continue
        old = pair_results.get((seed, 0, e))
        if old is None:
            continue
        for estimator in ("is", "prompt_wis"):
            ro = old[estimator]; rr = results[estimator]
            if ro.get("estimate") is None or rr.get("estimate") is None:
                delta_err = None
            else:
                delta_err = float(ro["absolute_error"]) - float(rr["absolute_error"])
            out.append({
                "dataset": rr["dataset"], "split": split, "training_seed": seed, "target_step": e,
                "recent_behavior_step": b, "estimator": estimator, "online_reference": rr["online_reference"],
                "old_behavior_step": 0, "old_estimate": ro["estimate"], "recent_estimate": rr["estimate"],
                "old_absolute_error": ro["absolute_error"], "recent_absolute_error": rr["absolute_error"],
                "delta_error_old_minus_recent": delta_err,
                "old_median_relative_ess": ro["median_prompt_relative_ess"],
                "recent_median_relative_ess": rr["median_prompt_relative_ess"],
                "delta_relative_ess_recent_minus_old": float(rr["median_prompt_relative_ess"]) - float(ro["median_prompt_relative_ess"]),
                "old_mean_d2": ro["mean_d2"], "recent_mean_d2": rr["mean_d2"],
                "old_mean_max_normalized_weight": ro["mean_max_normalized_weight"],
                "recent_mean_max_normalized_weight": rr["mean_max_normalized_weight"],
                "old_tokenwise_kl_proxy": ro["tokenwise_kl_proxy"], "recent_tokenwise_kl_proxy": rr["tokenwise_kl_proxy"],
                "K": rr["K"], "n_prompts": rr["n_prompts"],
            })
    return out


def sample_size_sensitivity(
    *, p: Pair, full_rows: Sequence[Mapping[str, Any]], online: Mapping[str, float], max_k: int,
    spec: AnalysisSpec,
) -> list[dict[str, Any]]:
    if p.split != "development" or p.purpose not in ("fixed", "rolling"):
        return []
    candidates = [k for k in (8, 16, 32) if k <= max_k]
    if max_k not in candidates:
        candidates.append(max_k)
    out: list[dict[str, Any]] = []
    for k in sorted(set(candidates)):
        subset = [r for r in full_rows if int(r["sample_index"]) < k]
        stats = pair_prompt_stats(subset)
        if not stats or any(s.K != k for s in stats):
            continue
        online_value = math.fsum(float(online[s.prompt_id]) for s in stats) / len(stats)
        diag = aggregate_pair_diagnostics(stats, subset)
        reps = min(spec.bootstrap_reps, 500)  # sensitivity diagnostic; main pair bootstrap remains full B.
        for est in ("is", "prompt_wis"):
            if est == "is":
                estimate, status, _ = estimate_is(stats)
            else:
                estimate = estimate_pwis(stats); status = "ok"
            _, bs = bootstrap_pair(p=p, stats=stats, online=online, estimator=est, reps=reps, base_seed=spec.bootstrap_seed + k)
            width = None
            if bs["ope_ci_low"] is not None and bs["ope_ci_high"] is not None:
                width = float(bs["ope_ci_high"]) - float(bs["ope_ci_low"])
            out.append({
                "dataset": p.dataset, "split": p.split, "training_seed": p.training_seed,
                "behavior_step": p.behavior_step, "target_step": p.target_step, "purpose": p.purpose,
                "K": k, "estimator": est, "estimate": estimate, "online_reference": online_value,
                "absolute_error": None if estimate is None else abs(estimate - online_value),
                "median_prompt_relative_ess": diag["median_prompt_relative_ess"],
                "p10_prompt_relative_ess": diag["p10_prompt_relative_ess"],
                "bootstrap_ci_width": width, "bootstrap_nonfinite_fraction": bs["bootstrap_nonfinite_fraction"],
                "n_prompts": len(stats), "estimate_status": status,
            })
    return out


# ---------------------------------------------------------------------------
# Test firewall and source prerequisite checks
# ---------------------------------------------------------------------------


def verify_identity_prerequisite(data_root: Path, pairs: Sequence[Pair], k: int) -> None:
    identity = [p for p in pairs if p.purpose == "identity" and p.behavior_step == p.target_step]
    if not identity:
        raise Program06Error("Pair registry contains no identity pairs for this split.")
    for p in identity:
        rows, _ = load_rescore_rows(data_root, p, k)
        if any(r.get("identity_pass") is not True for r in rows):
            raise Program06Error(f"Program 04 identity prerequisite failed for seed={p.training_seed}, step={p.target_step}")


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Program 06: CPU-only OPE, overlap diagnostics, fixed/rolling/length experiments.")
    ap.add_argument("--config", default="configs/protocol.yaml")
    ap.add_argument("--mode", choices=("smoke", "pilot", "paper"), default="paper")
    ap.add_argument("--split", choices=("development", "test"), required=True)
    ap.add_argument("--resume", action="store_true", help="Accepted for unified CLI; Program 06 cheaply recomputes pair statistics and reuses immutable GPU assets.")
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--output-root", default=".")
    ap.add_argument("--analysis-k", type=int, default=None, help="Smoke/pilot only. Paper K must come from frozen protocol.")
    ap.add_argument("--sample-size-sensitivity", action="store_true", help="Development-only optional T07 K-sensitivity; off by default to save CPU time.")
    ap.add_argument("--seed", type=int, action="append", dest="selected_seeds", help="Optional subset of training seeds; repeat flag. Test requires the frozen paper seed set.")
    return ap.parse_args(argv)


def validate_mode(args: argparse.Namespace) -> None:
    if args.device != "cpu":
        raise Program06Error("Program 06 is CPU-only by design. Use --device cpu; it must not consume GPU for bootstrap/statistics.")
    if args.mode != "paper" and args.split == "test":
        raise Program06Error("smoke/pilot modes are development-only; official test is paper mode only.")
    if args.sample_size_sensitivity and args.split != "development":
        raise Program06Error("Sample-size sensitivity is development-only and forbidden on official test.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv); validate_mode(args)
    root = Path(args.output_root).resolve(); config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    cfg, config_sha = load_protocol(config_path)
    spec = resolve_spec(cfg, args.mode, args.analysis_k)
    if args.selected_seeds:
        chosen = tuple(int(x) for x in args.selected_seeds)
        if args.mode == "paper" and args.split == "test" and set(chosen) != set(spec.seeds):
            raise Program06Error("Official test must evaluate the complete frozen training-seed set.")
        spec = AnalysisSpec(chosen, spec.target_steps, spec.bootstrap_reps, spec.bootstrap_seed, spec.k_main)
    pv = protocol_version(cfg)
    lock: dict[str, Any] | None = None; gate_sha: str | None = None
    if args.mode == "paper":
        lock = verify_protocol_lock(root, config_sha)
        if args.split == "test":
            gate_sha = verify_frozen_gate_for_test(root, lock)

    upstream = verify_upstream_static_manifests(root)
    pairs, pair_manifest = load_pairs(root, args.mode, args.split, spec, pv, config_sha, upstream)
    rroot = rescore_data_root(root, args.mode); oroot = online_data_root(root, args.mode)
    tables_dir, diag_dir = output_dirs(root, args.mode); tables_dir.mkdir(parents=True, exist_ok=True); diag_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("PROGRAM 06 — OPE / OVERLAP DIAGNOSTICS")
    print(f"protocol version : {pv}")
    print(f"config SHA-256   : {config_sha}")
    print(f"mode / split     : {args.mode} / {args.split}")
    print(f"device           : cpu")
    print(f"seeds            : {list(spec.seeds)}")
    print(f"analysis K       : {spec.k_main}")
    print(f"bootstrap reps   : {spec.bootstrap_reps}")
    print(f"pair count       : {len(pairs)}")
    print(f"tables dir       : {tables_dir}")
    print(f"diagnostics dir  : {diag_dir}")
    print("=" * 78)

    verify_identity_prerequisite(rroot, pairs, spec.k_main)
    analysis_pairs = [p for p in pairs if p.purpose in ("fixed", "rolling")]
    if not analysis_pairs:
        raise Program06Error("No fixed/rolling pairs available after identity gate.")

    fixed_rows_all: list[dict[str, Any]] = []
    overlap_rows_all: list[dict[str, Any]] = []
    sample_rows_all: list[dict[str, Any]] = []
    pair_results: dict[tuple[int, int, int], dict[str, dict[str, Any]]] = {}
    source_rescore_manifests: list[dict[str, Any]] = []
    source_online: dict[tuple[int, int], dict[str, Any]] = {}
    bootstrap_outputs: list[dict[str, Any]] = []

    online_cache: dict[tuple[int, int], dict[str, float]] = {}
    for i, p in enumerate(analysis_pairs, 1):
        key_online = (p.training_seed, p.target_step)
        if key_online not in online_cache:
            online_map, online_ref, prov = load_online_prompt_summary(oroot, args.split, p.training_seed, p.target_step)
            online_cache[key_online] = online_map; source_online[key_online] = {**prov, "reference": online_ref}
        rows, manifests = load_rescore_rows(rroot, p, spec.k_main)
        source_rescore_manifests.extend(manifests)
        target_hashes = {str(r["target_adapter_sha256"]) for r in rows}
        dataset_revisions = {str(r["dataset_revision"]) for r in rows}
        online_ref_meta = source_online[key_online]["reference"]
        if len(target_hashes) != 1 or next(iter(target_hashes)) != str(online_ref_meta.get("target_adapter_sha256")):
            raise Program06Error(f"Program 04 and Program 05 do not refer to the same target adapter for seed={p.training_seed}, target={p.target_step}.")
        if len(dataset_revisions) != 1 or next(iter(dataset_revisions)) != str(online_ref_meta.get("dataset_revision")):
            raise Program06Error(f"Program 04 and Program 05 dataset revisions differ for seed={p.training_seed}, target={p.target_step}.")
        frows, orows, results, boots = analyze_pair(p=p, rescore_rows=rows, online=online_cache[key_online], spec=spec, diagnostics_dir=diag_dir)
        fixed_rows_all.extend(frows); overlap_rows_all.extend(orows); pair_results[(p.training_seed, p.behavior_step, p.target_step)] = results; bootstrap_outputs.extend(boots)
        if args.sample_size_sensitivity:
            sample_rows_all.extend(sample_size_sensitivity(p=p, full_rows=rows, online=online_cache[key_online], max_k=spec.k_main, spec=spec))
        print(f"[{i:03d}/{len(analysis_pairs):03d}] seed={p.training_seed} b={p.behavior_step:03d} e={p.target_step:03d} {p.purpose} PASS")

    rolling_rows = build_rolling_rows(pair_results, args.split)

    t02 = tables_dir / "T02_fixed_reuse.csv"; t03 = tables_dir / "T03_overlap.csv"; t04 = tables_dir / "T04_rolling_comparison.csv"; t07 = tables_dir / "T07_sample_size.csv"
    merge_split_csv(t02, fixed_rows_all, FIXED_COLUMNS, args.split)
    merge_split_csv(t03, overlap_rows_all, OVERLAP_COLUMNS, args.split)
    merge_split_csv(t04, rolling_rows, ROLLING_COLUMNS, args.split)
    if args.sample_size_sensitivity:
        merge_split_csv(t07, sample_rows_all, SAMPLE_SIZE_COLUMNS, args.split)

    # Compact per-split provenance manifest; deduplicate source manifest records.
    unique_rescore: dict[str, dict[str, Any]] = {}
    for x in source_rescore_manifests:
        unique_rescore[x["path"]] = x
    output_files = [t02, t03, t04] + ([t07] if args.sample_size_sensitivity else [])
    manifest = {
        "schema_version": OUTPUT_SCHEMA,
        "manifest_type": "program06_analysis_manifest",
        "created_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "mode": args.mode, "split": args.split, "protocol_version": pv,
        "protocol_config_sha256": config_sha, "frozen_gate_sha256": gate_sha,
        "analysis_spec": {"seeds": list(spec.seeds), "K": spec.k_main, "bootstrap_reps": spec.bootstrap_reps, "bootstrap_seed": spec.bootstrap_seed},
        "mathematical_contract": {
            "ordinary_is": "prompt-level IS then equal-prompt average; no clipping; non-representable float64 estimates flagged",
            "primary_wis": "prompt-normalized WIS",
            "global_wis": "appendix-only diagnostic",
            "ess": "(sum w)^2/sum w^2",
            "relative_ess": "ESS/K",
            "d2_proxy": "log((1/K)sum w^2)",
            "bootstrap_cluster": "prompt",
            "online_reference": "high-precision on-policy Monte Carlo reference; not exact truth",
        },
        "inputs": {
            "upstream_static_manifests": upstream,
            "pair_registry_manifest_sha256": sha256_file(pair_registry_paths(root, args.mode)[1]),
            "rescore_manifests": list(unique_rescore.values()),
            "online_targets": [{"training_seed": k[0], "target_step": k[1], **v} for k, v in sorted(source_online.items())],
        },
        "outputs": [
            {
                "path": rel(p, root),
                "sha256_at_write_time": sha256_file(p),
                "bytes_at_write_time": p.stat().st_size,
                "semantic_scope": {"column": "split", "value": args.split},
                "semantic_scope_empty": bool(
                    args.mode != "paper" and p == t04 and len(rolling_rows) == 0
                ),
                "semantic_sha256": semantic_csv_scope_sha256(
                    p,
                    "split",
                    args.split,
                    allow_empty=bool(
                        args.mode != "paper" and p == t04 and len(rolling_rows) == 0
                    ),
                ),
            }
            for p in output_files if p.exists()
        ],
        "bootstrap_outputs": [{**x, "path": rel(Path(x["path"]), root)} for x in bootstrap_outputs],
        "row_counts_current_split": {"T02_fixed_reuse": len(fixed_rows_all), "T03_overlap": len(overlap_rows_all), "T04_rolling_comparison": len(rolling_rows), "T07_sample_size": len(sample_rows_all)},
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": __import__("numpy").__version__, "pyarrow": __import__("pyarrow").__version__},
    }
    mpath = diag_dir / f"program06_{args.split}_manifest.json"
    atomic_write_json(mpath, manifest)

    print("\nPROGRAM 06 PASSED")
    print(f"T02 fixed rows   : {len(fixed_rows_all)}")
    print(f"T03 overlap rows : {len(overlap_rows_all)}")
    print(f"T04 rolling rows : {len(rolling_rows)}")
    if args.sample_size_sensitivity: print(f"T07 sample rows  : {len(sample_rows_all)}")
    print(f"manifest         : {mpath}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Program06Error as exc:
        print(f"\nPROGRAM 06 FAILED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
