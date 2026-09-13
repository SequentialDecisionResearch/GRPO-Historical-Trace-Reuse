#!/usr/bin/env python3
"""
Post-hoc alpha mechanism analysis for the GRPO-OPE trace-reuse study.

Research status
---------------
This is an EXPLORATORY / POST-HOC analysis. It does not alter Programs 01-09,
does not call a model, does not use CUDA, does not refit the frozen gate, and
does not select new checkpoint pairs. It reads the exact matched-refresh pairs
already present in outputs/tables/T04_rolling_comparison.csv and reuses the
immutable Program 04 rescored trajectories.

For each prompt i and behavior/target pair, with log weights z_ij = log W_ij
and binary rewards R_ij, it computes the empirical squared-weight allocation

    alpha_hat_i = sum_j exp(2 z_ij) R_ij / sum_j exp(2 z_ij),

plus

    log kappa2_hat_i = log[(1/K) sum_j exp(2 z_ij)],
    A_proxy_i(V)     = V^2 + (1 - 2V) alpha_hat_i,
    log m2_proxy_i   = log kappa2_hat_i + log A_proxy_i(V),

where V is the matched pair's aggregate independent finite on-policy reference from T04.
Using one aggregate V across prompts follows the pre-specified post-results memo; therefore A_proxy and m2_proxy are mechanism diagnostics, not estimates of each prompt's latent theoretical v_i-specific coefficient. The old and recent behavior logs share the same target and V, so the comparison still isolates how squared-weight allocation changes under refresh.

Primary output scope
--------------------
- GSM8K development matched-refresh pairs: 45 comparisons (exploratory primary)
- GSM8K official test matched-refresh pairs: 9 comparisons (exploratory replication)

Expected project layout
-----------------------
<root>/
  outputs/tables/T04_rolling_comparison.csv
  outputs/diagnostics/program06_development_manifest.json
  outputs/diagnostics/program06_test_manifest.json
  data/target_rescores/gsm8k/split=<...>/seed=<...>/
      behavior_step=<bbbb>/target_step=<eeee>/.../shard=<sssss>/
          manifest.json
          rescored.parquet

Outputs
-------
<root>/outputs/posthoc_alpha/
  alpha_pair_summary.csv
  alpha_split_summary.csv
  alpha_prompt_detail.parquet
  alpha_mechanism_manifest.json
  README.txt

Typical command
---------------
  python posthoc_alpha_mechanism_analysis_v2.py --root "C:\\lsg\\grpo_ope_reuse"

Deep audit (slower; verifies every rescored.parquet SHA-256 against its shard
manifest):
  python posthoc_alpha_mechanism_analysis_v2.py --root "C:\\lsg\\grpo_ope_reuse" --verify-shard-hashes

A wall-clock guard defaults to 7.5 hours so the program stops before an 8-hour
budget is exceeded. In normal use this CPU-only pass should be far cheaper than
any model-running program because it only reads existing Parquet files.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROGRAM = "posthoc_alpha_mechanism_analysis_v2.py"
PROGRAM_VERSION = "1.1"
OUTPUT_SCHEMA = "grpo_ope_posthoc_alpha_v2"
EXPECTED_PROGRAM06_SCHEMA = "grpo_ope_program06_v1"
PRIMARY_ESTIMATOR = "prompt_wis"
DEFAULT_SPLITS = ("development", "test")
EXPECTED_SEEDS = (20260826, 20260827, 20260828)
EXPECTED_K = 16
EXPECTED_N_PROMPTS = {"development": 1473, "test": 1319}
EXPECTED_PAIR_COUNTS = {"development": 45, "test": 9}
EXPECTED_PAIRS_PER_SEED = {"development": 15, "test": 3}
EXPECTED_RECENT_BEHAVIORS = {100, 200, 300}
FLOAT64_LOG_MAX = math.log(sys.float_info.max)
FLOAT64_LOG_MIN = math.log(float.fromhex("0x0.0000000000001p-1022"))

PAIR_COLUMNS = (
    "split", "training_seed", "target_step", "old_behavior_step", "recent_behavior_step",
    "K", "n_prompts", "online_reference", "alpha_favorable_direction",
    "old_median_relative_ess", "recent_median_relative_ess",
    "old_mean_alpha", "recent_mean_alpha", "delta_mean_alpha_recent_minus_old",
    "old_median_alpha", "recent_median_alpha",
    "old_mean_A_proxy", "recent_mean_A_proxy", "delta_mean_A_proxy_recent_minus_old",
    "mean_delta_log_kappa2_recent_minus_old", "geometric_kappa2_ratio_recent_over_old",
    "mean_delta_log_A_proxy_recent_minus_old", "geometric_A_proxy_ratio_recent_over_old",
    "mean_delta_log_m2_proxy_recent_minus_old", "geometric_m2_proxy_ratio_recent_over_old",
    "log_mean_m2_proxy_old", "log_mean_m2_proxy_recent", "aggregate_m2_proxy_ratio_recent_over_old",
    "fraction_prompts_overlap_channel_favorable", "fraction_prompts_reward_proxy_channel_favorable",
    "fraction_prompts_combined_proxy_favorable", "fraction_prompts_alpha_moves_favorable",
    "raw_old_pair_id", "raw_recent_pair_id",
)

SPLIT_COLUMNS = (
    "split", "n_pairs", "n_training_seeds", "n_prompt_pair_rows",
    "pairs_rESS_improved", "pairs_reward_proxy_channel_favorable",
    "pairs_prompt_geometric_m2_proxy_favorable", "pairs_aggregate_m2_proxy_favorable",
    "pairs_alpha_mean_moved_favorable", "mean_old_rESS", "mean_recent_rESS",
    "mean_old_alpha", "mean_recent_alpha", "mean_old_A_proxy", "mean_recent_A_proxy",
    "geometric_mean_kappa2_ratio_recent_over_old",
    "geometric_mean_A_proxy_ratio_recent_over_old",
    "geometric_mean_promptwise_m2_proxy_ratio_recent_over_old",
    "mean_pair_aggregate_m2_proxy_ratio_recent_over_old",
    "median_pair_aggregate_m2_proxy_ratio_recent_over_old",
    "geometric_mean_pair_aggregate_m2_proxy_ratio_recent_over_old",
    "mechanism_pattern",
    "target_reference_min", "target_reference_max",
)

PROMPT_COLUMNS = (
    "split", "training_seed", "target_step", "old_behavior_step", "recent_behavior_step",
    "prompt_id", "K", "online_reference",
    "old_alpha", "recent_alpha", "delta_alpha_recent_minus_old",
    "old_log_kappa2", "recent_log_kappa2", "delta_log_kappa2_recent_minus_old",
    "old_relative_ess", "recent_relative_ess", "delta_relative_ess_recent_minus_old",
    "old_A_proxy", "recent_A_proxy", "delta_A_proxy_recent_minus_old",
    "old_log_m2_proxy", "recent_log_m2_proxy", "delta_log_m2_proxy_recent_minus_old",
    "overlap_channel_favorable", "reward_proxy_channel_favorable", "combined_proxy_favorable",
    "alpha_moves_favorable",
)

RESCORE_COLUMNS = (
    "training_seed", "split", "behavior_step", "target_step", "purpose",
    "prompt_id", "sample_index", "log_weight", "correctness_reward",
)


class AlphaMechanismError(RuntimeError):
    """Controlled, actionable failure."""


@dataclass(frozen=True)
class Comparison:
    split: str
    training_seed: int
    target_step: int
    old_behavior_step: int
    recent_behavior_step: int
    online_reference: float
    K: int
    n_prompts: int
    old_median_relative_ess_t04: float
    recent_median_relative_ess_t04: float


@dataclass(frozen=True)
class PromptMechanism:
    prompt_id: str
    alpha: float
    log_kappa2: float
    relative_ess: float
    log_sq_correct_sum: float
    log_sq_incorrect_sum: float


@dataclass(frozen=True)
class PairLoad:
    prompts: Mapping[str, PromptMechanism]
    pair_id: str
    purpose: str
    shard_count: int
    parquet_count: int
    source_fingerprint_sha256: str


# ---------------------------------------------------------------------------
# Deterministic utilities
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(
        obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AlphaMechanismError(f"Missing required JSON file: {path}")
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AlphaMechanismError(f"Cannot parse JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise AlphaMechanismError(f"Expected JSON object in {path}")
    return obj


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise AlphaMechanismError(f"Missing required CSV file: {path}")
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return [dict(r) for r in csv.DictReader(f)]
    except Exception as exc:
        raise AlphaMechanismError(f"Cannot read CSV {path}: {exc}") from exc


def fsync_directory(path: Path) -> None:
    # Windows directory handles do not support POSIX fsync semantics.
    if os.name == "nt":
        return
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


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=str) + "\n"
    atomic_write_text(path, text)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({c: row.get(c) for c in columns})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise AlphaMechanismError(f"pyarrow is required: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        normalized = [{c: r.get(c) for c in columns} for r in rows]
        table = pa.Table.from_pylist(normalized)
        pq.write_table(table, tmp, compression="zstd", use_dictionary=True)
        with tmp.open("rb+") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        tmp.unlink(missing_ok=True)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise AlphaMechanismError("Mean requested on an empty sequence.")
    return math.fsum(values) / len(values)


def median(values: Sequence[float]) -> float:
    if not values:
        raise AlphaMechanismError("Median requested on an empty sequence.")
    return float(statistics.median(values))


def logsumexp(values: Sequence[float]) -> float:
    if not values:
        return -math.inf
    if any(math.isnan(x) for x in values):
        raise AlphaMechanismError("NaN encountered in logsumexp input.")
    if any(x == math.inf for x in values):
        return math.inf
    finite = [x for x in values if x != -math.inf]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(math.fsum(math.exp(x - m) for x in finite))


def logmeanexp(values: Sequence[float]) -> float:
    if not values:
        raise AlphaMechanismError("logmeanexp requested on an empty sequence.")
    return logsumexp(values) - math.log(len(values))


def safe_exp(logx: float) -> float | None:
    if math.isnan(logx):
        return None
    if logx == -math.inf or logx < FLOAT64_LOG_MIN:
        return 0.0
    if logx == math.inf or logx > FLOAT64_LOG_MAX:
        return None
    out = math.exp(logx)
    return out if math.isfinite(out) else None


def check_deadline(deadline: float, context: str) -> None:
    if time.monotonic() >= deadline:
        raise AlphaMechanismError(
            f"Wall-clock guard reached before {context}. The program stopped cleanly rather than exceed the runtime budget."
        )


def start_hard_watchdog(max_runtime_hours: float) -> threading.Timer:
    """Start a process-level hard stop so a blocking I/O call cannot run past the budget.

    Normal deadline checks raise clean Python exceptions. This independent daemon timer
    is the last-resort guard for a filesystem/Parquet call that does not return. Outputs
    are written atomically only after all analysis completes, so a hard stop cannot
    publish a half-written final file.
    """
    seconds = max_runtime_hours * 3600.0

    def _hard_stop() -> None:
        try:
            sys.stderr.write(
                "\nPOST-HOC ALPHA ANALYSIS HARD STOP\n"
                f"The {max_runtime_hours:.2f}-hour wall-clock cap was reached. "
                "The process is exiting before the 8-hour budget.\n"
            )
            sys.stderr.flush()
        finally:
            os._exit(3)

    timer = threading.Timer(seconds, _hard_stop)
    timer.daemon = True
    timer.start()
    return timer


def isclose(a: float, b: float, *, atol: float = 1e-9, rtol: float = 1e-10) -> bool:
    return math.isclose(a, b, abs_tol=atol, rel_tol=rtol)


# ---------------------------------------------------------------------------
# Input firewall and exact matched-pair selection
# ---------------------------------------------------------------------------


def validate_program06_manifest(path: Path, split: str) -> dict[str, Any]:
    m = read_json(path)
    if m.get("manifest_type") != "program06_analysis_manifest":
        raise AlphaMechanismError(f"Unexpected Program 06 manifest type: {path}")
    if str(m.get("schema_version")) != EXPECTED_PROGRAM06_SCHEMA:
        raise AlphaMechanismError(
            f"Unsupported Program 06 schema in {path}: {m.get('schema_version')!r}"
        )
    if str(m.get("split")) != split:
        raise AlphaMechanismError(f"Program 06 manifest split mismatch: {path}")
    return m


def load_comparisons(root: Path, splits: Sequence[str]) -> tuple[list[Comparison], dict[str, Any]]:
    t04_path = root / "outputs" / "tables" / "T04_rolling_comparison.csv"
    rows = read_csv(t04_path)
    required = {
        "split", "training_seed", "target_step", "recent_behavior_step", "estimator",
        "online_reference", "old_behavior_step", "old_median_relative_ess",
        "recent_median_relative_ess", "K", "n_prompts",
    }
    if not rows:
        raise AlphaMechanismError("T04_rolling_comparison.csv is empty.")
    missing = required - set(rows[0])
    if missing:
        raise AlphaMechanismError(f"T04 is missing required columns: {sorted(missing)}")

    selected: dict[tuple[str, int, int, int], Comparison] = {}
    for r in rows:
        split = str(r["split"])
        if split not in splits or str(r["estimator"]) != PRIMARY_ESTIMATOR:
            continue
        c = Comparison(
            split=split,
            training_seed=int(r["training_seed"]),
            target_step=int(r["target_step"]),
            old_behavior_step=int(r["old_behavior_step"]),
            recent_behavior_step=int(r["recent_behavior_step"]),
            online_reference=float(r["online_reference"]),
            K=int(r["K"]),
            n_prompts=int(r["n_prompts"]),
            old_median_relative_ess_t04=float(r["old_median_relative_ess"]),
            recent_median_relative_ess_t04=float(r["recent_median_relative_ess"]),
        )
        if c.old_behavior_step != 0:
            raise AlphaMechanismError(
                f"Unexpected old_behavior_step={c.old_behavior_step}; matched-refresh analysis expects step 0."
            )
        if not (0.0 < c.online_reference < 1.0):
            raise AlphaMechanismError(
                f"online_reference must be strictly inside (0,1) for the log-risk decomposition; observed {c.online_reference} for {c}"
            )
        if c.K <= 0 or c.n_prompts <= 0:
            raise AlphaMechanismError(f"Invalid K/n_prompts for {c}")
        if c.K != EXPECTED_K:
            raise AlphaMechanismError(f"Expected frozen K={EXPECTED_K}; observed K={c.K} for {c}")
        if c.n_prompts != EXPECTED_N_PROMPTS[split]:
            raise AlphaMechanismError(
                f"Expected n_prompts={EXPECTED_N_PROMPTS[split]} for split={split}; "
                f"observed {c.n_prompts} for {c}"
            )
        if c.training_seed not in EXPECTED_SEEDS:
            raise AlphaMechanismError(f"Unexpected training seed in T04: {c.training_seed}")
        if c.recent_behavior_step not in EXPECTED_RECENT_BEHAVIORS or c.target_step <= c.recent_behavior_step:
            raise AlphaMechanismError(
                f"Unexpected matched-refresh geometry: recent={c.recent_behavior_step}, target={c.target_step}"
            )
        if not (1.0 / c.K - 1e-12 <= c.old_median_relative_ess_t04 <= 1.0 + 1e-12):
            raise AlphaMechanismError(f"T04 old median rESS outside [1/K,1]: {c}")
        if not (1.0 / c.K - 1e-12 <= c.recent_median_relative_ess_t04 <= 1.0 + 1e-12):
            raise AlphaMechanismError(f"T04 recent median rESS outside [1/K,1]: {c}")
        key = (split, c.training_seed, c.target_step, c.recent_behavior_step)
        if key in selected:
            raise AlphaMechanismError(f"Duplicate prompt-WIS matched comparison in T04: {key}")
        selected[key] = c

    out = sorted(selected.values(), key=lambda x: (x.split, x.training_seed, x.target_step, x.recent_behavior_step))
    if not out:
        raise AlphaMechanismError(f"No prompt-WIS T04 comparisons found for splits={list(splits)}")

    # Paper-run invariants from the already-completed frozen study.
    counts = {sp: sum(c.split == sp for c in out) for sp in splits}
    for sp in splits:
        if counts.get(sp) != EXPECTED_PAIR_COUNTS[sp]:
            raise AlphaMechanismError(
                f"Expected exactly {EXPECTED_PAIR_COUNTS[sp]} {sp} matched-refresh comparisons; "
                f"found {counts.get(sp)}."
            )
        seed_counts = {seed: sum(c.split == sp and c.training_seed == seed for c in out) for seed in EXPECTED_SEEDS}
        if any(seed_counts[seed] != EXPECTED_PAIRS_PER_SEED[sp] for seed in EXPECTED_SEEDS):
            raise AlphaMechanismError(
                f"Unexpected per-seed matched-pair counts for split={sp}: {seed_counts}; "
                f"expected {EXPECTED_PAIRS_PER_SEED[sp]} per seed."
            )

    provenance: dict[str, Any] = {
        "T04_path": str(t04_path),
        "T04_sha256": sha256_file(t04_path),
        "comparison_counts": counts,
        "program06_manifests": {},
    }
    for split in splits:
        mp = root / "outputs" / "diagnostics" / f"program06_{split}_manifest.json"
        m = validate_program06_manifest(mp, split)
        provenance["program06_manifests"][split] = {
            "path": str(mp),
            "sha256": sha256_file(mp),
            "created_by": m.get("created_by"),
            "protocol_config_sha256": m.get("protocol_config_sha256"),
        }
    return out, provenance


# ---------------------------------------------------------------------------
# Immutable Program 04 rescore reader
# ---------------------------------------------------------------------------


def pair_prefix(root: Path, split: str, seed: int, behavior_step: int, target_step: int) -> Path:
    return (
        root / "data" / "target_rescores" / "gsm8k"
        / f"split={split}" / f"seed={seed}"
        / f"behavior_step={behavior_step:04d}" / f"target_step={target_step:04d}"
    )


def load_pair(
    *, root: Path, split: str, seed: int, behavior_step: int, target_step: int, K: int,
    verify_shard_hashes: bool, deadline: float,
) -> PairLoad:
    check_deadline(deadline, f"loading split={split}, seed={seed}, b={behavior_step}, e={target_step}")
    prefix = pair_prefix(root, split, seed, behavior_step, target_step)
    if not prefix.exists():
        raise AlphaMechanismError(f"Missing Program 04 rescore prefix: {prefix}")

    manifest_paths = sorted(
        p for p in prefix.rglob("manifest.json") if p.parent.name.startswith("shard=")
    )
    if not manifest_paths:
        raise AlphaMechanismError(f"No shard manifests found under {prefix}")

    parquet_paths: list[Path] = []
    pair_ids: set[str] = set()
    purposes: set[str] = set()
    fingerprint_records: list[dict[str, Any]] = []

    for idx, mp in enumerate(manifest_paths):
        if idx % 64 == 0:
            check_deadline(deadline, f"validating shard manifests under {prefix}")
        m = read_json(mp)
        if m.get("manifest_type") != "target_rescore_shard" or str(m.get("schema_version")) != "1.0":
            raise AlphaMechanismError(f"Invalid target-rescore shard manifest: {mp}")
        pid = str(m.get("pair_id", ""))
        if not pid:
            raise AlphaMechanismError(f"Shard manifest lacks pair_id: {mp}")
        pair_ids.add(pid)
        pp = mp.parent / "rescored.parquet"
        if not pp.exists():
            raise AlphaMechanismError(f"Missing rescored.parquet beside {mp}")
        prec = m.get("parquet") or {}
        expected_sha = prec.get("file_sha256") if isinstance(prec, Mapping) else None
        if verify_shard_hashes:
            if not isinstance(expected_sha, str) or not expected_sha:
                raise AlphaMechanismError(f"Cannot verify Parquet hash; manifest lacks file_sha256: {mp}")
            observed_sha = sha256_file(pp)
            if observed_sha != expected_sha:
                raise AlphaMechanismError(f"Rescored Parquet SHA-256 mismatch: {pp}")
        parquet_paths.append(pp)
        fingerprint_records.append({
            "manifest_rel": mp.relative_to(root).as_posix(),
            "manifest_sha256": sha256_file(mp),
            "parquet_rel": pp.relative_to(root).as_posix(),
            "manifest_parquet_sha256": expected_sha,
        })

    if len(pair_ids) != 1:
        raise AlphaMechanismError(f"Multiple pair_ids under one pair prefix {prefix}: {sorted(pair_ids)[:3]}")

    try:
        import pyarrow.dataset as ds  # type: ignore
    except Exception as exc:
        raise AlphaMechanismError(f"pyarrow is required to scan immutable rescore Parquet files: {exc}") from exc

    check_deadline(deadline, f"scanning Parquet files under {prefix}")
    try:
        dataset = ds.dataset([str(p) for p in parquet_paths], format="parquet")
        schema_names = set(dataset.schema.names)
        missing = set(RESCORE_COLUMNS) - schema_names
        if missing:
            raise AlphaMechanismError(f"Rescore Parquet schema under {prefix} misses columns: {sorted(missing)}")
        filt = ds.field("sample_index") < K
        table = dataset.to_table(columns=list(RESCORE_COLUMNS), filter=filt, use_threads=True)
        cols = table.to_pydict()
    except AlphaMechanismError:
        raise
    except Exception as exc:
        raise AlphaMechanismError(f"Cannot scan rescored Parquet files under {prefix}: {exc}") from exc

    nrows = table.num_rows
    if nrows <= 0:
        raise AlphaMechanismError(f"No rescore rows with sample_index < K={K} under {prefix}")

    grouped: dict[str, list[tuple[int, float, float]]] = {}
    for i in range(nrows):
        if i % 100000 == 0:
            check_deadline(deadline, f"validating rows under {prefix}")
        if int(cols["training_seed"][i]) != seed:
            raise AlphaMechanismError(f"training_seed mismatch in {prefix}")
        if str(cols["split"][i]) != split:
            raise AlphaMechanismError(f"split mismatch in {prefix}")
        if int(cols["behavior_step"][i]) != behavior_step or int(cols["target_step"][i]) != target_step:
            raise AlphaMechanismError(f"behavior/target step mismatch in {prefix}")
        purposes.add(str(cols["purpose"][i]))
        prompt_id = str(cols["prompt_id"][i])
        sample_index = int(cols["sample_index"][i])
        z = float(cols["log_weight"][i])
        reward = float(cols["correctness_reward"][i])
        if sample_index < 0 or sample_index >= K:
            raise AlphaMechanismError(f"sample_index outside 0..{K-1} after filter in {prefix}")
        if not math.isfinite(z):
            raise AlphaMechanismError(f"Non-finite log_weight under {prefix}, prompt={prompt_id}")
        if reward not in (0.0, 1.0):
            raise AlphaMechanismError(f"correctness_reward must be binary 0/1 under {prefix}")
        grouped.setdefault(prompt_id, []).append((sample_index, z, reward))

    if len(purposes) != 1:
        raise AlphaMechanismError(f"Multiple purpose values under one pair prefix {prefix}: {sorted(purposes)}")

    prompts: dict[str, PromptMechanism] = {}
    expected_idx = list(range(K))
    for prompt_id in sorted(grouped):
        triplets = sorted(grouped[prompt_id], key=lambda x: x[0])
        indices = [x[0] for x in triplets]
        if indices != expected_idx:
            raise AlphaMechanismError(
                f"Prompt {prompt_id} under {prefix} does not contain exactly sample indices 0..{K-1}."
            )
        z = [x[1] for x in triplets]
        rewards = [x[2] for x in triplets]
        lse1 = logsumexp(z)
        lse2 = logsumexp([2.0 * x for x in z])
        if not (math.isfinite(lse1) and math.isfinite(lse2)):
            raise AlphaMechanismError(f"Non-finite log-sum-exp for prompt {prompt_id} under {prefix}")
        positive_sq_logs = [2.0 * zz for zz, rr in zip(z, rewards) if rr == 1.0]
        negative_sq_logs = [2.0 * zz for zz, rr in zip(z, rewards) if rr == 0.0]
        log_sq_correct = logsumexp(positive_sq_logs)
        log_sq_incorrect = logsumexp(negative_sq_logs)
        alpha = math.exp(log_sq_correct - lse2) if positive_sq_logs else 0.0
        alpha = min(1.0, max(0.0, alpha))
        log_kappa2 = lse2 - math.log(K)
        log_ess = 2.0 * lse1 - lse2
        ess = math.exp(log_ess) if log_ess <= FLOAT64_LOG_MAX else float("inf")
        if not math.isfinite(ess) or ess < 1.0 - 1e-8 or ess > K + 1e-7:
            raise AlphaMechanismError(f"ESS invariant failed for prompt {prompt_id}: ESS={ess}, K={K}")
        ess = min(float(K), max(1.0, ess))
        prompts[prompt_id] = PromptMechanism(
            prompt_id=prompt_id,
            alpha=alpha,
            log_kappa2=log_kappa2,
            relative_ess=ess / K,
            log_sq_correct_sum=log_sq_correct,
            log_sq_incorrect_sum=log_sq_incorrect,
        )

    fp = sha256_bytes(canonical_bytes(fingerprint_records))
    return PairLoad(
        prompts=prompts,
        pair_id=next(iter(pair_ids)),
        purpose=next(iter(purposes)),
        shard_count=len(manifest_paths),
        parquet_count=len(parquet_paths),
        source_fingerprint_sha256=fp,
    )


# ---------------------------------------------------------------------------
# Mechanism decomposition
# ---------------------------------------------------------------------------


def reward_factor(v: float, alpha: float) -> float:
    # Binary-reward identity: A = v^2 + (1 - 2v) alpha.
    A = v * v + (1.0 - 2.0 * v) * alpha
    if A < -1e-12 or A > 1.0 + 1e-12:
        raise AlphaMechanismError(f"Reward factor A outside [0,1]: v={v}, alpha={alpha}, A={A}")
    return min(1.0, max(0.0, A))


def alpha_moves_favorable(v: float, old_alpha: float, recent_alpha: float) -> bool:
    if v < 0.5:
        return recent_alpha < old_alpha
    if v > 0.5:
        return recent_alpha > old_alpha
    return False


def alpha_direction_label(v: float) -> str:
    if v < 0.5:
        return "lower_alpha_favorable"
    if v > 0.5:
        return "higher_alpha_favorable"
    return "alpha_neutral_at_v_equals_half"


def analyze_comparison(
    *, root: Path, c: Comparison, verify_shard_hashes: bool, deadline: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    old = load_pair(
        root=root, split=c.split, seed=c.training_seed, behavior_step=c.old_behavior_step,
        target_step=c.target_step, K=c.K, verify_shard_hashes=verify_shard_hashes, deadline=deadline,
    )
    recent = load_pair(
        root=root, split=c.split, seed=c.training_seed, behavior_step=c.recent_behavior_step,
        target_step=c.target_step, K=c.K, verify_shard_hashes=verify_shard_hashes, deadline=deadline,
    )
    if old.purpose != "fixed":
        raise AlphaMechanismError(
            f"Old comparator purpose must be 'fixed'; observed {old.purpose!r} for split={c.split}, seed={c.training_seed}, target={c.target_step}"
        )
    if recent.purpose != "rolling":
        raise AlphaMechanismError(
            f"Recent comparator purpose must be 'rolling'; observed {recent.purpose!r} for split={c.split}, seed={c.training_seed}, target={c.target_step}"
        )
    if old.pair_id == recent.pair_id:
        raise AlphaMechanismError("Old and recent matched comparison unexpectedly share the same pair_id.")

    old_ids = set(old.prompts)
    recent_ids = set(recent.prompts)
    if old_ids != recent_ids:
        raise AlphaMechanismError(
            f"Prompt set mismatch for matched comparison split={c.split}, seed={c.training_seed}, target={c.target_step}."
        )
    if len(old_ids) != c.n_prompts:
        raise AlphaMechanismError(
            f"Prompt count mismatch for split={c.split}, seed={c.training_seed}, target={c.target_step}: "
            f"T04={c.n_prompts}, raw={len(old_ids)}"
        )

    old_ress = median([x.relative_ess for x in old.prompts.values()])
    recent_ress = median([x.relative_ess for x in recent.prompts.values()])
    if not isclose(old_ress, c.old_median_relative_ess_t04):
        raise AlphaMechanismError(
            f"Raw/T04 old median rESS mismatch for split={c.split}, seed={c.training_seed}, target={c.target_step}: "
            f"raw={old_ress:.15g}, T04={c.old_median_relative_ess_t04:.15g}"
        )
    if not isclose(recent_ress, c.recent_median_relative_ess_t04):
        raise AlphaMechanismError(
            f"Raw/T04 recent median rESS mismatch for split={c.split}, seed={c.training_seed}, target={c.target_step}: "
            f"raw={recent_ress:.15g}, T04={c.recent_median_relative_ess_t04:.15g}"
        )

    v = c.online_reference
    prompt_rows: list[dict[str, Any]] = []
    old_alpha_values: list[float] = []
    recent_alpha_values: list[float] = []
    old_A_proxy_values: list[float] = []
    recent_A_proxy_values: list[float] = []
    delta_log_kappa: list[float] = []
    delta_log_A: list[float] = []
    delta_log_m2: list[float] = []
    old_log_m2_proxy_values: list[float] = []
    recent_log_m2_proxy_values: list[float] = []
    overlap_fav = reward_fav = combined_fav = alpha_fav = 0

    for prompt_id in sorted(old_ids):
        o = old.prompts[prompt_id]
        r = recent.prompts[prompt_id]
        Ao = reward_factor(v, o.alpha)
        Ar = reward_factor(v, r.alpha)
        logAo = math.log(Ao) if Ao > 0.0 else -math.inf
        logAr = math.log(Ar) if Ar > 0.0 else -math.inf
        logm_o = o.log_kappa2 + logAo if logAo != -math.inf else -math.inf
        logm_r = r.log_kappa2 + logAr if logAr != -math.inf else -math.inf

        # Independent algebraic cross-check of the binary-stratum identity using
        # raw squared-weight mass in the correct and incorrect strata:
        # mean W^2(R-v)^2 = [sum_correct W^2(1-v)^2 + sum_incorrect W^2 v^2] / K.
        def direct_log_m2(pm: PromptMechanism) -> float:
            terms: list[float] = []
            if pm.log_sq_correct_sum != -math.inf:
                terms.append(pm.log_sq_correct_sum + 2.0 * math.log1p(-v))
            if pm.log_sq_incorrect_sum != -math.inf:
                terms.append(pm.log_sq_incorrect_sum + 2.0 * math.log(v))
            return logsumexp(terms) - math.log(c.K)

        direct_o = direct_log_m2(o)
        direct_r = direct_log_m2(r)
        if not isclose(logm_o, direct_o, atol=3e-12, rtol=3e-12):
            raise AlphaMechanismError(
                f"Binary-stratum identity failed for old log, prompt={prompt_id}: factorized={logm_o}, direct={direct_o}"
            )
        if not isclose(logm_r, direct_r, atol=3e-12, rtol=3e-12):
            raise AlphaMechanismError(
                f"Binary-stratum identity failed for recent log, prompt={prompt_id}: factorized={logm_r}, direct={direct_r}"
            )

        dk = r.log_kappa2 - o.log_kappa2
        if logAo == -math.inf and logAr == -math.inf:
            dAlog = 0.0
        elif logAo == -math.inf:
            dAlog = math.inf
        elif logAr == -math.inf:
            dAlog = -math.inf
        else:
            dAlog = logAr - logAo
        if logm_o == -math.inf and logm_r == -math.inf:
            dmlog = 0.0
        elif logm_o == -math.inf:
            dmlog = math.inf
        elif logm_r == -math.inf:
            dmlog = -math.inf
        else:
            dmlog = logm_r - logm_o

        of = dk < 0.0
        rf = Ar < Ao
        cf = dmlog < 0.0
        af = alpha_moves_favorable(v, o.alpha, r.alpha)
        overlap_fav += int(of)
        reward_fav += int(rf)
        combined_fav += int(cf)
        alpha_fav += int(af)

        old_alpha_values.append(o.alpha)
        recent_alpha_values.append(r.alpha)
        old_A_proxy_values.append(Ao)
        recent_A_proxy_values.append(Ar)
        delta_log_kappa.append(dk)
        if math.isfinite(dAlog):
            delta_log_A.append(dAlog)
        if math.isfinite(dmlog):
            delta_log_m2.append(dmlog)
        old_log_m2_proxy_values.append(logm_o)
        recent_log_m2_proxy_values.append(logm_r)

        prompt_rows.append({
            "split": c.split,
            "training_seed": c.training_seed,
            "target_step": c.target_step,
            "old_behavior_step": c.old_behavior_step,
            "recent_behavior_step": c.recent_behavior_step,
            "prompt_id": prompt_id,
            "K": c.K,
            "online_reference": v,
            "old_alpha": o.alpha,
            "recent_alpha": r.alpha,
            "delta_alpha_recent_minus_old": r.alpha - o.alpha,
            "old_log_kappa2": o.log_kappa2,
            "recent_log_kappa2": r.log_kappa2,
            "delta_log_kappa2_recent_minus_old": dk,
            "old_relative_ess": o.relative_ess,
            "recent_relative_ess": r.relative_ess,
            "delta_relative_ess_recent_minus_old": r.relative_ess - o.relative_ess,
            "old_A_proxy": Ao,
            "recent_A_proxy": Ar,
            "delta_A_proxy_recent_minus_old": Ar - Ao,
            "old_log_m2_proxy": logm_o,
            "recent_log_m2_proxy": logm_r,
            "delta_log_m2_proxy_recent_minus_old": dmlog,
            "overlap_channel_favorable": of,
            "reward_proxy_channel_favorable": rf,
            "combined_proxy_favorable": cf,
            "alpha_moves_favorable": af,
        })

    n = len(prompt_rows)
    if n == 0:
        raise AlphaMechanismError("Matched comparison unexpectedly has zero prompts.")
    if len(delta_log_A) != n or len(delta_log_m2) != n:
        raise AlphaMechanismError(
            "A zero-valued reward factor produced an infinite log ratio; this edge case requires explicit review."
        )

    mean_dk = mean(delta_log_kappa)
    mean_dA = mean(delta_log_A)
    mean_dm = mean(delta_log_m2)
    # Exact promptwise log decomposition check.
    if not isclose(mean_dm, mean_dk + mean_dA, atol=2e-12, rtol=2e-12):
        raise AlphaMechanismError(
            f"Mechanism log decomposition failed: mean Δlog m2={mean_dm}, "
            f"mean Δlog kappa2 + mean Δlog A={mean_dk + mean_dA}"
        )

    log_mean_m2_proxy_old = logmeanexp(old_log_m2_proxy_values)
    log_mean_m2_proxy_recent = logmeanexp(recent_log_m2_proxy_values)
    aggregate_ratio = safe_exp(log_mean_m2_proxy_recent - log_mean_m2_proxy_old)

    pair_row = {
        "split": c.split,
        "training_seed": c.training_seed,
        "target_step": c.target_step,
        "old_behavior_step": c.old_behavior_step,
        "recent_behavior_step": c.recent_behavior_step,
        "K": c.K,
        "n_prompts": n,
        "online_reference": v,
        "alpha_favorable_direction": alpha_direction_label(v),
        "old_median_relative_ess": old_ress,
        "recent_median_relative_ess": recent_ress,
        "old_mean_alpha": mean(old_alpha_values),
        "recent_mean_alpha": mean(recent_alpha_values),
        "delta_mean_alpha_recent_minus_old": mean(recent_alpha_values) - mean(old_alpha_values),
        "old_median_alpha": median(old_alpha_values),
        "recent_median_alpha": median(recent_alpha_values),
        "old_mean_A_proxy": mean(old_A_proxy_values),
        "recent_mean_A_proxy": mean(recent_A_proxy_values),
        "delta_mean_A_proxy_recent_minus_old": mean(recent_A_proxy_values) - mean(old_A_proxy_values),
        "mean_delta_log_kappa2_recent_minus_old": mean_dk,
        "geometric_kappa2_ratio_recent_over_old": safe_exp(mean_dk),
        "mean_delta_log_A_proxy_recent_minus_old": mean_dA,
        "geometric_A_proxy_ratio_recent_over_old": safe_exp(mean_dA),
        "mean_delta_log_m2_proxy_recent_minus_old": mean_dm,
        "geometric_m2_proxy_ratio_recent_over_old": safe_exp(mean_dm),
        "log_mean_m2_proxy_old": log_mean_m2_proxy_old,
        "log_mean_m2_proxy_recent": log_mean_m2_proxy_recent,
        "aggregate_m2_proxy_ratio_recent_over_old": aggregate_ratio,
        "fraction_prompts_overlap_channel_favorable": overlap_fav / n,
        "fraction_prompts_reward_proxy_channel_favorable": reward_fav / n,
        "fraction_prompts_combined_proxy_favorable": combined_fav / n,
        "fraction_prompts_alpha_moves_favorable": alpha_fav / n,
        "raw_old_pair_id": old.pair_id,
        "raw_recent_pair_id": recent.pair_id,
    }
    source_rec = {
        "split": c.split,
        "training_seed": c.training_seed,
        "target_step": c.target_step,
        "old": {
            "behavior_step": c.old_behavior_step,
            "pair_id": old.pair_id,
            "purpose": old.purpose,
            "shard_count": old.shard_count,
            "source_fingerprint_sha256": old.source_fingerprint_sha256,
        },
        "recent": {
            "behavior_step": c.recent_behavior_step,
            "pair_id": recent.pair_id,
            "purpose": recent.purpose,
            "shard_count": recent.shard_count,
            "source_fingerprint_sha256": recent.source_fingerprint_sha256,
        },
    }
    return pair_row, prompt_rows, source_rec


def summarize_splits(pair_rows: Sequence[Mapping[str, Any]], prompt_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in sorted({str(r["split"]) for r in pair_rows}):
        ps = [r for r in pair_rows if str(r["split"]) == split]
        prs = [r for r in prompt_rows if str(r["split"]) == split]
        mean_log_k_ratio = mean([float(r["mean_delta_log_kappa2_recent_minus_old"]) for r in ps])
        mean_log_A_ratio = mean([float(r["mean_delta_log_A_proxy_recent_minus_old"]) for r in ps])
        mean_log_m_ratio = mean([float(r["mean_delta_log_m2_proxy_recent_minus_old"]) for r in ps])
        aggregate_logs = [
            float(r["log_mean_m2_proxy_recent"]) - float(r["log_mean_m2_proxy_old"]) for r in ps
        ]
        refs = [float(r["online_reference"]) for r in ps]
        pair_aggregate_ratios = [float(r["aggregate_m2_proxy_ratio_recent_over_old"]) for r in ps]
        n_ress_fav = sum(float(r["recent_median_relative_ess"]) > float(r["old_median_relative_ess"]) for r in ps)
        n_A_fav = sum(float(r["recent_mean_A_proxy"]) < float(r["old_mean_A_proxy"]) for r in ps)
        n_prompt_geom_m2_fav = sum(float(r["mean_delta_log_m2_proxy_recent_minus_old"]) < 0.0 for r in ps)
        n_agg_m2_fav = sum(float(r["aggregate_m2_proxy_ratio_recent_over_old"]) < 1.0 for r in ps)
        n_alpha_fav = sum(
            alpha_moves_favorable(float(r["online_reference"]), float(r["old_mean_alpha"]), float(r["recent_mean_alpha"]))
            for r in ps
        )
        # Publication-facing mechanism classification.  Corollary 2 concerns the
        # arithmetic mean of promptwise m2 coefficients, so pair_aggregate_ratios
        # (ratio of arithmetic prompt means) is the relevant aggregate leading-risk
        # proxy.  The geometric mean of promptwise ratios is retained only as a
        # heterogeneity diagnostic and must not be used to classify aggregate risk.
        if n_ress_fav == len(ps) and n_agg_m2_fav == len(ps) and n_alpha_fav < len(ps) / 2:
            mechanism_pattern = "C_overlap_dominates_adverse_reward_stratum_movement"
        elif n_ress_fav == len(ps) and n_agg_m2_fav == len(ps) and n_alpha_fav >= len(ps) / 2:
            mechanism_pattern = "B_overlap_and_reward_stratum_mostly_align"
        elif n_ress_fav == len(ps):
            mechanism_pattern = "A_overlap_repairs_but_aggregate_risk_not_uniformly_improved"
        else:
            mechanism_pattern = "mixed_or_unclassified"
        out.append({
            "split": split,
            "n_pairs": len(ps),
            "n_training_seeds": len({int(r["training_seed"]) for r in ps}),
            "n_prompt_pair_rows": len(prs),
            "pairs_rESS_improved": n_ress_fav,
            "pairs_reward_proxy_channel_favorable": n_A_fav,
            "pairs_prompt_geometric_m2_proxy_favorable": n_prompt_geom_m2_fav,
            "pairs_aggregate_m2_proxy_favorable": n_agg_m2_fav,
            "pairs_alpha_mean_moved_favorable": n_alpha_fav,
            "mean_old_rESS": mean([float(r["old_median_relative_ess"]) for r in ps]),
            "mean_recent_rESS": mean([float(r["recent_median_relative_ess"]) for r in ps]),
            "mean_old_alpha": mean([float(r["old_mean_alpha"]) for r in ps]),
            "mean_recent_alpha": mean([float(r["recent_mean_alpha"]) for r in ps]),
            "mean_old_A_proxy": mean([float(r["old_mean_A_proxy"]) for r in ps]),
            "mean_recent_A_proxy": mean([float(r["recent_mean_A_proxy"]) for r in ps]),
            "geometric_mean_kappa2_ratio_recent_over_old": safe_exp(mean_log_k_ratio),
            "geometric_mean_A_proxy_ratio_recent_over_old": safe_exp(mean_log_A_ratio),
            "geometric_mean_promptwise_m2_proxy_ratio_recent_over_old": safe_exp(mean_log_m_ratio),
            "mean_pair_aggregate_m2_proxy_ratio_recent_over_old": mean(pair_aggregate_ratios),
            "median_pair_aggregate_m2_proxy_ratio_recent_over_old": median(pair_aggregate_ratios),
            "geometric_mean_pair_aggregate_m2_proxy_ratio_recent_over_old": safe_exp(mean(aggregate_logs)),
            "mechanism_pattern": mechanism_pattern,
            "target_reference_min": min(refs),
            "target_reference_max": max(refs),
        })
    return out


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------


def run_analysis(
    *, root: Path, splits: Sequence[str], verify_shard_hashes: bool,
    max_runtime_hours: float, output_dir: Path | None = None,
) -> dict[str, Any]:
    start_wall = time.monotonic()
    deadline = start_wall + max_runtime_hours * 3600.0
    started_utc = now_utc()

    comparisons, input_provenance = load_comparisons(root, splits)
    outdir = output_dir or (root / "outputs" / "posthoc_alpha")
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("POST-HOC ALPHA MECHANISM ANALYSIS")
    print(f"root                   : {root}")
    print(f"splits                 : {list(splits)}")
    print(f"matched comparisons    : {len(comparisons)}")
    print(f"verify shard hashes    : {verify_shard_hashes}")
    print(f"wall-clock guard       : {max_runtime_hours:.2f} hours")
    print("calls model            : NO")
    print("uses CUDA              : NO")
    print("refits gate            : NO")
    print("selects new pairs      : NO (exact T04 prompt-WIS matched pairs only)")
    print("research status        : EXPLORATORY / POST-HOC")
    print("=" * 88)

    pair_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    for idx, c in enumerate(comparisons, start=1):
        check_deadline(deadline, f"comparison {idx}/{len(comparisons)}")
        print(
            f"[{idx:02d}/{len(comparisons):02d}] {c.split:<11} seed={c.training_seed} "
            f"target={c.target_step:04d} old={c.old_behavior_step:04d} recent={c.recent_behavior_step:04d}"
        )
        prow, details, source = analyze_comparison(
            root=root, c=c, verify_shard_hashes=verify_shard_hashes, deadline=deadline
        )
        pair_rows.append(prow)
        prompt_rows.extend(details)
        sources.append(source)

    split_rows = summarize_splits(pair_rows, prompt_rows)

    pair_path = outdir / "alpha_pair_summary.csv"
    split_path = outdir / "alpha_split_summary.csv"
    prompt_path = outdir / "alpha_prompt_detail.parquet"
    readme_path = outdir / "README.txt"
    manifest_path = outdir / "alpha_mechanism_manifest.json"

    atomic_write_csv(pair_path, pair_rows, PAIR_COLUMNS)
    atomic_write_csv(split_path, split_rows, SPLIT_COLUMNS)
    atomic_write_parquet(prompt_path, prompt_rows, PROMPT_COLUMNS)

    readme = "Post-hoc alpha mechanism analysis\n\n"
    readme += "Research status: EXPLORATORY / POST-HOC. Not a frozen confirmatory gate.\n"
    readme += "No model calls, no CUDA, no gate refit, and no new pair selection.\n\n"
    readme += "Definitions (prompt i):\n"
    readme += "  alpha_hat_i = sum_j W_ij^2 R_ij / sum_j W_ij^2\n"
    readme += "  log kappa2_hat_i = log[(1/K) sum_j W_ij^2]\n"
    readme += "  A_proxy_i(V) = V^2 + (1 - 2V) alpha_hat_i\n"
    readme += "  log m2_proxy_i = log kappa2_hat_i + log A_proxy_i(V)\n\n"
    readme += "The matched pair uses the same target and the same finite on-policy reference v.\n"
    readme += "Development results are the main exploratory mechanism analysis; test results are exploratory replication.\n"
    readme += "The program validates raw recomputed median rESS against T04 before accepting any mechanism result.\n"
    atomic_write_text(readme_path, readme)

    ended_utc = now_utc()
    elapsed = time.monotonic() - start_wall
    output_records = []
    for p in (pair_path, split_path, prompt_path, readme_path):
        output_records.append({
            "path": p.relative_to(root).as_posix() if root in p.parents else str(p),
            "sha256": sha256_file(p),
            "bytes": p.stat().st_size,
        })

    manifest: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "manifest_type": "posthoc_alpha_mechanism_manifest",
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "created_at_utc": ended_utc,
        "started_at_utc": started_utc,
        "runtime_seconds": elapsed,
        "max_runtime_hours_guard": max_runtime_hours,
        "research_status": "exploratory_post_hoc",
        "firewall": {
            "calls_model": False,
            "uses_cuda": False,
            "refits_gate": False,
            "mutates_upstream_results": False,
            "selects_new_pairs": False,
            "pair_selection_source": "exact prompt_wis matched-refresh rows already present in T04_rolling_comparison.csv",
            "verify_shard_hashes": verify_shard_hashes,
        },
        "mathematical_contract": {
            "alpha_hat_i": "sum_j W_ij^2 R_ij / sum_j W_ij^2",
            "log_kappa2_hat_i": "log((1/K) sum_j W_ij^2)",
            "A_proxy_i": "V^2 + (1 - 2V) alpha_hat_i, using the matched pair aggregate finite on-policy reference V from T04",
            "log_m2_proxy_i": "log_kappa2_hat_i + log(A_proxy_i)",
            "scope_note": "A_proxy/m2_proxy use aggregate V rather than latent prompt-specific v_i; they are post-hoc mechanism diagnostics, not promptwise theoretical coefficient estimates",
            "aggregation_contract": "For the prompt-averaged pWIS leading coefficient in Corollary 2, the publication-facing pair summary uses the arithmetic mean of promptwise m2 proxies. A geometric mean of promptwise m2 ratios is retained only as a heterogeneity diagnostic and is not used to classify aggregate risk.",
            "interpretation": "mechanism diagnostic only; not a finite-sample certificate and not confirmatory",
        },
        "inputs": input_provenance,
        "comparisons": {
            "count": len(comparisons),
            "development": sum(c.split == "development" for c in comparisons),
            "test": sum(c.split == "test" for c in comparisons),
            "source_records": sources,
        },
        "row_counts": {
            "alpha_pair_summary": len(pair_rows),
            "alpha_split_summary": len(split_rows),
            "alpha_prompt_detail": len(prompt_rows),
        },
        "outputs": output_records,
    }
    if Path(__file__).exists():
        manifest["source_code_sha256"] = sha256_file(Path(__file__).resolve())
    atomic_write_json(manifest_path, manifest)

    # Add the manifest itself to the final human-visible audit without trying to self-hash it.
    print("\nPOST-HOC ALPHA ANALYSIS PASSED")
    print(f"pair summary   : {pair_path}")
    print(f"split summary  : {split_path}")
    print(f"prompt detail  : {prompt_path}")
    print(f"manifest       : {manifest_path}")
    print(f"runtime        : {elapsed / 60.0:.2f} minutes")
    return manifest


# ---------------------------------------------------------------------------
# Built-in synthetic I/O self-test
# ---------------------------------------------------------------------------


def _write_synthetic_rescore_pair(
    root: Path, *, split: str, seed: int, b: int, e: int, purpose: str, K: int,
    prompts: Mapping[str, Sequence[tuple[float, float]]], pair_id: str,
) -> float:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise AlphaMechanismError(f"pyarrow is required for self-test: {exc}") from exc

    unit = pair_prefix(root, split, seed, b, e) / f"sample_block=0000-{K-1:04d}" / "shard=00000"
    unit.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    ress_values: list[float] = []
    for prompt_id, zr in prompts.items():
        if len(zr) != K:
            raise AlphaMechanismError("Synthetic prompt does not have K samples.")
        z = [float(x[0]) for x in zr]
        lse1 = logsumexp(z); lse2 = logsumexp([2*x for x in z])
        ress_values.append(math.exp(2*lse1-lse2)/K)
        for j, (zz, rr) in enumerate(zr):
            rows.append({
                "training_seed": seed, "split": split, "behavior_step": b,
                "target_step": e, "purpose": purpose, "prompt_id": prompt_id,
                "sample_index": j, "log_weight": float(zz), "correctness_reward": float(rr),
            })
    pp = unit / "rescored.parquet"
    pq.write_table(pa.Table.from_pylist(rows), pp, compression="zstd")
    mp = unit / "manifest.json"
    atomic_write_json(mp, {
        "schema_version": "1.0", "manifest_type": "target_rescore_shard", "pair_id": pair_id,
        "parquet": {"file_sha256": sha256_file(pp), "row_count": len(rows)},
    })
    return median(ress_values)


def run_self_test() -> int:
    try:
        import pyarrow  # type: ignore  # noqa: F401
        has_pyarrow = True
    except Exception:
        has_pyarrow = False

    if not has_pyarrow:
        # Math/control-flow fallback for environments that do not have the project's
        # Parquet dependency installed. The actual grpo_ope environment already
        # requires pyarrow for Programs 04/06.
        old_prompts = {
            "p1": PromptMechanism("p1", 0.90, 2.0, 0.10, 2.0 + math.log(16.0) + math.log(0.90), 2.0 + math.log(16.0) + math.log(0.10)),
            "p2": PromptMechanism("p2", 0.40, 1.0, 0.20, 1.0 + math.log(16.0) + math.log(0.40), 1.0 + math.log(16.0) + math.log(0.60)),
        }
        recent_prompts = {
            "p1": PromptMechanism("p1", 0.20, 0.1, 0.95, 0.1 + math.log(16.0) + math.log(0.20), 0.1 + math.log(16.0) + math.log(0.80)),
            "p2": PromptMechanism("p2", 0.30, 0.2, 0.90, 0.2 + math.log(16.0) + math.log(0.30), 0.2 + math.log(16.0) + math.log(0.70)),
        }
        original_load_pair = globals()["load_pair"]
        def fake_load_pair(**kwargs: Any) -> PairLoad:
            use_old = int(kwargs["behavior_step"]) == 0
            return PairLoad(
                prompts=old_prompts if use_old else recent_prompts,
                pair_id="selftest-old" if use_old else "selftest-recent",
                purpose="fixed" if use_old else "rolling",
                shard_count=1, parquet_count=1, source_fingerprint_sha256="selftest",
            )
        globals()["load_pair"] = fake_load_pair
        try:
            c = Comparison(
                "development", 20260826, 160, 0, 100, 0.25, 16, 2,
                median([0.10, 0.20]), median([0.95, 0.90]),
            )
            pair_row, detail, _ = analyze_comparison(
                root=Path("."), c=c, verify_shard_hashes=False, deadline=time.monotonic() + 30.0
            )
            if len(detail) != 2:
                raise AlphaMechanismError("Fallback self-test prompt-detail row count mismatch.")
            if not isclose(
                float(pair_row["mean_delta_log_m2_proxy_recent_minus_old"]),
                float(pair_row["mean_delta_log_kappa2_recent_minus_old"]) + float(pair_row["mean_delta_log_A_proxy_recent_minus_old"]),
                atol=1e-12, rtol=1e-12,
            ):
                raise AlphaMechanismError("Fallback self-test mechanism decomposition mismatch.")
        finally:
            globals()["load_pair"] = original_load_pair
        print("SELF-TEST PASSED (math/control-flow; pyarrow unavailable, Parquet I/O portion skipped)")
        return 0

    with tempfile.TemporaryDirectory(prefix="alpha_mechanism_selftest_") as td:
        root = Path(td)
        (root / "outputs" / "tables").mkdir(parents=True)
        (root / "outputs" / "diagnostics").mkdir(parents=True)
        K = 4
        seed = 20260826
        t04_rows: list[dict[str, Any]] = []
        for split in ("development", "test"):
            # Self-test only bypasses the paper-count invariant by calling analyze_comparison directly below.
            old_prompts = {
                "p1": [(0.0,1.0), (-1.0,0.0), (-2.0,0.0), (-3.0,0.0)],
                "p2": [(0.2,0.0), (0.0,1.0), (-0.2,0.0), (-0.4,1.0)],
            }
            recent_prompts = {
                "p1": [(0.0,1.0), (0.0,0.0), (0.0,0.0), (0.0,0.0)],
                "p2": [(0.0,0.0), (0.0,1.0), (0.0,0.0), (0.0,1.0)],
            }
            old_ress = _write_synthetic_rescore_pair(
                root, split=split, seed=seed, b=0, e=160, purpose="fixed", K=K,
                prompts=old_prompts, pair_id=f"{split}-old",
            )
            recent_ress = _write_synthetic_rescore_pair(
                root, split=split, seed=seed, b=100, e=160, purpose="rolling", K=K,
                prompts=recent_prompts, pair_id=f"{split}-recent",
            )
            t04_rows.append({
                "dataset":"GSM8K", "split":split, "training_seed":seed, "target_step":160,
                "recent_behavior_step":100, "estimator":"prompt_wis", "online_reference":0.25,
                "old_behavior_step":0, "old_estimate":0.0, "recent_estimate":0.0,
                "old_absolute_error":0.0, "recent_absolute_error":0.0,
                "delta_error_old_minus_recent":0.0,
                "old_median_relative_ess":old_ress, "recent_median_relative_ess":recent_ress,
                "delta_relative_ess_recent_minus_old":recent_ress-old_ress,
                "old_mean_d2":0.0, "recent_mean_d2":0.0,
                "old_mean_max_normalized_weight":0.0, "recent_mean_max_normalized_weight":0.0,
                "old_tokenwise_kl_proxy":0.0, "recent_tokenwise_kl_proxy":0.0,
                "K":K, "n_prompts":2,
            })
        atomic_write_csv(root/"outputs"/"tables"/"T04_rolling_comparison.csv", t04_rows, tuple(t04_rows[0].keys()))
        for split in ("development", "test"):
            atomic_write_json(root/"outputs"/"diagnostics"/f"program06_{split}_manifest.json", {
                "manifest_type":"program06_analysis_manifest", "schema_version":EXPECTED_PROGRAM06_SCHEMA,
                "split":split, "created_by":{"program":"synthetic"}, "protocol_config_sha256":"synthetic",
            })

        deadline = time.monotonic() + 300.0
        for split in ("development", "test"):
            c = Comparison(split, seed, 160, 0, 100, 0.25, K, 2,
                           float(t04_rows[0 if split=="development" else 1]["old_median_relative_ess"]),
                           float(t04_rows[0 if split=="development" else 1]["recent_median_relative_ess"]))
            pair_row, detail, _ = analyze_comparison(
                root=root, c=c, verify_shard_hashes=True, deadline=deadline
            )
            if len(detail) != 2:
                raise AlphaMechanismError("Self-test prompt-detail row count mismatch.")
            # p1 recent alpha is exactly 1/4 because squared weights are equal.
            p1 = next(r for r in detail if r["prompt_id"] == "p1")
            if not isclose(float(p1["recent_alpha"]), 0.25, atol=1e-14, rtol=0.0):
                raise AlphaMechanismError(f"Self-test alpha formula mismatch: {p1['recent_alpha']}")
            # Recent p1 weights are uniform, so rESS must be exactly 1.
            if not isclose(float(p1["recent_relative_ess"]), 1.0, atol=1e-14, rtol=0.0):
                raise AlphaMechanismError("Self-test rESS formula mismatch.")
            if not isclose(
                float(pair_row["mean_delta_log_m2_proxy_recent_minus_old"]),
                float(pair_row["mean_delta_log_kappa2_recent_minus_old"]) + float(pair_row["mean_delta_log_A_proxy_recent_minus_old"]),
                atol=1e-12, rtol=1e-12,
            ):
                raise AlphaMechanismError("Self-test mechanism decomposition mismatch.")
    print("SELF-TEST PASSED")
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Low-cost exploratory alpha/reward-stratum mechanism analysis from immutable Program 04/06 artifacts."
    )
    p.add_argument("--root", default=".", help="Project root, e.g. C:\\lsg\\grpo_ope_reuse")
    p.add_argument(
        "--splits", default="development,test",
        help="Comma-separated subset of development,test. Default: development,test",
    )
    p.add_argument(
        "--verify-shard-hashes", action="store_true",
        help="Deep audit: hash every rescored.parquet and compare to its shard manifest. Slower but still model-free.",
    )
    p.add_argument(
        "--max-runtime-hours", type=float, default=7.5,
        help="Wall-clock guard. Default 7.5 hours, leaving margin under an 8-hour budget.",
    )
    p.add_argument("--self-test", action="store_true", help="Run a synthetic I/O/math self-test and exit.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.max_runtime_hours <= 0.0 or args.max_runtime_hours > 7.9:
        raise AlphaMechanismError("--max-runtime-hours must be > 0 and <= 7.9.")
    splits = tuple(x.strip() for x in str(args.splits).split(",") if x.strip())
    if not splits or any(x not in DEFAULT_SPLITS for x in splits) or len(set(splits)) != len(splits):
        raise AlphaMechanismError("--splits must be a unique comma-separated subset of development,test")
    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise AlphaMechanismError(f"Project root does not exist: {root}")
    watchdog = start_hard_watchdog(float(args.max_runtime_hours))
    try:
        run_analysis(
            root=root,
            splits=splits,
            verify_shard_hashes=bool(args.verify_shard_hashes),
            max_runtime_hours=float(args.max_runtime_hours),
        )
    finally:
        watchdog.cancel()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AlphaMechanismError as exc:
        print(f"\nPOST-HOC ALPHA ANALYSIS FAILED\n{exc}", file=sys.stderr)
        raise SystemExit(2)
