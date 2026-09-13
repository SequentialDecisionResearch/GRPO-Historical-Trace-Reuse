#!/usr/bin/env python3
"""GRPO-OPE Program 01: immutable GSM8K split registry.

Research boundary
-----------------
This program does exactly one job: it converts the GSM8K snapshot pinned by
Program 00 into a deterministic, immutable prompt registry for the paper:

    7473 upstream train rows -> 6000 GRPO training + 1473 development
    1319 upstream test rows  -> 1319 official test

It does NOT construct prompts/chat templates, parse mathematical answers, train
GRPO, generate completions, compute rewards/OPE, fit a reuse gate, or inspect
any official-test result.  The registry deliberately stores hashes and source
row locations, not plaintext questions/answers, so later scripts must return to
the already-pinned raw GSM8K asset and explicitly choose an allowed split.

Determinism
-----------
* prompt_id is a stable SHA-256 identity based on pinned dataset revision,
  upstream split, source row index, and exact question hash.
* only the upstream *train* partition is subdivided.  Its assignment is obtained
  by sorting rows by SHA256(split_seed, prompt_id) and taking exactly the first
  6000 as research_split='training'; the remaining 1473 become 'development'.
* the upstream test partition is never mixed with training/development and is
  always research_split='test'.
* Python RNG state, row iteration timing, process count, and --mode do not affect
  the registry.

Immutability
------------
The first successful run writes:

    data/splits/gsm8k_split_registry.parquet
    manifests/split_registry_manifest.json

Both are content-hashed.  A later run verifies them and never silently rewrites
or re-splits them.  Partial publication can be repaired only with --resume and
only when regenerated content is identical to the already-frozen side.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import stat
import sys
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROGRAM = "01_prepare_splits.py"
PROGRAM_VERSION = "1.0.0"
MANIFEST_SCHEMA = "1.0"
REGISTRY_SCHEMA = "1.0"
SPLIT_ALGORITHM = "sha256_rank_v1"

PROJECT_NAME = "grpo_ope_reuse"
GSM8K_REPO = "openai/gsm8k"
GSM8K_CONFIG = "main"
EXPECTED_UPSTREAM_ROWS = {"train": 7473, "test": 1319}
EXPECTED_RESEARCH_ROWS = {"training": 6000, "development": 1473, "test": 1319}
REQUIRED_FIELDS = {"question", "answer"}
DEFAULT_SPLIT_SEED = 20260826

CORE_PACKAGES = (
    "torch",
    "transformers",
    "trl",
    "peft",
    "accelerate",
    "datasets",
    "huggingface_hub",
)
EXTRA_PACKAGES = ("tokenizers", "safetensors", "pyarrow", "numpy", "PyYAML")
HASH_EXCLUDE_DIRS = {".git", ".cache", "__pycache__"}

REGISTRY_COLUMNS = (
    "dataset",
    "dataset_revision",
    "source_row_index",
    "prompt_id",
    "original_split",
    "research_split",
    "question_hash",
    "gold_answer_hash",
    "split_assignment_hash",
)


class Program01Error(RuntimeError):
    """Controlled Program 01 failure with an actionable message."""


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        raise Program01Error(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise Program01Error(f"Expected a JSON object in {path}.")
    return obj


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def payload_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise Program01Error(f"Asset directory is missing: {root}")
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in HASH_EXCLUDE_DIRS for part in p.relative_to(root).parts):
            continue
        out.append(p)
    return sorted(out, key=lambda p: p.as_posix())


def tree_hash(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total = 0
    for p in payload_files(root):
        size = p.stat().st_size
        records.append({"path": rel(p, root), "size_bytes": size, "sha256": sha256_file(p)})
        total += size
    if not records:
        raise Program01Error(f"No files found in asset directory: {root}")
    return {
        "tree_sha256": sha256_bytes(canonical_bytes(records)),
        "file_count": len(records),
        "total_bytes": total,
        "files": records,
    }


def make_file_read_only(path: Path) -> None:
    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
    except OSError as exc:
        print(f"[WARN] Could not set read-only flag on {path}: {exc}")
        print("       Hash verification remains authoritative.")


def atomic_write_json_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Crash-safe publication.  Existing manifests are never overwritten."""
    if path.exists():
        raise Program01Error(f"Refusing to overwrite immutable manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            raise Program01Error(f"Manifest appeared concurrently: {path}")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def inspect_protocol_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "path": str(path),
            "sha256": None,
            "project_name": None,
            "protocol_version": None,
            "configured_split_seed": None,
        }
    if package_version("PyYAML") is None:
        raise Program01Error(f"{path} exists but PyYAML is not installed.")
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        raise Program01Error(f"Cannot parse protocol config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise Program01Error("protocol.yaml must contain a YAML mapping.")

    project = cfg.get("project") or {}
    splits = cfg.get("splits") or {}
    if project and not isinstance(project, dict):
        raise Program01Error("protocol.yaml: project must be a mapping.")
    if splits and not isinstance(splits, dict):
        raise Program01Error("protocol.yaml: splits must be a mapping when present.")

    project_name = project.get("name") if isinstance(project, dict) else None
    if project_name not in (None, PROJECT_NAME):
        raise Program01Error(f"Unexpected project.name={project_name!r}; expected {PROJECT_NAME!r}.")

    configured_seed = splits.get("seed") if isinstance(splits, dict) else None
    if configured_seed is not None:
        if isinstance(configured_seed, bool) or not isinstance(configured_seed, int):
            raise Program01Error("protocol.yaml: splits.seed must be an integer.")
        if configured_seed < 0:
            raise Program01Error("protocol.yaml: splits.seed must be >= 0.")

    return {
        "exists": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "project_name": project_name,
        "protocol_version": project.get("protocol_version") if isinstance(project, dict) else None,
        "configured_split_seed": configured_seed,
    }


def resolve_split_seed(
    cli_seed: int | None, config_info: Mapping[str, Any], frozen_seed: int | None = None
) -> tuple[int, str]:
    config_seed = config_info.get("configured_split_seed")
    if cli_seed is not None and cli_seed < 0:
        raise Program01Error("--split-seed must be >= 0.")
    if cli_seed is not None and config_seed is not None and cli_seed != config_seed:
        raise Program01Error(
            f"--split-seed={cli_seed} conflicts with protocol.yaml splits.seed={config_seed}."
        )
    if cli_seed is not None:
        return cli_seed, "cli"
    if isinstance(config_seed, int):
        return config_seed, "protocol.yaml:splits.seed"
    if frozen_seed is not None:
        if isinstance(frozen_seed, bool) or not isinstance(frozen_seed, int) or frozen_seed < 0:
            raise Program01Error("Existing split manifest contains an invalid split seed.")
        return frozen_seed, "frozen_split_manifest"
    return DEFAULT_SPLIT_SEED, "program_default"


def current_environment_fingerprint() -> tuple[str, dict[str, Any]]:
    """Reproduce Program 00's environment fingerprint basis exactly."""
    packages = {p: package_version(p) for p in CORE_PACKAGES + EXTRA_PACKAGES}
    missing = [p for p in CORE_PACKAGES if packages[p] is None]
    if missing:
        raise Program01Error(
            "Current environment is missing packages frozen/required by Program 00: " + ", ".join(missing)
        )
    if packages.get("pyarrow") is None:
        raise Program01Error(
            "pyarrow is required to create/read the immutable Parquet registry. "
            "Install the same project environment used by Program 00."
        )

    try:
        import torch  # type: ignore

        torch_cuda_build = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
    except Exception as exc:
        raise Program01Error(f"Cannot inspect frozen PyTorch environment: {exc}") from exc

    basis = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": packages,
        "torch_cuda_build": torch_cuda_build,
        "cudnn_version": cudnn_version,
    }
    return sha256_bytes(canonical_bytes(basis)), basis


def verify_environment_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise Program01Error(
            f"Missing {path}. Program 00 must complete successfully before Program 01."
        )
    env = read_json(path)
    if env.get("manifest_type") != "environment":
        raise Program01Error(f"Wrong manifest type in {path}.")
    expected = env.get("environment_fingerprint_sha256")
    observed, _ = current_environment_fingerprint()
    if expected != observed:
        raise Program01Error(
            "Current software environment differs from Program 00's frozen environment.\n"
            f"expected={expected}\nobserved={observed}\n"
            "Use the frozen environment or create a new project/protocol version."
        )
    return env


def verify_data_manifest(path: Path, gsm8k_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists():
        raise Program01Error(
            f"Missing {path}. Program 00 must freeze GSM8K before Program 01."
        )
    manifest = read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("manifest_type") != "data":
        raise Program01Error(f"Invalid Program 00 data manifest header: {path}")

    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(datasets.get("gsm8k"), dict):
        raise Program01Error("data_manifest.json lacks datasets.gsm8k.")
    gsm = datasets["gsm8k"]
    if gsm.get("repo_id") != GSM8K_REPO or gsm.get("role") != "primary_dataset":
        raise Program01Error("Frozen data manifest does not identify openai/gsm8k as the primary dataset.")
    revision = gsm.get("resolved_revision")
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise Program01Error("Frozen GSM8K resolved_revision is not a valid 40-character commit SHA.")

    scope = manifest.get("research_scope")
    expected_fingerprint = manifest.get("content_fingerprint_sha256")
    observed_fingerprint = sha256_bytes(
        canonical_bytes({"research_scope": scope, "datasets": datasets})
    )
    if expected_fingerprint != observed_fingerprint:
        raise Program01Error(
            "data_manifest.json content fingerprint mismatch; the Program 00 manifest appears to have changed."
        )

    observed_tree = tree_hash(gsm8k_dir)
    if observed_tree["tree_sha256"] != gsm.get("tree_sha256"):
        raise Program01Error(
            "Pinned GSM8K local files differ from Program 00's frozen tree hash.\n"
            f"expected={gsm.get('tree_sha256')}\nobserved={observed_tree['tree_sha256']}"
        )

    validation = gsm.get("validation")
    if not isinstance(validation, dict):
        raise Program01Error("Program 00 GSM8K record lacks validation metadata.")
    splits = validation.get("splits")
    if not isinstance(splits, dict):
        raise Program01Error("Program 00 GSM8K validation lacks split metadata.")
    for split, expected_rows in EXPECTED_UPSTREAM_ROWS.items():
        info = splits.get(split)
        if not isinstance(info, dict) or info.get("row_count") != expected_rows:
            raise Program01Error(
                f"Program 00 manifest does not confirm GSM8K {split} row_count={expected_rows}."
            )
    return manifest, gsm


def model_revision_for_header(path: Path) -> str:
    if not path.exists():
        return "missing-model-manifest"
    try:
        m = read_json(path)
        return str(m["models"]["primary"]["resolved_revision"])
    except Exception:
        return "invalid-model-manifest"


def discover_gsm8k_parquet(root: Path) -> dict[str, list[Path]]:
    train = sorted((root / GSM8K_CONFIG).glob("train*.parquet"))
    test = sorted((root / GSM8K_CONFIG).glob("test*.parquet"))
    if not train or not test:
        train = sorted(
            p for p in root.rglob("train*.parquet") if GSM8K_CONFIG in p.relative_to(root).parts
        )
        test = sorted(
            p for p in root.rglob("test*.parquet") if GSM8K_CONFIG in p.relative_to(root).parts
        )
    if not train or not test:
        raise Program01Error(
            "Pinned GSM8K snapshot does not contain main/train*.parquet and main/test*.parquet."
        )
    return {"train": train, "test": test}


def load_pinned_gsm8k(root: Path) -> dict[str, list[dict[str, str]]]:
    """Load only the exact local Parquet files.  No Hub/network lookup is performed."""
    files = discover_gsm8k_parquet(root)
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise Program01Error(f"Cannot import datasets: {exc}") from exc

    try:
        ds = load_dataset(
            "parquet",
            data_files={
                "train": [str(p) for p in files["train"]],
                "test": [str(p) for p in files["test"]],
            },
        )
    except Exception as exc:
        raise Program01Error(f"Cannot load pinned local GSM8K parquet snapshot: {exc}") from exc

    out: dict[str, list[dict[str, str]]] = {}
    for split, expected in EXPECTED_UPSTREAM_ROWS.items():
        d = ds[split]
        if len(d) != expected:
            raise Program01Error(f"GSM8K {split}: expected {expected} rows, found {len(d)}.")
        missing = REQUIRED_FIELDS - set(d.column_names)
        if missing:
            raise Program01Error(f"GSM8K {split}: missing required fields {sorted(missing)}.")
        rows: list[dict[str, str]] = []
        for i in range(len(d)):
            q = d[i]["question"]
            a = d[i]["answer"]
            if not isinstance(q, str) or not isinstance(a, str):
                raise Program01Error(f"GSM8K {split} row {i}: question/answer must both be strings.")
            rows.append({"question": q, "answer": a})
        out[split] = rows
    return out


def prompt_id_for(
    *, dataset_revision: str, original_split: str, source_row_index: int, question_hash: str
) -> str:
    basis = {
        "dataset": GSM8K_REPO,
        "dataset_config": GSM8K_CONFIG,
        "dataset_revision": dataset_revision,
        "original_split": original_split,
        "source_row_index": source_row_index,
        "question_hash": question_hash,
    }
    return sha256_bytes(canonical_bytes(basis))


def split_assignment_hash(split_seed: int, prompt_id: str) -> str:
    return sha256_bytes(canonical_bytes({"split_seed": split_seed, "prompt_id": prompt_id}))


def build_registry_rows(
    raw: Mapping[str, Sequence[Mapping[str, str]]], dataset_revision: str, split_seed: int
) -> list[dict[str, Any]]:
    if len(raw.get("train", ())) != EXPECTED_UPSTREAM_ROWS["train"]:
        raise Program01Error("Registry builder received the wrong upstream train row count.")
    if len(raw.get("test", ())) != EXPECTED_UPSTREAM_ROWS["test"]:
        raise Program01Error("Registry builder received the wrong upstream test row count.")

    train_rows: list[dict[str, Any]] = []
    for i, row in enumerate(raw["train"]):
        q = row["question"]
        a = row["answer"]
        qh = sha256_text(q)
        ah = sha256_text(a)
        pid = prompt_id_for(
            dataset_revision=dataset_revision,
            original_split="train",
            source_row_index=i,
            question_hash=qh,
        )
        train_rows.append(
            {
                "dataset": GSM8K_REPO,
                "dataset_revision": dataset_revision,
                "source_row_index": i,
                "prompt_id": pid,
                "original_split": "train",
                "research_split": None,
                "question_hash": qh,
                "gold_answer_hash": ah,
                "split_assignment_hash": split_assignment_hash(split_seed, pid),
            }
        )

    ranked = sorted(train_rows, key=lambda r: (r["split_assignment_hash"], r["prompt_id"]))
    training_ids = {r["prompt_id"] for r in ranked[: EXPECTED_RESEARCH_ROWS["training"]]}
    if len(training_ids) != EXPECTED_RESEARCH_ROWS["training"]:
        raise Program01Error("Unexpected prompt_id collision while selecting the 6000 training prompts.")
    for r in train_rows:
        r["research_split"] = "training" if r["prompt_id"] in training_ids else "development"

    test_rows: list[dict[str, Any]] = []
    for i, row in enumerate(raw["test"]):
        q = row["question"]
        a = row["answer"]
        qh = sha256_text(q)
        ah = sha256_text(a)
        pid = prompt_id_for(
            dataset_revision=dataset_revision,
            original_split="test",
            source_row_index=i,
            question_hash=qh,
        )
        test_rows.append(
            {
                "dataset": GSM8K_REPO,
                "dataset_revision": dataset_revision,
                "source_row_index": i,
                "prompt_id": pid,
                "original_split": "test",
                "research_split": "test",
                "question_hash": qh,
                "gold_answer_hash": ah,
                # Stored for schema uniformity/audit only.  It is NOT used to assign official test.
                "split_assignment_hash": split_assignment_hash(split_seed, pid),
            }
        )

    # Canonical file order is upstream train order followed by upstream test order.
    rows = train_rows + test_rows
    validate_registry_rows(rows, dataset_revision=dataset_revision, split_seed=split_seed)
    return rows


def validate_registry_rows(
    rows: Sequence[Mapping[str, Any]], *, dataset_revision: str, split_seed: int
) -> dict[str, Any]:
    expected_total = sum(EXPECTED_UPSTREAM_ROWS.values())
    if len(rows) != expected_total:
        raise Program01Error(f"Registry expected {expected_total} rows, found {len(rows)}.")

    prompt_ids = [r.get("prompt_id") for r in rows]
    if any(not isinstance(x, str) or re.fullmatch(r"[0-9a-f]{64}", x) is None for x in prompt_ids):
        raise Program01Error("Registry contains an invalid prompt_id.")
    if len(set(prompt_ids)) != len(prompt_ids):
        raise Program01Error("Registry prompt_id values are not unique.")

    original_counts = Counter(str(r.get("original_split")) for r in rows)
    research_counts = Counter(str(r.get("research_split")) for r in rows)
    if dict(original_counts) != EXPECTED_UPSTREAM_ROWS:
        raise Program01Error(
            f"Unexpected original split counts: {dict(original_counts)}; expected {EXPECTED_UPSTREAM_ROWS}."
        )
    if dict(research_counts) != EXPECTED_RESEARCH_ROWS:
        raise Program01Error(
            f"Unexpected research split counts: {dict(research_counts)}; expected {EXPECTED_RESEARCH_ROWS}."
        )

    for r in rows:
        if r.get("dataset") != GSM8K_REPO or r.get("dataset_revision") != dataset_revision:
            raise Program01Error("Registry contains a mixed dataset/revision.")
        original = r.get("original_split")
        research = r.get("research_split")
        if original == "test" and research != "test":
            raise Program01Error("Official GSM8K test row leaked into a non-test research split.")
        if original == "train" and research not in {"training", "development"}:
            raise Program01Error("Upstream train row received an invalid research split.")
        pid = r.get("prompt_id")
        if r.get("split_assignment_hash") != split_assignment_hash(split_seed, str(pid)):
            raise Program01Error("split_assignment_hash is inconsistent with split_seed + prompt_id.")
        for key in ("question_hash", "gold_answer_hash", "split_assignment_hash"):
            x = r.get(key)
            if not isinstance(x, str) or re.fullmatch(r"[0-9a-f]{64}", x) is None:
                raise Program01Error(f"Registry contains an invalid {key}.")

    # Re-derive the training membership from seed + prompt_id.  This catches any
    # accidental/manual change to research_split while preserving exact 6000/1473.
    train = [r for r in rows if r["original_split"] == "train"]
    ranked = sorted(train, key=lambda r: (r["split_assignment_hash"], r["prompt_id"]))
    expected_training = {r["prompt_id"] for r in ranked[: EXPECTED_RESEARCH_ROWS["training"]]}
    observed_training = {r["prompt_id"] for r in train if r["research_split"] == "training"}
    if expected_training != observed_training:
        raise Program01Error("Research split membership is not reproducible from split_seed + prompt_id.")

    q_counts = Counter(str(r["question_hash"]) for r in rows)
    duplicate_question_hash_rows = sum(n - 1 for n in q_counts.values() if n > 1)
    return {
        "status": "passed",
        "row_count": len(rows),
        "original_split_counts": dict(original_counts),
        "research_split_counts": dict(research_counts),
        "unique_prompt_ids": len(set(prompt_ids)),
        "duplicate_question_hash_extra_rows": duplicate_question_hash_rows,
        "plaintext_question_stored": False,
        "plaintext_gold_answer_stored": False,
        "assignment_rederived_from_seed_and_prompt_id": True,
        "official_test_disjoint": True,
    }


def registry_content_sha256(rows: Iterable[Mapping[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        canonical = {k: row[k] for k in REGISTRY_COLUMNS}
        h.update(canonical_bytes(canonical))
        h.update(b"\n")
    return h.hexdigest()


def write_registry_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise Program01Error(f"Refusing to overwrite immutable registry: {path}")
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise Program01Error(f"Cannot import pyarrow: {exc}") from exc

    schema = pa.schema(
        [
            pa.field("dataset", pa.string(), nullable=False),
            pa.field("dataset_revision", pa.string(), nullable=False),
            pa.field("source_row_index", pa.int64(), nullable=False),
            pa.field("prompt_id", pa.string(), nullable=False),
            pa.field("original_split", pa.string(), nullable=False),
            pa.field("research_split", pa.string(), nullable=False),
            pa.field("question_hash", pa.string(), nullable=False),
            pa.field("gold_answer_hash", pa.string(), nullable=False),
            pa.field("split_assignment_hash", pa.string(), nullable=False),
        ]
    )
    normalized = [{k: r[k] for k in REGISTRY_COLUMNS} for r in rows]
    table = pa.Table.from_pylist(normalized, schema=schema)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        pq.write_table(
            table,
            tmp,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
            row_group_size=1024,
        )
        # On Windows, fsync() can fail on a read-only file descriptor.
        # Reopen the completed Parquet file read/write, flush Python's buffer,
        # then fsync before the atomic os.replace().
        with tmp.open("rb+") as f:
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            raise Program01Error(f"Registry appeared concurrently: {path}")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def read_registry_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise Program01Error(f"Cannot import pyarrow: {exc}") from exc
    try:
        table = pq.read_table(path, columns=list(REGISTRY_COLUMNS))
    except Exception as exc:
        raise Program01Error(f"Cannot read split registry {path}: {exc}") from exc
    return table.to_pylist()


def make_manifest(
    *,
    root: Path,
    config_info: Mapping[str, Any],
    split_seed: int,
    split_seed_source: str,
    data_manifest: Mapping[str, Any],
    gsm_record: Mapping[str, Any],
    registry_path: Path,
    rows: Sequence[Mapping[str, Any]],
    validation: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    revision = str(gsm_record["resolved_revision"])
    registry = {
        "schema_version": REGISTRY_SCHEMA,
        "local_path": rel(registry_path, root),
        "format": "parquet",
        "columns": list(REGISTRY_COLUMNS),
        "row_count": len(rows),
        "file_size_bytes": registry_path.stat().st_size,
        "file_sha256": sha256_file(registry_path),
        "content_sha256": registry_content_sha256(rows),
        "validation": dict(validation),
    }
    source = {
        "dataset": GSM8K_REPO,
        "dataset_config": GSM8K_CONFIG,
        "dataset_revision": revision,
        "gsm8k_tree_sha256": gsm_record.get("tree_sha256"),
        "program00_data_manifest_path": "manifests/data_manifest.json",
        "program00_data_manifest_content_fingerprint_sha256": data_manifest.get(
            "content_fingerprint_sha256"
        ),
    }
    policy = {
        "algorithm": SPLIT_ALGORITHM,
        "split_seed": split_seed,
        "split_seed_source": split_seed_source,
        "upstream_train_rule": (
            "rank every upstream train row by SHA256(canonical_json({split_seed,prompt_id})); "
            "lowest 6000 -> training; remaining 1473 -> development"
        ),
        "upstream_test_rule": "all 1319 upstream test rows -> research_split='test'",
        "expected_research_rows": EXPECTED_RESEARCH_ROWS,
        "mode_invariant": True,
    }
    firewall = {
        "official_test_is_never_used_for_split_tuning": True,
        "official_test_count": EXPECTED_RESEARCH_ROWS["test"],
        "registry_contains_plaintext_question": False,
        "registry_contains_plaintext_gold_answer": False,
        "test_reward_or_parser_output_present": False,
        "note": (
            "Program 01 records only source location and integrity hashes. Downstream development code "
            "must not dereference research_split='test' before the frozen-protocol test stage."
        ),
    }
    fingerprint_basis = {
        "source": source,
        "split_policy": policy,
        "registry_content_sha256": registry["content_sha256"],
        "registry_file_sha256": registry["file_sha256"],
        "research_firewall": firewall,
    }
    return {
        "schema_version": MANIFEST_SCHEMA,
        "manifest_type": "split_registry",
        "created_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "creation_mode": mode,
        "protocol_config_observed": {
            "path": config_info.get("path"),
            "sha256_at_creation": config_info.get("sha256"),
            "project_name": config_info.get("project_name"),
            "protocol_version_at_creation": config_info.get("protocol_version"),
            "note": "Full protocol.yaml is not frozen by Program 01; only the split policy is immutable here.",
        },
        "source": source,
        "split_policy": policy,
        "registry": registry,
        "research_firewall": firewall,
        "content_fingerprint_sha256": sha256_bytes(canonical_bytes(fingerprint_basis)),
    }


def verify_split_manifest(
    *,
    manifest_path: Path,
    registry_path: Path,
    data_manifest: Mapping[str, Any],
    gsm_record: Mapping[str, Any],
    requested_seed: int,
) -> dict[str, Any]:
    m = read_json(manifest_path)
    if m.get("schema_version") != MANIFEST_SCHEMA or m.get("manifest_type") != "split_registry":
        raise Program01Error(f"Invalid split registry manifest header: {manifest_path}")

    source = m.get("source")
    policy = m.get("split_policy")
    registry = m.get("registry")
    firewall = m.get("research_firewall")
    if not all(isinstance(x, dict) for x in (source, policy, registry, firewall)):
        raise Program01Error("split_registry_manifest.json is missing required sections.")

    if source.get("dataset") != GSM8K_REPO:
        raise Program01Error("Split registry manifest points to an unexpected dataset.")
    if source.get("dataset_revision") != gsm_record.get("resolved_revision"):
        raise Program01Error("Split registry dataset revision differs from Program 00's frozen revision.")
    if source.get("gsm8k_tree_sha256") != gsm_record.get("tree_sha256"):
        raise Program01Error("Split registry source tree hash differs from Program 00's frozen GSM8K tree.")
    if source.get("program00_data_manifest_content_fingerprint_sha256") != data_manifest.get(
        "content_fingerprint_sha256"
    ):
        raise Program01Error("Split registry was created from a different Program 00 data manifest.")
    if policy.get("algorithm") != SPLIT_ALGORITHM:
        raise Program01Error("Unexpected split algorithm in frozen manifest.")
    if policy.get("split_seed") != requested_seed:
        raise Program01Error(
            f"Split registry is already frozen with seed={policy.get('split_seed')}; "
            f"requested/resolved seed={requested_seed}. Create a new project/protocol version instead."
        )
    if policy.get("expected_research_rows") != EXPECTED_RESEARCH_ROWS:
        raise Program01Error("Frozen split manifest has unexpected research split counts/policy.")
    if firewall.get("official_test_is_never_used_for_split_tuning") is not True:
        raise Program01Error("Frozen split manifest does not preserve the official-test firewall.")

    if not registry_path.exists():
        raise Program01Error(f"Frozen manifest exists but registry file is missing: {registry_path}")
    observed_file_sha = sha256_file(registry_path)
    if observed_file_sha != registry.get("file_sha256"):
        raise Program01Error(
            "Immutable split registry file hash mismatch.\n"
            f"expected={registry.get('file_sha256')}\nobserved={observed_file_sha}"
        )
    rows = read_registry_parquet(registry_path)
    validation = validate_registry_rows(
        rows,
        dataset_revision=str(gsm_record["resolved_revision"]),
        split_seed=requested_seed,
    )
    observed_content = registry_content_sha256(rows)
    if observed_content != registry.get("content_sha256"):
        raise Program01Error(
            "Immutable split registry content hash mismatch.\n"
            f"expected={registry.get('content_sha256')}\nobserved={observed_content}"
        )
    if registry.get("row_count") != len(rows):
        raise Program01Error("Frozen split manifest row_count does not match the registry.")
    old_validation = registry.get("validation")
    if isinstance(old_validation, dict):
        for key in (
            "row_count",
            "original_split_counts",
            "research_split_counts",
            "unique_prompt_ids",
            "official_test_disjoint",
        ):
            if old_validation.get(key) != validation.get(key):
                raise Program01Error(f"Frozen split validation field changed: {key}.")

    fingerprint_basis = {
        "source": source,
        "split_policy": policy,
        "registry_content_sha256": registry.get("content_sha256"),
        "registry_file_sha256": registry.get("file_sha256"),
        "research_firewall": firewall,
    }
    observed_manifest_fp = sha256_bytes(canonical_bytes(fingerprint_basis))
    if m.get("content_fingerprint_sha256") != observed_manifest_fp:
        raise Program01Error("split_registry_manifest.json content fingerprint mismatch.")
    return m


def regenerate_to_temp(
    *,
    root: Path,
    gsm8k_dir: Path,
    dataset_revision: str,
    split_seed: int,
) -> tuple[Path, list[dict[str, Any]], dict[str, Any]]:
    raw = load_pinned_gsm8k(gsm8k_dir)
    rows = build_registry_rows(raw, dataset_revision, split_seed)
    validation = validate_registry_rows(rows, dataset_revision=dataset_revision, split_seed=split_seed)
    temp_dir = root / "data" / "splits"
    temp_dir.mkdir(parents=True, exist_ok=True)
    tmp = temp_dir / f".gsm8k_split_registry.rebuild-{uuid.uuid4().hex}.parquet"
    write_registry_parquet(tmp, rows)
    return tmp, rows, validation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="GRPO-OPE Program 01: create/verify the immutable GSM8K split registry."
    )
    p.add_argument("--config", default="configs/protocol.yaml")
    p.add_argument("--mode", choices=("smoke", "pilot", "paper"), default="paper")
    p.add_argument("--output-root", default=".")
    p.add_argument(
        "--split-seed",
        type=int,
        default=None,
        help=(
            "Immutable train/development split seed. Priority: CLI, then protocol.yaml splits.seed, "
            f"then fixed default {DEFAULT_SPLIT_SEED}."
        ),
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Safely repair a partial Program 01 publication; never changes a completed frozen registry.",
    )
    p.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify Program 00 assets plus the existing frozen split registry; write nothing.",
    )
    p.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Accepted for CLI consistency. Program 01 is CPU-only and rejects cuda.",
    )
    args = p.parse_args(argv)
    if args.device != "cpu":
        p.error("Program 01 is a deterministic CPU data-registry step; use --device cpu.")
    if args.split_seed is not None and args.split_seed < 0:
        p.error("--split-seed must be >= 0")
    if args.verify_only and args.resume:
        p.error("--verify-only and --resume are mutually exclusive")
    return args


def print_header(
    *,
    root: Path,
    mode: str,
    config_info: Mapping[str, Any],
    data_revision: str,
    model_revision: str,
    split_seed: int,
    registry_path: Path,
    manifest_path: Path,
) -> None:
    complete = registry_path.exists() and manifest_path.exists()
    unfinished = 0 if complete else 1
    print("=" * 82)
    print("GRPO-OPE Program 01 — Immutable Split Registry")
    print(f"project root       : {root}")
    print(f"mode               : {mode} (does not change the canonical split)")
    print(f"protocol version   : {config_info.get('protocol_version') or 'not locked'}")
    print(f"config hash        : {config_info.get('sha256') or 'none'}")
    print(f"GSM8K revision     : {data_revision}")
    print(f"model revision     : {model_revision}")
    print(f"split / seed       : all / {split_seed}")
    print(f"unfinished units   : {unfinished}")
    print(f"registry path      : {registry_path}")
    print(f"manifest path      : {manifest_path}")
    print("research boundary  : split selection + integrity only; no reward/OPE/test tuning")
    print("=" * 82)


def first_run(
    *,
    root: Path,
    config_info: Mapping[str, Any],
    mode: str,
    split_seed: int,
    split_seed_source: str,
    data_manifest: Mapping[str, Any],
    gsm_record: Mapping[str, Any],
    gsm8k_dir: Path,
    registry_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    if registry_path.exists() or manifest_path.exists():
        raise Program01Error("Internal error: first_run called with pre-existing Program 01 output.")

    raw = load_pinned_gsm8k(gsm8k_dir)
    revision = str(gsm_record["resolved_revision"])
    rows = build_registry_rows(raw, revision, split_seed)
    validation = validate_registry_rows(rows, dataset_revision=revision, split_seed=split_seed)

    write_registry_parquet(registry_path, rows)
    try:
        manifest = make_manifest(
            root=root,
            config_info=config_info,
            split_seed=split_seed,
            split_seed_source=split_seed_source,
            data_manifest=data_manifest,
            gsm_record=gsm_record,
            registry_path=registry_path,
            rows=rows,
            validation=validation,
            mode=mode,
        )
        atomic_write_json_once(manifest_path, manifest)
    except Exception:
        # The registry is deterministic and can be safely adopted with --resume.
        # Never silently delete it here; preserving it makes crash state auditable.
        raise

    make_file_read_only(registry_path)
    make_file_read_only(manifest_path)
    return verify_split_manifest(
        manifest_path=manifest_path,
        registry_path=registry_path,
        data_manifest=data_manifest,
        gsm_record=gsm_record,
        requested_seed=split_seed,
    )


def resume_partial(
    *,
    root: Path,
    config_info: Mapping[str, Any],
    mode: str,
    split_seed: int,
    split_seed_source: str,
    data_manifest: Mapping[str, Any],
    gsm_record: Mapping[str, Any],
    gsm8k_dir: Path,
    registry_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    revision = str(gsm_record["resolved_revision"])

    # Case A: registry was published, crash occurred before manifest publication.
    if registry_path.exists() and not manifest_path.exists():
        existing_rows = read_registry_parquet(registry_path)
        validation = validate_registry_rows(
            existing_rows, dataset_revision=revision, split_seed=split_seed
        )

        # Rebuild from pinned raw data and compare semantic content, not just row counts.
        raw = load_pinned_gsm8k(gsm8k_dir)
        expected_rows = build_registry_rows(raw, revision, split_seed)
        if registry_content_sha256(existing_rows) != registry_content_sha256(expected_rows):
            raise Program01Error(
                "Orphan registry does not equal deterministic regeneration from pinned GSM8K; refusing adoption."
            )
        manifest = make_manifest(
            root=root,
            config_info=config_info,
            split_seed=split_seed,
            split_seed_source=split_seed_source,
            data_manifest=data_manifest,
            gsm_record=gsm_record,
            registry_path=registry_path,
            rows=existing_rows,
            validation=validation,
            mode=mode,
        )
        atomic_write_json_once(manifest_path, manifest)
        make_file_read_only(registry_path)
        make_file_read_only(manifest_path)
        return verify_split_manifest(
            manifest_path=manifest_path,
            registry_path=registry_path,
            data_manifest=data_manifest,
            gsm_record=gsm_record,
            requested_seed=split_seed,
        )

    # Case B: manifest exists, registry was lost after publication.  Regenerate,
    # require both semantic and exact file hash agreement with the frozen manifest,
    # then restore.  Environment verification above makes exact Parquet reproduction
    # a reasonable hard requirement instead of silently changing provenance.
    if manifest_path.exists() and not registry_path.exists():
        frozen = read_json(manifest_path)
        policy = frozen.get("split_policy")
        registry = frozen.get("registry")
        source = frozen.get("source")
        if not all(isinstance(x, dict) for x in (policy, registry, source)):
            raise Program01Error("Cannot resume from malformed split manifest.")
        if policy.get("split_seed") != split_seed:
            raise Program01Error("Frozen split seed differs from requested seed; cannot restore registry.")
        if source.get("dataset_revision") != revision:
            raise Program01Error("Frozen split manifest refers to a different GSM8K revision.")

        tmp, rows, _ = regenerate_to_temp(
            root=root, gsm8k_dir=gsm8k_dir, dataset_revision=revision, split_seed=split_seed
        )
        try:
            if registry_content_sha256(rows) != registry.get("content_sha256"):
                raise Program01Error("Regenerated registry content differs from frozen manifest.")
            if sha256_file(tmp) != registry.get("file_sha256"):
                raise Program01Error(
                    "Regenerated Parquet bytes differ from frozen manifest. "
                    "Do not replace it under a changed writer/environment; restore the original frozen environment/file."
                )
            registry_path.parent.mkdir(parents=True, exist_ok=True)
            os.replace(tmp, registry_path)
        finally:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        make_file_read_only(registry_path)
        return verify_split_manifest(
            manifest_path=manifest_path,
            registry_path=registry_path,
            data_manifest=data_manifest,
            gsm_record=gsm_record,
            requested_seed=split_seed,
        )

    raise Program01Error("--resume requested, but Program 01 is not in a recognized partial state.")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    config = Path(args.config).expanduser()
    if not config.is_absolute():
        config = root / config
    config_info = inspect_protocol_yaml(config)

    manifests_dir = root / "manifests"
    gsm8k_dir = root / "data" / "raw" / "gsm8k"
    data_manifest_path = manifests_dir / "data_manifest.json"
    model_manifest_path = manifests_dir / "model_manifest.json"
    environment_manifest_path = manifests_dir / "environment_manifest.json"
    registry_path = root / "data" / "splits" / "gsm8k_split_registry.parquet"
    split_manifest_path = manifests_dir / "split_registry_manifest.json"

    frozen_seed = None
    if split_manifest_path.exists():
        try:
            frozen_policy = read_json(split_manifest_path).get("split_policy")
            if isinstance(frozen_policy, dict):
                frozen_seed = frozen_policy.get("split_seed")
        except Program01Error:
            # Full manifest validation below will provide the authoritative error.
            frozen_seed = None
    split_seed, split_seed_source = resolve_split_seed(
        args.split_seed, config_info, frozen_seed=frozen_seed
    )

    # Program 00 is a hard dependency.  Verify the environment and the exact local
    # GSM8K tree before even looking at an existing Program 01 output.
    verify_environment_manifest(environment_manifest_path)
    data_manifest, gsm_record = verify_data_manifest(data_manifest_path, gsm8k_dir)
    data_revision = str(gsm_record["resolved_revision"])
    model_revision = model_revision_for_header(model_manifest_path)

    print_header(
        root=root,
        mode=args.mode,
        config_info=config_info,
        data_revision=data_revision,
        model_revision=model_revision,
        split_seed=split_seed,
        registry_path=registry_path,
        manifest_path=split_manifest_path,
    )

    registry_exists = registry_path.exists()
    manifest_exists = split_manifest_path.exists()

    if registry_exists and manifest_exists:
        manifest = verify_split_manifest(
            manifest_path=split_manifest_path,
            registry_path=registry_path,
            data_manifest=data_manifest,
            gsm_record=gsm_record,
            requested_seed=split_seed,
        )
        print(f"[OK] frozen split registry verified: {registry_path}")
        print(f"[OK] split registry manifest verified: {split_manifest_path}")
    elif args.verify_only:
        raise Program01Error(
            "--verify-only requires both the frozen registry and split_registry_manifest.json."
        )
    elif registry_exists != manifest_exists:
        if not args.resume:
            present = str(registry_path if registry_exists else split_manifest_path)
            missing = str(split_manifest_path if registry_exists else registry_path)
            raise Program01Error(
                "Partial Program 01 publication detected.\n"
                f"present={present}\nmissing={missing}\n"
                "Re-run with --resume to perform content-verified recovery."
            )
        manifest = resume_partial(
            root=root,
            config_info=config_info,
            mode=args.mode,
            split_seed=split_seed,
            split_seed_source=split_seed_source,
            data_manifest=data_manifest,
            gsm_record=gsm_record,
            gsm8k_dir=gsm8k_dir,
            registry_path=registry_path,
            manifest_path=split_manifest_path,
        )
        print("[OK] partial Program 01 publication recovered without changing frozen content.")
    else:
        manifest = first_run(
            root=root,
            config_info=config_info,
            mode=args.mode,
            split_seed=split_seed,
            split_seed_source=split_seed_source,
            data_manifest=data_manifest,
            gsm_record=gsm_record,
            gsm8k_dir=gsm8k_dir,
            registry_path=registry_path,
            manifest_path=split_manifest_path,
        )
        print(f"[WRITE] {registry_path}")
        print(f"[WRITE] {split_manifest_path}")

    counts = manifest["registry"]["validation"]["research_split_counts"]
    print("-" * 82)
    print("PROGRAM 01 PASSED")
    print(f"split seed           : {manifest['split_policy']['split_seed']}")
    print(f"training prompts     : {counts['training']}")
    print(f"development prompts  : {counts['development']}")
    print(f"official test prompts: {counts['test']}")
    print(f"registry SHA-256      : {manifest['registry']['file_sha256']}")
    print(f"content SHA-256       : {manifest['registry']['content_sha256']}")
    print("next step             : Experiment 0 correctness gate / development GPU pilot")
    print("-" * 82)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Program01Error as exc:
        print(f"\n[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(2)
    except KeyboardInterrupt:
        print("\n[FAIL] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
