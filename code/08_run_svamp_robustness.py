#!/usr/bin/env python3
"""GRPO-OPE Program 08: frozen SVAMP external robustness evaluation.

Research role
-------------
Program 08 is deliberately *not* a second main experiment. It reuses the
already-trained GRPO checkpoints and the already-frozen reuse gate to test
external prompt-distribution robustness on pinned SVAMP.

The program performs, only for a small frozen representative pair registry:

    frozen GSM8K-development pair selection (before SVAMP row content is parsed)
        -> SVAMP behavior generation
        -> exact target teacher-forcing rescoring
        -> independent target on-policy Monte Carlo reference
        -> IS / prompt-normalized WIS + overlap diagnostics
        -> application of the *already frozen* rESS gate and KL baseline
        -> T08_svamp.csv + audit manifests

Hard research invariants
------------------------
* No GRPO training.
* No gate refitting, threshold override, or target-pair tuning.
* No pair selection using SVAMP questions, rewards, OPE errors, or diagnostics.
* Representative pair templates are frozen from GSM8K development diagnostics
  before any SVAMP row content or reward is parsed.
* The frozen pair registry covers high / medium / low overlap using only the
  calibration seeds from the frozen gate.
* The same model revision, tokenizer/chat template, prompt template, sampling
  policy, LoRA adapters, K/L rules, parser and correctness reward are reused.
* Exact behavior token log-probabilities are captured at generation time.
* Target rescoring teacher-forces the exact saved prompt/completion token IDs.
* Identity rescoring is rechecked on SVAMP for every behavior checkpoint used.
* All OPE/overlap calculations start in log-space. Ordinary IS is never saved
  by clipping extreme weights.
* Bootstrap resamples prompts as clusters.
* Expensive behavior/rescore/online prompt shards are immutable, content-
  hashed, atomically published, and resumeable.
* Paper robustness runs always evaluate all frozen training seeds and all 1000
  pinned SVAMP prompts. Engineering smoke/pilot outputs are isolated.

This script is intentionally self-contained so the final robustness result can
be audited without editing Programs 03-07.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import math
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

PROGRAM = "08_run_svamp_robustness.py"
PROGRAM_VERSION = "1.0.0"
PROJECT_NAME = "grpo_ope_reuse"
MANIFEST_SCHEMA = "1.0"
ROBUSTNESS_SCHEMA = "grpo_ope_program08_v1"
SVAMP_PAIR_SCHEMA = "1.0"

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
SVAMP_EXPECTED_ROWS = 1000
DEFAULT_SEEDS = (20260826, 20260827, 20260828)
DEFAULT_REPRESENTATIVE_PAIRS = 8
DEFAULT_BOOTSTRAP_REPS = 2000
DEFAULT_BOOTSTRAP_SEED = 20260826
DEFAULT_SAMPLE_BLOCK_SIZE = 8
DEFAULT_PROMPTS_PER_SHARD = 8
DEFAULT_GENERATION_BATCH_SIZE = 4
PAPER_BEHAVIOR_K_ALLOWED = (16, 32)
PAPER_L_MAIN_ALLOWED = (8, 16)
PAPER_L_AUDIT = 32
PAPER_AUDIT_STEPS = (0, 100, 200, 300, 400)

CORE_PACKAGES = (
    "torch", "transformers", "trl", "peft", "accelerate", "datasets", "huggingface_hub"
)
EXTRA_PACKAGES = ("tokenizers", "safetensors", "pyarrow", "numpy", "PyYAML")
HASH_EXCLUDE_DIRS = {".git", ".cache", "__pycache__"}

PROMPT_TEMPLATE = (
    "Solve the following math problem. Show your reasoning briefly.\n"
    "Put only the final numerical answer inside <answer>...</answer>.\n\n"
    "Problem:\n{question}"
)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d[\d,]*)?")
HEX64_RE = re.compile(r"[0-9a-f]{64}")

T08_COLUMNS = (
    "dataset", "dataset_revision", "training_seed", "behavior_step", "target_step",
    "pair_purpose", "overlap_band", "representative_rank", "estimator",
    "estimate", "estimate_status", "online_reference", "signed_error", "absolute_error",
    "ope_ci_low", "ope_ci_high", "online_ci_low", "online_ci_high",
    "signed_error_ci_low", "signed_error_ci_high", "absolute_error_ci_low",
    "absolute_error_ci_high", "bootstrap_reps", "bootstrap_nonfinite_fraction",
    "n_prompts", "K", "L", "median_prompt_ess", "median_prompt_relative_ess",
    "p10_prompt_relative_ess", "mean_d2", "median_d2", "mean_max_normalized_weight",
    "p90_max_normalized_weight", "tokenwise_kl_proxy", "mean_abs_log_weight",
    "mean_log_weight", "sd_log_weight", "mean_completion_length", "truncation_rate",
    "gate_kind", "gate_tolerance", "gate_diagnostic", "gate_threshold", "gate_decision",
    "gate_accepted", "gate_reliable", "gate_false_accept", "gate_false_reject",
)

BEHAVIOR_COLUMNS = (
    "trajectory_id", "dataset", "dataset_revision", "protocol_version", "training_seed",
    "behavior_step", "prompt_id", "source_row_index", "sample_index", "prompt_token_ids",
    "completion_token_ids", "behavior_token_logprobs", "behavior_sequence_logprob",
    "completion_length", "terminated_with_eos", "was_truncated", "parsed_answer",
    "correctness_reward", "parser_status", "temperature", "top_p", "top_k",
    "repetition_penalty", "max_completion_length", "generation_seed", "generation_call_id",
    "model_revision", "tokenizer_revision", "chat_template_hash", "prompt_template_hash",
    "behavior_adapter_sha256", "generation_config_sha256", "behavior_logprob_source",
    "completion_text",
)

RESCORE_COLUMNS = (
    "rescore_id", "trajectory_id", "dataset", "dataset_revision", "protocol_version",
    "training_seed", "behavior_step", "target_step", "prompt_id", "sample_index",
    "behavior_adapter_sha256", "target_adapter_sha256", "behavior_sequence_logprob",
    "target_sequence_logprob", "log_weight", "mean_log_ratio_per_token",
    "completion_length", "correctness_reward", "terminated_with_eos", "was_truncated",
    "identity_abs_sequence_logprob_diff", "identity_max_abs_token_logprob_diff",
    "identity_mean_abs_token_logprob_diff", "identity_pass", "source_behavior_content_sha256",
)

ONLINE_COLUMNS = (
    "online_id", "dataset", "dataset_revision", "protocol_version", "training_seed",
    "target_step", "prompt_id", "source_row_index", "sample_index", "completion_token_ids",
    "completion_length", "terminated_with_eos", "was_truncated", "parsed_answer", "correct",
    "parser_status", "temperature", "top_p", "top_k", "repetition_penalty",
    "max_completion_length", "generation_seed", "generation_call_id", "model_revision",
    "tokenizer_revision", "chat_template_hash", "prompt_template_hash",
    "target_adapter_sha256", "generation_config_sha256", "completion_text",
)


class Program08Error(RuntimeError):
    """Controlled Program 08 failure with an actionable message."""


@dataclass(frozen=True)
class RobustnessSpec:
    seeds: tuple[int, ...]
    calibration_seeds: tuple[int, ...]
    representative_pairs: int
    k: int
    l_main: int
    l_audit: int
    audit_steps: tuple[int, ...]
    bootstrap_reps: int
    bootstrap_seed: int
    sample_block_size: int
    prompts_per_shard: int
    generation_batch_size: int
    max_completion_length: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    source_row_index: int
    source_id: str
    question: str
    raw_gold_answer: str
    gold_answer: str
    svamp_type: str


@dataclass(frozen=True)
class AdapterRecord:
    step: int
    path: Path
    payload_sha256: str
    manifest: Mapping[str, Any]


@dataclass(frozen=True)
class PairTemplate:
    behavior_step: int
    target_step: int
    purpose: str
    representative_rank: int
    overlap_band: str
    source_median_relative_ess: float
    source_tokenwise_kl_proxy: float


# ---------------------------------------------------------------------------
# Generic integrity utilities
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


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
    if not path.exists():
        raise Program08Error(f"Missing required JSON file: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        raise Program08Error(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise Program08Error(f"Expected JSON object in {path}")
    return obj


def fsync_directory(path: Path) -> None:
    try:
        fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(path: Path, obj: Mapping[str, Any]) -> None:
    atomic_write_bytes(path, json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n")


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in columns})
    atomic_write_bytes(path, buf.getvalue().encode("utf-8"))


def rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def make_read_only(path: Path) -> None:
    if not path.exists():
        return
    targets = [path]
    if path.is_dir():
        targets.extend(sorted(path.rglob("*"), reverse=True))
    for p in targets:
        try:
            mode = stat.S_IMODE(p.stat().st_mode)
            p.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        except OSError:
            pass


def payload_files(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.is_dir():
        raise Program08Error(f"Asset directory is missing: {root}")
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
        raise Program08Error(f"No payload files under {root}")
    return {
        "tree_sha256": sha256_bytes(canonical_bytes(records)),
        "file_count": len(records),
        "total_bytes": total,
        "files": records,
    }


def adapter_payload_hash(root: Path) -> str:
    excluded = {"adapter_manifest.json", "program02_resume_manifest.json", "trainer_state.json"}
    records: list[dict[str, Any]] = []
    for p in payload_files(root):
        if p.name in excluded:
            continue
        records.append({"path": rel(p, root), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)})
    if not records:
        raise Program08Error(f"Adapter directory has no payload files: {root}")
    return sha256_bytes(canonical_bytes(records))


def git_commit(root: Path) -> str | None:
    try:
        cp = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, timeout=5)
    except Exception:
        return None
    s = cp.stdout.strip()
    return s if cp.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", s) else None


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def current_environment_fingerprint() -> tuple[str, dict[str, Any]]:
    packages = {name: package_version(name) for name in (*CORE_PACKAGES, *EXTRA_PACKAGES)}
    if packages.get("pyarrow") is None:
        raise Program08Error("pyarrow is required for immutable Program 08 Parquet assets.")
    try:
        import torch  # type: ignore
        cuda_build = torch.version.cuda
        cudnn = torch.backends.cudnn.version()
    except Exception as exc:
        raise Program08Error(f"Cannot inspect PyTorch environment: {exc}") from exc
    basis = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": packages,
        "torch_cuda_build": cuda_build,
        "cudnn_version": cudnn,
    }
    return sha256_bytes(canonical_bytes(basis)), basis


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise Program08Error(f"Missing CSV: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_parquet(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise Program08Error(f"pyarrow is required: {exc}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(r) for r in rows])
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    with path.open("rb") as f:
        os.fsync(f.fileno())


def read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise Program08Error(f"pyarrow is required: {exc}") from exc
    return pq.read_table(path).to_pylist()


def rows_hash(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], sort_keys: Sequence[str]) -> str:
    normalized = [{k: r.get(k) for k in columns} for r in rows]
    normalized.sort(key=lambda r: tuple(r.get(k) for k in sort_keys))
    return sha256_bytes(canonical_bytes(normalized))


def first_present(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        cur: Any = mapping
        ok = True
        for key in path:
            if not isinstance(cur, Mapping) or key not in cur:
                ok = False
                break
            cur = cur[key]
        if ok:
            return cur
    return None


# ---------------------------------------------------------------------------
# Protocol / frozen upstream verification
# ---------------------------------------------------------------------------


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise Program08Error(f"Missing protocol file: {path}")
    raw = path.read_bytes()
    try:
        import yaml  # type: ignore
        obj = yaml.safe_load(raw)
    except Exception as exc:
        raise Program08Error(f"Cannot parse protocol YAML: {exc}") from exc
    if not isinstance(obj, dict):
        raise Program08Error("protocol.yaml must contain a mapping.")
    return obj, sha256_bytes(raw)


def protocol_version(cfg: Mapping[str, Any]) -> str:
    p = cfg.get("project") or {}
    return str(p.get("protocol_version", "1.0")) if isinstance(p, Mapping) else "1.0"


def verify_environment_manifest(path: Path) -> dict[str, Any]:
    env = read_json(path)
    if env.get("manifest_type") != "environment":
        raise Program08Error("Invalid Program 00 environment manifest.")
    observed, _ = current_environment_fingerprint()
    if env.get("environment_fingerprint_sha256") != observed:
        raise Program08Error(
            "Current software environment differs from Program 00 frozen environment.\n"
            f"expected={env.get('environment_fingerprint_sha256')}\nobserved={observed}"
        )
    return env


def verify_data_and_svamp(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(root / "manifests" / "data_manifest.json")
    if manifest.get("manifest_type") != "data" or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise Program08Error("Invalid Program 00 data manifest.")
    datasets = manifest.get("datasets")
    if manifest.get("content_fingerprint_sha256") != sha256_bytes(canonical_bytes({"research_scope": manifest.get("research_scope"), "datasets": datasets})):
        raise Program08Error("Program 00 data manifest content fingerprint mismatch.")
    if not isinstance(datasets, Mapping) or not isinstance(datasets.get("svamp"), Mapping):
        raise Program08Error(
            "Program 08 requires pinned SVAMP in Program 00. data_manifest.json has no datasets.svamp. "
            "Do not mutate an old manifest in place; use the protocol/versioned project that pinned SVAMP."
        )
    s = dict(datasets["svamp"])
    svamp_dir = root / "data" / "raw" / "svamp"
    if tree_hash(svamp_dir)["tree_sha256"] != s.get("tree_sha256"):
        raise Program08Error("Pinned local SVAMP tree differs from Program 00 manifest.")
    validation = s.get("validation") or {}
    if int(validation.get("row_count", -1)) != SVAMP_EXPECTED_ROWS:
        raise Program08Error("Program 00 SVAMP validation did not freeze exactly 1000 rows.")
    json_rel = validation.get("svamp_json_path")
    if not isinstance(json_rel, str):
        raise Program08Error("SVAMP manifest lacks validation.svamp_json_path.")
    svamp_json = svamp_dir / json_rel
    if not svamp_json.exists() or sha256_file(svamp_json) != validation.get("svamp_json_sha256"):
        raise Program08Error("SVAMP.json file hash differs from Program 00.")
    return manifest, s


def tokenizer_template_hash(model_dir: Path) -> tuple[str, str]:
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise Program08Error(f"Cannot load pinned tokenizer: {exc}") from exc
    template = getattr(tok, "chat_template", None)
    if template is None:
        raise Program08Error("Pinned tokenizer has no chat_template.")
    text = template if isinstance(template, str) else json.dumps(template, ensure_ascii=False, sort_keys=True, default=str)
    return sha256_text(text), tok.__class__.__name__


def verify_model_manifest(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = root / "manifests" / "model_manifest.json"
    manifest = read_json(path)
    if manifest.get("manifest_type") != "model" or manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise Program08Error("Invalid Program 00 model manifest.")
    models = manifest.get("models")
    if manifest.get("content_fingerprint_sha256") != sha256_bytes(canonical_bytes(models)):
        raise Program08Error("Program 00 model manifest content fingerprint mismatch.")
    if not isinstance(models, Mapping) or not isinstance(models.get("primary"), Mapping):
        raise Program08Error("model_manifest.json lacks models.primary.")
    model = dict(models["primary"])
    if model.get("repo_id") != MODEL_REPO:
        raise Program08Error(f"Frozen primary model is not {MODEL_REPO}.")
    model_dir = root / "models" / "qwen25_05b"
    if tree_hash(model_dir)["tree_sha256"] != model.get("tree_sha256"):
        raise Program08Error("Pinned local Qwen model differs from Program 00 manifest.")
    observed_chat, _ = tokenizer_template_hash(model_dir)
    expected_chat = (((model.get("validation") or {}).get("tokenizer") or {}).get("chat_template_sha256"))
    if observed_chat != expected_chat:
        raise Program08Error("Tokenizer chat-template hash differs from Program 00.")
    return manifest, model


def verify_protocol_lock(root: Path, cfg_sha: str) -> dict[str, Any]:
    lock = read_json(root / "manifests" / "protocol_lock.json")
    expected = first_present(lock, (("config_sha256",), ("protocol_config_sha256",), ("inputs", "config_sha256")))
    if not isinstance(expected, str) or not HEX64_RE.fullmatch(expected):
        raise Program08Error("protocol_lock.json does not contain a valid frozen config SHA-256.")
    if expected != cfg_sha:
        raise Program08Error("Current protocol.yaml differs from protocol_lock.json.")
    return lock


def verify_frozen_gate(root: Path, cfg_sha: str, pv: str) -> tuple[dict[str, Any], dict[str, Any]]:
    gate_path = root / "outputs" / "frozen_gate.json"
    side_path = root / "manifests" / "frozen_gate_manifest.json"
    gate = read_json(gate_path)
    side = read_json(side_path)
    observed = sha256_file(gate_path)
    expected = first_present(side, (("gate_file_sha256",), ("file_sha256",), ("frozen_gate_sha256",)))
    if observed != expected:
        raise Program08Error("Frozen gate file hash mismatch.")
    if gate.get("manifest_type") != "frozen_reuse_gate":
        raise Program08Error("outputs/frozen_gate.json is not a frozen reuse gate.")
    if str(gate.get("protocol_config_sha256")) != cfg_sha or str(gate.get("protocol_version")) != pv:
        raise Program08Error("Frozen gate belongs to a different protocol/config.")
    if str(gate.get("estimator")) != "prompt_wis":
        raise Program08Error("Program 08 external gate validation requires frozen primary estimator prompt_wis.")
    if gate.get("test_firewall", {}).get("test_mode_refit_allowed") is not False:
        raise Program08Error("Frozen gate does not explicitly prohibit test-time refitting.")
    return gate, side


def verify_main_official_test_complete(root: Path, expected_seeds: Sequence[int]) -> dict[str, Any]:
    p06 = root / "outputs" / "diagnostics" / "program06_test_manifest.json"
    m06 = read_json(p06)
    if m06.get("manifest_type") != "program06_analysis_manifest" or m06.get("split") != "test" or m06.get("mode") != "paper":
        raise Program08Error("SVAMP robustness is post-main-result only: Program 06 official GSM8K test must complete first.")
    t06 = root / "outputs" / "tables" / "T06_gate_test.csv"
    rows = read_csv_rows(t06)
    official = [r for r in rows if r.get("split") == "test" and r.get("evaluation_role") == "official_test"]
    if not official:
        raise Program08Error("Program 07 test-only official GSM8K gate evaluation is missing from T06_gate_test.csv.")
    observed_seeds = {int(r["training_seed"]) for r in official}
    missing = set(int(x) for x in expected_seeds) - observed_seeds
    if missing:
        raise Program08Error(f"Program 07 official test does not cover all frozen training seeds; missing={sorted(missing)}")
    return {"program06_test_manifest_sha256": sha256_file(p06), "T06_gate_test_sha256": sha256_file(t06)}


def verify_program06_development(root: Path) -> tuple[dict[str, Any], str, str]:
    mp = root / "outputs" / "diagnostics" / "program06_development_manifest.json"
    m = read_json(mp)
    if m.get("manifest_type") != "program06_analysis_manifest" or m.get("split") != "development":
        raise Program08Error("Program 08 requires completed Program 06 development analysis.")
    t02 = root / "outputs" / "tables" / "T02_fixed_reuse.csv"
    t04 = root / "outputs" / "tables" / "T04_rolling_comparison.csv"
    if not t02.exists() or not t04.exists():
        raise Program08Error("Program 06 development T02/T04 tables are missing.")
    # Program 06's manifest contains output hashes; verify if present.
    outputs = m.get("outputs") or []
    hashed: dict[str, str] = {}
    if isinstance(outputs, list):
        for rec in outputs:
            if isinstance(rec, Mapping) and isinstance(rec.get("path"), str) and isinstance(rec.get("sha256"), str):
                hashed[Path(str(rec["path"])).name] = str(rec["sha256"])
    elif isinstance(outputs, Mapping):
        for label, rec in outputs.items():
            if isinstance(rec, Mapping) and isinstance(rec.get("sha256"), str):
                hashed[str(label)] = str(rec["sha256"])
    for p, legacy_label in ((t02, "T02_fixed_reuse"), (t04, "T04_rolling_comparison")):
        expected = hashed.get(p.name, hashed.get(legacy_label))
        if expected is None:
            raise Program08Error(f"Program 06 development manifest does not hash required table {p.name}.")
        if sha256_file(p) != expected:
            raise Program08Error(f"Program 06 table {p.name} differs from its development manifest.")
    return m, sha256_file(t02), sha256_file(t04)


def resolve_spec(cfg: Mapping[str, Any], gate: Mapping[str, Any]) -> RobustnessSpec:
    training = cfg.get("training") or {}
    behavior = cfg.get("behavior") or {}
    online = cfg.get("online_reference") or {}
    ope = cfg.get("ope") or {}
    robustness = cfg.get("robustness") or {}
    if not all(isinstance(x, Mapping) for x in (training, behavior, online, ope, robustness)):
        raise Program08Error("training/behavior/online_reference/ope/robustness protocol sections must be mappings.")
    seeds = tuple(int(x) for x in training.get("seeds", DEFAULT_SEEDS))
    calibration = tuple(int(x) for x in gate.get("calibration_seeds", seeds[:2]))
    k = int(behavior.get("K_main", behavior.get("K", 16)))
    l_main = int(online.get("L_main", 8))
    l_audit = int(online.get("L_audit", PAPER_L_AUDIT))
    audits = tuple(int(x) for x in online.get("audit_steps", PAPER_AUDIT_STEPS))
    rep_pairs = int(robustness.get("svamp_representative_pairs", DEFAULT_REPRESENTATIVE_PAIRS))
    spec = RobustnessSpec(
        seeds=seeds,
        calibration_seeds=calibration,
        representative_pairs=rep_pairs,
        k=k,
        l_main=l_main,
        l_audit=l_audit,
        audit_steps=audits,
        bootstrap_reps=int(ope.get("bootstrap_reps", DEFAULT_BOOTSTRAP_REPS)),
        bootstrap_seed=int(ope.get("bootstrap_seed", DEFAULT_BOOTSTRAP_SEED)),
        sample_block_size=int(robustness.get("sample_block_size", DEFAULT_SAMPLE_BLOCK_SIZE)),
        prompts_per_shard=int(robustness.get("prompts_per_shard", DEFAULT_PROMPTS_PER_SHARD)),
        generation_batch_size=int(robustness.get("generation_batch_size", DEFAULT_GENERATION_BATCH_SIZE)),
        max_completion_length=int(training.get("max_completion_length", 128)),
        temperature=float(training.get("temperature", 1.0)),
        top_p=float(training.get("top_p", 1.0)),
        top_k=int(training.get("top_k", 0)),
        repetition_penalty=float(training.get("repetition_penalty", 1.0)),
    )
    if len(spec.seeds) < 3 or any(s not in spec.seeds for s in spec.calibration_seeds):
        raise Program08Error("Paper robustness requires the frozen multi-seed design and valid calibration seed subset.")
    if spec.representative_pairs < 6 or spec.representative_pairs > 8:
        raise Program08Error("SVAMP representative-pair count must remain within the memo's frozen 6-8 range.")
    if spec.k not in PAPER_BEHAVIOR_K_ALLOWED:
        raise Program08Error(f"Paper robustness K must be frozen at one of {PAPER_BEHAVIOR_K_ALLOWED}.")
    if spec.l_main not in PAPER_L_MAIN_ALLOWED or spec.l_audit != PAPER_L_AUDIT:
        raise Program08Error("Paper robustness must reuse frozen L_main (8/16) and L_audit=32.")
    if not math.isclose(spec.temperature, 1.0, abs_tol=1e-12) or not math.isclose(spec.top_p, 1.0, abs_tol=1e-12):
        raise Program08Error("Exact robustness OPE requires temperature=1 and top_p=1.")
    if spec.top_k != 0 or not math.isclose(spec.repetition_penalty, 1.0, abs_tol=1e-12):
        raise Program08Error("Exact robustness OPE requires top_k=0 and repetition_penalty=1.")
    return spec


# ---------------------------------------------------------------------------
# Freeze representative pair registry BEFORE reading SVAMP
# ---------------------------------------------------------------------------


def _float(row: Mapping[str, Any], key: str) -> float:
    try:
        v = float(row[key])
    except Exception as exc:
        raise Program08Error(f"Invalid numeric {key} in Program 06 table") from exc
    if not math.isfinite(v):
        raise Program08Error(f"Non-finite {key} in Program 06 table")
    return v


def candidate_pair_templates(root: Path, calibration_seeds: Sequence[int]) -> list[dict[str, Any]]:
    cal = set(int(x) for x in calibration_seeds)
    grouped: dict[tuple[int, int, str], dict[str, list[float]]] = {}
    for r in read_csv_rows(root / "outputs" / "tables" / "T02_fixed_reuse.csv"):
        if r.get("split") != "development" or r.get("estimator") != "prompt_wis":
            continue
        if int(r["training_seed"]) not in cal:
            continue
        key = (int(r["behavior_step"]), int(r["target_step"]), "fixed")
        g = grouped.setdefault(key, {"ress": [], "kl": []})
        g["ress"].append(_float(r, "median_prompt_relative_ess"))
        g["kl"].append(_float(r, "tokenwise_kl_proxy"))
    for r in read_csv_rows(root / "outputs" / "tables" / "T04_rolling_comparison.csv"):
        if r.get("split") != "development" or r.get("estimator") != "prompt_wis":
            continue
        if int(r["training_seed"]) not in cal:
            continue
        key = (int(r["recent_behavior_step"]), int(r["target_step"]), "rolling")
        g = grouped.setdefault(key, {"ress": [], "kl": []})
        g["ress"].append(_float(r, "recent_median_relative_ess"))
        g["kl"].append(_float(r, "recent_tokenwise_kl_proxy"))
    out: list[dict[str, Any]] = []
    import statistics
    for (b, e, purpose), g in grouped.items():
        if len(g["ress"]) != len(cal):
            continue
        out.append({
            "behavior_step": b,
            "target_step": e,
            "purpose": purpose,
            "pooled_median_relative_ess": float(statistics.median(g["ress"])),
            "pooled_tokenwise_kl_proxy": float(statistics.median(g["kl"])),
        })
    out.sort(key=lambda r: (r["pooled_median_relative_ess"], r["behavior_step"], r["target_step"]))
    if len(out) < 8:
        raise Program08Error("Too few complete GSM8K development candidate pair templates to freeze SVAMP robustness registry.")
    return out


def evenly_spaced_distinct_indices(n: int, k: int) -> list[int]:
    if k > n:
        raise Program08Error("Cannot select more representative pairs than candidates.")
    if k == 1:
        return [n // 2]
    raw = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    out: list[int] = []
    used: set[int] = set()
    for x in raw:
        y = int(x)
        if y in used:
            for d in range(1, n):
                for z in (y - d, y + d):
                    if 0 <= z < n and z not in used:
                        y = z
                        break
                if y not in used:
                    break
        used.add(y); out.append(y)
    return sorted(out)


def band_for_rank(rank: int, total: int) -> str:
    # candidates are sorted low-rESS -> high-rESS; label overlap accordingly.
    frac = rank / max(total - 1, 1)
    if frac < 1 / 3:
        return "low_overlap"
    if frac < 2 / 3:
        return "medium_overlap"
    return "high_overlap"


def pair_registry_paths(root: Path) -> tuple[Path, Path]:
    return root / "manifests" / "svamp_pair_registry.parquet", root / "manifests" / "svamp_pair_registry_manifest.json"


def freeze_or_verify_svamp_pair_registry(
    *, root: Path, spec: RobustnessSpec, pv: str, cfg_sha: str,
    program06_sha: str, t02_sha: str, t04_sha: str, gate_sha: str,
) -> list[PairTemplate]:
    pp, mp = pair_registry_paths(root)
    if pp.exists() != mp.exists():
        raise Program08Error("Partial SVAMP pair registry publication detected.")
    if not pp.exists():
        # IMPORTANT: this function must run before SVAMP row content is parsed.
        candidates = candidate_pair_templates(root, spec.calibration_seeds)
        idxs = evenly_spaced_distinct_indices(len(candidates), spec.representative_pairs)
        chosen = [candidates[i] for i in idxs]
        rows: list[dict[str, Any]] = []
        for rep_rank, (cand_idx, c) in enumerate(zip(idxs, chosen), start=1):
            rows.append({
                "protocol_version": pv,
                "behavior_step": int(c["behavior_step"]),
                "target_step": int(c["target_step"]),
                "purpose": str(c["purpose"]),
                "representative_rank": rep_rank,
                "source_candidate_rank_low_to_high_overlap": cand_idx + 1,
                "source_candidate_count": len(candidates),
                "overlap_band": band_for_rank(cand_idx, len(candidates)),
                "source_median_relative_ess": float(c["pooled_median_relative_ess"]),
                "source_tokenwise_kl_proxy": float(c["pooled_tokenwise_kl_proxy"]),
            })
        pp.parent.mkdir(parents=True, exist_ok=True)
        stage = pp.parent / f".{pp.name}.stage-{uuid.uuid4().hex}"
        write_parquet(stage, rows)
        os.replace(stage, pp); fsync_directory(pp.parent)
        manifest = {
            "schema_version": SVAMP_PAIR_SCHEMA,
            "manifest_type": "svamp_frozen_representative_pair_registry",
            "created_at_utc": now_utc(),
            "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
            "protocol_version": pv,
            "protocol_config_sha256": cfg_sha,
            "selection_data": "GSM8K development only",
            "selection_seeds": list(spec.calibration_seeds),
            "selection_rule": (
                "Aggregate each fixed/recent-rolling checkpoint template across frozen calibration seeds by median prompt rESS; "
                "sort low-to-high rESS; choose evenly spaced distinct ranks to cover the full overlap range."
            ),
            "svamp_information_used_for_selection": False,
            "representative_pair_count": len(rows),
            "source_program06_development_manifest_sha256": program06_sha,
            "source_T02_sha256": t02_sha,
            "source_T04_sha256": t04_sha,
            "frozen_gate_sha256": gate_sha,
            "registry_path": rel(pp, root),
            "registry_file_sha256": sha256_file(pp),
            "registry_content_sha256": sha256_bytes(canonical_bytes(rows)),
            "immutable": True,
        }
        atomic_write_json(mp, manifest)
        make_read_only(pp); make_read_only(mp)
    manifest = read_json(mp)
    if manifest.get("manifest_type") != "svamp_frozen_representative_pair_registry":
        raise Program08Error("Invalid SVAMP pair registry manifest.")
    if str(manifest.get("protocol_config_sha256")) != cfg_sha or str(manifest.get("protocol_version")) != pv:
        raise Program08Error("SVAMP pair registry belongs to a different protocol/config.")
    if manifest.get("svamp_information_used_for_selection") is not False:
        raise Program08Error("SVAMP pair registry does not certify outcome-independent selection.")
    if sha256_file(pp) != manifest.get("registry_file_sha256"):
        raise Program08Error("SVAMP pair registry file hash mismatch.")
    rows = read_parquet(pp)
    if len(rows) != spec.representative_pairs:
        raise Program08Error("Frozen representative-pair count differs from protocol.")
    out = [PairTemplate(
        behavior_step=int(r["behavior_step"]), target_step=int(r["target_step"]), purpose=str(r["purpose"]),
        representative_rank=int(r["representative_rank"]), overlap_band=str(r["overlap_band"]),
        source_median_relative_ess=float(r["source_median_relative_ess"]),
        source_tokenwise_kl_proxy=float(r["source_tokenwise_kl_proxy"]),
    ) for r in rows]
    keys = [(p.behavior_step, p.target_step) for p in out]
    if len(keys) != len(set(keys)) or any(p.target_step <= p.behavior_step for p in out):
        raise Program08Error("Frozen SVAMP pair registry contains duplicate/non-forward pairs.")
    return sorted(out, key=lambda p: p.representative_rank)


# ---------------------------------------------------------------------------
# Program 02 adapter verification
# ---------------------------------------------------------------------------


def training_manifest_path(root: Path, seed: int) -> Path:
    return root / "manifests" / "training" / f"seed_{seed}.json"


def verify_training_seed(root: Path, seed: int, required_steps: Sequence[int]) -> tuple[dict[str, Any], dict[int, AdapterRecord]]:
    tm = read_json(training_manifest_path(root, seed))
    if tm.get("status") != "complete" or int(tm.get("training_seed", tm.get("seed", -1))) != seed:
        raise Program08Error(f"Program 02 training manifest for seed {seed} is not complete.")
    base = root / "checkpoints" / f"seed_{seed}" / "adapters"
    amap: dict[int, AdapterRecord] = {}
    for step in sorted(set(required_steps)):
        adir = base / f"step_{step:04d}"
        amp = adir / "adapter_manifest.json"
        am = read_json(amp)
        if int(am.get("training_seed", am.get("seed", seed))) != seed or int(am.get("step", -1)) != step:
            raise Program08Error(f"Adapter manifest seed/step mismatch: {amp}")
        observed = adapter_payload_hash(adir)
        expected = first_present(am, (("adapter_payload_tree_sha256",), ("adapter_payload_sha256",), ("payload_sha256",), ("adapter_sha256",)))
        if not isinstance(expected, str) or observed != expected:
            raise Program08Error(f"Adapter payload hash mismatch at seed={seed}, step={step}.")
        amap[step] = AdapterRecord(step, adir, observed, am)
    return tm, amap


# ---------------------------------------------------------------------------
# SVAMP prompt construction (only after pair registry freeze)
# ---------------------------------------------------------------------------


def normalize_numeric_text(text: str) -> Fraction | None:
    s = text.strip().replace(",", "").replace("$", "")
    if not s or not NUMBER_RE.fullmatch(s):
        return None
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            den = int(b)
            return None if den == 0 else Fraction(int(a), den)
        return Fraction(Decimal(s))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def canonical_fraction(v: Fraction) -> str:
    return str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}"


def parse_completion_answer(text: str) -> tuple[Fraction | None, str]:
    matches = ANSWER_RE.findall(text)
    if len(matches) != 1:
        return None, "missing_answer_tag" if not matches else "multiple_answer_tags"
    if len(re.findall(r"<answer\b", text, re.I)) != 1 or len(re.findall(r"</answer\s*>", text, re.I)) != 1:
        return None, "malformed_answer_tags"
    val = normalize_numeric_text(matches[0])
    return (val, "ok") if val is not None else (None, "invalid_numeric_answer")


def locate_svamp_json(root: Path, svamp_record: Mapping[str, Any]) -> Path:
    relp = ((svamp_record.get("validation") or {}).get("svamp_json_path"))
    if not isinstance(relp, str):
        raise Program08Error("Frozen SVAMP manifest lacks svamp_json_path.")
    return root / "data" / "raw" / "svamp" / relp


def load_svamp_prompts(root: Path, svamp_record: Mapping[str, Any], limit: int | None = None) -> list[PromptRecord]:
    path = locate_svamp_json(root, svamp_record)
    # This is intentionally the first function in the pipeline that opens the
    # SVAMP row data; pair registry must already be frozen before this call.
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list) or len(rows) != SVAMP_EXPECTED_ROWS:
        raise Program08Error(f"Expected exactly {SVAMP_EXPECTED_ROWS} SVAMP rows.")
    revision = str(svamp_record.get("resolved_revision"))
    out: list[PromptRecord] = []
    seen_ids: set[str] = set()
    for i, r in enumerate(rows):
        if not isinstance(r, Mapping):
            raise Program08Error(f"SVAMP row {i} is not an object.")
        for key in ("ID", "Body", "Question", "Answer"):
            if key not in r:
                raise Program08Error(f"SVAMP row {i} lacks required field {key}.")
        sid = str(r["ID"])
        if sid in seen_ids:
            raise Program08Error(f"Duplicate SVAMP ID: {sid}")
        seen_ids.add(sid)
        body = str(r["Body"]).strip(); q = str(r["Question"]).strip()
        if not body or not q:
            raise Program08Error(f"Empty Body/Question in SVAMP row {sid}.")
        # Crucial leakage firewall: Equation is NEVER included in the prompt.
        question = f"{body}\n{q}"
        gold = normalize_numeric_text(str(r["Answer"]))
        if gold is None:
            raise Program08Error(f"SVAMP Answer is not parseable as an exact numeric value: ID={sid}")
        pid = sha256_bytes(canonical_bytes({
            "dataset": "SVAMP", "dataset_revision": revision, "source_id": sid,
            "source_row_index": i, "question_sha256": sha256_text(question),
        }))
        out.append(PromptRecord(pid, i, sid, question, str(r["Answer"]), canonical_fraction(gold), str(r.get("Type", ""))))
    if limit is not None:
        out = out[:limit]
    return out


# ---------------------------------------------------------------------------
# Tokenizer, model and generation
# ---------------------------------------------------------------------------


def load_tokenizer(model_dir: Path, model_record: Mapping[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise Program08Error(f"Cannot load pinned tokenizer: {exc}") from exc
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise Program08Error("Tokenizer has neither pad nor EOS token.")
        tok.pad_token = tok.eos_token
    template = tok.chat_template
    text = template if isinstance(template, str) else json.dumps(template, ensure_ascii=False, sort_keys=True, default=str)
    expected = (((model_record.get("validation") or {}).get("tokenizer") or {}).get("chat_template_sha256"))
    if sha256_text(text) != expected:
        raise Program08Error("Loaded chat-template hash differs from Program 00.")
    return tok


def render_prompt_ids(tokenizer: Any, question: str) -> list[int]:
    messages = [{"role": "user", "content": PROMPT_TEMPLATE.format(question=question)}]
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    if not isinstance(ids, list) or not ids or not all(isinstance(x, int) for x in ids):
        raise Program08Error("Tokenizer did not return a flat prompt token-ID sequence.")
    return [int(x) for x in ids]


def resolve_dtype(device: str) -> tuple[Any, str]:
    import torch  # type: ignore
    if device == "cuda":
        if not torch.cuda.is_available():
            raise Program08Error("--device cuda requested but CUDA is unavailable.")
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bf16"
        return torch.float16, "fp16"
    return torch.float32, "fp32"


def load_adapter_model(model_dir: Path, adapter: AdapterRecord, device: str) -> tuple[Any, str]:
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
        from peft import PeftModel  # type: ignore
    except Exception as exc:
        raise Program08Error(f"Cannot import inference stack: {exc}") from exc
    dtype, precision = resolve_dtype(device)
    try:
        try:
            base = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False, dtype=dtype)
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, str(adapter.path), is_trainable=False)
        model.to(device); model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        for p in model.parameters():
            p.requires_grad_(False)
        return model, precision
    except Exception as exc:
        raise Program08Error(f"Cannot load Qwen + adapter {adapter.path}: {exc}") from exc


def eos_ids(tokenizer: Any) -> set[int]:
    raw = tokenizer.eos_token_id
    if isinstance(raw, int):
        return {raw}
    if isinstance(raw, (list, tuple, set)) and raw:
        return {int(x) for x in raw}
    raise Program08Error("Tokenizer has no usable eos_token_id.")


def completion_cut_length(tokens: Sequence[int], eos_set: set[int], n_steps: int, max_new_tokens: int) -> tuple[int, bool, bool]:
    for i, tok in enumerate(tokens[:n_steps]):
        if int(tok) in eos_set:
            return i + 1, True, False
    if n_steps == max_new_tokens:
        return n_steps, False, True
    raise Program08Error("Generation ended early without EOS; refusing to guess termination semantics.")


def stable_seed(*parts: Any) -> int:
    value = int.from_bytes(hashlib.sha256(canonical_bytes(list(parts))).digest()[:8], "big")
    return value % (2**31 - 1)


def generation_record(spec: RobustnessSpec, tokenizer: Any, output_scores: bool) -> dict[str, Any]:
    return {
        "do_sample": True, "num_beams": 1, "temperature": spec.temperature,
        "top_p": spec.top_p, "top_k": spec.top_k, "repetition_penalty": spec.repetition_penalty,
        "max_new_tokens": spec.max_completion_length, "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": sorted(eos_ids(tokenizer)), "return_dict_in_generate": True,
        "output_scores": output_scores,
    }


def seeded_generate(model: Any, input_ids: Any, attention_mask: Any, kwargs: Mapping[str, Any], seed: int, device: str) -> Any:
    import torch  # type: ignore
    from transformers import GenerationConfig  # type: ignore
    gc = GenerationConfig(**dict(kwargs))
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
        with torch.inference_mode():
            return model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=gc)


def trajectory_id(pv: str, dataset_revision: str, model_revision: str, seed: int, b: int, prompt_id: str, sample_index: int) -> str:
    return sha256_bytes(canonical_bytes([pv, dataset_revision, model_revision, seed, b, "svamp", prompt_id, sample_index]))


def online_id(pv: str, dataset_revision: str, seed: int, e: int, prompt_id: str, sample_index: int) -> str:
    return sha256_bytes(canonical_bytes([pv, dataset_revision, seed, e, "svamp", prompt_id, sample_index]))


def generate_prompt_samples(
    *, model: Any, tokenizer: Any, prompt: PromptRecord, sample_start: int, sample_end: int,
    spec: RobustnessSpec, pv: str, dataset_revision: str, model_revision: str,
    training_seed: int, step: int, adapter_sha: str, device: str, behavior: bool,
) -> list[dict[str, Any]]:
    import torch  # type: ignore
    prompt_ids = render_prompt_ids(tokenizer, prompt.question)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(input_ids)
    eos_set = eos_ids(tokenizer)
    out_rows: list[dict[str, Any]] = []
    gen_cfg = generation_record(spec, tokenizer, output_scores=behavior)
    gen_hash = sha256_bytes(canonical_bytes(gen_cfg))
    chat_text = tokenizer.chat_template if isinstance(tokenizer.chat_template, str) else json.dumps(tokenizer.chat_template, ensure_ascii=False, sort_keys=True, default=str)
    chat_hash = sha256_text(chat_text); prompt_hash = sha256_text(PROMPT_TEMPLATE)
    for sub_start in range(sample_start, sample_end, spec.generation_batch_size):
        sub_end = min(sample_end, sub_start + spec.generation_batch_size)
        n = sub_end - sub_start
        call_seed = stable_seed("svamp_behavior" if behavior else "svamp_online", pv, dataset_revision, model_revision, training_seed, step, prompt.prompt_id, sub_start, sub_end, gen_hash)
        call_id = sha256_bytes(canonical_bytes([call_seed, training_seed, step, prompt.prompt_id, sub_start, sub_end, "behavior" if behavior else "online"]))
        kwargs = dict(gen_cfg); kwargs["num_return_sequences"] = n
        outputs = seeded_generate(model, input_ids, attention, kwargs, call_seed, device)
        if not hasattr(outputs, "sequences"):
            raise Program08Error("generate() did not return sequences.")
        if behavior:
            scores = outputs.scores
            if scores is None or len(scores) == 0:
                raise Program08Error("Behavior generation did not return token scores.")
            transition = model.compute_transition_scores(outputs.sequences, scores, normalize_logits=True)
            n_steps = len(scores)
        else:
            # output_scores=False still returns a structured generation object.
            generated_total = int(outputs.sequences.shape[1]) - len(prompt_ids)
            n_steps = generated_total
            transition = None
        generated = outputs.sequences[:, len(prompt_ids): len(prompt_ids) + n_steps].detach().cpu()
        if int(generated.shape[0]) != n:
            raise Program08Error("Unexpected generated sequence count.")
        if behavior:
            transition = transition.detach().to(dtype=torch.float64, device="cpu")
            if int(transition.shape[1]) != n_steps:
                raise Program08Error("Behavior transition-score/token alignment failure.")
        gold = Fraction(prompt.gold_answer)
        for local in range(n):
            idx = sub_start + local
            all_ids = [int(x) for x in generated[local].tolist()]
            cut, ended_eos, truncated = completion_cut_length(all_ids, eos_set, n_steps, spec.max_completion_length)
            comp = all_ids[:cut]
            text = tokenizer.decode(comp, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            parsed, status = parse_completion_answer(text)
            parsed_s = None if parsed is None else canonical_fraction(parsed)
            correct = float(parsed is not None and parsed == gold)
            common = {
                "dataset": "SVAMP", "dataset_revision": dataset_revision, "protocol_version": pv,
                "training_seed": training_seed, "prompt_id": prompt.prompt_id, "source_row_index": prompt.source_row_index,
                "sample_index": idx, "completion_token_ids": comp, "completion_length": len(comp),
                "terminated_with_eos": ended_eos, "was_truncated": truncated, "parsed_answer": parsed_s,
                "parser_status": status, "temperature": spec.temperature, "top_p": spec.top_p, "top_k": spec.top_k,
                "repetition_penalty": spec.repetition_penalty, "max_completion_length": spec.max_completion_length,
                "generation_seed": call_seed, "generation_call_id": call_id, "model_revision": model_revision,
                "tokenizer_revision": model_revision, "chat_template_hash": chat_hash, "prompt_template_hash": prompt_hash,
                "generation_config_sha256": gen_hash, "completion_text": text,
            }
            if behavior:
                logps = [float(x) for x in transition[local, :cut].tolist()]
                if len(logps) != len(comp) or not all(math.isfinite(x) for x in logps):
                    raise Program08Error("Behavior token/logprob alignment or finiteness failure.")
                out_rows.append({
                    **common,
                    "trajectory_id": trajectory_id(pv, dataset_revision, model_revision, training_seed, step, prompt.prompt_id, idx),
                    "behavior_step": step, "prompt_token_ids": prompt_ids, "behavior_token_logprobs": logps,
                    "behavior_sequence_logprob": math.fsum(logps), "correctness_reward": correct,
                    "behavior_adapter_sha256": adapter_sha,
                    "behavior_logprob_source": "generate_scores+compute_transition_scores(normalize_logits=True)",
                })
            else:
                out_rows.append({
                    **common,
                    "online_id": online_id(pv, dataset_revision, training_seed, step, prompt.prompt_id, idx),
                    "target_step": step, "correct": correct, "target_adapter_sha256": adapter_sha,
                })
    return out_rows


# ---------------------------------------------------------------------------
# Immutable expensive-shard I/O
# ---------------------------------------------------------------------------


def sample_blocks(n: int, block: int) -> list[tuple[int, int]]:
    return [(i, min(n, i + block)) for i in range(0, n, block)]


def prompt_shards(prompts: Sequence[PromptRecord], n: int) -> list[list[PromptRecord]]:
    return [list(prompts[i:i+n]) for i in range(0, len(prompts), n)]


def robustness_root(root: Path, mode: str) -> Path:
    return root / "data" / "svamp_robustness" if mode == "paper" else root / "data" / "svamp_robustness" / f"_{mode}"


def unit_dir(base: Path, kind: str, seed: int, b: int | None, e: int | None, s0: int, s1: int, shard: int) -> Path:
    p = base / kind / f"seed={seed}"
    if b is not None:
        p /= f"behavior_step={b:04d}"
    if e is not None:
        p /= f"target_step={e:04d}"
    return p / f"sample_block={s0:04d}-{s1-1:04d}" / f"shard={shard:05d}"


def verify_unit(path: Path, expected_kind: str, expected_content_sha: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mp = path / "manifest.json"
    if not mp.exists():
        raise Program08Error(f"Missing shard manifest: {mp}")
    m = read_json(mp)
    if m.get("manifest_type") != expected_kind:
        raise Program08Error(f"Wrong shard type in {mp}")
    data_name = str(m.get("data_file"))
    pp = path / data_name
    if not pp.exists() or sha256_file(pp) != m.get("file_sha256"):
        raise Program08Error(f"Shard file hash mismatch: {pp}")
    rows = read_parquet(pp)
    content_sha = sha256_bytes(canonical_bytes(rows))
    if content_sha != m.get("content_sha256"):
        raise Program08Error(f"Shard semantic hash mismatch: {pp}")
    if expected_content_sha is not None and content_sha != expected_content_sha:
        raise Program08Error("Completed immutable shard differs from expected content.")
    return rows, m


def publish_unit(path: Path, rows: Sequence[Mapping[str, Any]], kind: str, data_name: str, metadata_fields: Mapping[str, Any]) -> None:
    if path.exists():
        existing, _ = verify_unit(path, kind)
        if sha256_bytes(canonical_bytes(existing)) != sha256_bytes(canonical_bytes(list(rows))):
            raise Program08Error(f"Refusing to overwrite immutable shard with different content: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.parent / f".{path.name}.stage-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        pp = stage / data_name
        write_parquet(pp, rows)
        content_sha = sha256_bytes(canonical_bytes(list(rows)))
        manifest = {
            "schema_version": "1.0", "manifest_type": kind, "created_at_utc": now_utc(),
            "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
            **dict(metadata_fields), "data_file": data_name, "row_count": len(rows),
            "file_sha256": sha256_file(pp), "content_sha256": content_sha,
        }
        atomic_write_json(stage / "manifest.json", manifest)
        fsync_directory(stage)
        os.replace(stage, path); fsync_directory(path.parent)
        make_read_only(path)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def cleanup_staging(base: Path, resume: bool) -> None:
    if not base.exists():
        return
    stale = [p for p in base.rglob(".*.stage-*") if p.is_dir()]
    if stale and not resume:
        raise Program08Error(f"Found {len(stale)} interrupted staging directories. Re-run with --resume.")
    for p in stale:
        shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------
# Behavior, exact rescoring and online collection
# ---------------------------------------------------------------------------


def collect_behavior(
    *, base: Path, model_dir: Path, tokenizer: Any, model_record: Mapping[str, Any], adapter: AdapterRecord,
    seed: int, b: int, prompts: Sequence[PromptRecord], spec: RobustnessSpec, pv: str,
    dataset_revision: str, device: str, resume: bool,
) -> None:
    model, _ = load_adapter_model(model_dir, adapter, device)
    try:
        for s0, s1 in sample_blocks(spec.k, spec.sample_block_size):
            for shard_idx, pshard in enumerate(prompt_shards(prompts, spec.prompts_per_shard)):
                outdir = unit_dir(base, "behavior", seed, b, None, s0, s1, shard_idx)
                if outdir.exists():
                    verify_unit(outdir, "svamp_behavior_shard"); continue
                rows: list[dict[str, Any]] = []
                for p in pshard:
                    rows.extend(generate_prompt_samples(
                        model=model, tokenizer=tokenizer, prompt=p, sample_start=s0, sample_end=s1, spec=spec,
                        pv=pv, dataset_revision=dataset_revision, model_revision=str(model_record.get("resolved_revision")),
                        training_seed=seed, step=b, adapter_sha=adapter.payload_sha256, device=device, behavior=True,
                    ))
                publish_unit(outdir, rows, "svamp_behavior_shard", "behavior.parquet", {
                    "training_seed": seed, "behavior_step": b, "sample_start": s0, "sample_end_exclusive": s1,
                    "shard_index": shard_idx, "adapter_sha256": adapter.payload_sha256,
                })
    finally:
        del model
        try:
            import torch  # type: ignore
            if device == "cuda": torch.cuda.empty_cache()
        except Exception:
            pass


def load_behavior_rows(base: Path, seed: int, b: int) -> list[dict[str, Any]]:
    p = base / "behavior" / f"seed={seed}" / f"behavior_step={b:04d}"
    if not p.exists():
        raise Program08Error(f"Missing SVAMP behavior assets for seed={seed}, b={b}")
    out: list[dict[str, Any]] = []
    for mp in sorted(p.rglob("manifest.json")):
        rows, _ = verify_unit(mp.parent, "svamp_behavior_shard")
        out.extend(rows)
    if not out:
        raise Program08Error("No SVAMP behavior rows found.")
    return out


def teacher_force_rows(model: Any, rows: Sequence[Mapping[str, Any]], device: str) -> list[tuple[float, list[float]]]:
    import torch  # type: ignore
    if not rows:
        return []
    # Correctness > throughput: batch by compatible lengths to keep shift logic explicit.
    results: list[tuple[float, list[float]]] = []
    for r in rows:
        prompt = [int(x) for x in r["prompt_token_ids"]]
        comp = [int(x) for x in r["completion_token_ids"]]
        if not prompt or not comp:
            raise Program08Error("Teacher forcing received empty prompt/completion tokens.")
        seq = prompt + comp
        ids = torch.tensor([seq], dtype=torch.long, device=device)
        attn = torch.ones_like(ids)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=attn).logits
        pred = logits[0, len(prompt)-1:len(prompt)+len(comp)-1, :].float()
        if int(pred.shape[0]) != len(comp):
            raise Program08Error("Teacher-forcing causal shift alignment failure.")
        targets = torch.tensor(comp, dtype=torch.long, device=pred.device)
        token_logps = torch.log_softmax(pred, dim=-1).gather(1, targets[:, None]).squeeze(1).double().cpu().tolist()
        vals = [float(x) for x in token_logps]
        if not all(math.isfinite(x) for x in vals):
            raise Program08Error("Non-finite target token log-probability.")
        results.append((math.fsum(vals), vals))
    return results


def rescore_id(traj: str, target_step: int, adapter_sha: str) -> str:
    return sha256_bytes(canonical_bytes([traj, target_step, adapter_sha]))


def identity_check_behavior(
    *, base: Path, model_dir: Path, adapter: AdapterRecord, seed: int, b: int,
    device: str, token_atol: float = 5e-4, sequence_atol: float = 2e-3,
) -> dict[str, Any]:
    rows = load_behavior_rows(base, seed, b)
    model, _ = load_adapter_model(model_dir, adapter, device)
    max_tok = 0.0; max_seq = 0.0; n = 0
    try:
        for r, (seq_lp, token_lp) in zip(rows, teacher_force_rows(model, rows, device)):
            btoken = [float(x) for x in r["behavior_token_logprobs"]]
            if len(btoken) != len(token_lp):
                raise Program08Error("SVAMP identity token-length mismatch.")
            td = max(abs(a-b) for a,b in zip(btoken, token_lp))
            sd = abs(float(r["behavior_sequence_logprob"]) - seq_lp)
            max_tok = max(max_tok, td); max_seq = max(max_seq, sd); n += 1
    finally:
        del model
        try:
            import torch  # type: ignore
            if device == "cuda": torch.cuda.empty_cache()
        except Exception:
            pass
    passed = max_tok <= token_atol and max_seq <= sequence_atol
    if not passed:
        raise Program08Error(
            f"SVAMP identity check FAILED seed={seed}, step={b}: max_token_diff={max_tok:.3g}, max_sequence_diff={max_seq:.3g}"
        )
    return {"training_seed": seed, "behavior_step": b, "n_trajectories": n, "max_abs_token_diff": max_tok, "max_abs_sequence_diff": max_seq, "identity_pass": True}


def collect_rescore(
    *, base: Path, model_dir: Path, seed: int, pair: PairTemplate, target_adapter: AdapterRecord,
    device: str, spec: RobustnessSpec,
) -> None:
    behavior = load_behavior_rows(base, seed, pair.behavior_step)
    model, _ = load_adapter_model(model_dir, target_adapter, device)
    try:
        by_unit: dict[tuple[int,int,int], list[dict[str, Any]]] = {}
        # Preserve the behavior shard partitioning by using sample block and deterministic prompt-shard index.
        for r in behavior:
            s0 = (int(r["sample_index"]) // spec.sample_block_size) * spec.sample_block_size
            s1 = min(spec.k, s0 + spec.sample_block_size)
            shard_idx = int(r["source_row_index"]) // spec.prompts_per_shard
            by_unit.setdefault((s0,s1,shard_idx), []).append(r)
        for (s0,s1,shard_idx), rows in sorted(by_unit.items()):
            outdir = unit_dir(base, "rescore", seed, pair.behavior_step, pair.target_step, s0, s1, shard_idx)
            if outdir.exists():
                verify_unit(outdir, "svamp_rescore_shard"); continue
            scored = teacher_force_rows(model, rows, device)
            out: list[dict[str, Any]] = []
            source_sha = sha256_bytes(canonical_bytes(rows))
            for r, (target_seq, target_tokens) in zip(rows, scored):
                blogp = float(r["behavior_sequence_logprob"]); logw = target_seq - blogp
                out.append({
                    "rescore_id": rescore_id(str(r["trajectory_id"]), pair.target_step, target_adapter.payload_sha256),
                    "trajectory_id": r["trajectory_id"], "dataset": "SVAMP", "dataset_revision": r["dataset_revision"],
                    "protocol_version": r["protocol_version"], "training_seed": seed,
                    "behavior_step": pair.behavior_step, "target_step": pair.target_step,
                    "prompt_id": r["prompt_id"], "sample_index": int(r["sample_index"]),
                    "behavior_adapter_sha256": r["behavior_adapter_sha256"],
                    "target_adapter_sha256": target_adapter.payload_sha256,
                    "behavior_sequence_logprob": blogp, "target_sequence_logprob": target_seq,
                    "log_weight": logw, "mean_log_ratio_per_token": logw / int(r["completion_length"]),
                    "completion_length": int(r["completion_length"]), "correctness_reward": float(r["correctness_reward"]),
                    "terminated_with_eos": bool(r["terminated_with_eos"]), "was_truncated": bool(r["was_truncated"]),
                    "identity_abs_sequence_logprob_diff": None, "identity_max_abs_token_logprob_diff": None,
                    "identity_mean_abs_token_logprob_diff": None, "identity_pass": None,
                    "source_behavior_content_sha256": source_sha,
                })
            publish_unit(outdir, out, "svamp_rescore_shard", "rescore.parquet", {
                "training_seed": seed, "behavior_step": pair.behavior_step, "target_step": pair.target_step,
                "sample_start": s0, "sample_end_exclusive": s1, "shard_index": shard_idx,
                "target_adapter_sha256": target_adapter.payload_sha256,
            })
    finally:
        del model
        try:
            import torch  # type: ignore
            if device == "cuda": torch.cuda.empty_cache()
        except Exception:
            pass


def load_rescore_rows(base: Path, seed: int, b: int, e: int) -> list[dict[str, Any]]:
    p = base / "rescore" / f"seed={seed}" / f"behavior_step={b:04d}" / f"target_step={e:04d}"
    if not p.exists():
        raise Program08Error(f"Missing SVAMP rescore assets seed={seed}, b={b}, e={e}")
    out: list[dict[str, Any]] = []
    for mp in sorted(p.rglob("manifest.json")):
        rows, _ = verify_unit(mp.parent, "svamp_rescore_shard"); out.extend(rows)
    return out


def target_l(spec: RobustnessSpec, step: int) -> int:
    return spec.l_audit if step in spec.audit_steps else spec.l_main


def collect_online(
    *, base: Path, model_dir: Path, tokenizer: Any, model_record: Mapping[str, Any], adapter: AdapterRecord,
    seed: int, e: int, prompts: Sequence[PromptRecord], spec: RobustnessSpec, pv: str,
    dataset_revision: str, device: str,
) -> None:
    L = target_l(spec, e)
    model, _ = load_adapter_model(model_dir, adapter, device)
    try:
        for s0, s1 in sample_blocks(L, spec.sample_block_size):
            for shard_idx, pshard in enumerate(prompt_shards(prompts, spec.prompts_per_shard)):
                outdir = unit_dir(base, "online", seed, None, e, s0, s1, shard_idx)
                if outdir.exists():
                    verify_unit(outdir, "svamp_online_shard"); continue
                rows: list[dict[str, Any]] = []
                for p in pshard:
                    rows.extend(generate_prompt_samples(
                        model=model, tokenizer=tokenizer, prompt=p, sample_start=s0, sample_end=s1, spec=spec,
                        pv=pv, dataset_revision=dataset_revision, model_revision=str(model_record.get("resolved_revision")),
                        training_seed=seed, step=e, adapter_sha=adapter.payload_sha256, device=device, behavior=False,
                    ))
                publish_unit(outdir, rows, "svamp_online_shard", "online.parquet", {
                    "training_seed": seed, "target_step": e, "sample_start": s0, "sample_end_exclusive": s1,
                    "shard_index": shard_idx, "target_adapter_sha256": adapter.payload_sha256,
                })
    finally:
        del model
        try:
            import torch  # type: ignore
            if device == "cuda": torch.cuda.empty_cache()
        except Exception:
            pass


def load_online_prompt_accuracy(base: Path, seed: int, e: int, expected_L: int) -> dict[str, float]:
    p = base / "online" / f"seed={seed}" / f"target_step={e:04d}"
    rows: list[dict[str, Any]] = []
    for mp in sorted(p.rglob("manifest.json")):
        r, _ = verify_unit(mp.parent, "svamp_online_shard"); rows.extend(r)
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(str(r["prompt_id"]), []).append(float(r["correct"]))
    if not by or any(len(v) != expected_L for v in by.values()):
        raise Program08Error(f"Online reference for seed={seed}, e={e} does not have exactly L={expected_L} per prompt.")
    return {k: math.fsum(v)/len(v) for k,v in by.items()}


# ---------------------------------------------------------------------------
# OPE / overlap diagnostics / bootstrap / gate application
# ---------------------------------------------------------------------------


def logsumexp(xs: Sequence[float]) -> float:
    if not xs:
        return -math.inf
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(math.fsum(math.exp(x-m) for x in xs))


def prompt_stats(rows: Sequence[Mapping[str, Any]], k_expected: int) -> list[dict[str, Any]]:
    import statistics
    by: dict[str, list[Mapping[str, Any]]] = {}
    for r in rows:
        by.setdefault(str(r["prompt_id"]), []).append(r)
    out: list[dict[str, Any]] = []
    for pid, rs in sorted(by.items()):
        if len(rs) != k_expected:
            raise Program08Error(f"Prompt {pid} has K={len(rs)}, expected {k_expected}.")
        logw = [float(r["log_weight"]) for r in rs]
        reward = [float(r["correctness_reward"]) for r in rs]
        lse1 = logsumexp(logw); lse2 = logsumexp([2*x for x in logw])
        log_ess = 2*lse1-lse2
        ess = math.exp(log_ess) if log_ess < math.log(sys.float_info.max) else math.inf
        ress = ess / k_expected
        d2 = lse2 - math.log(k_expected)
        cmax = math.exp(max(logw)-lse1)
        # Stable pWIS numerator; if no rewards=1, numerator is zero.
        pos = [lw for lw,r in zip(logw,reward) if r > 0.5]
        pwis = 0.0 if not pos else math.exp(logsumexp(pos)-lse1)
        mean_logw = math.fsum(logw)/k_expected
        sd_logw = statistics.pstdev(logw) if k_expected > 1 else 0.0
        lengths = [int(r["completion_length"]) for r in rs]
        token_count = sum(lengths)
        token_kl = -math.fsum(float(r["log_weight"]) for r in rs) / token_count
        out.append({
            "prompt_id": pid, "K": k_expected, "logw": logw, "reward": reward,
            "pwis": pwis, "ess": ess, "ress": ress, "d2": d2, "cmax": cmax,
            "mean_logw": mean_logw, "sd_logw": sd_logw,
            "mean_abs_logw": math.fsum(abs(x) for x in logw)/k_expected,
            "tokenwise_kl_proxy": token_kl,
            "mean_length": math.fsum(lengths)/k_expected,
            "truncation_rate": math.fsum(float(bool(r["was_truncated"])) for r in rs)/k_expected,
        })
    return out


def ordinary_is(stats: Sequence[Mapping[str, Any]]) -> tuple[float | None, str]:
    # Work with prompt log-numerators and only exponentiate at the final scale.
    vals_log: list[float] = []
    for s in stats:
        pos = [lw for lw,r in zip(s["logw"], s["reward"]) if r > 0.5]
        vals_log.append(-math.inf if not pos else logsumexp(pos)-math.log(int(s["K"])))
    finite = [x for x in vals_log if x != -math.inf]
    if not finite:
        return 0.0, "ok"
    log_est = logsumexp(finite)-math.log(len(stats))
    if log_est > math.log(sys.float_info.max):
        return None, "nonfinite_estimator"
    return math.exp(log_est), "ok"


def percentile(xs: Sequence[float], q: float) -> float:
    import numpy as np  # type: ignore
    return float(np.quantile(np.asarray(xs, dtype=float), q))


def aggregate_diagnostics(stats: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    import statistics
    return {
        "median_prompt_ess": float(statistics.median(float(s["ess"]) for s in stats)),
        "median_prompt_relative_ess": float(statistics.median(float(s["ress"]) for s in stats)),
        "p10_prompt_relative_ess": percentile([float(s["ress"]) for s in stats], .10),
        "mean_d2": math.fsum(float(s["d2"]) for s in stats)/len(stats),
        "median_d2": float(statistics.median(float(s["d2"]) for s in stats)),
        "mean_max_normalized_weight": math.fsum(float(s["cmax"]) for s in stats)/len(stats),
        "p90_max_normalized_weight": percentile([float(s["cmax"]) for s in stats], .90),
        "tokenwise_kl_proxy": math.fsum(float(s["tokenwise_kl_proxy"])*float(s["mean_length"])*int(s["K"]) for s in stats) / math.fsum(float(s["mean_length"])*int(s["K"]) for s in stats),
        "mean_abs_log_weight": math.fsum(float(s["mean_abs_logw"]) for s in stats)/len(stats),
        "mean_log_weight": math.fsum(float(s["mean_logw"]) for s in stats)/len(stats),
        "sd_log_weight": float(statistics.pstdev([float(s["mean_logw"]) for s in stats])) if len(stats)>1 else 0.0,
        "mean_completion_length": math.fsum(float(s["mean_length"]) for s in stats)/len(stats),
        "truncation_rate": math.fsum(float(s["truncation_rate"]) for s in stats)/len(stats),
    }


def bootstrap_pair(stats: Sequence[Mapping[str, Any]], online: Mapping[str,float], estimator: str, reps: int, seed: int) -> dict[str, Any]:
    import numpy as np  # type: ignore
    pids = [str(s["prompt_id"]) for s in stats]
    if set(pids) != set(online):
        raise Program08Error("Rescore and online prompt sets differ.")
    pw = np.asarray([float(s["pwis"]) for s in stats], dtype=np.float64)
    on = np.asarray([float(online[p]) for p in pids], dtype=np.float64)
    if estimator == "is":
        prompt_is = []
        for s in stats:
            pos = [float(lw) for lw,r in zip(s["logw"], s["reward"]) if float(r) > .5]
            if not pos:
                prompt_is.append(0.0)
                continue
            log_prompt_is = logsumexp(pos) - math.log(int(s["K"]))
            if log_prompt_is > math.log(sys.float_info.max):
                prompt_is.append(math.inf)
            else:
                prompt_is.append(math.exp(log_prompt_is))
        ope = np.asarray(prompt_is, dtype=np.float64)
    else:
        ope = pw
    rng = np.random.default_rng(seed)
    vals_ope: list[float] = []; vals_on: list[float] = []; vals_signed: list[float] = []; vals_abs: list[float] = []
    nonfinite = 0
    M = len(stats); chunk = 256
    for start in range(0, reps, chunk):
        n = min(chunk, reps-start)
        idx = rng.integers(0, M, size=(n,M), endpoint=False)
        bo = np.mean(ope[idx], axis=1); bn = np.mean(on[idx], axis=1)
        for a,b in zip(bo.tolist(), bn.tolist()):
            if not math.isfinite(a):
                nonfinite += 1; continue
            vals_ope.append(a); vals_on.append(b); vals_signed.append(a-b); vals_abs.append(abs(a-b))
    def ci(v: Sequence[float]) -> tuple[float|None,float|None]:
        return (None,None) if not v else (percentile(v,.025), percentile(v,.975))
    o1,o2=ci(vals_ope); n1,n2=ci(vals_on); s1,s2=ci(vals_signed); a1,a2=ci(vals_abs)
    return {
        "ope_ci_low":o1,"ope_ci_high":o2,"online_ci_low":n1,"online_ci_high":n2,
        "signed_error_ci_low":s1,"signed_error_ci_high":s2,"absolute_error_ci_low":a1,"absolute_error_ci_high":a2,
        "bootstrap_nonfinite_fraction": nonfinite/reps,
    }


def frozen_thresholds(gate: Mapping[str, Any], section: str) -> list[dict[str, Any]]:
    sec = gate.get(section)
    if not isinstance(sec, Mapping) or not isinstance(sec.get("thresholds"), list):
        raise Program08Error(f"Frozen gate missing {section}.thresholds")
    return [dict(x) for x in sec["thresholds"] if isinstance(x, Mapping)]


def apply_gate(value: float, threshold: float, direction: str) -> bool:
    if direction == "ge": return value >= threshold
    if direction == "le": return value <= threshold
    raise Program08Error(f"Unknown frozen gate direction: {direction}")


def analyze_pair(
    *, rows: Sequence[Mapping[str, Any]], online: Mapping[str,float], pair: PairTemplate,
    seed: int, dataset_revision: str, K: int, L: int, bootstrap_reps: int, bootstrap_seed: int,
    gate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    stats = prompt_stats(rows, K)
    if len(stats) != len(online):
        raise Program08Error("SVAMP rescore and online references do not cover the same number of prompts.")
    online_value = math.fsum(float(online[s["prompt_id"]]) for s in stats)/len(stats)
    diag = aggregate_diagnostics(stats)
    out: list[dict[str, Any]] = []
    for estimator in ("is", "prompt_wis"):
        if estimator == "is": estimate,status = ordinary_is(stats)
        else: estimate,status = math.fsum(float(s["pwis"]) for s in stats)/len(stats), "ok"
        signed = None if estimate is None else estimate-online_value
        bsum = bootstrap_pair(stats, online, estimator, bootstrap_reps, stable_seed("svamp_bootstrap",bootstrap_seed,seed,pair.behavior_step,pair.target_step,estimator))
        base = {
            "dataset":"SVAMP","dataset_revision":dataset_revision,"training_seed":seed,
            "behavior_step":pair.behavior_step,"target_step":pair.target_step,"pair_purpose":pair.purpose,
            "overlap_band":pair.overlap_band,"representative_rank":pair.representative_rank,"estimator":estimator,
            "estimate":estimate,"estimate_status":status,"online_reference":online_value,"signed_error":signed,
            "absolute_error":None if signed is None else abs(signed),**bsum,"bootstrap_reps":bootstrap_reps,
            "n_prompts":len(stats),"K":K,"L":L,**diag,
        }
        if estimator != "prompt_wis":
            out.append({**base,"gate_kind":"not_applicable","gate_tolerance":None,"gate_diagnostic":None,
                        "gate_threshold":None,"gate_decision":None,"gate_accepted":None,"gate_reliable":None,
                        "gate_false_accept":None,"gate_false_reject":None})
            continue
        # Apply primary frozen gate and the one predeclared KL baseline. No fitting.
        for section, kind, diagnostic, value in (
            ("primary_gate","primary_ress","median_prompt_relative_ess",float(diag["median_prompt_relative_ess"])),
            ("baseline","kl_baseline","tokenwise_kl_proxy",float(diag["tokenwise_kl_proxy"])),
        ):
            for tr in frozen_thresholds(gate, section):
                tol=float(tr["tolerance"]); threshold=float(tr["threshold"]); direction=str(tr["direction"])
                accepted=apply_gate(value,threshold,direction)
                reliable = False if estimate is None else abs(float(estimate)-online_value) <= tol
                out.append({**base,"gate_kind":kind,"gate_tolerance":tol,"gate_diagnostic":diagnostic,
                            "gate_threshold":threshold,"gate_decision":"accept" if accepted else "refresh",
                            "gate_accepted":accepted,"gate_reliable":reliable,
                            "gate_false_accept":bool(accepted and not reliable),
                            "gate_false_reject":bool((not accepted) and reliable)})
    return out


# ---------------------------------------------------------------------------
# Final summaries / manifests
# ---------------------------------------------------------------------------


def summarize_gate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str,float,str], list[Mapping[str,Any]]] = {}
    for r in rows:
        if r.get("estimator") != "prompt_wis" or r.get("gate_kind") not in ("primary_ress","kl_baseline"):
            continue
        grouped.setdefault((str(r["gate_kind"]),float(r["gate_tolerance"]),str(r["overlap_band"])),[]).append(r)
    out=[]
    for (kind,tol,band), rs in sorted(grouped.items()):
        acc=[r for r in rs if bool(r["gate_accepted"])]
        rej=[r for r in rs if not bool(r["gate_accepted"])]
        fa=sum(bool(r["gate_false_accept"]) for r in rs)
        out.append({
            "dataset":"SVAMP","gate_kind":kind,"tolerance":tol,"overlap_band":band,"n_pair_seed_cases":len(rs),
            "accepted_count":len(acc),"accept_rate":len(acc)/len(rs),"false_accept_count":fa,
            "false_accept_rate_among_accepted":0.0 if not acc else fa/len(acc),
            "false_accept_fraction_all":fa/len(rs),
            "accepted_mae":None if not acc else math.fsum(float(r["absolute_error"]) for r in acc)/len(acc),
            "rejected_mae":None if not rej else math.fsum(float(r["absolute_error"]) for r in rej)/len(rej),
        })
    return out


def safe_reset(path: Path) -> None:
    if os.environ.get("GRPO_OPE_ALLOW_RESET") != "YES":
        raise Program08Error("Dangerous reset requires environment variable GRPO_OPE_ALLOW_RESET=YES.")
    if path.exists():
        for p in path.rglob("*"):
            if p.is_file():
                try: p.chmod(stat.S_IWUSR | stat.S_IRUSR)
                except OSError: pass
        shutil.rmtree(path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap=argparse.ArgumentParser(description="Program 08: frozen SVAMP external robustness using existing GRPO checkpoints and frozen reuse gate.")
    ap.add_argument("--config",default="configs/protocol.yaml")
    ap.add_argument("--mode",choices=("smoke","pilot","paper"),default="paper")
    ap.add_argument("--split",choices=("test",),default="test",help="SVAMP is external robustness only; kept for unified CLI compatibility.")
    ap.add_argument("--resume",action="store_true")
    ap.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    ap.add_argument("--output-root",default=".")
    ap.add_argument("--verify-only",action="store_true")
    ap.add_argument("--reset-checkpoints",action="store_true")
    return ap.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> int:
    args=parse_args(argv); started=time.time(); root=Path(args.output_root).resolve(); cfg_path=root/args.config
    cfg,cfg_sha=load_protocol(cfg_path); pv=protocol_version(cfg)
    env=verify_environment_manifest(root/"manifests"/"environment_manifest.json")
    data_manifest,svamp_record=verify_data_and_svamp(root)
    _,model_record=verify_model_manifest(root)
    verify_protocol_lock(root,cfg_sha)
    gate,gate_manifest=verify_frozen_gate(root,cfg_sha,pv)
    p06,t02_sha,t04_sha=verify_program06_development(root)
    spec=resolve_spec(cfg,gate)
    main_test_proof=verify_main_official_test_complete(root,spec.seeds)

    gate_sha=sha256_file(root/"outputs"/"frozen_gate.json")
    p06_sha=sha256_file(root/"outputs"/"diagnostics"/"program06_development_manifest.json")

    # Critical ordering guarantee: freeze/verify pair registry before opening SVAMP.json.
    pairs=freeze_or_verify_svamp_pair_registry(root=root,spec=spec,pv=pv,cfg_sha=cfg_sha,program06_sha=p06_sha,t02_sha=t02_sha,t04_sha=t04_sha,gate_sha=gate_sha)

    prompt_limit=None
    active_pairs=pairs
    active_seeds=list(spec.seeds)
    active_spec=spec
    if args.mode=="smoke":
        prompt_limit=8; active_pairs=pairs[:1]; active_seeds=[spec.seeds[0]]
        active_spec=RobustnessSpec(**{**spec.__dict__,"k":2,"l_main":2,"l_audit":2,"sample_block_size":2,"prompts_per_shard":4,"bootstrap_reps":100})
    elif args.mode=="pilot":
        prompt_limit=50; active_pairs=pairs[:2]; active_seeds=[spec.seeds[0]]
        active_spec=RobustnessSpec(**{**spec.__dict__,"k":4,"l_main":4,"l_audit":4,"sample_block_size":4,"prompts_per_shard":8,"bootstrap_reps":200})

    required_steps=sorted({x for p in active_pairs for x in (p.behavior_step,p.target_step)})
    trainings:dict[int,dict[str,Any]]={}; adapters:dict[int,dict[int,AdapterRecord]]={}
    for seed in active_seeds:
        tm,am=verify_training_seed(root,seed,required_steps); trainings[seed]=tm; adapters[seed]=am

    # Only now may SVAMP question/reward rows be opened.
    prompts=load_svamp_prompts(root,svamp_record,limit=prompt_limit)
    dataset_revision=str(svamp_record.get("resolved_revision")); model_revision=str(model_record.get("resolved_revision"))
    model_dir=root/"models"/"qwen25_05b"; tokenizer=load_tokenizer(model_dir,model_record)
    base=robustness_root(root,args.mode)
    if args.reset_checkpoints: safe_reset(base)
    cleanup_staging(base,args.resume)

    print("="*78)
    print("PROGRAM 08 — FROZEN SVAMP EXTERNAL ROBUSTNESS")
    print(f"mode / dataset          : {args.mode} / SVAMP")
    print(f"protocol version        : {pv}")
    print(f"config SHA              : {cfg_sha}")
    print(f"SVAMP revision          : {dataset_revision}")
    print(f"SVAMP prompts           : {len(prompts)}")
    print(f"training seeds          : {active_seeds}")
    print(f"representative pairs    : {[(p.behavior_step,p.target_step,p.overlap_band) for p in active_pairs]}")
    print(f"frozen gate SHA         : {gate_sha}")
    print("pair selection leakage  : NONE — registry frozen before SVAMP row content/rewards parsed")
    print(f"output root             : {base}")
    print("="*78)

    if args.verify_only:
        # Verification mode intentionally does not create missing expensive assets.
        for seed in active_seeds:
            for p in active_pairs:
                load_behavior_rows(base,seed,p.behavior_step)
                load_rescore_rows(base,seed,p.behavior_step,p.target_step)
                load_online_prompt_accuracy(base,seed,p.target_step,target_l(active_spec,p.target_step))
        print("PROGRAM 08 VERIFY-ONLY PASSED")
        return 0

    # 1) behavior logs shared across selected pairs with the same b.
    for seed in active_seeds:
        for b in sorted({p.behavior_step for p in active_pairs}):
            collect_behavior(base=base,model_dir=model_dir,tokenizer=tokenizer,model_record=model_record,adapter=adapters[seed][b],seed=seed,b=b,prompts=prompts,spec=active_spec,pv=pv,dataset_revision=dataset_revision,device=args.device,resume=args.resume)

    # 2) identity check all behavior checkpoints BEFORE any nonidentity target rescore.
    identity=[]
    for seed in active_seeds:
        for b in sorted({p.behavior_step for p in active_pairs}):
            identity.append(identity_check_behavior(base=base,model_dir=model_dir,adapter=adapters[seed][b],seed=seed,b=b,device=args.device))
    identity_path=base/"identity_checks.json"
    atomic_write_json(identity_path,{"schema_version":"1.0","dataset":"SVAMP","identity_checks":identity,"all_pass":True})

    # 3) selected target rescoring only.
    for seed in active_seeds:
        for p in active_pairs:
            collect_rescore(base=base,model_dir=model_dir,seed=seed,pair=p,target_adapter=adapters[seed][p.target_step],device=args.device,spec=active_spec)

    # 4) online reference once per unique target, shared across all selected behavior pairs.
    for seed in active_seeds:
        for e in sorted({p.target_step for p in active_pairs}):
            collect_online(base=base,model_dir=model_dir,tokenizer=tokenizer,model_record=model_record,adapter=adapters[seed][e],seed=seed,e=e,prompts=prompts,spec=active_spec,pv=pv,dataset_revision=dataset_revision,device=args.device)

    # 5) CPU statistics + frozen gate application.
    all_rows:list[dict[str,Any]]=[]
    for seed in active_seeds:
        for p in active_pairs:
            rr=load_rescore_rows(base,seed,p.behavior_step,p.target_step)
            L=target_l(active_spec,p.target_step)
            on=load_online_prompt_accuracy(base,seed,p.target_step,L)
            all_rows.extend(analyze_pair(rows=rr,online=on,pair=p,seed=seed,dataset_revision=dataset_revision,K=active_spec.k,L=L,bootstrap_reps=active_spec.bootstrap_reps,bootstrap_seed=active_spec.bootstrap_seed,gate=gate))

    tables=root/"outputs"/"tables"; diagnostics=root/"outputs"/"diagnostics"; tables.mkdir(parents=True,exist_ok=True); diagnostics.mkdir(parents=True,exist_ok=True)
    t08=tables/"T08_svamp.csv" if args.mode=="paper" else tables/f"T08_svamp_{args.mode}.csv"
    atomic_write_csv(t08,all_rows,T08_COLUMNS)
    summary=summarize_gate_rows(all_rows)
    sp=diagnostics/("T08_svamp_gate_summary.csv" if args.mode=="paper" else f"T08_svamp_gate_summary_{args.mode}.csv")
    summary_cols=("dataset","gate_kind","tolerance","overlap_band","n_pair_seed_cases","accepted_count","accept_rate","false_accept_count","false_accept_rate_among_accepted","false_accept_fraction_all","accepted_mae","rejected_mae")
    atomic_write_csv(sp,summary,summary_cols)

    manifest={
        "schema_version":ROBUSTNESS_SCHEMA,"manifest_type":"program08_svamp_robustness_manifest","created_at_utc":now_utc(),
        "created_by":{"program":PROGRAM,"program_version":PROGRAM_VERSION},"mode":args.mode,"dataset":"SVAMP",
        "protocol_version":pv,"protocol_config_sha256":cfg_sha,"git_commit":git_commit(root),
        "research_boundary":{"retrained_grpo":False,"refit_gate":False,"selected_pairs_using_svamp":False,"new_ablation_matrix":False},
        "inputs":{
            "environment_manifest_fingerprint":env.get("environment_fingerprint_sha256"),
            "svamp_revision":dataset_revision,"svamp_tree_sha256":svamp_record.get("tree_sha256"),
            "model_revision":model_revision,"frozen_gate_sha256":gate_sha,
            "svamp_pair_registry_sha256":sha256_file(pair_registry_paths(root)[0]),
            "program06_development_manifest_sha256":p06_sha,"T02_development_sha256":t02_sha,"T04_development_sha256":t04_sha,
            **main_test_proof,
        },
        "sampling":{"K":active_spec.k,"L_main":active_spec.l_main,"L_audit":active_spec.l_audit,"audit_steps":list(active_spec.audit_steps),
                    "temperature":active_spec.temperature,"top_p":active_spec.top_p,"top_k":active_spec.top_k,"repetition_penalty":active_spec.repetition_penalty,
                    "max_completion_length":active_spec.max_completion_length},
        "counts":{"prompts":len(prompts),"seeds":len(active_seeds),"representative_pairs":len(active_pairs),"T08_rows":len(all_rows)},
        "identity_checks_path":rel(identity_path,root),"identity_checks_sha256":sha256_file(identity_path),
        "outputs":{"T08_svamp":{"path":rel(t08,root),"sha256":sha256_file(t08)},"gate_summary":{"path":rel(sp,root),"sha256":sha256_file(sp)}},
        "interpretation":"External prompt-distribution validation of the already-frozen GSM8K reuse gate; no SVAMP-based tuning is permitted.",
    }
    mp=diagnostics/("program08_svamp_manifest.json" if args.mode=="paper" else f"program08_svamp_{args.mode}_manifest.json")
    atomic_write_json(mp,manifest)

    elapsed=time.time()-started
    print("\n"+"="*78)
    print("PROGRAM 08 PASSED")
    print(f"T08 rows              : {len(all_rows)}")
    print(f"T08                    : {t08}")
    print(f"gate summary           : {sp}")
    print(f"manifest               : {mp}")
    print(f"elapsed                : {elapsed/60:.1f} min")
    print("No GRPO retraining, pair reselection, or gate refitting occurred.")
    print("="*78)
    return 0


def main(argv: Sequence[str] | None=None) -> int:
    try:
        return run(argv)
    except Program08Error as exc:
        print(f"\nPROGRAM 08 FAILED\n{exc}",file=sys.stderr); return 2
    except KeyboardInterrupt:
        print("\nPROGRAM 08 INTERRUPTED. Published shards remain valid; resume with --resume.",file=sys.stderr); return 130
    except Exception:
        print("\nPROGRAM 08 UNEXPECTED FAILURE",file=sys.stderr); traceback.print_exc(); return 1


if __name__=="__main__":
    raise SystemExit(main())
