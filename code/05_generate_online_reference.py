#!/usr/bin/env python3
"""GRPO-OPE Program 05: generate immutable on-policy Monte Carlo references.

Research boundary
-----------------
Program 05 directly samples every target GRPO checkpoint on the frozen prompt
set so later OPE estimates have an independent reference value.  For training
seed s, target checkpoint e, prompt x_j and sample index l,

    Y^(e)_{j,l} ~ pi_e(. | x_j),     l = 0,...,L_e-1,

and the paper reference is

    V_online(pi_e) = (1/M) sum_j (1/L_e) sum_l R(x_j, Y^(e)_{j,l}).

This is a high-precision on-policy Monte Carlo reference, NOT exact truth.
Program 05 does not compute IS/pWIS/DR, overlap diagnostics, a reuse gate, or
paper figures.

Paper invariants
----------------
* Program 00 data/model/environment manifests are re-verified.
* Program 01's immutable split registry is re-verified.
* Program 02's completed GRPO training paths and exact target adapters are
  re-verified before generation.
* Program 04's identity gate must have passed for every selected seed/split
  before Program 05 spends GPU time producing references.
* Target checkpoints are the frozen OPE grid 0,20,...,400.
* Online generation uses exactly the same stochastic policy definition as OPE:
  temperature=1, top_p=1, top_k=0, repetition_penalty=1, same tokenizer/chat
  template/prompt template/max completion length.
* Online samples depend only on (seed,target,prompt,sample_index), never on a
  behavior anchor. Fixed-log and rolling-log experiments therefore share the
  same on-policy samples.
* Stable online IDs follow the code-design memo:

      hash(protocol_version, dataset_revision, training_seed, target_step,
           split, prompt_id, sample_index)

* L is append-only. Development may extend L_main from 8 to 16; predeclared
  audit steps use L_audit=32. Existing samples are never overwritten.
* Official test forbids sample-size overrides and is legal only after the
  frozen protocol and frozen reuse gate are verifiable.
* Each expensive prompt shard is published by staging -> fsync -> atomic rename
  with file and semantic SHA-256 hashes. Resume skips only verified units.
* Prompt-level success counts and target-level reference summaries are derived
  from immutable samples and can be rebuilt without regenerating the model.

Engineering modes
-----------------
--mode smoke and --mode pilot are development-only, isolated from paper assets,
and use only a small prompt/checkpoint/sample subset.
"""

from __future__ import annotations

import argparse
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
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROGRAM = "05_generate_online_reference.py"
PROGRAM_VERSION = "1.1.1"
MANIFEST_SCHEMA = "1.0"
ONLINE_SCHEMA = "1.0"
POLICY_INFERENCE_PRECISION = "fp32"

PROJECT_NAME = "grpo_ope_reuse"
GSM8K_REPO = "openai/gsm8k"
GSM8K_CONFIG = "main"
MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"

EXPECTED_RESEARCH_ROWS = {"training": 6000, "development": 1473, "test": 1319}
PAPER_TARGET_STEPS = tuple(range(0, 401, 20))
PAPER_AUDIT_STEPS = (0, 100, 200, 300, 400)
PAPER_ALLOWED_L_MAIN = (8, 16)
PAPER_L_AUDIT = 32
DEFAULT_SAMPLE_BLOCK_SIZE = 8
DEFAULT_GENERATION_BATCH_SIZE = 4
DEFAULT_PROMPTS_PER_SHARD = 8

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
GSM8K_GOLD_RE = re.compile(r"####\s*(.*?)\s*$", re.DOTALL)
NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/[-+]?\d[\d,]*)?")
HEX64_RE = re.compile(r"[0-9a-f]{64}")

ONLINE_COLUMNS = (
    "online_id", "dataset", "dataset_revision", "protocol_version",
    "training_seed", "target_step", "split", "prompt_id", "source_row_index",
    "sample_index", "completion_token_ids", "completion_length",
    "terminated_with_eos", "was_truncated", "parsed_answer", "correct",
    "parser_status", "temperature", "top_p", "top_k", "repetition_penalty",
    "max_completion_length", "generation_seed", "generation_call_id",
    "model_revision", "tokenizer_revision", "chat_template_hash",
    "prompt_template_hash", "target_adapter_sha256", "generation_config_sha256",
    "completion_text",
)


class Program05Error(RuntimeError):
    """Controlled Program 05 failure with an actionable message."""


@dataclass(frozen=True)
class OnlineSpec:
    target_steps: tuple[int, ...]
    l_main: int
    l_audit: int
    audit_steps: tuple[int, ...]
    sample_block_size: int
    generation_batch_size: int
    prompts_per_shard: int
    max_completion_length: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float


@dataclass(frozen=True)
class PromptRecord:
    prompt_id: str
    original_split: str
    research_split: str
    source_row_index: int
    question: str
    raw_gold_answer: str
    gold_answer: str


@dataclass(frozen=True)
class AdapterRecord:
    step: int
    path: Path
    payload_sha256: str
    manifest: Mapping[str, Any]


# ---------------------------------------------------------------------------
# Generic integrity utilities
# ---------------------------------------------------------------------------


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def canonical_bytes(obj: Any) -> bytes:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


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
        raise Program05Error(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise Program05Error(f"Expected a JSON object in {path}.")
    return obj


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


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


def payload_files(root: Path) -> list[Path]:
    if not root.exists():
        raise Program05Error(f"Missing asset directory: {root}")
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        parts = set(p.relative_to(root).parts)
        if parts & HASH_EXCLUDE_DIRS:
            continue
        out.append(p)
    return sorted(out, key=lambda p: p.relative_to(root).as_posix())


def tree_hash(root: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total = 0
    for p in payload_files(root):
        size = p.stat().st_size
        records.append({"path": rel(p, root), "size_bytes": size, "sha256": sha256_file(p)})
        total += size
    if not records:
        raise Program05Error(f"No files found in asset directory: {root}")
    return {
        "tree_sha256": sha256_bytes(canonical_bytes(records)),
        "file_count": len(records),
        "total_bytes": total,
    }


def adapter_payload_hash(root: Path) -> str:
    records: list[dict[str, Any]] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.name == "adapter_manifest.json":
            continue
        records.append({
            "path": p.relative_to(root).as_posix(),
            "size_bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })
    records.sort(key=lambda x: x["path"])
    if not records:
        raise Program05Error(f"Adapter contains no payload files: {root}")
    return sha256_bytes(canonical_bytes(records))


def make_read_only_tree(root: Path) -> None:
    failures: list[str] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            mode = stat.S_IMODE(p.stat().st_mode)
            p.chmod(mode & ~stat.S_IWUSR & ~stat.S_IWGRP & ~stat.S_IWOTH)
        except OSError as exc:
            failures.append(f"{p}: {exc}")
    if failures:
        print("[WARN] Could not mark every shard file read-only; hashes remain authoritative.")
        for msg in failures[:5]:
            print(f"       {msg}")


def git_commit(root: Path) -> str | None:
    try:
        cp = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10, check=False,
        )
        value = cp.stdout.strip()
        if cp.returncode == 0 and re.fullmatch(r"[0-9a-fA-F]{40}", value):
            return value.lower()
    except Exception:
        pass
    return None


def package_version(name: str) -> str | None:
    aliases = {"torch": "torch", "PyYAML": "PyYAML"}
    try:
        return metadata.version(aliases.get(name, name))
    except metadata.PackageNotFoundError:
        return None


def current_environment_fingerprint() -> tuple[str, dict[str, Any]]:
    packages = {p: package_version(p) for p in CORE_PACKAGES + EXTRA_PACKAGES}
    missing = [p for p in CORE_PACKAGES if packages[p] is None]
    if missing:
        raise Program05Error(
            "Current environment is missing packages frozen/required by Program 00: " + ", ".join(missing)
        )
    if packages.get("pyarrow") is None:
        raise Program05Error("pyarrow is required for immutable Parquet online-reference shards.")
    try:
        import torch  # type: ignore
        torch_cuda_build = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
    except Exception as exc:
        raise Program05Error(f"Cannot inspect PyTorch environment: {exc}") from exc
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
        raise Program05Error(f"Missing {path}. Program 00 must complete first.")
    env = read_json(path)
    if env.get("manifest_type") != "environment":
        raise Program05Error(f"Wrong manifest type in {path}.")
    expected = env.get("environment_fingerprint_sha256")
    observed, _ = current_environment_fingerprint()
    if expected != observed:
        raise Program05Error(
            "Current software environment differs from Program 00's frozen environment.\n"
            f"expected={expected}\nobserved={observed}"
        )
    return env


def verify_data_manifest(path: Path, gsm8k_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists():
        raise Program05Error(f"Missing {path}. Program 00 must freeze GSM8K first.")
    manifest = read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("manifest_type") != "data":
        raise Program05Error(f"Invalid Program 00 data manifest: {path}")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not isinstance(datasets.get("gsm8k"), dict):
        raise Program05Error("data_manifest.json lacks datasets.gsm8k.")
    gsm = datasets["gsm8k"]
    if gsm.get("repo_id") != GSM8K_REPO:
        raise Program05Error("Frozen primary dataset is not openai/gsm8k.")
    expected_fp = manifest.get("content_fingerprint_sha256")
    observed_fp = sha256_bytes(canonical_bytes({"research_scope": manifest.get("research_scope"), "datasets": datasets}))
    if expected_fp != observed_fp:
        raise Program05Error("Program 00 data manifest content fingerprint mismatch.")
    observed_tree = tree_hash(gsm8k_dir)["tree_sha256"]
    if observed_tree != gsm.get("tree_sha256"):
        raise Program05Error(
            "Pinned GSM8K local files differ from Program 00.\n"
            f"expected={gsm.get('tree_sha256')}\nobserved={observed_tree}"
        )
    return manifest, gsm


def tokenizer_template_hash(model_dir: Path) -> tuple[str, str]:
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise Program05Error(f"Cannot load pinned tokenizer locally: {exc}") from exc
    template = getattr(tok, "chat_template", None)
    if template is None:
        raise Program05Error("Pinned tokenizer has no chat_template.")
    text = template if isinstance(template, str) else json.dumps(template, ensure_ascii=False, sort_keys=True, default=str)
    return sha256_text(text), tok.__class__.__name__


def verify_model_manifest(path: Path, model_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.exists():
        raise Program05Error(f"Missing {path}. Program 00 must freeze the model first.")
    manifest = read_json(path)
    if manifest.get("schema_version") != MANIFEST_SCHEMA or manifest.get("manifest_type") != "model":
        raise Program05Error(f"Invalid Program 00 model manifest: {path}")
    models = manifest.get("models")
    if not isinstance(models, dict) or not isinstance(models.get("primary"), dict):
        raise Program05Error("model_manifest.json lacks models.primary.")
    model = models["primary"]
    if model.get("repo_id") != MODEL_REPO:
        raise Program05Error(f"Frozen primary model is not {MODEL_REPO}.")
    if manifest.get("content_fingerprint_sha256") != sha256_bytes(canonical_bytes(models)):
        raise Program05Error("Program 00 model manifest content fingerprint mismatch.")
    observed_tree = tree_hash(model_dir)["tree_sha256"]
    if observed_tree != model.get("tree_sha256"):
        raise Program05Error(
            "Pinned model local files differ from Program 00.\n"
            f"expected={model.get('tree_sha256')}\nobserved={observed_tree}"
        )
    validation = model.get("validation") or {}
    tok_validation = validation.get("tokenizer") if isinstance(validation, dict) else None
    if not isinstance(tok_validation, dict):
        raise Program05Error("Frozen model manifest lacks tokenizer validation metadata.")
    observed_chat, tok_class = tokenizer_template_hash(model_dir)
    if observed_chat != tok_validation.get("chat_template_sha256"):
        raise Program05Error("Pinned tokenizer chat template differs from Program 00 manifest.")
    if tok_class != tok_validation.get("class"):
        raise Program05Error("Tokenizer class differs from Program 00 validation.")
    return manifest, model


def verify_split_registry(
    manifest_path: Path,
    registry_path: Path,
    data_manifest: Mapping[str, Any],
    gsm_record: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not manifest_path.exists() or not registry_path.exists():
        raise Program05Error("Program 01 split registry/manifest is missing.")
    m = read_json(manifest_path)
    if m.get("schema_version") != MANIFEST_SCHEMA or m.get("manifest_type") != "split_registry":
        raise Program05Error("Invalid Program 01 split registry manifest.")
    source = m.get("source")
    registry = m.get("registry")
    firewall = m.get("research_firewall")
    if not all(isinstance(x, dict) for x in (source, registry, firewall)):
        raise Program05Error("Split registry manifest is missing required sections.")
    if source.get("dataset_revision") != gsm_record.get("resolved_revision"):
        raise Program05Error("Split registry dataset revision differs from Program 00.")
    if source.get("gsm8k_tree_sha256") != gsm_record.get("tree_sha256"):
        raise Program05Error("Split registry source tree differs from Program 00.")
    if source.get("program00_data_manifest_content_fingerprint_sha256") != data_manifest.get("content_fingerprint_sha256"):
        raise Program05Error("Split registry was created from a different Program 00 data manifest.")
    if firewall.get("official_test_is_never_used_for_split_tuning") is not True:
        raise Program05Error("Split registry does not preserve the official-test firewall.")
    if registry.get("file_sha256") != sha256_file(registry_path):
        raise Program05Error("Immutable split registry file SHA-256 mismatch.")
    try:
        import pyarrow.parquet as pq  # type: ignore
        rows = pq.read_table(registry_path).to_pylist()
    except Exception as exc:
        raise Program05Error(f"Cannot read immutable split registry: {exc}") from exc
    counts = {"training": 0, "development": 0, "test": 0}
    seen: set[str] = set()
    for r in rows:
        split = r.get("research_split")
        if split not in counts:
            raise Program05Error(f"Unexpected research_split={split!r} in registry.")
        counts[str(split)] += 1
        pid = r.get("prompt_id")
        if not isinstance(pid, str) or HEX64_RE.fullmatch(pid) is None or pid in seen:
            raise Program05Error("Split registry contains invalid/duplicate prompt_id.")
        seen.add(pid)
        if r.get("original_split") == "test" and split != "test":
            raise Program05Error("Official GSM8K test row leaked into another research split.")
    if counts != EXPECTED_RESEARCH_ROWS:
        raise Program05Error(f"Unexpected split counts: {counts}; expected {EXPECTED_RESEARCH_ROWS}.")
    columns = registry.get("columns")
    if isinstance(columns, list) and columns:
        h = hashlib.sha256()
        for row in rows:
            try:
                canonical = {k: row[k] for k in columns}
            except KeyError as exc:
                raise Program05Error(f"Split registry missing canonical column {exc}.") from exc
            h.update(canonical_bytes(canonical))
            h.update(b"\n")
        if h.hexdigest() != registry.get("content_sha256"):
            raise Program05Error("Immutable split registry semantic content SHA-256 mismatch.")
    return m, rows


def load_protocol(path: Path) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise Program05Error(f"Missing protocol config {path}.")
    try:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        raise Program05Error(f"Cannot parse protocol config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise Program05Error("protocol.yaml must contain a YAML mapping.")
    project = cfg.get("project") or {}
    if not isinstance(project, dict):
        raise Program05Error("protocol.yaml project must be a mapping.")
    if project.get("name") not in (None, PROJECT_NAME):
        raise Program05Error(f"Unexpected project.name={project.get('name')!r}.")
    return cfg, sha256_file(path)


def _dig(mapping: Mapping[str, Any], path: Sequence[str]) -> Any:
    cur: Any = mapping
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return None
        cur = cur[key]
    return cur


def first_present(mapping: Mapping[str, Any], paths: Sequence[Sequence[str]]) -> Any:
    for path in paths:
        value = _dig(mapping, path)
        if value is not None:
            return value
    return None


def protocol_version(cfg: Mapping[str, Any]) -> str:
    project = cfg.get("project") or {}
    value = project.get("protocol_version") if isinstance(project, Mapping) else None
    return str(value if value is not None else "1.0")


def configured_seeds(cfg: Mapping[str, Any]) -> tuple[int, ...]:
    t = cfg.get("training") or {}
    raw = t.get("seeds", [20260826, 20260827, 20260828]) if isinstance(t, Mapping) else []
    if not isinstance(raw, list) or not raw:
        raise Program05Error("training.seeds must be a non-empty list.")
    out: list[int] = []
    for x in raw:
        if isinstance(x, bool) or not isinstance(x, int) or x < 0:
            raise Program05Error("training.seeds must contain non-negative integers.")
        out.append(x)
    if len(set(out)) != len(out):
        raise Program05Error("training.seeds contains duplicates.")
    return tuple(out)


def _positive_int(mapping: Mapping[str, Any], name: str, default: int) -> int:
    value = mapping.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Program05Error(f"{name} must be a positive integer.")
    return int(value)


def parse_online_spec(cfg: Mapping[str, Any]) -> OnlineSpec:
    online = cfg.get("online_reference") or {}
    behavior = cfg.get("behavior") or {}
    training = cfg.get("training") or {}
    ope = cfg.get("ope") or {}
    if not all(isinstance(x, Mapping) for x in (online, behavior, training, ope)):
        raise Program05Error("protocol online_reference/behavior/training/ope sections must be mappings.")

    raw_steps = ope.get("target_steps", list(PAPER_TARGET_STEPS))
    if not isinstance(raw_steps, list) or not raw_steps:
        raise Program05Error("ope.target_steps must be a non-empty integer list.")
    target_steps = tuple(int(x) for x in raw_steps if isinstance(x, int) and not isinstance(x, bool))
    if len(target_steps) != len(raw_steps) or len(set(target_steps)) != len(target_steps) or any(x < 0 for x in target_steps):
        raise Program05Error("ope.target_steps must contain unique non-negative integers.")
    if tuple(sorted(target_steps)) != target_steps:
        raise Program05Error("ope.target_steps must be strictly increasing.")

    raw_audit = online.get("audit_steps", list(PAPER_AUDIT_STEPS))
    if not isinstance(raw_audit, list):
        raise Program05Error("online_reference.audit_steps must be an integer list.")
    audit_steps = tuple(int(x) for x in raw_audit if isinstance(x, int) and not isinstance(x, bool))
    if len(audit_steps) != len(raw_audit) or len(set(audit_steps)) != len(audit_steps):
        raise Program05Error("online_reference.audit_steps must contain unique integers.")
    if any(x not in target_steps for x in audit_steps):
        raise Program05Error("Every online_reference.audit_steps entry must also be an OPE target step.")

    l_main = _positive_int(online, "L_main", 8)
    l_audit = _positive_int(online, "L_audit", 32)
    sample_block_size = _positive_int(online, "sample_block_size", DEFAULT_SAMPLE_BLOCK_SIZE)
    generation_batch_size = _positive_int(online, "generation_batch_size", DEFAULT_GENERATION_BATCH_SIZE)
    prompts_per_shard = _positive_int(online, "prompts_per_shard", DEFAULT_PROMPTS_PER_SHARD)
    if generation_batch_size > sample_block_size:
        raise Program05Error("online_reference.generation_batch_size may not exceed sample_block_size.")

    # The on-policy reference MUST sample the exact same stochastic policy class
    # whose probabilities are used by Programs 03/04.  Program 03 resolves
    # behavior.* first and training.* second; mirror that contract here.
    def policy_value(name: str, default: Any) -> Any:
        return behavior.get(name, training.get(name, default))

    canonical_policy = {
        "temperature": float(policy_value("temperature", 1.0)),
        "top_p": float(policy_value("top_p", 1.0)),
        "top_k": int(policy_value("top_k", 0)),
        "repetition_penalty": float(policy_value("repetition_penalty", 1.0)),
        "max_completion_length": policy_value("max_completion_length", 128),
    }
    if isinstance(canonical_policy["max_completion_length"], bool) or not isinstance(canonical_policy["max_completion_length"], int):
        raise Program05Error("behavior/training max_completion_length must be an integer.")
    if int(canonical_policy["max_completion_length"]) <= 0:
        raise Program05Error("behavior/training max_completion_length must be positive.")

    # online_reference may redundantly state policy fields for readability, but
    # it is forbidden to define a different policy.
    for name, expected in canonical_policy.items():
        if name not in online:
            continue
        observed = online[name]
        if name in {"temperature", "top_p", "repetition_penalty"}:
            if isinstance(observed, bool) or not isinstance(observed, (int, float)) or not math.isclose(float(observed), float(expected)):
                raise Program05Error(f"online_reference.{name} conflicts with the Program 03 behavior-policy definition.")
        else:
            if isinstance(observed, bool) or not isinstance(observed, int) or int(observed) != int(expected):
                raise Program05Error(f"online_reference.{name} conflicts with the Program 03 behavior-policy definition.")

    spec = OnlineSpec(
        target_steps=target_steps, l_main=l_main, l_audit=l_audit, audit_steps=audit_steps,
        sample_block_size=sample_block_size, generation_batch_size=generation_batch_size,
        prompts_per_shard=prompts_per_shard, max_completion_length=int(canonical_policy["max_completion_length"]),
        temperature=float(canonical_policy["temperature"]), top_p=float(canonical_policy["top_p"]),
        top_k=int(canonical_policy["top_k"]), repetition_penalty=float(canonical_policy["repetition_penalty"]),
    )
    if spec.temperature <= 0 or not (0 < spec.top_p <= 1) or spec.top_k < 0 or spec.repetition_penalty <= 0:
        raise Program05Error("Invalid shared target-policy sampling configuration.")
    if spec.l_audit < spec.l_main:
        raise Program05Error("L_audit cannot be smaller than L_main.")
    return spec


def online_spec_fingerprint(spec: OnlineSpec) -> str:
    return sha256_bytes(canonical_bytes(asdict(spec)))


def validate_paper_contract(spec: OnlineSpec, seeds: Sequence[int]) -> None:
    if len(seeds) < 3:
        raise Program05Error("Paper mode requires at least three GRPO training seeds.")
    if spec.target_steps != PAPER_TARGET_STEPS:
        raise Program05Error("Paper target checkpoint grid must be exactly 0,20,...,400.")
    if spec.audit_steps != PAPER_AUDIT_STEPS:
        raise Program05Error(f"Paper audit checkpoint grid must be {list(PAPER_AUDIT_STEPS)}.")
    if spec.l_main not in PAPER_ALLOWED_L_MAIN:
        raise Program05Error(f"Paper L_main must be one of {PAPER_ALLOWED_L_MAIN}.")
    if spec.l_audit != PAPER_L_AUDIT:
        raise Program05Error(f"Paper L_audit must equal {PAPER_L_AUDIT}.")
    if not math.isclose(spec.temperature, 1.0) or not math.isclose(spec.top_p, 1.0):
        raise Program05Error("Paper online policy requires temperature=1 and top_p=1.")
    if spec.top_k != 0 or not math.isclose(spec.repetition_penalty, 1.0):
        raise Program05Error("Paper online policy requires top_k=0 and repetition_penalty=1.")


def verify_protocol_lock(
    path: Path,
    *,
    config_sha256: str,
    data_manifest: Mapping[str, Any],
    model_record: Mapping[str, Any],
    split_manifest: Mapping[str, Any],
    online_spec: OnlineSpec,
) -> dict[str, Any]:
    if not path.exists():
        raise Program05Error(f"Paper mode requires frozen {path}.")
    lock = read_json(path)
    locked_config = first_present(lock, (("config_sha256",), ("protocol_config_sha256",), ("protocol", "config_sha256"), ("inputs", "config_sha256")))
    if not isinstance(locked_config, str):
        raise Program05Error("protocol_lock.json must contain a frozen config SHA-256.")
    if locked_config != config_sha256:
        raise Program05Error(f"protocol.yaml differs from protocol_lock.json. locked={locked_config} observed={config_sha256}")
    locked_data = first_present(lock, (("data_manifest_sha256",), ("inputs", "data_manifest_sha256"), ("data", "content_fingerprint_sha256")))
    if locked_data is not None and locked_data != data_manifest.get("content_fingerprint_sha256"):
        raise Program05Error("Program 00 data manifest differs from protocol lock.")
    locked_model = first_present(lock, (("model_sha",), ("model_revision",), ("model", "resolved_revision"), ("inputs", "model_revision")))
    if locked_model is not None and locked_model != model_record.get("resolved_revision"):
        raise Program05Error("Pinned Qwen revision differs from protocol lock.")
    locked_split = first_present(lock, (("split_registry_sha256",), ("inputs", "split_registry_sha256"), ("split", "content_fingerprint_sha256")))
    if locked_split is not None and locked_split != split_manifest.get("content_fingerprint_sha256"):
        raise Program05Error("Program 01 split registry differs from protocol lock.")
    locked_online = first_present(lock, (("online_reference_config_sha256",), ("online_reference", "resolved_config_sha256"), ("frozen", "online_reference_config_sha256")))
    if locked_online is not None and locked_online != online_spec_fingerprint(online_spec):
        raise Program05Error("Resolved online-reference configuration differs from protocol lock.")
    return lock


def verify_frozen_gate_for_test(root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    gate_path = root / "outputs" / "frozen_gate.json"
    if not gate_path.exists():
        raise Program05Error(
            "Official test online generation is forbidden before outputs/frozen_gate.json exists. "
            "Calibrate/freeze the gate on development first."
        )
    observed = sha256_file(gate_path)
    expected = first_present(lock, (("frozen_gate_sha256",), ("gate", "file_sha256"), ("inputs", "frozen_gate_sha256"), ("frozen", "gate_sha256")))
    sidecar_path = root / "manifests" / "frozen_gate_manifest.json"
    if expected is None and sidecar_path.exists():
        sm = read_json(sidecar_path)
        expected = first_present(sm, (("gate_file_sha256",), ("file_sha256",), ("gate", "file_sha256")))
    if not isinstance(expected, str) or HEX64_RE.fullmatch(expected) is None:
        raise Program05Error("Cannot verify frozen gate SHA-256 for official test.")
    if observed != expected:
        raise Program05Error(f"Frozen gate hash mismatch: expected={expected}, observed={observed}")
    return {"path": str(gate_path), "sha256": observed}


# ---------------------------------------------------------------------------
# Program 02 training-path verification
# ---------------------------------------------------------------------------


def training_manifest_path(root: Path, mode: str, seed: int) -> Path:
    if mode == "paper":
        return root / "manifests" / "training" / f"seed_{seed}.json"
    return root / "manifests" / f"_{mode}" / "training" / f"seed_{seed}.json"


def checkpoint_seed_root(root: Path, mode: str, seed: int) -> Path:
    if mode == "paper":
        return root / "checkpoints" / f"seed_{seed}"
    return root / "checkpoints" / f"_{mode}" / "program02" / f"seed_{seed}"


def verify_training_seed(
    *, root: Path, mode: str, seed: int, model_record: Mapping[str, Any],
    data_manifest: Mapping[str, Any], split_manifest: Mapping[str, Any], config_sha256: str,
) -> tuple[dict[str, Any], dict[int, AdapterRecord]]:
    path = training_manifest_path(root, mode, seed)
    if not path.exists():
        raise Program05Error(f"Missing completed Program 02 training manifest: {path}")
    m = read_json(path)
    if m.get("manifest_type") != "grpo_training_seed" or m.get("status") != "complete":
        raise Program05Error(f"Program 02 seed manifest is not complete: {path}")
    if m.get("training_seed") != seed or m.get("mode") != mode:
        raise Program05Error(f"Program 02 seed manifest metadata mismatch: {path}")
    inputs = m.get("inputs") or {}
    if not isinstance(inputs, Mapping):
        raise Program05Error(f"Program 02 seed manifest lacks inputs: {path}")
    source = split_manifest.get("source") or {}
    if inputs.get("dataset_revision") != source.get("dataset_revision"):
        raise Program05Error(f"Seed {seed} was trained against a different dataset revision.")
    if inputs.get("data_manifest_content_fingerprint_sha256") != data_manifest.get("content_fingerprint_sha256"):
        raise Program05Error(f"Seed {seed} was trained against a different data manifest.")
    if inputs.get("split_registry_content_fingerprint_sha256") != split_manifest.get("content_fingerprint_sha256"):
        raise Program05Error(f"Seed {seed} was trained against a different split registry.")
    if inputs.get("model_revision") != model_record.get("resolved_revision"):
        raise Program05Error(f"Seed {seed} was trained from a different base model revision.")
    expected_chat = ((model_record.get("validation") or {}).get("tokenizer") or {}).get("chat_template_sha256")
    if inputs.get("chat_template_sha256") != expected_chat:
        raise Program05Error(f"Seed {seed} chat template differs from Program 00.")
    if inputs.get("prompt_template_sha256") != sha256_text(PROMPT_TEMPLATE):
        raise Program05Error(f"Seed {seed} prompt template differs from Program 05 frozen template.")
    if mode == "paper" and inputs.get("protocol_config_sha256") != config_sha256:
        raise Program05Error(f"Seed {seed} was trained under a different frozen protocol.yaml.")

    seed_root = checkpoint_seed_root(root, mode, seed)
    adapters = m.get("permanent_adapters")
    if not isinstance(adapters, list) or not adapters:
        raise Program05Error(f"Seed {seed} manifest lacks permanent adapters.")
    by_step: dict[int, AdapterRecord] = {}
    for rec in adapters:
        if not isinstance(rec, Mapping) or "step" not in rec:
            continue
        step = int(rec["step"])
        p = seed_root / "adapters" / f"step_{step:04d}"
        if not p.exists():
            raise Program05Error(f"Missing permanent adapter: {p}")
        observed = adapter_payload_hash(p)
        if observed != rec.get("payload_sha256"):
            raise Program05Error(f"Permanent adapter hash mismatch: seed={seed}, step={step}")
        amp = p / "adapter_manifest.json"
        if not amp.exists():
            raise Program05Error(f"Permanent adapter lacks adapter_manifest.json: {p}")
        am = read_json(amp)
        if am.get("training_seed") != seed or am.get("step") != step:
            raise Program05Error(f"Adapter manifest metadata mismatch: {p}")
        if am.get("base_model_revision") != model_record.get("resolved_revision"):
            raise Program05Error(f"Adapter base revision mismatch: {p}")
        if am.get("adapter_payload_tree_sha256") != observed:
            raise Program05Error(f"Adapter sidecar hash mismatch: {p}")
        by_step[step] = AdapterRecord(step=step, path=p, payload_sha256=observed, manifest=am)
    return m, by_step


# ---------------------------------------------------------------------------
# Program 04 correctness prerequisite
# ---------------------------------------------------------------------------


def rescore_manifest_root(root: Path, mode: str, split: str) -> Path:
    if mode == "paper":
        return root / "manifests" / "target_rescores" / split
    return root / "manifests" / f"_{mode}" / "target_rescores" / split


def verify_program04_identity_gate(root: Path, mode: str, split: str, seeds: Sequence[int]) -> dict[str, Any]:
    """Require verified identity-rescore evidence before online generation.

    Program 04's full distant-pair completion is not mathematically required to
    sample pi_e, but the identity gate is the project's strongest practical check
    that tokenization, adapters, EOS and policy-probability semantics are aligned.
    """
    mroot = rescore_manifest_root(root, mode, split)
    summary_path = mroot / "pair_summaries.json"
    index_path = mroot / "collection_index.json"
    if not summary_path.exists() or not index_path.exists():
        raise Program05Error(
            "Program 05 requires Program 04's identity gate to have passed first. "
            f"Missing {summary_path} or {index_path}."
        )
    summaries = read_json(summary_path)
    index = read_json(index_path)
    if summaries.get("manifest_type") != "target_rescore_pair_summaries":
        raise Program05Error("Invalid Program 04 pair_summaries.json.")
    rows = summaries.get("summaries")
    if not isinstance(rows, list):
        raise Program05Error("Program 04 pair_summaries.json lacks summaries list.")
    by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for r in rows:
        if isinstance(r, Mapping) and r.get("purpose") == "identity":
            try:
                key = (int(r.get("training_seed")), int(r.get("target_step")))
            except Exception:
                continue
            by_key[key] = r
    # Paper anchors are fixed. Engineering modes may only have 0/final identity;
    # require whatever Program 04 actually registered for the selected seed.
    expected_steps: set[int] = set()
    for shard in index.get("shards", []) if isinstance(index.get("shards"), list) else []:
        if isinstance(shard, Mapping) and shard.get("purpose") == "identity":
            try:
                expected_steps.add(int(shard.get("target_step")))
            except Exception:
                pass
    if mode == "paper":
        expected_steps = {0, 100, 200, 300}
    if not expected_steps:
        raise Program05Error("Program 04 collection index contains no identity evidence.")
    missing: list[tuple[int, int]] = []
    failed: list[tuple[int, int]] = []
    for seed in seeds:
        for step in sorted(expected_steps):
            r = by_key.get((int(seed), step))
            if r is None:
                missing.append((int(seed), step))
            elif not bool(r.get("identity_pass")):
                failed.append((int(seed), step))
    if missing:
        raise Program05Error(f"Program 04 identity summaries missing for {missing}.")
    if failed:
        raise Program05Error(f"Program 04 identity gate failed for {failed}; online generation is blocked.")
    return {
        "pair_summaries_path": str(summary_path),
        "pair_summaries_sha256": sha256_file(summary_path),
        "collection_index_path": str(index_path),
        "collection_index_sha256": sha256_file(index_path),
        "identity_steps": sorted(expected_steps),
    }


# ---------------------------------------------------------------------------
# GSM8K dereference and frozen parser/reward
# ---------------------------------------------------------------------------


def discover_gsm8k_parquet(root: Path) -> dict[str, list[Path]]:
    """Discover only the frozen GSM8K ``main`` configuration.

    The upstream GSM8K repository also contains the ``socratic`` configuration,
    which has the same 7,473/1,319 train/test row counts.  A recursive search
    across all Parquet files therefore silently doubles the rows.  Program 01
    already fixes the research dataset to ``main``; Program 05 must dereference
    exactly the same files.
    """
    train = sorted((root / GSM8K_CONFIG).glob("train*.parquet"))
    test = sorted((root / GSM8K_CONFIG).glob("test*.parquet"))

    # Fallback for a snapshot layout with an extra directory level, but still
    # require the exact configured GSM8K config in the relative path.
    if not train or not test:
        train = sorted(
            p
            for p in root.rglob("train*.parquet")
            if p.is_file() and GSM8K_CONFIG in p.relative_to(root).parts
        )
        test = sorted(
            p
            for p in root.rglob("test*.parquet")
            if p.is_file() and GSM8K_CONFIG in p.relative_to(root).parts
        )

    if not train or not test:
        raise Program05Error(
            f"Pinned GSM8K snapshot does not contain "
            f"{GSM8K_CONFIG}/train*.parquet and {GSM8K_CONFIG}/test*.parquet."
        )

    return {"train": train, "test": test}


def load_upstream_rows(gsm8k_dir: Path, original_split: str) -> list[dict[str, str]]:
    files = discover_gsm8k_parquet(gsm8k_dir).get(original_split, [])
    if not files:
        raise Program05Error(f"Cannot find pinned GSM8K {original_split} Parquet files under {gsm8k_dir}.")
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset("parquet", data_files={original_split: [str(p) for p in files]}, split=original_split)
    except Exception as exc:
        raise Program05Error(f"Cannot load pinned GSM8K {original_split} Parquet locally: {exc}") from exc
    rows: list[dict[str, str]] = []
    for row in ds:
        if "question" not in row or "answer" not in row:
            raise Program05Error("Pinned GSM8K row lacks question/answer fields.")
        rows.append({"question": str(row["question"]), "answer": str(row["answer"])})
    expected = 7473 if original_split == "train" else 1319
    if len(rows) != expected:
        raise Program05Error(f"Pinned GSM8K {original_split} has {len(rows)} rows, expected {expected}.")
    return rows


def normalize_numeric_text(text: str) -> Fraction | None:
    cleaned = text.strip().replace(",", "").replace("$", "")
    if NUMBER_RE.fullmatch(cleaned) is None:
        return None
    try:
        if "/" in cleaned:
            numerator, denominator = cleaned.split("/", maxsplit=1)
            return Fraction(int(numerator), int(denominator))
        return Fraction(Decimal(cleaned))
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def canonical_fraction(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def extract_gsm8k_gold(answer_field: str) -> Fraction | None:
    match = GSM8K_GOLD_RE.search(answer_field)
    return None if match is None else normalize_numeric_text(match.group(1))


def parse_completion_answer(text: str) -> tuple[Fraction | None, str]:
    matches = ANSWER_RE.findall(text)
    if len(matches) == 0:
        return None, "missing_answer_tag"
    if len(matches) != 1:
        return None, "multiple_answer_tags"
    parsed = normalize_numeric_text(matches[0])
    if parsed is None:
        return None, "invalid_numeric_answer"
    return parsed, "ok"


def build_prompt_records(
    *, registry_rows: Sequence[Mapping[str, Any]], raw_rows: Sequence[Mapping[str, str]],
    research_split: str, limit: int | None,
) -> list[PromptRecord]:
    selected = [dict(r) for r in registry_rows if r.get("research_split") == research_split]
    selected.sort(key=lambda r: str(r.get("prompt_id")))
    if limit is not None:
        selected = selected[:limit]
    records: list[PromptRecord] = []
    for r in selected:
        idx = r.get("source_row_index")
        if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0 or idx >= len(raw_rows):
            raise Program05Error(f"Invalid source_row_index in split registry: {idx!r}")
        source = raw_rows[idx]
        question = str(source["question"])
        answer = str(source["answer"])
        q_hash = sha256_text(question)
        a_hash = sha256_text(answer)
        if r.get("question_hash") != q_hash or r.get("gold_answer_hash") != a_hash:
            raise Program05Error(f"Split registry source hash mismatch for prompt_id={r.get('prompt_id')}.")
        gold = extract_gsm8k_gold(answer)
        if gold is None:
            raise Program05Error(f"Cannot parse GSM8K gold answer for prompt_id={r.get('prompt_id')}.")
        records.append(PromptRecord(
            prompt_id=str(r["prompt_id"]), original_split=str(r["original_split"]),
            research_split=str(r["research_split"]), source_row_index=int(idx), question=question,
            raw_gold_answer=answer, gold_answer=canonical_fraction(gold),
        ))
    expected = EXPECTED_RESEARCH_ROWS[research_split]
    if limit is None and len(records) != expected:
        raise Program05Error(f"Research split {research_split} yielded {len(records)} prompts, expected {expected}.")
    if not records:
        raise Program05Error(f"No prompts selected for research split {research_split}.")
    return records


# ---------------------------------------------------------------------------
# Stable IDs, sample blocks, tokenizer and target-policy generation
# ---------------------------------------------------------------------------


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256(canonical_bytes(list(parts))).digest()
    # torch.Generator.manual_seed accepts signed/unsigned 64-bit-like values;
    # keep within positive 63-bit range for cross-platform safety.
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def online_id(
    *, protocol_version_value: str, dataset_revision: str, training_seed: int,
    target_step: int, split: str, prompt_id: str, sample_index: int,
) -> str:
    return sha256_bytes(canonical_bytes([
        protocol_version_value, dataset_revision, int(training_seed), int(target_step),
        split, prompt_id, int(sample_index),
    ]))


def sample_blocks(target_l: int, block_size: int) -> list[tuple[int, int]]:
    if target_l <= 0 or block_size <= 0:
        raise Program05Error("target L and sample block size must be positive.")
    return [(i, min(i + block_size, target_l)) for i in range(0, target_l, block_size)]


def generation_subblocks(start: int, end: int, batch_size: int) -> list[tuple[int, int]]:
    if not (0 <= start < end) or batch_size <= 0:
        raise Program05Error("Invalid generation subblock bounds.")
    return [(i, min(i + batch_size, end)) for i in range(start, end, batch_size)]


def prompt_shards(prompts: Sequence[PromptRecord], prompts_per_shard: int) -> list[list[PromptRecord]]:
    return [list(prompts[i:i + prompts_per_shard]) for i in range(0, len(prompts), prompts_per_shard)]


def load_tokenizer(model_dir: Path, model_record: Mapping[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer  # type: ignore
        tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise Program05Error(f"Cannot load pinned tokenizer: {exc}") from exc
    if tok.pad_token_id is None:
        if tok.eos_token_id is None:
            raise Program05Error("Tokenizer has neither pad_token_id nor eos_token_id.")
        tok.pad_token = tok.eos_token
    template = getattr(tok, "chat_template", None)
    if template is None:
        raise Program05Error("Pinned tokenizer has no chat template.")
    text = template if isinstance(template, str) else json.dumps(template, ensure_ascii=False, sort_keys=True, default=str)
    expected = ((model_record.get("validation") or {}).get("tokenizer") or {}).get("chat_template_sha256")
    if expected is not None and sha256_text(text) != expected:
        raise Program05Error("Loaded tokenizer chat template hash differs from Program 00.")
    return tok


def prompt_content(question: str) -> str:
    return PROMPT_TEMPLATE.format(question=question)


def render_prompt_ids(tokenizer: Any, question: str) -> list[int]:
    messages = [{"role": "user", "content": prompt_content(question)}]
    try:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=False)
    except TypeError:
        ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise Program05Error("Unexpected batched prompt IDs from single conversation.")
        ids = ids[0]
    if not isinstance(ids, list) or not ids or not all(isinstance(x, int) for x in ids):
        raise Program05Error("Tokenizer.apply_chat_template did not return a flat non-empty token ID list.")
    return [int(x) for x in ids]


def resolve_dtype(device: str, preferred: str | None) -> tuple[Any, str]:
    """Return the fixed precision used to sample the on-policy reference.

    Program 02 training precision is intentionally decoupled from the policy
    inference precision used by Programs 03--05.  The OPE probability evidence
    and the independent on-policy reference therefore evaluate/sample the same
    FP32 inference policy even when GRPO training used BF16.

    ``preferred`` is retained only for backward-compatible call signatures and
    is deliberately ignored.
    """
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise Program05Error(f"Cannot import torch: {exc}") from exc
    if device == "cuda" and not torch.cuda.is_available():
        raise Program05Error("--device cuda requested but torch.cuda.is_available() is False.")
    return torch.float32, POLICY_INFERENCE_PRECISION


def load_target_model(
    *, model_dir: Path, adapter_dir: Path, device: str, preferred_precision: str | None,
) -> tuple[Any, str]:
    try:
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM  # type: ignore
    except Exception as exc:
        raise Program05Error(f"Cannot import inference stack: {exc}") from exc
    dtype, resolved = resolve_dtype(device, preferred_precision)
    try:
        try:
            base = AutoModelForCausalLM.from_pretrained(
                str(model_dir), local_files_only=True, trust_remote_code=False, dtype=dtype
            )
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(
                str(model_dir), local_files_only=True, trust_remote_code=False, torch_dtype=dtype
            )
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False)
        model.to(device)
        model.eval()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = True
        for p in model.parameters():
            p.requires_grad_(False)
        return model, resolved
    except Exception as exc:
        raise Program05Error(f"Cannot load base model + target LoRA adapter {adapter_dir}: {exc}") from exc


def eos_ids(tokenizer: Any) -> set[int]:
    raw = tokenizer.eos_token_id
    if raw is None:
        raise Program05Error("Tokenizer has no eos_token_id; termination semantics would be undefined.")
    if isinstance(raw, int):
        return {int(raw)}
    if isinstance(raw, (list, tuple, set)) and raw:
        return {int(x) for x in raw}
    raise Program05Error(f"Unsupported eos_token_id value: {raw!r}")


def completion_cut_length(tokens: Sequence[int], eos_set: set[int], generated_width: int, max_new_tokens: int) -> tuple[int, bool, bool]:
    if generated_width < 0 or generated_width > len(tokens):
        raise Program05Error("Generated sequence width alignment is invalid.")
    for i, token in enumerate(tokens[:generated_width]):
        if int(token) in eos_set:
            return i + 1, True, False
    if generated_width == max_new_tokens:
        return generated_width, False, True
    raise Program05Error(
        "Generation ended before max_new_tokens without EOS. Refusing to guess stopping semantics."
    )


def generation_config_record(spec: OnlineSpec, tokenizer: Any) -> dict[str, Any]:
    return {
        "generation_config_source": "fresh_transformers.GenerationConfig_not_model_generation_config",
        "do_sample": True,
        "num_beams": 1,
        "temperature": spec.temperature,
        "top_p": spec.top_p,
        "top_k": spec.top_k,
        "repetition_penalty": spec.repetition_penalty,
        "max_new_tokens": spec.max_completion_length,
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token_id": sorted(eos_ids(tokenizer)),
        "bos_token_id": None if tokenizer.bos_token_id is None else int(tokenizer.bos_token_id),
        "return_dict_in_generate": True,
        "output_scores": False,
        "policy_inference_precision": POLICY_INFERENCE_PRECISION,
        "policy_definition": "stochastic target-policy generation used for on-policy Monte Carlo reference",
    }


def _seeded_generate(model: Any, input_ids: Any, attention_mask: Any, kwargs: Mapping[str, Any], seed: int, device: str) -> Any:
    try:
        import torch  # type: ignore
        from transformers import GenerationConfig  # type: ignore
    except Exception as exc:
        raise Program05Error(f"Cannot import generation stack: {exc}") from exc
    generation_config = GenerationConfig(**dict(kwargs))
    devices: list[int] = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed(seed)
        with torch.inference_mode():
            return model.generate(input_ids=input_ids, attention_mask=attention_mask, generation_config=generation_config)


def generate_prompt_block(
    *, model: Any, tokenizer: Any, prompt: PromptRecord, sample_start: int, sample_end: int,
    spec: OnlineSpec, protocol_version_value: str, dataset_revision: str, model_revision: str,
    tokenizer_revision: str, training_seed: int, target_step: int, split: str,
    adapter_sha256: str, device: str,
) -> list[dict[str, Any]]:
    try:
        import torch  # type: ignore
    except Exception as exc:
        raise Program05Error(f"Cannot import torch during generation: {exc}") from exc
    prompt_ids = render_prompt_ids(tokenizer, prompt.question)
    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention = torch.ones_like(prompt_tensor)
    eos_set = eos_ids(tokenizer)
    gen_record = generation_config_record(spec, tokenizer)
    gen_hash = sha256_bytes(canonical_bytes(gen_record))
    chat_hash = sha256_text(tokenizer.chat_template if isinstance(tokenizer.chat_template, str) else json.dumps(tokenizer.chat_template, ensure_ascii=False, sort_keys=True, default=str))
    prompt_hash = sha256_text(PROMPT_TEMPLATE)
    gold = Fraction(prompt.gold_answer)

    rows: list[dict[str, Any]] = []
    for sub_start, sub_end in generation_subblocks(sample_start, sample_end, spec.generation_batch_size):
        n = sub_end - sub_start
        call_seed = stable_seed(
            "online_generation", protocol_version_value, dataset_revision, model_revision,
            training_seed, target_step, split, prompt.prompt_id, sub_start, sub_end, gen_hash,
        )
        call_id = sha256_bytes(canonical_bytes({
            "seed": call_seed, "prompt_id": prompt.prompt_id, "sample_start": sub_start,
            "sample_end": sub_end, "target_step": target_step, "training_seed": training_seed,
            "split": split,
        }))
        kwargs = {
            "do_sample": True,
            "num_beams": 1,
            "num_return_sequences": n,
            "max_new_tokens": spec.max_completion_length,
            "temperature": spec.temperature,
            "top_p": spec.top_p,
            "top_k": spec.top_k,
            "repetition_penalty": spec.repetition_penalty,
            "pad_token_id": int(tokenizer.pad_token_id),
            "eos_token_id": sorted(eos_set) if len(eos_set) > 1 else next(iter(eos_set)),
            "use_cache": True,
            "return_dict_in_generate": True,
            "output_scores": False,
        }
        outputs = _seeded_generate(model, prompt_tensor, attention, kwargs, call_seed, device)
        if not hasattr(outputs, "sequences"):
            raise Program05Error("model.generate did not return sequences under return_dict_in_generate=True.")
        sequences = outputs.sequences
        if int(sequences.shape[0]) != n:
            raise Program05Error(f"Generation returned {int(sequences.shape[0])} sequences, expected {n}.")
        if int(sequences.shape[1]) < len(prompt_ids):
            raise Program05Error("Generated sequence is shorter than the prompt token sequence.")
        generated = sequences[:, len(prompt_ids):]
        width = int(generated.shape[1])
        if width <= 0 or width > spec.max_completion_length:
            raise Program05Error(f"Unexpected generated width {width}.")
        generated_cpu = generated.detach().to("cpu")
        for local_i in range(n):
            sample_index = sub_start + local_i
            token_all = [int(x) for x in generated_cpu[local_i].tolist()]
            cut, ended_eos, truncated = completion_cut_length(token_all, eos_set, width, spec.max_completion_length)
            comp_ids = token_all[:cut]
            if not comp_ids:
                raise Program05Error("Online completion contains zero generated tokens.")
            text = tokenizer.decode(comp_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            parsed, parser_status = parse_completion_answer(text)
            parsed_str = None if parsed is None else canonical_fraction(parsed)
            correct = float(parsed is not None and parsed == gold)
            oid = online_id(
                protocol_version_value=protocol_version_value, dataset_revision=dataset_revision,
                training_seed=training_seed, target_step=target_step, split=split,
                prompt_id=prompt.prompt_id, sample_index=sample_index,
            )
            rows.append({
                "online_id": oid,
                "dataset": "GSM8K",
                "dataset_revision": dataset_revision,
                "protocol_version": protocol_version_value,
                "training_seed": training_seed,
                "target_step": target_step,
                "split": split,
                "prompt_id": prompt.prompt_id,
                "source_row_index": prompt.source_row_index,
                "sample_index": sample_index,
                "completion_token_ids": comp_ids,
                "completion_length": len(comp_ids),
                "terminated_with_eos": ended_eos,
                "was_truncated": truncated,
                "parsed_answer": parsed_str,
                "correct": correct,
                "parser_status": parser_status,
                "temperature": spec.temperature,
                "top_p": spec.top_p,
                "top_k": spec.top_k,
                "repetition_penalty": spec.repetition_penalty,
                "max_completion_length": spec.max_completion_length,
                "generation_seed": call_seed,
                "generation_call_id": call_id,
                "model_revision": model_revision,
                "tokenizer_revision": tokenizer_revision,
                "chat_template_hash": chat_hash,
                "prompt_template_hash": prompt_hash,
                "target_adapter_sha256": adapter_sha256,
                "generation_config_sha256": gen_hash,
                "completion_text": text,
            })
    if [int(r["sample_index"]) for r in rows] != list(range(sample_start, sample_end)):
        raise Program05Error("Generated online sample indices are not contiguous/deterministic.")
    return rows


# ---------------------------------------------------------------------------
# Immutable sample shard storage
# ---------------------------------------------------------------------------


def online_parquet_schema() -> Any:
    try:
        import pyarrow as pa  # type: ignore
    except Exception as exc:
        raise Program05Error(f"Cannot import pyarrow: {exc}") from exc
    return pa.schema([
        pa.field("online_id", pa.string(), nullable=False),
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("dataset_revision", pa.string(), nullable=False),
        pa.field("protocol_version", pa.string(), nullable=False),
        pa.field("training_seed", pa.int64(), nullable=False),
        pa.field("target_step", pa.int64(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("prompt_id", pa.string(), nullable=False),
        pa.field("source_row_index", pa.int64(), nullable=False),
        pa.field("sample_index", pa.int64(), nullable=False),
        pa.field("completion_token_ids", pa.list_(pa.int64()), nullable=False),
        pa.field("completion_length", pa.int64(), nullable=False),
        pa.field("terminated_with_eos", pa.bool_(), nullable=False),
        pa.field("was_truncated", pa.bool_(), nullable=False),
        pa.field("parsed_answer", pa.string(), nullable=True),
        pa.field("correct", pa.float64(), nullable=False),
        pa.field("parser_status", pa.string(), nullable=False),
        pa.field("temperature", pa.float64(), nullable=False),
        pa.field("top_p", pa.float64(), nullable=False),
        pa.field("top_k", pa.int64(), nullable=False),
        pa.field("repetition_penalty", pa.float64(), nullable=False),
        pa.field("max_completion_length", pa.int64(), nullable=False),
        pa.field("generation_seed", pa.int64(), nullable=False),
        pa.field("generation_call_id", pa.string(), nullable=False),
        pa.field("model_revision", pa.string(), nullable=False),
        pa.field("tokenizer_revision", pa.string(), nullable=False),
        pa.field("chat_template_hash", pa.string(), nullable=False),
        pa.field("prompt_template_hash", pa.string(), nullable=False),
        pa.field("target_adapter_sha256", pa.string(), nullable=False),
        pa.field("generation_config_sha256", pa.string(), nullable=False),
        pa.field("completion_text", pa.string(), nullable=False),
    ])


def row_for_hash(row: Mapping[str, Any]) -> dict[str, Any]:
    return {k: row.get(k) for k in ONLINE_COLUMNS}


def rows_content_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    normalized = [row_for_hash(r) for r in rows]
    normalized.sort(key=lambda r: (str(r["prompt_id"]), int(r["sample_index"])))
    return sha256_bytes(canonical_bytes(normalized))


def validate_online_rows(
    rows: Sequence[Mapping[str, Any]], *, expected_prompts: Sequence[PromptRecord],
    sample_start: int, sample_end: int, training_seed: int, target_step: int,
    split: str, protocol_version_value: str, dataset_revision: str, model_revision: str,
    adapter_sha256: str, spec: OnlineSpec,
) -> dict[str, Any]:
    expected_prompt_ids = [p.prompt_id for p in expected_prompts]
    expected_count = len(expected_prompts) * (sample_end - sample_start)
    if len(rows) != expected_count:
        raise Program05Error(f"Online shard has {len(rows)} rows, expected {expected_count}.")
    seen: set[str] = set()
    by_prompt: dict[str, set[int]] = {p: set() for p in expected_prompt_ids}
    lengths: list[int] = []
    trunc = 0
    correct = 0
    parser_ok = 0
    for r in rows:
        for col in ONLINE_COLUMNS:
            if col not in r:
                raise Program05Error(f"Online row missing required field {col}.")
        if int(r["training_seed"]) != training_seed or int(r["target_step"]) != target_step or r["split"] != split:
            raise Program05Error("Online row seed/target/split mismatch.")
        if r["dataset_revision"] != dataset_revision or r["model_revision"] != model_revision:
            raise Program05Error("Online row model/data revision mismatch.")
        if r["protocol_version"] != protocol_version_value:
            raise Program05Error("Online row protocol version mismatch.")
        if r["target_adapter_sha256"] != adapter_sha256:
            raise Program05Error("Online row target adapter hash mismatch.")
        pid = str(r["prompt_id"])
        if pid not in by_prompt:
            raise Program05Error(f"Unexpected prompt_id in online shard: {pid}")
        idx = int(r["sample_index"])
        if idx < sample_start or idx >= sample_end or idx in by_prompt[pid]:
            raise Program05Error("Online shard has out-of-range or duplicate sample_index.")
        by_prompt[pid].add(idx)
        expected_id = online_id(
            protocol_version_value=protocol_version_value, dataset_revision=dataset_revision,
            training_seed=training_seed, target_step=target_step, split=split,
            prompt_id=pid, sample_index=idx,
        )
        if r["online_id"] != expected_id or expected_id in seen:
            raise Program05Error("Online ID is not reproducible/unique from the stable key.")
        seen.add(expected_id)
        comp = r["completion_token_ids"]
        if not isinstance(comp, list) or not comp or int(r["completion_length"]) != len(comp):
            raise Program05Error("Online completion token/length mismatch.")
        if bool(r["terminated_with_eos"]) and bool(r["was_truncated"]):
            raise Program05Error("Online completion cannot be both EOS-terminated and truncated.")
        if bool(r["was_truncated"]) and len(comp) != spec.max_completion_length:
            raise Program05Error("Truncated online completion does not reach max_completion_length.")
        c = float(r["correct"])
        if c not in (0.0, 1.0):
            raise Program05Error("Online correctness reward must be binary 0/1.")
        if not math.isclose(float(r["temperature"]), spec.temperature) or not math.isclose(float(r["top_p"]), spec.top_p):
            raise Program05Error("Online row temperature/top_p mismatch.")
        if int(r["top_k"]) != spec.top_k or not math.isclose(float(r["repetition_penalty"]), spec.repetition_penalty):
            raise Program05Error("Online row top_k/repetition_penalty mismatch.")
        lengths.append(len(comp))
        trunc += int(bool(r["was_truncated"]))
        correct += int(c)
        parser_ok += int(r["parser_status"] == "ok")
    expected_indices = set(range(sample_start, sample_end))
    for pid, got in by_prompt.items():
        if got != expected_indices:
            raise Program05Error(f"Online shard sample indices incomplete for prompt {pid}.")
    return {
        "row_count": len(rows),
        "prompt_count": len(expected_prompts),
        "samples_per_prompt": sample_end - sample_start,
        "correct_count": correct,
        "parser_ok_count": parser_ok,
        "truncated_count": trunc,
        "mean_completion_length": math.fsum(lengths) / len(lengths) if lengths else None,
    }


def online_data_root(root: Path, mode: str) -> Path:
    return root / "data" / "online_reference" if mode == "paper" else root / "data" / "online_reference" / f"_{mode}"


def online_manifest_root(root: Path, mode: str, split: str) -> Path:
    if mode == "paper":
        return root / "manifests" / "online_reference" / split
    return root / "manifests" / f"_{mode}" / "online_reference" / split


def online_unit_dir(
    data_root: Path, *, split: str, training_seed: int, target_step: int,
    sample_start: int, sample_end: int, shard_index: int,
) -> Path:
    return (
        data_root / "gsm8k" / f"split={split}" / f"seed={training_seed}" /
        f"target_step={target_step:04d}" / f"sample_block={sample_start:04d}-{sample_end-1:04d}" /
        f"shard={shard_index:05d}"
    )


def read_online_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
        return [dict(x) for x in pq.read_table(path).to_pylist()]
    except Exception as exc:
        raise Program05Error(f"Cannot read online Parquet {path}: {exc}") from exc


def verify_online_unit(
    unit_dir: Path, *, root: Path, expected_prompts: Sequence[PromptRecord], sample_start: int,
    sample_end: int, training_seed: int, target_step: int, split: str,
    protocol_version_value: str, dataset_revision: str, model_revision: str,
    adapter_sha256: str, spec: OnlineSpec,
) -> dict[str, Any]:
    mp = unit_dir / "manifest.json"
    pp = unit_dir / "online_samples.parquet"
    if not mp.exists() or not pp.exists():
        raise Program05Error(f"Partial/corrupt online shard unit: {unit_dir}")
    m = read_json(mp)
    if m.get("schema_version") != ONLINE_SCHEMA or m.get("manifest_type") != "online_reference_shard":
        raise Program05Error(f"Invalid online shard manifest: {mp}")
    fixed = {
        "training_seed": training_seed, "target_step": target_step, "split": split,
        "sample_start": sample_start, "sample_end_exclusive": sample_end,
        "dataset_revision": dataset_revision, "protocol_version": protocol_version_value,
        "model_revision": model_revision, "target_adapter_sha256": adapter_sha256,
        "online_config_sha256": online_spec_fingerprint(spec),
    }
    for k, v in fixed.items():
        if m.get(k) != v:
            raise Program05Error(f"Existing online shard {unit_dir} has incompatible {k}: {m.get(k)!r} != {v!r}")
    if m.get("policy_inference_precision") != POLICY_INFERENCE_PRECISION:
        raise Program05Error(
            f"Existing online shard does not use the required "
            f"{POLICY_INFERENCE_PRECISION} policy-inference precision: {unit_dir}"
        )
    expected_pids = [p.prompt_id for p in expected_prompts]
    if m.get("prompt_ids") != expected_pids:
        raise Program05Error(f"Existing online shard prompt IDs differ: {unit_dir}")
    observed_file = sha256_file(pp)
    if observed_file != (m.get("parquet") or {}).get("file_sha256"):
        raise Program05Error(f"Online shard file hash mismatch: {unit_dir}")
    rows = read_online_rows(pp)
    validation = validate_online_rows(
        rows, expected_prompts=expected_prompts, sample_start=sample_start, sample_end=sample_end,
        training_seed=training_seed, target_step=target_step, split=split,
        protocol_version_value=protocol_version_value, dataset_revision=dataset_revision,
        model_revision=model_revision, adapter_sha256=adapter_sha256, spec=spec,
    )
    observed_content = rows_content_sha256(rows)
    if observed_content != (m.get("parquet") or {}).get("content_sha256"):
        raise Program05Error(f"Online shard semantic content hash mismatch: {unit_dir}")
    if validation.get("row_count") != (m.get("validation") or {}).get("row_count"):
        raise Program05Error(f"Online shard validation count mismatch: {unit_dir}")
    return m


def publish_online_unit(
    *, unit_dir: Path, root: Path, rows: Sequence[Mapping[str, Any]], expected_prompts: Sequence[PromptRecord],
    sample_start: int, sample_end: int, training_seed: int, target_step: int, split: str, shard_index: int,
    protocol_version_value: str, dataset_revision: str, model_revision: str, tokenizer_revision: str,
    chat_template_hash: str, adapter_sha256: str, online_config_sha256: str,
    generation_config_sha256: str, spec: OnlineSpec,
) -> dict[str, Any]:
    if unit_dir.exists():
        return verify_online_unit(
            unit_dir, root=root, expected_prompts=expected_prompts, sample_start=sample_start,
            sample_end=sample_end, training_seed=training_seed, target_step=target_step, split=split,
            protocol_version_value=protocol_version_value, dataset_revision=dataset_revision,
            model_revision=model_revision, adapter_sha256=adapter_sha256, spec=spec,
        )
    validation = validate_online_rows(
        rows, expected_prompts=expected_prompts, sample_start=sample_start, sample_end=sample_end,
        training_seed=training_seed, target_step=target_step, split=split,
        protocol_version_value=protocol_version_value, dataset_revision=dataset_revision,
        model_revision=model_revision, adapter_sha256=adapter_sha256, spec=spec,
    )
    unit_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = unit_dir.parent / f".{unit_dir.name}.staging-{uuid.uuid4().hex}"
    try:
        stage.mkdir(parents=False, exist_ok=False)
        pp = stage / "online_samples.parquet"
        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
            table = pa.Table.from_pylist([dict(r) for r in rows], schema=online_parquet_schema())
            pq.write_table(table, pp, compression="zstd", use_dictionary=True)
        except Exception as exc:
            raise Program05Error(f"Cannot write immutable online-reference Parquet shard: {exc}") from exc
        with pp.open("rb+") as f:
            f.flush(); os.fsync(f.fileno())
        content_sha = rows_content_sha256(rows)
        file_sha = sha256_file(pp)
        manifest = {
            "schema_version": ONLINE_SCHEMA,
            "manifest_type": "online_reference_shard",
            "created_at_utc": now_utc(),
            "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
            "dataset": "GSM8K",
            "dataset_revision": dataset_revision,
            "protocol_version": protocol_version_value,
            "training_seed": training_seed,
            "target_step": target_step,
            "split": split,
            "sample_start": sample_start,
            "sample_end_exclusive": sample_end,
            "shard_index": shard_index,
            "prompt_ids": [p.prompt_id for p in expected_prompts],
            "online_config_sha256": online_config_sha256,
            "model_revision": model_revision,
            "tokenizer_revision": tokenizer_revision,
            "chat_template_hash": chat_template_hash,
            "target_adapter_sha256": adapter_sha256,
            "generation_config_sha256": generation_config_sha256,
            "policy_inference_precision": POLICY_INFERENCE_PRECISION,
            "parquet": {
                "local_path": rel(unit_dir / "online_samples.parquet", root),
                "file_sha256": file_sha,
                "content_sha256": content_sha,
                "row_count": len(rows),
                "columns": list(ONLINE_COLUMNS),
            },
            "validation": validation,
        }
        atomic_write_json(stage / "manifest.json", manifest)
        fsync_directory(stage)
        if unit_dir.exists():
            raise Program05Error(f"Online shard appeared concurrently: {unit_dir}")
        os.replace(stage, unit_dir)
        fsync_directory(unit_dir.parent)
        make_read_only_tree(unit_dir)
        return verify_online_unit(
            unit_dir, root=root, expected_prompts=expected_prompts, sample_start=sample_start,
            sample_end=sample_end, training_seed=training_seed, target_step=target_step, split=split,
            protocol_version_value=protocol_version_value, dataset_revision=dataset_revision,
            model_revision=model_revision, adapter_sha256=adapter_sha256, spec=spec,
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def cleanup_staging(data_root: Path, resume: bool) -> None:
    stale = [p for p in data_root.rglob(".*.staging-*") if p.is_dir()] if data_root.exists() else []
    if not stale:
        return
    if not resume:
        raise Program05Error(
            f"Found {len(stale)} stale Program 05 staging directories. Re-run with --resume to discard only unpublished staging state."
        )
    for p in stale:
        print(f"[RESUME] removing unpublished staging directory {p}")
        shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------
# Collection indices and derived prompt/reference summaries
# ---------------------------------------------------------------------------


def rebuild_collection_index(
    *, root: Path, data_root: Path, manifest_root: Path, mode: str, split: str,
    target_l_main: int, spec: OnlineSpec, seeds: Sequence[int], target_steps: Sequence[int],
    dataset_revision: str, model_revision: str, protocol_version_value: str, config_sha256: str,
    program04_identity: Mapping[str, Any],
) -> Path:
    records: list[dict[str, Any]] = []
    if data_root.exists():
        for mp in sorted(data_root.rglob("manifest.json")):
            if not mp.parent.name.startswith("shard="):
                continue
            m = read_json(mp)
            if m.get("manifest_type") != "online_reference_shard" or m.get("split") != split:
                continue
            if int(m.get("training_seed", -1)) not in set(int(x) for x in seeds):
                continue
            if int(m.get("target_step", -1)) not in set(int(x) for x in target_steps):
                continue
            records.append({
                "manifest_path": rel(mp, root),
                "manifest_sha256": sha256_file(mp),
                "parquet_path": (m.get("parquet") or {}).get("local_path"),
                "parquet_sha256": (m.get("parquet") or {}).get("file_sha256"),
                "content_sha256": (m.get("parquet") or {}).get("content_sha256"),
                "training_seed": m.get("training_seed"),
                "target_step": m.get("target_step"),
                "sample_start": m.get("sample_start"),
                "sample_end_exclusive": m.get("sample_end_exclusive"),
                "shard_index": m.get("shard_index"),
                "row_count": (m.get("parquet") or {}).get("row_count"),
                "target_adapter_sha256": m.get("target_adapter_sha256"),
                "policy_inference_precision": m.get("policy_inference_precision"),
            })
    records.sort(key=lambda x: (
        int(x.get("training_seed", -1)), int(x.get("target_step", -1)),
        int(x.get("sample_start", -1)), int(x.get("shard_index", -1)),
    ))
    payload = {
        "schema_version": ONLINE_SCHEMA,
        "manifest_type": "online_reference_collection_index",
        "updated_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "mode": mode, "split": split,
        "protocol_version": protocol_version_value,
        "protocol_config_sha256": config_sha256,
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "target_l_main": target_l_main,
        "l_audit": spec.l_audit,
        "audit_steps": list(spec.audit_steps),
        "seeds": list(seeds),
        "target_steps": list(target_steps),
        "online_spec": asdict(spec),
        "online_config_sha256": online_spec_fingerprint(spec),
        "policy_inference_precision": POLICY_INFERENCE_PRECISION,
        "program04_identity_gate": dict(program04_identity),
        "shard_count": len(records),
        "trajectory_rows": sum(int(r.get("row_count") or 0) for r in records),
        "shards": records,
        "shard_set_sha256": sha256_bytes(canonical_bytes(records)),
    }
    manifest_root.mkdir(parents=True, exist_ok=True)
    path = manifest_root / "collection_index.json"
    atomic_write_json(path, payload)
    return path


def target_sample_limit(spec: OnlineSpec, target_step: int, target_l_main: int) -> int:
    return spec.l_audit if target_step in set(spec.audit_steps) else target_l_main


def target_prefix(data_root: Path, split: str, seed: int, step: int) -> Path:
    return data_root / "gsm8k" / f"split={split}" / f"seed={seed}" / f"target_step={step:04d}"


def collect_target_rows(data_root: Path, split: str, seed: int, step: int, target_l: int) -> list[dict[str, Any]]:
    prefix = target_prefix(data_root, split, seed, step)
    rows: list[dict[str, Any]] = []
    if not prefix.exists():
        return rows
    for mp in sorted(prefix.rglob("manifest.json")):
        if not mp.parent.name.startswith("shard="):
            continue
        m = read_json(mp)
        s0 = int(m.get("sample_start", -1)); s1 = int(m.get("sample_end_exclusive", -1))
        if s0 < 0 or s1 <= s0 or s0 >= target_l:
            continue
        pp = mp.parent / "online_samples.parquet"
        for r in read_online_rows(pp):
            if int(r["sample_index"]) < target_l:
                rows.append(r)
    rows.sort(key=lambda r: (str(r["prompt_id"]), int(r["sample_index"])))
    return rows


def prompt_summary_schema() -> Any:
    try:
        import pyarrow as pa  # type: ignore
    except Exception as exc:
        raise Program05Error(f"Cannot import pyarrow: {exc}") from exc
    return pa.schema([
        pa.field("dataset", pa.string(), nullable=False),
        pa.field("dataset_revision", pa.string(), nullable=False),
        pa.field("training_seed", pa.int64(), nullable=False),
        pa.field("split", pa.string(), nullable=False),
        pa.field("target_step", pa.int64(), nullable=False),
        pa.field("prompt_id", pa.string(), nullable=False),
        pa.field("success_count", pa.int64(), nullable=False),
        pa.field("n_samples", pa.int64(), nullable=False),
        pa.field("prompt_accuracy", pa.float64(), nullable=False),
    ])


def atomic_write_parquet(path: Path, rows: Sequence[Mapping[str, Any]], schema: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        table = pa.Table.from_pylist([dict(r) for r in rows], schema=schema)
        pq.write_table(table, tmp, compression="zstd", use_dictionary=True)
        with tmp.open("rb+") as f:
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        fsync_directory(path.parent)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def rebuild_target_summary(
    *, root: Path, data_root: Path, split: str, seed: int, target_step: int, target_l: int,
    prompts: Sequence[PromptRecord], dataset_revision: str, adapter_sha256: str,
) -> tuple[Path, Path, dict[str, Any]]:
    rows = collect_target_rows(data_root, split, seed, target_step, target_l)
    expected_total = len(prompts) * target_l
    if len(rows) != expected_total:
        raise Program05Error(
            f"Cannot summarize seed={seed}, step={target_step}: observed {len(rows)} online rows, expected {expected_total}."
        )
    by_prompt: dict[str, list[dict[str, Any]]] = {p.prompt_id: [] for p in prompts}
    seen_ids: set[str] = set()
    for r in rows:
        pid = str(r["prompt_id"])
        if pid not in by_prompt:
            raise Program05Error(f"Unexpected prompt {pid} while rebuilding target summary.")
        if r["online_id"] in seen_ids:
            raise Program05Error(f"Duplicate online_id while rebuilding target summary: {r['online_id']}")
        seen_ids.add(str(r["online_id"])); by_prompt[pid].append(r)
    summaries: list[dict[str, Any]] = []
    prompt_accs: list[float] = []
    mc_var_terms: list[float] = []
    total_success = 0
    for p in prompts:
        xs = sorted(by_prompt[p.prompt_id], key=lambda r: int(r["sample_index"]))
        indices = [int(r["sample_index"]) for r in xs]
        if indices != list(range(target_l)):
            raise Program05Error(f"Online samples are not complete 0..{target_l-1} for prompt {p.prompt_id}.")
        success = sum(int(float(r["correct"])) for r in xs)
        acc = success / target_l
        total_success += success
        prompt_accs.append(acc)
        # Unbiased Bernoulli sample-mean variance estimate for fixed prompt j:
        # s_j^2/L = p_hat(1-p_hat)/(L-1), L>1.
        if target_l > 1:
            mc_var_terms.append(acc * (1.0 - acc) / (target_l - 1))
        summaries.append({
            "dataset": "GSM8K", "dataset_revision": dataset_revision, "training_seed": seed,
            "split": split, "target_step": target_step, "prompt_id": p.prompt_id,
            "success_count": success, "n_samples": target_l, "prompt_accuracy": acc,
        })
    m = len(prompt_accs)
    online_ref = math.fsum(prompt_accs) / m
    pooled = total_success / expected_total
    if not math.isclose(online_ref, pooled, rel_tol=0.0, abs_tol=1e-15):
        raise Program05Error("Prompt-equal online reference differs from pooled rate despite uniform L.")
    mean = online_ref
    prompt_var = math.fsum((x - mean) ** 2 for x in prompt_accs) / (m - 1) if m > 1 else 0.0
    naive_prompt_se = math.sqrt(prompt_var / m) if m > 0 else float("nan")
    mc_conditional_se = math.sqrt(math.fsum(mc_var_terms)) / m if m > 0 else float("nan")

    out_dir = target_prefix(data_root, split, seed, target_step)
    summary_path = out_dir / "prompt_summary.parquet"
    reference_path = out_dir / "reference_summary.json"
    atomic_write_parquet(summary_path, summaries, prompt_summary_schema())
    prompt_summary_sha = sha256_file(summary_path)
    shard_manifests = sorted(str(p.relative_to(root)) for p in out_dir.rglob("manifest.json") if p.parent.name.startswith("shard="))
    shard_set_sha = sha256_bytes(canonical_bytes([
        {"path": x, "sha256": sha256_file(root / x)} for x in shard_manifests
    ]))
    payload = {
        "schema_version": ONLINE_SCHEMA,
        "manifest_type": "online_reference_target_summary",
        "updated_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "dataset": "GSM8K", "dataset_revision": dataset_revision,
        "training_seed": seed, "split": split, "target_step": target_step,
        "target_adapter_sha256": adapter_sha256,
        "n_prompts": m, "samples_per_prompt": target_l,
        "total_samples": expected_total, "total_success": total_success,
        "online_reference": online_ref,
        "reference_name": "high-precision on-policy Monte Carlo reference",
        "not_exact_truth": True,
        "uncertainty": {
            "naive_prompt_se": naive_prompt_se,
            "conditional_generation_mc_se": mc_conditional_se,
            "primary_paper_ci_note": "Program 06 should use prompt-level paired/bootstrap inference; these SEs are descriptive diagnostics.",
        },
        "prompt_summary": {
            "local_path": rel(summary_path, root), "file_sha256": prompt_summary_sha,
            "row_count": len(summaries),
        },
        "source_online_shard_set_sha256": shard_set_sha,
    }
    atomic_write_json(reference_path, payload)
    return summary_path, reference_path, payload


# ---------------------------------------------------------------------------
# Planning and target-checkpoint collection
# ---------------------------------------------------------------------------


def engineering_spec(base: OnlineSpec, mode: str, available_steps: Sequence[int]) -> tuple[OnlineSpec, int | None]:
    if mode == "paper":
        return base, None
    steps = sorted(set(int(x) for x in available_steps))
    if not steps or 0 not in steps:
        raise Program05Error(f"Program 02 {mode} checkpoints do not contain step 0.")
    final = max(steps)
    targets = (0,) if final == 0 else (0, final)
    if mode == "pilot":
        spec = OnlineSpec(
            **{**asdict(base), "target_steps": targets, "audit_steps": tuple(), "l_main": min(4, base.l_main),
               "l_audit": min(4, base.l_audit), "sample_block_size": min(4, base.sample_block_size),
               "generation_batch_size": min(4, base.generation_batch_size), "prompts_per_shard": min(8, base.prompts_per_shard)}
        )
        return spec, 50
    spec = OnlineSpec(
        **{**asdict(base), "target_steps": targets, "audit_steps": tuple(), "l_main": min(2, base.l_main),
           "l_audit": min(2, base.l_audit), "sample_block_size": min(2, base.sample_block_size),
           "generation_batch_size": min(2, base.generation_batch_size), "prompts_per_shard": min(4, base.prompts_per_shard),
           "max_completion_length": min(64, base.max_completion_length)}
    )
    return spec, 4


def expected_unit_count(prompt_count: int, prompts_per_shard: int, target_l: int, sample_block_size: int) -> int:
    n_prompt_shards = math.ceil(prompt_count / prompts_per_shard)
    n_sample_blocks = math.ceil(target_l / sample_block_size)
    return n_prompt_shards * n_sample_blocks


def count_missing_units(
    *, data_root: Path, prompts: Sequence[PromptRecord], seeds: Sequence[int], target_steps: Sequence[int],
    target_l_main: int, spec: OnlineSpec,
) -> int:
    pshards = prompt_shards(prompts, spec.prompts_per_shard)
    missing = 0
    for seed in seeds:
        for step in target_steps:
            tl = target_sample_limit(spec, step, target_l_main)
            for s0, s1 in sample_blocks(tl, spec.sample_block_size):
                for shard_index, _ in enumerate(pshards):
                    unit = online_unit_dir(
                        data_root, split=prompts[0].research_split, training_seed=seed, target_step=step,
                        sample_start=s0, sample_end=s1, shard_index=shard_index,
                    )
                    if not unit.exists():
                        missing += 1
    return missing


def target_has_any_state(data_root: Path, split: str, seed: int, step: int) -> bool:
    return target_prefix(data_root, split, seed, step).exists()


def collect_seed_target(
    *, root: Path, mode: str, split: str, seed: int, target_step: int, target_l: int,
    spec: OnlineSpec, prompts: Sequence[PromptRecord], data_root: Path, model_dir: Path,
    model_record: Mapping[str, Any], training_manifest: Mapping[str, Any], adapter: AdapterRecord,
    protocol_version_value: str, dataset_revision: str, device: str, resume: bool,
) -> None:
    tokenizer = load_tokenizer(model_dir, model_record)
    tokenizer_revision = str(model_record.get("tokenizer_revision") or model_record.get("resolved_revision"))
    chat_hash = sha256_text(tokenizer.chat_template if isinstance(tokenizer.chat_template, str) else json.dumps(tokenizer.chat_template, ensure_ascii=False, sort_keys=True, default=str))
    gen_hash = sha256_bytes(canonical_bytes(generation_config_record(spec, tokenizer)))
    pshards = prompt_shards(prompts, spec.prompts_per_shard)
    blocks = sample_blocks(target_l, spec.sample_block_size)

    existing = target_has_any_state(data_root, split, seed, target_step)
    if existing and not resume:
        # A fully complete target can be verified below; partial state requires
        # explicit resume so accidental reruns do not silently mix assets.
        all_exist = True
        for s0, s1 in blocks:
            for shard_index, pshard in enumerate(pshards):
                unit = online_unit_dir(data_root, split=split, training_seed=seed, target_step=target_step,
                                       sample_start=s0, sample_end=s1, shard_index=shard_index)
                if not unit.exists():
                    all_exist = False; break
                verify_online_unit(
                    unit, root=root, expected_prompts=pshard, sample_start=s0, sample_end=s1,
                    training_seed=seed, target_step=target_step, split=split,
                    protocol_version_value=protocol_version_value, dataset_revision=dataset_revision,
                    model_revision=str(model_record.get("resolved_revision")), adapter_sha256=adapter.payload_sha256,
                    spec=spec,
                )
            if not all_exist:
                break
        if all_exist:
            print(f"[SKIP VERIFIED TARGET] seed={seed} e={target_step} L={target_l}")
            return
        raise Program05Error(
            f"Partial Program 05 state exists for seed={seed}, target={target_step}. Re-run with --resume."
        )

    missing_units: list[tuple[int, int, int, list[PromptRecord], Path]] = []
    for s0, s1 in blocks:
        for shard_index, pshard in enumerate(pshards):
            unit = online_unit_dir(data_root, split=split, training_seed=seed, target_step=target_step,
                                   sample_start=s0, sample_end=s1, shard_index=shard_index)
            if unit.exists():
                verify_online_unit(
                    unit, root=root, expected_prompts=pshard, sample_start=s0, sample_end=s1,
                    training_seed=seed, target_step=target_step, split=split,
                    protocol_version_value=protocol_version_value, dataset_revision=dataset_revision,
                    model_revision=str(model_record.get("resolved_revision")), adapter_sha256=adapter.payload_sha256,
                    spec=spec,
                )
            else:
                missing_units.append((s0, s1, shard_index, pshard, unit))
    if not missing_units:
        print(f"[TARGET VERIFIED] seed={seed} e={target_step} L={target_l}")
        return

    training_precision = training_manifest.get("resolved_precision")
    print(f"[LOAD TARGET] seed={seed} target_step={target_step} adapter={adapter.path} missing_units={len(missing_units)}")
    model, precision = load_target_model(
        model_dir=model_dir, adapter_dir=adapter.path, device=device,
        preferred_precision=POLICY_INFERENCE_PRECISION,
    )
    if precision != POLICY_INFERENCE_PRECISION:
        raise Program05Error(
            f"Policy-inference precision contract violated: expected "
            f"{POLICY_INFERENCE_PRECISION}, got {precision}."
        )
    print(
        f"[TARGET READY] seed={seed} e={target_step} "
        f"policy precision={precision}; training precision={training_precision}"
    )
    try:
        for s0, s1, shard_index, pshard, unit in missing_units:
            shard_rows: list[dict[str, Any]] = []
            for prompt in pshard:
                shard_rows.extend(generate_prompt_block(
                    model=model, tokenizer=tokenizer, prompt=prompt, sample_start=s0, sample_end=s1,
                    spec=spec, protocol_version_value=protocol_version_value,
                    dataset_revision=dataset_revision, model_revision=str(model_record.get("resolved_revision")),
                    tokenizer_revision=tokenizer_revision, training_seed=seed, target_step=target_step,
                    split=split, adapter_sha256=adapter.payload_sha256, device=device,
                ))
            publish_online_unit(
                unit_dir=unit, root=root, rows=shard_rows, expected_prompts=pshard,
                sample_start=s0, sample_end=s1, training_seed=seed, target_step=target_step,
                split=split, shard_index=shard_index, protocol_version_value=protocol_version_value,
                dataset_revision=dataset_revision, model_revision=str(model_record.get("resolved_revision")),
                tokenizer_revision=tokenizer_revision, chat_template_hash=chat_hash,
                adapter_sha256=adapter.payload_sha256, online_config_sha256=online_spec_fingerprint(spec),
                generation_config_sha256=gen_hash, spec=spec,
            )
            print(f"[SHARD SAVED] seed={seed} e={target_step} samples={s0}:{s1} shard={shard_index}")
    finally:
        try:
            import torch  # type: ignore
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO-OPE Program 05: generate immutable on-policy Monte Carlo references.")
    p.add_argument("--config", default="configs/protocol.yaml")
    p.add_argument("--mode", choices=("smoke", "pilot", "paper"), default="paper")
    p.add_argument("--split", choices=("development", "test"), default="development")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--output-root", default=".")
    p.add_argument("--seed", type=int, default=None, help="Restrict to one legal configured training seed.")
    p.add_argument("--target-step", type=int, default=None, help="Restrict to one legal target checkpoint for debugging/resume.")
    p.add_argument(
        "--target-l-main", type=int, default=None,
        help="Paper development-only append target for non-audit checkpoints (8 or 16). Test forbids overrides.",
    )
    p.add_argument("--verify-only", action="store_true", help="Verify existing assets/summaries; never load a target model.")
    p.add_argument(
        "--reset-checkpoints", "--reset-online-reference", dest="reset_outputs", action="store_true",
        help="DANGEROUS: delete Program 05 outputs for selected mode/split. Requires GRPO_OPE_ALLOW_RESET=YES.",
    )
    return p.parse_args(argv)


def safe_reset(paths: Sequence[Path]) -> None:
    if os.environ.get("GRPO_OPE_ALLOW_RESET") != "YES":
        raise Program05Error("Reset requires environment variable GRPO_OPE_ALLOW_RESET=YES.")

    def _remove_readonly(func: Any, path: str, exc_info: Any) -> None:
        try:
            os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
            func(path)
        except Exception:
            raise

    for p in paths:
        if p.exists():
            print(f"[RESET] removing {p}")
            if p.is_dir():
                shutil.rmtree(p, onerror=_remove_readonly)
            else:
                os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
                p.unlink()


def print_header(
    *, root: Path, mode: str, split: str, cfg_path: Path, config_sha: str,
    protocol_version_value: str, data_revision: str, model_revision: str,
    seeds: Sequence[int], target_steps: Sequence[int], target_l_main: int,
    spec: OnlineSpec, prompt_count: int, missing_units: int, data_root: Path,
) -> None:
    print("=" * 90)
    print("GRPO-OPE Program 05 — on-policy Monte Carlo reference generation")
    print(f"program version        : {PROGRAM_VERSION}")
    print(f"project root           : {root}")
    print(f"mode / split           : {mode} / {split}")
    print(f"protocol version       : {protocol_version_value}")
    print(f"config                 : {cfg_path}")
    print(f"config SHA-256         : {config_sha}")
    print(f"online config hash     : {online_spec_fingerprint(spec)}")
    print(f"model revision         : {model_revision}")
    print(f"data revision          : {data_revision}")
    print(f"training seeds         : {list(seeds)}")
    print(f"target checkpoints     : {list(target_steps)}")
    print(f"prompts                : {prompt_count}")
    print(f"L_main / L_audit       : {target_l_main} / {spec.l_audit}")
    print(f"policy precision       : {POLICY_INFERENCE_PRECISION} (fixed; independent of Program 02 training precision)")
    print(f"audit steps            : {list(spec.audit_steps)}")
    print(f"sample block / gen grp : {spec.sample_block_size} / {spec.generation_batch_size}")
    print(f"prompt shard size      : {spec.prompts_per_shard}")
    print(f"max completion         : {spec.max_completion_length}")
    print(f"sampling               : T={spec.temperature}, top_p={spec.top_p}, top_k={spec.top_k}, rep={spec.repetition_penalty}")
    print(f"unfinished units       : {missing_units}")
    print(f"online output root     : {data_root}")
    print("reference semantics    : high-precision on-policy Monte Carlo; NOT exact truth")
    print("research boundary      : direct target sampling only; no OPE/ESS/gate fitting")
    print("=" * 90)


def run(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    start_time = time.time()
    root = Path(args.output_root).expanduser().resolve()
    cfg_path = Path(args.config).expanduser()
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path

    manifests = root / "manifests"
    gsm8k_dir = root / "data" / "raw" / "gsm8k"
    model_dir = root / "models" / "qwen25_05b"
    registry_path = root / "data" / "splits" / "gsm8k_split_registry.parquet"

    env_manifest = verify_environment_manifest(manifests / "environment_manifest.json")
    data_manifest, gsm_record = verify_data_manifest(manifests / "data_manifest.json", gsm8k_dir)
    model_manifest, model_record = verify_model_manifest(manifests / "model_manifest.json", model_dir)
    split_manifest, registry_rows = verify_split_registry(
        manifests / "split_registry_manifest.json", registry_path, data_manifest, gsm_record
    )
    cfg, config_sha = load_protocol(cfg_path)
    pversion = protocol_version(cfg)
    base_spec = parse_online_spec(cfg)
    base_seeds = configured_seeds(cfg)

    if args.mode == "paper":
        validate_paper_contract(base_spec, base_seeds)
        lock = verify_protocol_lock(
            manifests / "protocol_lock.json", config_sha256=config_sha, data_manifest=data_manifest,
            model_record=model_record, split_manifest=split_manifest, online_spec=base_spec,
        )
    else:
        lock = {}
        if args.split == "test":
            raise Program05Error("smoke/pilot modes are development-only; official test is forbidden.")

    if args.split == "test":
        if args.mode != "paper":
            raise Program05Error("Official test is legal only in paper mode.")
        if args.target_l_main is not None:
            raise Program05Error("Official test forbids --target-l-main; use the frozen L_main.")
        verify_frozen_gate_for_test(root, lock)

    selected_seeds = list(base_seeds if args.mode == "paper" else (base_seeds[0],))
    if args.seed is not None:
        if args.seed not in selected_seeds:
            raise Program05Error(f"--seed {args.seed} is not legal for mode={args.mode}; expected one of {selected_seeds}.")
        selected_seeds = [args.seed]

    training: dict[int, dict[str, Any]] = {}
    adapters: dict[int, dict[int, AdapterRecord]] = {}
    available_union: set[int] = set()
    for seed in selected_seeds:
        tm, amap = verify_training_seed(
            root=root, mode=args.mode, seed=seed, model_record=model_record,
            data_manifest=data_manifest, split_manifest=split_manifest, config_sha256=config_sha,
        )
        training[seed] = tm; adapters[seed] = amap; available_union.update(amap.keys())

    spec, prompt_limit = engineering_spec(base_spec, args.mode, sorted(available_union))
    if args.mode == "paper":
        spec = base_spec
    target_l_main = spec.l_main
    if args.target_l_main is not None:
        if args.mode != "paper" or args.split != "development":
            raise Program05Error("--target-l-main is allowed only for paper-mode development append experiments.")
        if args.target_l_main not in PAPER_ALLOWED_L_MAIN:
            raise Program05Error(f"Development --target-l-main must be one of {PAPER_ALLOWED_L_MAIN}.")
        if args.target_l_main < spec.l_main:
            raise Program05Error("--target-l-main cannot shrink below configured L_main; online samples are append-only.")
        target_l_main = int(args.target_l_main)

    selected_steps = list(spec.target_steps)
    if args.target_step is not None:
        if args.target_step not in selected_steps:
            raise Program05Error(f"--target-step {args.target_step} is not in the legal target grid {selected_steps}.")
        selected_steps = [args.target_step]

    for seed, amap in adapters.items():
        missing_steps = [s for s in selected_steps if s not in amap]
        if missing_steps:
            raise Program05Error(f"Seed {seed} lacks required target adapters {missing_steps}.")

    # Correctness gate before any target generation.
    program04_identity = verify_program04_identity_gate(root, args.mode, args.split, selected_seeds)

    original_split = "test" if args.split == "test" else "train"
    raw = load_upstream_rows(gsm8k_dir, original_split)
    prompts = build_prompt_records(
        registry_rows=registry_rows, raw_rows=raw, research_split=args.split, limit=prompt_limit,
    )

    data_root = online_data_root(root, args.mode)
    mroot = online_manifest_root(root, args.mode, args.split)
    if args.reset_outputs:
        safe_reset([data_root / "gsm8k" / f"split={args.split}", mroot])
    cleanup_staging(data_root, args.resume)

    missing_units = count_missing_units(
        data_root=data_root, prompts=prompts, seeds=selected_seeds, target_steps=selected_steps,
        target_l_main=target_l_main, spec=spec,
    )
    print_header(
        root=root, mode=args.mode, split=args.split, cfg_path=cfg_path, config_sha=config_sha,
        protocol_version_value=pversion, data_revision=str(gsm_record.get("resolved_revision")),
        model_revision=str(model_record.get("resolved_revision")), seeds=selected_seeds,
        target_steps=selected_steps, target_l_main=target_l_main, spec=spec,
        prompt_count=len(prompts), missing_units=missing_units, data_root=data_root,
    )
    print(f"git commit             : {git_commit(root) or 'not-a-git-checkout'}")
    print(f"environment manifest   : {env_manifest.get('environment_fingerprint_sha256')}")
    print(f"split registry hash    : {split_manifest.get('content_fingerprint_sha256')}")
    print(f"Program 04 identity    : PASS ({program04_identity['pair_summaries_sha256'][:12]}...)")

    if args.verify_only and missing_units:
        raise Program05Error(f"--verify-only found {missing_units} missing online shard units.")

    summaries: list[dict[str, Any]] = []
    for seed in selected_seeds:
        for step in selected_steps:
            tl = target_sample_limit(spec, step, target_l_main)
            if not args.verify_only:
                collect_seed_target(
                    root=root, mode=args.mode, split=args.split, seed=seed, target_step=step,
                    target_l=tl, spec=spec, prompts=prompts, data_root=data_root,
                    model_dir=model_dir, model_record=model_record, training_manifest=training[seed],
                    adapter=adapters[seed][step], protocol_version_value=pversion,
                    dataset_revision=str(gsm_record.get("resolved_revision")), device=args.device,
                    resume=args.resume,
                )
            # Verify all expected units before deriving the summary.
            pshards = prompt_shards(prompts, spec.prompts_per_shard)
            for s0, s1 in sample_blocks(tl, spec.sample_block_size):
                for shard_index, pshard in enumerate(pshards):
                    unit = online_unit_dir(data_root, split=args.split, training_seed=seed, target_step=step,
                                           sample_start=s0, sample_end=s1, shard_index=shard_index)
                    if not unit.exists():
                        raise Program05Error(f"Expected online shard is missing after collection: {unit}")
                    verify_online_unit(
                        unit, root=root, expected_prompts=pshard, sample_start=s0, sample_end=s1,
                        training_seed=seed, target_step=step, split=args.split,
                        protocol_version_value=pversion, dataset_revision=str(gsm_record.get("resolved_revision")),
                        model_revision=str(model_record.get("resolved_revision")),
                        adapter_sha256=adapters[seed][step].payload_sha256, spec=spec,
                    )
            sp, rp, summary = rebuild_target_summary(
                root=root, data_root=data_root, split=args.split, seed=seed, target_step=step,
                target_l=tl, prompts=prompts, dataset_revision=str(gsm_record.get("resolved_revision")),
                adapter_sha256=adapters[seed][step].payload_sha256,
            )
            summaries.append({
                "training_seed": seed, "target_step": step, "samples_per_prompt": tl,
                "online_reference": summary["online_reference"],
                "conditional_generation_mc_se": summary["uncertainty"]["conditional_generation_mc_se"],
                "prompt_summary_path": rel(sp, root), "reference_summary_path": rel(rp, root),
                "reference_summary_sha256": sha256_file(rp),
            })
            print(
                f"[REFERENCE] seed={seed} e={step} L={tl} "
                f"V_online={summary['online_reference']:.6f} "
                f"MC_SE={summary['uncertainty']['conditional_generation_mc_se']:.6f}"
            )
            rebuild_collection_index(
                root=root, data_root=data_root, manifest_root=mroot, mode=args.mode, split=args.split,
                target_l_main=target_l_main, spec=spec, seeds=selected_seeds, target_steps=selected_steps,
                dataset_revision=str(gsm_record.get("resolved_revision")),
                model_revision=str(model_record.get("resolved_revision")), protocol_version_value=pversion,
                config_sha256=config_sha, program04_identity=program04_identity,
            )

    final_missing = count_missing_units(
        data_root=data_root, prompts=prompts, seeds=selected_seeds, target_steps=selected_steps,
        target_l_main=target_l_main, spec=spec,
    )
    if final_missing != 0:
        raise Program05Error(f"Program finished with {final_missing} expected online shard units still missing.")
    index_path = rebuild_collection_index(
        root=root, data_root=data_root, manifest_root=mroot, mode=args.mode, split=args.split,
        target_l_main=target_l_main, spec=spec, seeds=selected_seeds, target_steps=selected_steps,
        dataset_revision=str(gsm_record.get("resolved_revision")), model_revision=str(model_record.get("resolved_revision")),
        protocol_version_value=pversion, config_sha256=config_sha, program04_identity=program04_identity,
    )
    summary_index = mroot / "reference_summaries.json"
    summaries.sort(key=lambda x: (int(x["training_seed"]), int(x["target_step"])))
    atomic_write_json(summary_index, {
        "schema_version": ONLINE_SCHEMA,
        "manifest_type": "online_reference_summary_index",
        "updated_at_utc": now_utc(),
        "mode": args.mode, "split": args.split,
        "reference_name": "high-precision on-policy Monte Carlo reference",
        "not_exact_truth": True,
        "policy_inference_precision": POLICY_INFERENCE_PRECISION,
        "target_l_main": target_l_main, "l_audit": spec.l_audit,
        "audit_steps": list(spec.audit_steps), "summaries": summaries,
        "summary_set_sha256": sha256_bytes(canonical_bytes(summaries)),
    })
    elapsed = time.time() - start_time
    print("\n" + "=" * 78)
    print("PROGRAM 05 PASSED")
    print(f"seed-target references : {len(summaries)}")
    print(f"collection index       : {index_path}")
    print(f"summary index          : {summary_index}")
    print(f"elapsed                : {elapsed/60:.1f} min")
    print("Reference semantics: independent on-policy Monte Carlo estimate, not exact truth.")
    print("=" * 78)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return run(argv)
    except Program05Error as exc:
        print(f"\nPROGRAM 05 FAILED\n{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nPROGRAM 05 INTERRUPTED. Published shards remain valid; resume with --resume.", file=sys.stderr)
        return 130
    except Exception as exc:
        print("\nPROGRAM 05 UNEXPECTED FAILURE", file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
