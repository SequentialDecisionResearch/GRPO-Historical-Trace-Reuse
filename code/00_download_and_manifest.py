#!/usr/bin/env python3
"""GRPO-OPE Program 00: download, revision pinning, and immutable manifests.

This script implements only the external-asset layer of the paper:
  * primary data: openai/gsm8k, config=main
  * primary model: Qwen/Qwen2.5-0.5B-Instruct
  * optional robustness data: official arkilpatel/SVAMP Git repository
  * exact local Python/ML package versions

It does not create research splits, train GRPO, inspect test rewards, compute OPE,
or fit any reuse gate.

First run resolves mutable upstream refs (normally "main") to exact commit SHAs,
downloads snapshots, validates them, hashes every file, writes immutable manifests,
and makes raw/model files read-only. Later runs verify the frozen local assets; they
do not silently follow a newer upstream "main" revision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


PROGRAM = "00_download_and_manifest.py"
PROGRAM_VERSION = "1.0.0"
MANIFEST_SCHEMA = "1.0"

GSM8K_REPO = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_ROWS = {"train": 7473, "test": 1319}
GSM8K_FIELDS = {"question", "answer"}

MODEL_REPO = "Qwen/Qwen2.5-0.5B-Instruct"
MIN_TRANSFORMERS = (4, 37, 0)

SVAMP_URL = "https://github.com/arkilpatel/SVAMP.git"
SVAMP_ROWS = 1000

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


class Program00Error(RuntimeError):
    pass


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


def json_safe(x: Any) -> Any:
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Mapping):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [json_safe(v) for v in x]
    if hasattr(x, "to_dict"):
        try:
            return json_safe(x.to_dict())
        except Exception:
            pass
    return str(x)


def rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception as exc:
        raise Program00Error(f"Cannot read {path}: {exc}") from exc
    if not isinstance(obj, dict):
        raise Program00Error(f"Manifest is not a JSON object: {path}")
    return obj


def write_manifest_once(path: Path, payload: Mapping[str, Any]) -> None:
    """Crash-safe publication; never overwrite an existing manifest."""
    if path.exists():
        raise Program00Error(f"Refusing to overwrite immutable manifest: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp-{uuid.uuid4().hex}"
    try:
        with tmp.open("x", encoding="utf-8", newline="\n") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            raise Program00Error(f"Manifest appeared concurrently: {path}")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def payload_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in HASH_EXCLUDE_DIRS for part in p.relative_to(root).parts):
            continue
        files.append(p)
    return sorted(files, key=lambda p: p.as_posix())


def tree_hash(root: Path) -> dict[str, Any]:
    if not root.is_dir():
        raise Program00Error(f"Asset directory is missing: {root}")
    records: list[dict[str, Any]] = []
    total = 0
    for p in payload_files(root):
        size = p.stat().st_size
        records.append({"path": rel(p, root), "size_bytes": size, "sha256": sha256_file(p)})
        total += size
    if not records:
        raise Program00Error(f"No files found in asset directory: {root}")
    return {
        "tree_sha256": sha256_bytes(canonical_bytes(records)),
        "file_count": len(records),
        "total_bytes": total,
        "files": records,
    }


def nonempty(path: Path) -> bool:
    return path.exists() and (not path.is_dir() or any(path.iterdir()))


def remove_transport_metadata(root: Path) -> None:
    def _remove_readonly(func, path, exc_info):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except Exception:
            raise

    for name in (".cache", ".git"):
        p = root / name
        if p.exists():
            shutil.rmtree(p, onerror=_remove_readonly)


def make_read_only(root: Path) -> None:
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
        print("[WARN] Some read-only flags could not be applied; hash verification remains authoritative.")
        for msg in failures[:5]:
            print(f"       {msg}")
        if len(failures) > 5:
            print(f"       ... {len(failures) - 5} more")


def run(cmd: Sequence[str], *, cwd: Path | None = None, timeout: int = 120) -> str:
    try:
        cp = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise Program00Error(f"Executable not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise Program00Error(f"Command timed out: {' '.join(cmd)}") from exc
    if cp.returncode != 0:
        raise Program00Error(
            f"Command failed ({cp.returncode}): {' '.join(cmd)}\n{cp.stderr.strip()}"
        )
    return cp.stdout.strip()


def version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def version_tuple(s: str) -> tuple[int, int, int]:
    nums = [int(x) for x in re.findall(r"\d+", s)[:3]]
    nums += [0] * (3 - len(nums))
    return nums[0], nums[1], nums[2]


def check_environment(config_path: Path) -> dict[str, Any]:
    packages = {p: version(p) for p in CORE_PACKAGES + EXTRA_PACKAGES}
    missing = [p for p in CORE_PACKAGES if packages[p] is None]
    if missing:
        raise Program00Error(
            "Missing required project packages: " + ", ".join(missing)
            + ". Install the GRPO-OPE environment before Program 00."
        )
    tv = packages["transformers"]
    assert tv is not None
    if version_tuple(tv) < MIN_TRANSFORMERS:
        raise Program00Error(f"Qwen2.5 requires transformers>=4.37.0; found {tv}.")

    config_info = inspect_protocol_yaml(config_path)

    try:
        import torch  # type: ignore

        torch_info: dict[str, Any] = {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_available": bool(torch.cuda.is_available()),
            "devices": [],
        }
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                torch_info["devices"].append(
                    {
                        "index": i,
                        "name": prop.name,
                        "memory_bytes": int(prop.total_memory),
                        "compute_capability": [int(prop.major), int(prop.minor)],
                    }
                )
    except Exception as exc:
        raise Program00Error(f"PyTorch environment inspection failed: {exc}") from exc

    fingerprint_basis = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": packages,
        "torch_cuda_build": torch_info["cuda_build"],
        "cudnn_version": torch_info["cudnn_version"],
    }
    return {
        "schema_version": MANIFEST_SCHEMA,
        "manifest_type": "environment",
        "created_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": packages,
        "torch_runtime": torch_info,
        "protocol_config_observed": config_info,
        "environment_fingerprint_basis": fingerprint_basis,
        "environment_fingerprint_sha256": sha256_bytes(canonical_bytes(fingerprint_basis)),
    }


def inspect_protocol_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path), "sha256": None}
    if version("PyYAML") is None:
        raise Program00Error(f"{path} exists but PyYAML is not installed.")
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        raise Program00Error(f"Cannot parse protocol config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        raise Program00Error("protocol.yaml must contain a YAML mapping.")

    project = cfg.get("project") or {}
    model = cfg.get("model") or {}
    if project and not isinstance(project, dict):
        raise Program00Error("protocol.yaml: project must be a mapping.")
    if model and not isinstance(model, dict):
        raise Program00Error("protocol.yaml: model must be a mapping.")
    project_name = project.get("name") if isinstance(project, dict) else None
    primary = model.get("primary") if isinstance(model, dict) else None
    if project_name not in (None, "grpo_ope_reuse"):
        raise Program00Error(f"Unexpected project.name={project_name!r}.")
    if primary not in (None, MODEL_REPO):
        raise Program00Error(f"protocol.yaml model.primary must be {MODEL_REPO!r}, found {primary!r}.")
    return {
        "exists": True,
        "path": str(path),
        "sha256": sha256_file(path),
        "project_name": project_name,
        "protocol_version": project.get("protocol_version") if isinstance(project, dict) else None,
        "model_primary": primary,
        "model_revision_field": model.get("revision") if isinstance(model, dict) else None,
    }


def verify_environment_manifest(path: Path, current: Mapping[str, Any]) -> None:
    old = read_json(path)
    if old.get("manifest_type") != "environment":
        raise Program00Error(f"Wrong manifest type in {path}")
    expected = old.get("environment_fingerprint_sha256")
    observed = current.get("environment_fingerprint_sha256")
    if expected != observed:
        raise Program00Error(
            "Current software environment differs from frozen environment_manifest.json.\n"
            f"expected={expected}\nobserved={observed}\n"
            "Use the frozen environment or create a new project/protocol version."
        )


def hf_tools() -> tuple[Any, Any]:
    try:
        from huggingface_hub import HfApi, snapshot_download  # type: ignore
    except Exception as exc:
        raise Program00Error(f"Cannot import huggingface_hub: {exc}") from exc
    return HfApi, snapshot_download


def card_dict(card: Any) -> dict[str, Any]:
    if card is None:
        return {}
    if isinstance(card, dict):
        return json_safe(card)
    if hasattr(card, "to_dict"):
        try:
            x = card.to_dict()
            return json_safe(x) if isinstance(x, dict) else {}
        except Exception:
            pass
    return json_safe(getattr(card, "__dict__", {}))


def hf_metadata(info: Any) -> dict[str, Any]:
    card = card_dict(getattr(info, "card_data", None))
    license_value = card.get("license")
    if license_value is None:
        tags = list(getattr(info, "tags", None) or [])
        license_tags = [t.split(":", 1)[1] for t in tags if isinstance(t, str) and t.startswith("license:")]
        license_value = license_tags or None
    return {
        "license": license_value,
        "private": getattr(info, "private", None),
        "gated": getattr(info, "gated", None),
        "last_modified": str(getattr(info, "last_modified", None)) if getattr(info, "last_modified", None) else None,
    }


def resolve_hf(repo_id: str, repo_type: str, requested: str, token: str | None) -> tuple[str, dict[str, Any]]:
    HfApi, _ = hf_tools()
    api = HfApi()
    try:
        if repo_type == "dataset":
            info = api.dataset_info(repo_id, revision=requested, token=token)
        elif repo_type == "model":
            info = api.model_info(repo_id, revision=requested, token=token)
        else:
            raise Program00Error(f"Unsupported repo_type={repo_type}")
    except Exception as exc:
        raise Program00Error(f"Cannot resolve {repo_id}@{requested}: {exc}") from exc
    sha = getattr(info, "sha", None)
    if not isinstance(sha, str) or re.fullmatch(r"[0-9a-fA-F]{40}", sha) is None:
        raise Program00Error(f"Hub did not return a valid commit SHA for {repo_id}.")
    return sha.lower(), hf_metadata(info)


def hf_download(repo_id: str, repo_type: str, sha: str, staging: Path, token: str | None, workers: int) -> None:
    _, snapshot_download = hf_tools()
    if staging.exists():
        raise Program00Error(f"Staging path already exists: {staging}")
    staging.mkdir(parents=True)
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=sha,
            local_dir=str(staging),
            token=token,
            max_workers=workers,
            force_download=False,
        )
        remove_transport_metadata(staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def publish(staging: Path, final: Path) -> None:
    if final.exists():
        raise Program00Error(f"Refusing to overwrite asset directory: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, final)


def validate_gsm8k(root: Path) -> dict[str, Any]:
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:
        raise Program00Error(f"Cannot import datasets: {exc}") from exc

    train = sorted((root / GSM8K_CONFIG).glob("train*.parquet"))
    test = sorted((root / GSM8K_CONFIG).glob("test*.parquet"))
    if not train or not test:
        # Conservative fallback for a future snapshot that keeps the main files
        # deeper in the repository while preserving their names.
        train = sorted(p for p in root.rglob("train*.parquet") if GSM8K_CONFIG in p.relative_to(root).parts)
        test = sorted(p for p in root.rglob("test*.parquet") if GSM8K_CONFIG in p.relative_to(root).parts)
    if not train or not test:
        raise Program00Error("Pinned GSM8K snapshot does not contain main/train*.parquet and main/test*.parquet.")

    try:
        ds = load_dataset(
            "parquet",
            data_files={"train": [str(p) for p in train], "test": [str(p) for p in test]},
        )
    except Exception as exc:
        raise Program00Error(f"Cannot load local GSM8K parquet snapshot: {exc}") from exc

    details: dict[str, Any] = {}
    for split, expected_rows in GSM8K_ROWS.items():
        observed_rows = len(ds[split])
        columns = set(ds[split].column_names)
        missing = GSM8K_FIELDS - columns
        if observed_rows != expected_rows:
            raise Program00Error(f"GSM8K {split}: expected {expected_rows} rows, found {observed_rows}.")
        if missing:
            raise Program00Error(f"GSM8K {split}: missing required fields {sorted(missing)}.")
        files = train if split == "train" else test
        details[split] = {
            "row_count": observed_rows,
            "expected_row_count": expected_rows,
            "columns": sorted(columns),
            "source_files": [rel(p, root) for p in files],
        }
    return {"config": GSM8K_CONFIG, "required_fields": sorted(GSM8K_FIELDS), "splits": details, "status": "passed"}


def tokenizer_files(root: Path) -> list[Path]:
    names = {
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "added_tokens.json", "vocab.json", "merges.txt", "spiece.model", "tokenizer.model",
    }
    return sorted(
        [p for p in payload_files(root) if p.name in names or p.suffix == ".model"],
        key=lambda p: p.as_posix(),
    )


def subset_hash(root: Path, files: Sequence[Path]) -> str:
    items = [{"path": rel(p, root), "size_bytes": p.stat().st_size, "sha256": sha256_file(p)} for p in files]
    return sha256_bytes(canonical_bytes(items))


def optional_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json_safe(json.load(f))
    except Exception as exc:
        raise Program00Error(f"Invalid JSON file {path}: {exc}") from exc


def validate_model(root: Path) -> dict[str, Any]:
    if not (root / "config.json").exists() or not (root / "tokenizer_config.json").exists():
        raise Program00Error("Pinned Qwen snapshot is missing config.json or tokenizer_config.json.")
    try:
        from transformers import AutoTokenizer  # type: ignore

        tok = AutoTokenizer.from_pretrained(str(root), local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        raise Program00Error(f"Cannot load pinned Qwen tokenizer locally: {exc}") from exc

    template = getattr(tok, "chat_template", None)
    if template is None:
        raise Program00Error("Pinned Qwen tokenizer has no chat_template.")
    template_text = template if isinstance(template, str) else json.dumps(json_safe(template), ensure_ascii=False, sort_keys=True)

    tfiles = tokenizer_files(root)
    if not tfiles:
        raise Program00Error("No tokenizer files found in pinned Qwen snapshot.")

    model_cfg = optional_json(root / "config.json") or {}
    gen_path = root / "generation_config.json"
    gen_cfg = optional_json(gen_path)
    summary_keys = (
        "model_type", "architectures", "vocab_size", "hidden_size", "intermediate_size",
        "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
        "max_position_embeddings", "torch_dtype", "tie_word_embeddings",
    )
    return {
        "status": "passed",
        "config_json_sha256": sha256_file(root / "config.json"),
        "model_config_summary": {k: model_cfg.get(k) for k in summary_keys if isinstance(model_cfg, dict) and k in model_cfg},
        "tokenizer": {
            "class": tok.__class__.__name__,
            "vocab_size": int(getattr(tok, "vocab_size", -1)),
            "model_max_length": int(getattr(tok, "model_max_length", -1)),
            "special_tokens_map": json_safe(getattr(tok, "special_tokens_map", {})),
            "tokenizer_files": [rel(p, root) for p in tfiles],
            "tokenizer_files_sha256": subset_hash(root, tfiles),
            "chat_template_sha256": sha256_text(template_text),
            "chat_template_length_chars": len(template_text),
        },
        "generation_config": {
            "present": gen_path.exists(),
            "path": "generation_config.json" if gen_path.exists() else None,
            "sha256": sha256_file(gen_path) if gen_path.exists() else None,
            "config": gen_cfg,
        },
    }


def verify_tree(root: Path, record: Mapping[str, Any], label: str) -> None:
    observed = tree_hash(root)["tree_sha256"]
    expected = record.get("tree_sha256")
    if observed != expected:
        raise Program00Error(f"{label} hash mismatch: expected {expected}, observed {observed}.")


def resolve_git(url: str, requested: str | None) -> str:
    run(["git", "--version"], timeout=15)
    if requested and re.fullmatch(r"[0-9a-fA-F]{40}", requested):
        # Checkout below is the authoritative existence check.  A raw commit SHA
        # need not be advertised as a branch/tag by git ls-remote.
        return requested.lower()
    ref = requested or "HEAD"
    try:
        out = run(["git", "ls-remote", url, ref], timeout=60)
    except Program00Error:
        out = ""
    if not out and requested:
        out = run(["git", "ls-remote", url, f"refs/heads/{requested}", f"refs/tags/{requested}"], timeout=60)
    lines = [x for x in out.splitlines() if x.strip()]
    if not lines:
        raise Program00Error(f"Cannot resolve SVAMP Git ref {ref!r}.")
    sha = lines[0].split()[0].lower()
    if re.fullmatch(r"[0-9a-f]{40}", sha) is None:
        raise Program00Error(f"Invalid Git SHA returned for SVAMP: {sha!r}")
    return sha


def git_download(url: str, sha: str, staging: Path) -> None:
    if staging.exists():
        raise Program00Error(f"Staging path already exists: {staging}")
    try:
        run(["git", "clone", "--quiet", "--no-checkout", url, str(staging)], timeout=600)
        run(["git", "checkout", "--quiet", "--detach", sha], cwd=staging, timeout=300)
        remove_transport_metadata(staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def svamp_json(root: Path) -> Path:
    candidates = [root / "SVAMP.json", root / "data" / "SVAMP.json"] + sorted(root.rglob("SVAMP.json"))
    seen: set[str] = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p
    raise Program00Error("SVAMP.json not found in pinned SVAMP snapshot.")


def validate_svamp(root: Path) -> dict[str, Any]:
    p = svamp_json(root)
    try:
        with p.open("r", encoding="utf-8") as f:
            rows = json.load(f)
    except Exception as exc:
        raise Program00Error(f"Cannot parse SVAMP.json: {exc}") from exc
    if not isinstance(rows, list) or len(rows) != SVAMP_ROWS:
        raise Program00Error(f"SVAMP: expected {SVAMP_ROWS} rows, found {len(rows) if isinstance(rows, list) else 'non-list'}.")
    licenses = []
    for x in root.rglob("*"):
        if x.is_file() and (x.name.upper().startswith("LICENSE") or x.name.upper().startswith("COPYING")):
            licenses.append({"path": rel(x, root), "sha256": sha256_file(x)})
    return {
        "status": "passed",
        "row_count": len(rows),
        "expected_row_count": SVAMP_ROWS,
        "svamp_json_path": rel(p, root),
        "svamp_json_sha256": sha256_file(p),
        "license": "MIT",
        "license_files": sorted(licenses, key=lambda x: x["path"]),
    }


def new_staging(final: Path) -> Path:
    final.parent.mkdir(parents=True, exist_ok=True)
    return final.parent / f".{final.name}.staging-{uuid.uuid4().hex}"


def build_gsm8k_record(root: Path, requested: str, resolved: str, hub: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "primary_dataset",
        "repo_id": GSM8K_REPO,
        "repo_type": "dataset",
        "source_url": f"https://huggingface.co/datasets/{GSM8K_REPO}",
        "requested_revision": requested,
        "resolved_revision": resolved,
        "downloaded_at_utc": now_utc(),
        "local_path": "data/raw/gsm8k",
        "license": hub.get("license"),
        "hub_metadata": json_safe(hub),
        "validation": validate_gsm8k(root),
        **tree_hash(root),
    }


def build_model_record(root: Path, requested: str, resolved: str, hub: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "role": "primary_model",
        "repo_id": MODEL_REPO,
        "repo_type": "model",
        "source_url": f"https://huggingface.co/{MODEL_REPO}",
        "requested_revision": requested,
        "resolved_revision": resolved,
        "tokenizer_revision": resolved,
        "downloaded_at_utc": now_utc(),
        "local_path": "models/qwen25_05b",
        "license": hub.get("license"),
        "hub_metadata": json_safe(hub),
        "validation": validate_model(root),
        **tree_hash(root),
    }


def build_svamp_record(root: Path, requested: str, resolved: str) -> dict[str, Any]:
    return {
        "role": "external_prompt_distribution_robustness",
        "source_type": "git",
        "source_url": SVAMP_URL,
        "requested_revision": requested,
        "resolved_revision": resolved,
        "downloaded_at_utc": now_utc(),
        "local_path": "data/raw/svamp",
        "validation": validate_svamp(root),
        **tree_hash(root),
    }


def data_manifest(gsm8k: Mapping[str, Any], svamp: Mapping[str, Any] | None) -> dict[str, Any]:
    datasets: dict[str, Any] = {"gsm8k": json_safe(gsm8k)}
    if svamp is not None:
        datasets["svamp"] = json_safe(svamp)
    scope = {
        "primary_dataset": GSM8K_REPO,
        "optional_robustness_dataset": "SVAMP" if svamp is not None else None,
        "excluded_from_main_paper": ["OBD", "CartPole", "M5/inventory", "D4RL/Minari", "CVRPLIB", "stock data"],
    }
    return {
        "schema_version": MANIFEST_SCHEMA,
        "manifest_type": "data",
        "created_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "research_scope": scope,
        "datasets": datasets,
        "content_fingerprint_sha256": sha256_bytes(canonical_bytes({"research_scope": scope, "datasets": datasets})),
    }


def model_manifest(model: Mapping[str, Any]) -> dict[str, Any]:
    models = {"primary": json_safe(model)}
    return {
        "schema_version": MANIFEST_SCHEMA,
        "manifest_type": "model",
        "created_at_utc": now_utc(),
        "created_by": {"program": PROGRAM, "program_version": PROGRAM_VERSION},
        "models": models,
        "content_fingerprint_sha256": sha256_bytes(canonical_bytes(models)),
    }


def verify_data_manifest(path: Path, gsm8k_dir: Path, svamp_dir: Path) -> dict[str, Any]:
    m = read_json(path)
    if m.get("schema_version") != MANIFEST_SCHEMA or m.get("manifest_type") != "data":
        raise Program00Error(f"Invalid data manifest header: {path}")
    dsets = m.get("datasets")
    if not isinstance(dsets, dict) or not isinstance(dsets.get("gsm8k"), dict):
        raise Program00Error("data_manifest.json lacks datasets.gsm8k.")
    g = dsets["gsm8k"]
    if g.get("repo_id") != GSM8K_REPO:
        raise Program00Error("Frozen data manifest points to an unexpected primary dataset.")
    validate_gsm8k(gsm8k_dir)
    verify_tree(gsm8k_dir, g, "GSM8K")
    if "svamp" in dsets:
        s = dsets["svamp"]
        if not isinstance(s, dict):
            raise Program00Error("datasets.svamp must be a JSON object.")
        validate_svamp(svamp_dir)
        verify_tree(svamp_dir, s, "SVAMP")
    return m


def verify_model_manifest(path: Path, model_dir: Path) -> dict[str, Any]:
    m = read_json(path)
    if m.get("schema_version") != MANIFEST_SCHEMA or m.get("manifest_type") != "model":
        raise Program00Error(f"Invalid model manifest header: {path}")
    models = m.get("models")
    if not isinstance(models, dict) or not isinstance(models.get("primary"), dict):
        raise Program00Error("model_manifest.json lacks models.primary.")
    r = models["primary"]
    if r.get("repo_id") != MODEL_REPO:
        raise Program00Error("Frozen model manifest points to an unexpected primary model.")
    current = validate_model(model_dir)
    verify_tree(model_dir, r, "Qwen model")
    old_validation = r.get("validation") or {}
    old_tok = old_validation.get("tokenizer") if isinstance(old_validation, dict) else None
    if not isinstance(old_tok, dict):
        raise Program00Error("Frozen model manifest lacks tokenizer validation fields.")
    for key in ("tokenizer_files_sha256", "chat_template_sha256"):
        if old_tok.get(key) != current["tokenizer"].get(key):
            raise Program00Error(f"Tokenizer {key} mismatch against frozen model manifest.")
    return m


def requested_revision_compatible(arg: str | None, record: Mapping[str, Any], label: str) -> None:
    if arg is None:
        return
    if arg not in {record.get("requested_revision"), record.get("resolved_revision")}:
        raise Program00Error(
            f"{label} is already frozen. --revision={arg!r} differs from the immutable manifest; "
            "create a new project/protocol version instead."
        )


def create_data_assets(
    gsm8k_dir: Path,
    svamp_dir: Path,
    *,
    gsm8k_revision: str | None,
    include_svamp: bool,
    svamp_revision: str | None,
    token: str | None,
    workers: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if nonempty(gsm8k_dir) or nonempty(svamp_dir):
        raise Program00Error("Raw data directory already contains unmanifested assets; refusing silent adoption/overwrite.")
    if gsm8k_dir.exists():
        gsm8k_dir.rmdir()
    if svamp_dir.exists():
        svamp_dir.rmdir()

    requested = gsm8k_revision or "main"
    resolved, hub = resolve_hf(GSM8K_REPO, "dataset", requested, token)
    stage = new_staging(gsm8k_dir)
    hf_download(GSM8K_REPO, "dataset", resolved, stage, token, workers)
    try:
        gsm8k_record = build_gsm8k_record(stage, requested, resolved, hub)
        publish(stage, gsm8k_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)

    svamp_record = None
    if include_svamp:
        requested_s = svamp_revision or "HEAD"
        resolved_s = resolve_git(SVAMP_URL, svamp_revision)
        stage_s = new_staging(svamp_dir)
        git_download(SVAMP_URL, resolved_s, stage_s)
        try:
            svamp_record = build_svamp_record(stage_s, requested_s, resolved_s)
            publish(stage_s, svamp_dir)
        finally:
            if stage_s.exists():
                shutil.rmtree(stage_s, ignore_errors=True)
    return gsm8k_record, svamp_record


def create_model_asset(
    model_dir: Path,
    *,
    model_revision: str | None,
    token: str | None,
    workers: int,
) -> dict[str, Any]:
    if nonempty(model_dir):
        raise Program00Error("Model directory already contains an unmanifested asset; refusing silent adoption/overwrite.")
    if model_dir.exists():
        model_dir.rmdir()
    requested = model_revision or "main"
    resolved, hub = resolve_hf(MODEL_REPO, "model", requested, token)
    stage = new_staging(model_dir)
    hf_download(MODEL_REPO, "model", resolved, stage, token, workers)
    try:
        record = build_model_record(stage, requested, resolved, hub)
        publish(stage, model_dir)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)
    return record


def restore_hf_asset_from_manifest(
    final: Path, record: Mapping[str, Any], *, token: str | None, workers: int
) -> None:
    repo = record.get("repo_id")
    repo_type = record.get("repo_type")
    sha = record.get("resolved_revision")
    if not all(isinstance(x, str) for x in (repo, repo_type, sha)):
        raise Program00Error("Manifest lacks repo_id/repo_type/resolved_revision for restoration.")
    stage = new_staging(final)
    hf_download(str(repo), str(repo_type), str(sha), stage, token, workers)
    try:
        verify_tree(stage, record, str(repo))
        if repo_type == "dataset":
            validate_gsm8k(stage)
        else:
            validate_model(stage)
        publish(stage, final)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def restore_svamp_from_manifest(final: Path, record: Mapping[str, Any]) -> None:
    sha = record.get("resolved_revision")
    if not isinstance(sha, str):
        raise Program00Error("SVAMP manifest lacks resolved_revision.")
    stage = new_staging(final)
    git_download(SVAMP_URL, sha, stage)
    try:
        verify_tree(stage, record, "SVAMP")
        validate_svamp(stage)
        publish(stage, final)
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GRPO-OPE Program 00: download, pin exact revisions, and write immutable manifests.")
    p.add_argument("--config", default="configs/protocol.yaml")
    p.add_argument("--mode", choices=("smoke", "pilot", "paper"), default="paper")
    p.add_argument("--output-root", default=".")
    p.add_argument("--gsm8k-revision", default=None, help="First-run HF branch/tag/SHA; default main.")
    p.add_argument("--model-revision", default=None, help="First-run HF branch/tag/SHA; default main.")
    p.add_argument("--include-svamp", action="store_true", help="Pin the optional official SVAMP repository on the first run.")
    p.add_argument("--svamp-revision", default=None, help="First-run Git ref/SHA; default HEAD.")
    p.add_argument("--hf-token-env", default="HF_TOKEN", help="Optional HF token environment variable; token is never logged.")
    p.add_argument("--max-workers", type=int, default=8)
    p.add_argument("--verify-only", action="store_true", help="Local verification only; no network/downloads.")
    args = p.parse_args(argv)
    if args.max_workers < 1:
        p.error("--max-workers must be >=1")
    return args


def print_header(root: Path, mode: str, env: Mapping[str, Any], data_manifest_path: Path, model_manifest_path: Path) -> None:
    cfg = env.get("protocol_config_observed") or {}
    data_sha = "unresolved"
    model_sha = "unresolved"
    if data_manifest_path.exists():
        try:
            data_sha = read_json(data_manifest_path)["datasets"]["gsm8k"]["resolved_revision"]
        except Exception:
            data_sha = "invalid-existing-manifest"
    if model_manifest_path.exists():
        try:
            model_sha = read_json(model_manifest_path)["models"]["primary"]["resolved_revision"]
        except Exception:
            model_sha = "invalid-existing-manifest"
    print("=" * 78)
    print("GRPO-OPE Program 00 — Download / Pin / Manifest")
    print(f"project root      : {root}")
    print(f"mode              : {mode}")
    print(f"protocol version  : {cfg.get('protocol_version') or 'not locked'}")
    print(f"config hash       : {cfg.get('sha256') or 'none'}")
    print(f"GSM8K revision    : {data_sha}")
    print(f"model revision    : {model_sha}")
    print("split / seed      : N/A / N/A")
    print("research boundary : external assets only; no statistical/OPE experiment")
    print("=" * 78)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    config = Path(args.config).expanduser()
    if not config.is_absolute():
        config = root / config

    gsm8k_dir = root / "data" / "raw" / "gsm8k"
    svamp_dir = root / "data" / "raw" / "svamp"
    model_dir = root / "models" / "qwen25_05b"
    manifests = root / "manifests"
    data_manifest_path = manifests / "data_manifest.json"
    model_manifest_path = manifests / "model_manifest.json"
    env_manifest_path = manifests / "environment_manifest.json"

    current_env = check_environment(config)
    print_header(root, args.mode, current_env, data_manifest_path, model_manifest_path)

    # Freeze/verify software environment first.  Config hash is recorded but is
    # intentionally not part of the environment fingerprint because protocol.yaml
    # may still change during the later development/pilot stage before protocol lock.
    if env_manifest_path.exists():
        verify_environment_manifest(env_manifest_path, current_env)
        print(f"[OK] environment: {env_manifest_path}")
    elif args.verify_only:
        raise Program00Error(f"Missing {env_manifest_path}")
    else:
        write_manifest_once(env_manifest_path, current_env)
        print(f"[WRITE] {env_manifest_path}")

    token = os.environ.get(args.hf_token_env) or None

    # DATA -------------------------------------------------------------------
    if data_manifest_path.exists():
        dm = read_json(data_manifest_path)
        dsets = dm.get("datasets") if isinstance(dm, dict) else None
        if not isinstance(dsets, dict) or not isinstance(dsets.get("gsm8k"), dict):
            raise Program00Error("Invalid data_manifest.json")
        requested_revision_compatible(args.gsm8k_revision, dsets["gsm8k"], "GSM8K")
        if args.include_svamp and "svamp" not in dsets:
            raise Program00Error(
                "data_manifest.json was frozen without SVAMP. Do not mutate it in place; "
                "use a new protocol/project version if SVAMP must be added later."
            )
        if "svamp" in dsets:
            requested_revision_compatible(args.svamp_revision, dsets["svamp"], "SVAMP")
        elif nonempty(svamp_dir):
            raise Program00Error(
                f"Unmanifested SVAMP content exists at {svamp_dir}; refusing to ignore it."
            )
        if not gsm8k_dir.exists():
            if args.verify_only:
                raise Program00Error(f"Pinned GSM8K directory is missing: {gsm8k_dir}")
            restore_hf_asset_from_manifest(gsm8k_dir, dsets["gsm8k"], token=token, workers=args.max_workers)
        if "svamp" in dsets and not svamp_dir.exists():
            if args.verify_only:
                raise Program00Error(f"Pinned SVAMP directory is missing: {svamp_dir}")
            restore_svamp_from_manifest(svamp_dir, dsets["svamp"])
        dm = verify_data_manifest(data_manifest_path, gsm8k_dir, svamp_dir)
        print(f"[OK] data manifest: {data_manifest_path}")
    else:
        if args.verify_only:
            raise Program00Error(f"Missing {data_manifest_path}")
        gsm, svamp = create_data_assets(
            gsm8k_dir,
            svamp_dir,
            gsm8k_revision=args.gsm8k_revision,
            include_svamp=args.include_svamp,
            svamp_revision=args.svamp_revision,
            token=token,
            workers=args.max_workers,
        )
        write_manifest_once(data_manifest_path, data_manifest(gsm, svamp))
        dm = verify_data_manifest(data_manifest_path, gsm8k_dir, svamp_dir)
        print(f"[WRITE] {data_manifest_path}")

    # MODEL ------------------------------------------------------------------
    if model_manifest_path.exists():
        mm = read_json(model_manifest_path)
        models = mm.get("models") if isinstance(mm, dict) else None
        if not isinstance(models, dict) or not isinstance(models.get("primary"), dict):
            raise Program00Error("Invalid model_manifest.json")
        requested_revision_compatible(args.model_revision, models["primary"], "Qwen model")
        if not model_dir.exists():
            if args.verify_only:
                raise Program00Error(f"Pinned model directory is missing: {model_dir}")
            restore_hf_asset_from_manifest(model_dir, models["primary"], token=token, workers=args.max_workers)
        mm = verify_model_manifest(model_manifest_path, model_dir)
        print(f"[OK] model manifest: {model_manifest_path}")
    else:
        if args.verify_only:
            raise Program00Error(f"Missing {model_manifest_path}")
        model_record = create_model_asset(
            model_dir,
            model_revision=args.model_revision,
            token=token,
            workers=args.max_workers,
        )
        write_manifest_once(model_manifest_path, model_manifest(model_record))
        mm = verify_model_manifest(model_manifest_path, model_dir)
        print(f"[WRITE] {model_manifest_path}")

    # Enforce the intended immutable local raw/model layer after all hashes pass.
    make_read_only(gsm8k_dir)
    if "svamp" in dm.get("datasets", {}):
        make_read_only(svamp_dir)
    make_read_only(model_dir)

    # One final verification after chmod; permissions must not affect content hashes.
    verify_data_manifest(data_manifest_path, gsm8k_dir, svamp_dir)
    verify_model_manifest(model_manifest_path, model_dir)
    verify_environment_manifest(env_manifest_path, current_env)

    print("=" * 78)
    print("PROGRAM 00 PASSED")
    print(f"GSM8K SHA : {dm['datasets']['gsm8k']['resolved_revision']}")
    print(f"MODEL SHA : {mm['models']['primary']['resolved_revision']}")
    if "svamp" in dm.get("datasets", {}):
        print(f"SVAMP SHA : {dm['datasets']['svamp']['resolved_revision']}")
    print(f"data manifest        : {data_manifest_path}")
    print(f"model manifest       : {model_manifest_path}")
    print(f"environment manifest : {env_manifest_path}")
    print("Next research step: Program 01 may build the immutable GSM8K split registry.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[ABORTED] User interrupted Program 00.", file=sys.stderr)
        raise SystemExit(130)
    except Program00Error as exc:
        print(f"\n[FAIL-FAST] {exc}", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print(f"\n[UNEXPECTED ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
