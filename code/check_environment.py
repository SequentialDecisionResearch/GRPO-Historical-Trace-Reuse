from __future__ import annotations

import importlib
import platform
import sys

PACKAGES = [
    "torch", "transformers", "datasets", "accelerate", "peft", "trl",
    "huggingface_hub", "tokenizers", "safetensors", "pyarrow", "numpy",
    "yaml", "scipy", "pandas", "matplotlib",
]

print("python =", sys.version.replace("\n", " "))
print("platform =", platform.platform())

missing = []
for name in PACKAGES:
    try:
        mod = importlib.import_module(name)
        print(f"{name} = {getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        missing.append((name, str(exc)))
        print(f"{name} = MISSING ({exc})")

try:
    import torch
    print("torch_cuda_build =", torch.version.cuda)
    print("cuda_available =", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu =", torch.cuda.get_device_name(0))
        print("bf16_supported =", torch.cuda.is_bf16_supported())
        props = torch.cuda.get_device_properties(0)
        print("gpu_memory_GB =", round(props.total_memory / 1024**3, 2))
except Exception as exc:
    print("torch_cuda_check_failed =", exc)

if missing:
    print("\nEnvironment is NOT ready. Missing packages:")
    for name, err in missing:
        print(f"  - {name}: {err}")
    raise SystemExit(1)

print("\nEnvironment dependency check passed.")
