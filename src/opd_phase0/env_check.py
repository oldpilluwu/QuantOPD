from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys

import torch

PACKAGES = ("torch", "transformers", "trl", "accelerate", "datasets", "bitsandbytes", "huggingface-hub")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Linux/CUDA environment used by Phase 0.")
    parser.add_argument("--allow-no-cuda", action="store_true", help="Report rather than fail when CUDA is absent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    versions = {package: importlib.metadata.version(package) for package in PACKAGES}
    import_checks = {}
    for module_name in ("bitsandbytes", "trl.experimental.distillation"):
        try:
            importlib.import_module(module_name)
            import_checks[module_name] = "ok"
        except Exception as error:  # pragma: no cover - depends on the server's native CUDA libraries
            import_checks[module_name] = f"{type(error).__name__}: {error}"
    cuda_available = torch.cuda.is_available()
    report = {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "import_checks": import_checks,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
        "gpus": [],
    }
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            report["gpus"].append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    print(json.dumps(report, indent=2))

    failures = []
    if platform.system() != "Linux":
        failures.append("Phase 0 model runs are supported on Linux only")
    if not cuda_available and not args.allow_no_cuda:
        failures.append("CUDA is not available")
    if cuda_available and not torch.cuda.is_bf16_supported():
        failures.append("The selected GPU does not report BF16 support")
    failed_imports = [name for name, result in import_checks.items() if result != "ok"]
    if failed_imports:
        failures.append(f"Required imports failed: {', '.join(failed_imports)}")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
