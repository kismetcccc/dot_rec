import hashlib
import json
import os
from pathlib import Path
import subprocess
import torch


def source_fingerprint(root):
    """Hash only regular Python source and JSON configuration files; no data/secrets."""
    root = Path(root).resolve()
    hashes = {}
    for directory, suffix in (("model", ".py"), ("common", ".py"), ("utils", ".py"), ("configs", ".json")):
        base = root / directory
        if not base.is_dir() or base.is_symlink():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix != suffix or not path.is_file():
                continue
            if any(part.is_symlink() for part in (path, *path.parents) if part != root):
                continue
            relative = path.relative_to(root).as_posix()
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("main.py", "evaluate.py"):
        path = root / name
        if path.is_file() and not path.is_symlink():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"algorithm": "sha256", "scope": "model/common/utils Python, main.py/evaluate.py and configs JSON; symlinks excluded",
            "files": hashes, "digest": hashlib.sha256(payload).hexdigest() if hashes else None}


def code_version():
    root = Path(__file__).resolve().parents[1]
    fingerprint = source_fingerprint(root)
    # Installed wheels may not have repository metadata; do not invent a SHA.
    try:
        result = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True)
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True)
    except OSError:
        return {"commit": "unknown", "dirty": None, "source_fingerprint": fingerprint}
    return {"commit": result.stdout.strip() if result.returncode == 0 else "unknown",
            "dirty": bool(status.stdout.strip()) if status.returncode == 0 else None,
            "source_fingerprint": fingerprint}


def configure_determinism(device):
    """Only called inside an explicit user run, before seeds or CUDA tensors."""
    if torch.device(device).type == "cuda":
        if torch.cuda.is_initialized():
            raise RuntimeError("Deterministic CUDA runs require a fresh process before CUDA initialization")
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if workspace is None:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        elif workspace not in (":4096:8", ":16:8"):
            raise ValueError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA")
    torch.use_deterministic_algorithms(True)
