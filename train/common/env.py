from __future__ import annotations

import json
import platform
from datetime import datetime
from pathlib import Path

import pandas as pd


def inspect_environment(runs_root: Path, weights_cache: Path) -> tuple[pd.DataFrame, str]:
    import torch
    import ultralytics

    cuda_ok = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "—"
    gpu_mem = (
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        if cuda_ok
        else "—"
    )
    device = "0" if cuda_ok else "cpu"
    rows = [
        ("Data/hora", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        ("SO", f"{platform.system()} {platform.release()}"),
        ("Python", platform.python_version()),
        ("PyTorch", torch.__version__),
        ("Ultralytics", ultralytics.__version__),
        ("CUDA disponível", str(cuda_ok)),
        ("CUDA (compilado)", getattr(torch.version, "cuda", None) or "—"),
        ("GPU", gpu_name),
        ("VRAM", gpu_mem),
        ("Dispositivo de treino", device),
        ("cuDNN", str(torch.backends.cudnn.enabled if cuda_ok else False)),
        ("cwd", str(Path.cwd())),
        ("runs_dir", str(runs_root)),
        ("weights_dir", str(weights_cache)),
    ]
    return pd.DataFrame(rows, columns=["Item", "Valor"]), device


def write_environment(path: Path, env_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {str(row["Item"]): str(row["Valor"]) for _, row in env_df.iterrows()}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_environment(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return {str(k): str(v) for k, v in data.items()}
