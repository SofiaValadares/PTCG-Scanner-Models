from __future__ import annotations

import os
from pathlib import Path


def find_repo() -> Path:
    here = Path(".").resolve()
    for folder in [here, *here.parents]:
        if (folder / "pipeline.py").is_file() and (
            folder / "src" / "cropper" / "rectify.py"
        ).is_file():
            return folder
    raise FileNotFoundError(
        "Rode o comando na raiz do repositório PTCG Scanner "
        "(pipeline.py e src/cropper)."
    )


def pretrained_dir(repo: Path) -> Path:
    preferred = repo / "pretrained"
    legacy = repo / "src" / "detection" / "weights"
    if preferred.is_dir() or not legacy.is_dir():
        return preferred
    return legacy


def export_models_dir(repo: Path) -> Path:
    dest = repo / "models"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def detection_runs(repo: Path) -> Path:
    return _prefer(repo / "runs" / "detection", repo / "src" / "detection" / "runs")


def roi_runs(repo: Path) -> Path:
    return _prefer(repo / "runs" / "roi", repo / "src" / "roi" / "runs")


def ocr_runs(repo: Path) -> Path:
    return _prefer(repo / "runs" / "ocr", repo / "src" / "ocr" / "runs")


def _prefer(new: Path, old: Path) -> Path:
    if new.is_dir() or not old.is_dir():
        return new
    return old


def configure_ultralytics_dirs(runs_root: Path, weights_cache: Path) -> None:
    runs_root.mkdir(parents=True, exist_ok=True)
    weights_cache.mkdir(parents=True, exist_ok=True)
    try:
        from ultralytics.utils import SETTINGS

        SETTINGS.update(
            {
                "runs_dir": str(runs_root),
                "weights_dir": str(weights_cache),
            }
        )
    except Exception as err:
        print(f"[aviso] não deu para fixar SETTINGS do Ultralytics: {err}")


def resolve_pretrained(name: str, weights_cache: Path) -> str:
    raw = Path(name)
    if raw.is_file():
        return str(raw.resolve())
    cached = weights_cache / raw.name
    if cached.is_file():
        return str(cached)
    return name


def chdir_repo(repo: Path) -> None:
    os.chdir(repo)
