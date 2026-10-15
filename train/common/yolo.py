from __future__ import annotations

import gc
import io
import logging
import os
import shutil
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from train.common.paths import configure_ultralytics_dirs, export_models_dir, resolve_pretrained
from train.common.progress import SweepProgress, write as term_write
from train.common.ranking import MAP_TOL, mark_yolo_winner, save_ranking


@dataclass
class SweepConfig:
    name: str
    kind: str
    repo: Path
    dataset_dir: Path
    data_yaml: Path
    weights_cache: Path
    runs_root: Path
    project_dir: Path
    sweep_dir: Path
    ranking_csv: Path
    docs_dir: Path
    candidates: list[dict]
    epochs: int = 300
    patience: int = 40
    seed: int = 42
    workers: int = 0
    skip_existing: bool = True
    cache_images: bool = True
    map_tol: float = MAP_TOL
    winner_weight_name: str = ""
    aabb_dir: Path | None = None
    aabb_yaml: Path | None = None
    class_names: dict[int, str] = field(default_factory=dict)
    optim: dict = field(default_factory=dict)
    loss: dict = field(default_factory=dict)
    augment: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    include_confusion: bool = False


@contextmanager
def quiet_ultralytics():
    import ultralytics.utils as uutils
    from ultralytics.utils import LOGGER

    prev_level = LOGGER.level
    prev_verbose = uutils.VERBOSE
    LOGGER.setLevel(logging.ERROR)
    uutils.VERBOSE = False
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            yield
    finally:
        LOGGER.setLevel(prev_level)
        uutils.VERBOSE = prev_verbose


@contextmanager
def cwd(path: Path):
    prev = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _is_oom(err: BaseException) -> bool:
    text = f"{type(err).__name__} {err}".lower()
    return "out of memory" in text or ("cuda" in text and "memory" in text)


def _load_model(path: str | Path, task: str):
    from ultralytics import YOLO

    if task == "detect":
        from ultralytics import RTDETR

        return RTDETR(str(path))
    return YOLO(str(path))


def ensure_aabb_dataset(cfg: SweepConfig) -> Path:
    if cfg.aabb_dir is None or cfg.aabb_yaml is None:
        raise RuntimeError("AABB não configurado neste sweep.")
    marker = cfg.aabb_dir / ".ok"
    if cfg.aabb_yaml.is_file() and marker.is_file():
        return cfg.aabb_yaml
    cfg.aabb_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ("train", "valid", "test"):
        img_src = cfg.dataset_dir / split_name / "images"
        lab_src = cfg.dataset_dir / split_name / "labels"
        if not img_src.is_dir():
            continue
        img_dst = cfg.aabb_dir / split_name / "images"
        lab_dst = cfg.aabb_dir / split_name / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lab_dst.mkdir(parents=True, exist_ok=True)
        for img in img_src.iterdir():
            if img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                continue
            dest_img = img_dst / img.name
            if not dest_img.exists():
                try:
                    os.link(img, dest_img)
                except OSError:
                    shutil.copy2(img, dest_img)
            lab = lab_src / f"{img.stem}.txt"
            lines_out = []
            if lab.is_file():
                for line in lab.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 9:
                        continue
                    cls = parts[0]
                    xy = list(map(float, parts[1:9]))
                    xs, ys = xy[0::2], xy[1::2]
                    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                    xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                    w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
                    lines_out.append(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
            (lab_dst / f"{img.stem}.txt").write_text("".join(lines_out), encoding="utf-8")
    cfg.aabb_yaml.write_text(
        f"path: {cfg.aabb_dir.as_posix()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names:\n"
        "  0: card\n",
        encoding="utf-8",
    )
    marker.write_text("ok\n", encoding="utf-8")
    return cfg.aabb_yaml


def data_yaml_for(cfg: SweepConfig, cand: dict) -> Path:
    if cand.get("task") == "detect":
        if cfg.aabb_yaml is None:
            raise RuntimeError("Candidato AABB sem aabb_yaml.")
        return cfg.aabb_yaml
    return cfg.data_yaml


def _obb_metrics(result, class_names: dict[int, str] | None = None) -> dict[str, float]:
    met = getattr(result, "obb", None) or getattr(result, "box", None)
    p = float(met.mp) if met is not None else float("nan")
    r = float(met.mr) if met is not None else float("nan")
    f1 = 2 * p * r / max(p + r, 1e-9)
    speed = getattr(result, "speed", {}) or {}
    inf_ms = float(speed.get("inference", float("nan")))
    map75 = getattr(met, "map75", None) if met is not None else None
    out = {
        "precision": p,
        "recall": r,
        "f1": f1,
        "map50": float(met.map50) if met is not None else float("nan"),
        "map75": float(map75) if map75 is not None else float("nan"),
        "map50_95": float(met.map) if met is not None else float("nan"),
        "infer_ms": inf_ms,
    }
    if class_names and len(class_names) > 1 and met is not None:
        maps = getattr(met, "maps", None)
        if maps is not None:
            arr = np.asarray(maps, dtype=float).ravel()
            for idx, name in class_names.items():
                if idx < len(arr):
                    out[f"{name}_map50_95"] = float(arr[idx])
    return out


def _run_finished(cand_dir: Path, epochs: int, patience: int) -> bool:
    best = cand_dir / "weights" / "best.pt"
    csv_path = cand_dir / "results.csv"
    if not best.is_file() or not csv_path.is_file():
        return False
    hist = pd.read_csv(csv_path)
    if len(hist) >= epochs:
        return True
    map_cols = [c for c in hist.columns if "mAP50-95" in str(c)]
    if not map_cols or len(hist) < patience + 1:
        return False
    series = hist[map_cols[0]]
    best_i = int(series.idxmax())
    return (len(hist) - 1 - best_i) >= patience


def _epochs_ran(cand_dir: Path) -> int:
    csv_path = cand_dir / "results.csv"
    if not csv_path.is_file():
        return 0
    try:
        return int(len(pd.read_csv(csv_path)))
    except Exception:
        return 0


def _build_model(cfg: SweepConfig, cand: dict):
    model_src = resolve_pretrained(cand["model"], cfg.weights_cache)
    init_src = (
        resolve_pretrained(cand["init_from"], cfg.weights_cache) if cand.get("init_from") else None
    )
    with cwd(cfg.weights_cache):
        model = _load_model(model_src, cand.get("task", "obb"))
        if init_src and cand.get("task", "obb") != "detect":
            try:
                model = model.load(init_src)
            except Exception:
                pass
        return model


def _train_kw(cfg: SweepConfig, cand: dict, imgsz: int, batch: int, device: str) -> dict:
    kw = dict(
        data=str(data_yaml_for(cfg, cand)),
        epochs=cfg.epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(cfg.sweep_dir),
        name=cand["id"],
        exist_ok=True,
        patience=cfg.patience,
        seed=cfg.seed,
        workers=cfg.workers,
        pretrained=str(cand["model"]).endswith(".pt"),
        plots=False,
        val=True,
        verbose=False,
        cache=cfg.cache_images,
        amp=True,
        save=True,
        **cfg.optim,
        **cfg.augment,
    )
    if cand.get("task", "obb") == "obb":
        kw.update(cfg.loss)
    else:
        kw.update({k: v for k, v in cfg.loss.items() if k != "angle"})
    return kw


def _map_from_trainer(trainer) -> float | None:
    metrics = getattr(trainer, "metrics", None)
    if not isinstance(metrics, dict):
        return None
    for key, value in metrics.items():
        if "mAP50-95" in str(key):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _attach_epoch_status(model, progress: SweepProgress, epochs: int) -> None:
    def on_fit_epoch_end(trainer):
        ep = int(getattr(trainer, "epoch", 0)) + 1
        n = int(getattr(trainer, "epochs", epochs) or epochs)
        progress.epoch_end(ep, n, _map_from_trainer(trainer))

    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)


def val_weights(
    cfg: SweepConfig,
    cand: dict,
    weights: Path,
    split: str,
    *,
    plots: bool,
    name: str,
    device: str,
):
    args_yaml = (cfg.sweep_dir / cand["id"] / "args.yaml")
    imgsz = int(cand["imgsz"])
    if args_yaml.is_file():
        saved = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
        imgsz = int(saved.get("imgsz", imgsz))
    data_yaml = data_yaml_for(cfg, cand)
    with quiet_ultralytics():
        model = _load_model(weights, cand.get("task", "obb"))
        return model.val(
            data=str(data_yaml),
            split=split,
            imgsz=imgsz,
            device=device,
            project=str(cfg.sweep_dir),
            name=name,
            exist_ok=True,
            plots=plots,
            workers=cfg.workers,
            verbose=False,
        )


def train_candidate(
    cfg: SweepConfig,
    cand: dict,
    device: str,
    progress: SweepProgress,
    index: int,
) -> dict:
    cand_id = cand["id"]
    family = cand["family"]
    cand_dir = cfg.sweep_dir / cand_id
    best_pt = cand_dir / "weights" / "best.pt"
    last_pt = cand_dir / "weights" / "last.pt"
    imgsz, batch = int(cand["imgsz"]), int(cand["batch"])
    started = time.time()

    if cand.get("task") == "detect":
        ensure_aabb_dataset(cfg)

    progress.start_family(index, family, cfg.epochs)
    if cfg.skip_existing and _run_finished(cand_dir, cfg.epochs, cfg.patience):
        progress.note("já treinado — reutilizando best.pt")
    elif last_pt.is_file() and not best_pt.is_file():
        progress.note("retomando last.pt")
        model = _load_model(last_pt, cand.get("task", "obb"))
        _attach_epoch_status(model, progress, cfg.epochs)
        with quiet_ultralytics():
            model.train(resume=True)
    elif last_pt.is_file() and not _run_finished(cand_dir, cfg.epochs, cfg.patience):
        progress.note("retomando last.pt")
        model = _load_model(last_pt, cand.get("task", "obb"))
        _attach_epoch_status(model, progress, cfg.epochs)
        with quiet_ultralytics():
            model.train(resume=True)
    else:
        while True:
            try:
                model = _build_model(cfg, cand)
                _attach_epoch_status(model, progress, cfg.epochs)
                with quiet_ultralytics():
                    model.train(**_train_kw(cfg, cand, imgsz, batch, device))
                break
            except Exception as err:
                if not _is_oom(err):
                    raise
                torch.cuda.empty_cache()
                gc.collect()
                if batch > 1:
                    batch = max(1, batch // 2)
                    progress.note(f"OOM → batch={batch}")
                    continue
                if imgsz > 800:
                    imgsz = 800
                    progress.note(f"OOM → imgsz={imgsz}")
                    continue
                if imgsz > 640:
                    imgsz = 640
                    progress.note(f"OOM → imgsz={imgsz}")
                    continue
                raise

    if not best_pt.is_file():
        raise FileNotFoundError(f"Sem best.pt em {cand_dir}")

    args_yaml = cand_dir / "args.yaml"
    if args_yaml.is_file():
        saved = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
        imgsz = int(saved.get("imgsz", imgsz))
        batch = int(saved.get("batch", batch))

    progress.note("validando val/test (sem plots)...")
    val_res = val_weights(cfg, cand, best_pt, "val", plots=False, name=f"{cand_id}-val", device=device)
    row: dict[str, Any] = {
        "id": cand_id,
        "family": family,
        "model": cand["model"],
        "task": cand.get("task", "obb"),
        "imgsz": imgsz,
        "batch": batch,
    }
    row.update({f"val_{k}": v for k, v in _obb_metrics(val_res, cfg.class_names).items()})
    data_yaml = data_yaml_for(cfg, cand)
    has_test = data_yaml.is_file() and "test:" in data_yaml.read_text(encoding="utf-8")
    if has_test:
        test_res = val_weights(
            cfg, cand, best_pt, "test", plots=False, name=f"{cand_id}-test", device=device
        )
        row.update({f"test_{k}": v for k, v in _obb_metrics(test_res, cfg.class_names).items()})
    row["hours"] = round((time.time() - started) / 3600, 3)
    row["best_pt"] = str(best_pt)
    row["epochs"] = _epochs_ran(cand_dir)
    summary = (
        f"épocas={row['epochs']}  val mAP50-95={row.get('val_map50_95', float('nan')):.4f}  "
        f"{row['hours']} h"
    )
    progress.family_done(summary)
    torch.cuda.empty_cache()
    gc.collect()
    return row


def run_sweep(cfg: SweepConfig, device: str) -> pd.DataFrame:
    configure_ultralytics_dirs(cfg.runs_root, cfg.weights_cache)
    cfg.sweep_dir.mkdir(parents=True, exist_ok=True)
    progress = SweepProgress(cfg.name, len(cfg.candidates))
    rows = []
    for index, cand in enumerate(cfg.candidates, start=1):
        family = cand["family"]
        try:
            rows.append(train_candidate(cfg, cand, device, progress, index))
        except Exception as err:
            progress.family_fail(f"{type(err).__name__}: {err}")
            rows.append(
                {
                    "id": cand["id"],
                    "family": family,
                    "model": cand["model"],
                    "task": cand.get("task", "obb"),
                    "imgsz": cand["imgsz"],
                    "batch": cand["batch"],
                    "error": str(err),
                }
            )
            torch.cuda.empty_cache()
            gc.collect()
    ranking = pd.DataFrame(rows)
    save_ranking(cfg.ranking_csv, ranking)
    term_write(f"Ranking bruto: {cfg.ranking_csv}")
    return ranking


def elect_winner(cfg: SweepConfig, ranking: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    ranked, winner = mark_yolo_winner(ranking, map_tol=cfg.map_tol)
    save_ranking(cfg.ranking_csv, ranked)
    src = Path(str(winner["best_pt"])) if "best_pt" in winner.index and pd.notna(winner.get("best_pt")) else None
    dest_dir = cfg.sweep_dir / "weights"
    dest_dir.mkdir(parents=True, exist_ok=True)
    final_name = cfg.winner_weight_name or f"{cfg.name}.pt"
    final_pt = dest_dir / final_name
    if src is not None and src.is_file():
        shutil.copy2(src, dest_dir / "best.pt")
        last_src = src.with_name("last.pt")
        if last_src.is_file():
            shutil.copy2(last_src, dest_dir / "last.pt")
        if final_pt.exists():
            final_pt.unlink()
        shutil.copy2(dest_dir / "best.pt", final_pt)
        export_pt = export_models_dir(cfg.repo) / final_name
        shutil.copy2(final_pt, export_pt)
        print(f"Vencedor: {winner.get('family')} ({winner.get('id')}) → {export_pt}")
    else:
        print(f"Vencedor: {winner.get('family')} ({winner.get('id')}) — peso local ausente, só o ranking foi atualizado.")
    return ranked, winner


def generate_candidate_plots(cfg: SweepConfig, cand: dict, device: str) -> None:
    cand_dir = cfg.sweep_dir / cand["id"]
    best_pt = cand_dir / "weights" / "best.pt"
    if not best_pt.is_file():
        term_write(f"  [sem best.pt] {cand['family']}")
        return
    term_write(f"  plots {cand['family']}...")
    val_weights(cfg, cand, best_pt, "val", plots=True, name="report", device=device)


def generate_all_plots(cfg: SweepConfig, device: str) -> None:
    term_write("Gerando gráficos de todas as famílias (val plots=True)...")
    for cand in cfg.candidates:
        generate_candidate_plots(cfg, cand, device)
