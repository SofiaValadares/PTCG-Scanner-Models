from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train.common.dataset import print_dataset_report, validate_dataset
from train.common.env import inspect_environment, write_environment
from train.common.paths import (
    chdir_repo,
    configure_ultralytics_dirs,
    detection_runs,
    find_repo,
    pretrained_dir,
)
from train.common.report import write_yolo_pdf, write_yolo_readme
from train.common.yolo import SweepConfig, elect_winner, generate_all_plots, run_sweep


def build_config(repo: Path, *, skip_existing: bool) -> SweepConfig:
    dataset_dir = repo / "data" / "detection"
    runs_root = detection_runs(repo)
    sweep_dir = runs_root / "obb" / "obb-v6"
    return SweepConfig(
        name="obb-v6",
        kind="detector",
        repo=repo,
        dataset_dir=dataset_dir,
        data_yaml=dataset_dir / "data.yaml",
        weights_cache=pretrained_dir(repo),
        runs_root=runs_root,
        project_dir=runs_root / "obb",
        sweep_dir=sweep_dir,
        ranking_csv=sweep_dir / "ranking.csv",
        docs_dir=repo / "train" / "detector",
        candidates=[
            {
                "id": "yolo26",
                "family": "YOLOv26",
                "model": "yolo26s-obb.pt",
                "task": "obb",
                "imgsz": 960,
                "batch": 4,
            },
            {
                "id": "yolo12",
                "family": "YOLOv12",
                "model": "yolo12n-obb.yaml",
                "init_from": "yolo12n.pt",
                "task": "obb",
                "imgsz": 960,
                "batch": 4,
            },
            {
                "id": "yolov8",
                "family": "YOLOv8",
                "model": "yolov8s-obb.pt",
                "task": "obb",
                "imgsz": 960,
                "batch": 4,
            },
            {
                "id": "rtdetrv2",
                "family": "RT-DETR-L",
                "model": "rtdetr-l.pt",
                "task": "detect",
                "imgsz": 640,
                "batch": 4,
            },
            {
                "id": "yolo11",
                "family": "YOLOv11",
                "model": "yolo11s-obb.pt",
                "task": "obb",
                "imgsz": 960,
                "batch": 4,
            },
        ],
        epochs=300,
        patience=40,
        skip_existing=skip_existing,
        map_tol=0.002,
        winner_weight_name="obb-v6.pt",
        aabb_dir=dataset_dir / "aabb",
        aabb_yaml=dataset_dir / "aabb" / "data.yaml",
        class_names={0: "card"},
        include_confusion=False,
        optim=dict(optimizer="AdamW", lr0=0.001, lrf=0.01),
        loss=dict(box=12.0, dfl=2.5, angle=2.0),
        augment=dict(
            degrees=30.0,
            flipud=0.5,
            fliplr=0.5,
            translate=0.10,
            scale=0.40,
            shear=2.0,
            perspective=0.0,
            hsv_h=0.02,
            hsv_s=0.70,
            hsv_v=0.50,
            bgr=0.0,
            mosaic=0.0,
            mixup=0.0,
            copy_paste=0.0,
            erasing=0.0,
            close_mosaic=0,
            cos_lr=True,
        ),
        notes=[
            "Receita de borda do obb-v5: mosaic/mixup/erasing desligados (o warp 63×88 mm precisa do retângulo da carta).",
            "RT-DETR-L não tem OBB oficial no Ultralytics: treina AABB convertido das mesmas anotações e não entra no cropper.",
            "YOLOv12 OBB usa o yaml + backbone `yolo12n.pt` (não há checkpoint OBB oficial).",
        ],
    )


def _load_ranking(cfg: SweepConfig) -> pd.DataFrame:
    if cfg.ranking_csv.is_file():
        return pd.read_csv(cfg.ranking_csv)
    docs_csv = cfg.docs_dir / "ranking.csv"
    if docs_csv.is_file():
        return pd.read_csv(docs_csv)
    raise FileNotFoundError(
        f"Sem ranking.csv em {cfg.ranking_csv} nem em {docs_csv}. Rode o treino primeiro."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Treina o sweep obb-v6 e grava o relatório em train/detector/."
    )
    parser.add_argument("--report-only", action="store_true", help="Não treina; só elege e gera o relatório.")
    parser.add_argument("--pdf", action="store_true", help="Também gera report.pdf comprimido (não versionado).")
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Retreina mesmo se já existir best.pt.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Não roda val(plots=True) de cada família.")
    args = parser.parse_args(argv)

    repo = find_repo()
    chdir_repo(repo)
    cfg = build_config(repo, skip_existing=not args.no_skip_existing)
    configure_ultralytics_dirs(cfg.runs_root, cfg.weights_cache)

    env_df, device = inspect_environment(cfg.runs_root, cfg.weights_cache)
    print(env_df.to_string(index=False))
    write_environment(cfg.sweep_dir / "environment.json", env_df)
    if device == "cpu":
        print("\n[AVISO] CUDA não detectada. CPU vai ser lenta.")

    if not args.report_only:
        ds = validate_dataset(cfg.dataset_dir, cfg.data_yaml)
        cfg.class_names = ds["names"]
        print_dataset_report(ds, cfg.data_yaml)
        ranking = run_sweep(cfg, device)
    else:
        ranking = _load_ranking(cfg)

    ranking, winner = elect_winner(cfg, ranking)

    if not args.no_plots:
        generate_all_plots(cfg, device)

    write_yolo_readme(cfg, ranking, winner)
    if args.pdf:
        write_yolo_pdf(cfg, ranking, winner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
