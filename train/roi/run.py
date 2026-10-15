from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from train.common.dataset import print_dataset_report, validate_dataset
from train.common.env import inspect_environment, write_environment
from train.common.paths import (
    chdir_repo,
    configure_ultralytics_dirs,
    find_repo,
    pretrained_dir,
    roi_runs,
)
from train.common.report import write_yolo_pdf, write_yolo_readme
from train.common.yolo import SweepConfig, elect_winner, generate_all_plots, run_sweep


def build_config(repo: Path, *, skip_existing: bool) -> SweepConfig:
    dataset_dir = repo / "data" / "roi"
    runs_root = roi_runs(repo)
    sweep_dir = runs_root / "ocr-roi-v3"
    imgsz, batch = 800, 8
    return SweepConfig(
        name="ocr-roi-v3",
        kind="roi",
        repo=repo,
        dataset_dir=dataset_dir,
        data_yaml=dataset_dir / "data.yaml",
        weights_cache=pretrained_dir(repo),
        runs_root=runs_root,
        project_dir=runs_root,
        sweep_dir=sweep_dir,
        ranking_csv=sweep_dir / "ranking.csv",
        docs_dir=repo / "train" / "roi",
        candidates=[
            {"id": "yolo26", "family": "YOLOv26", "model": "yolo26s-obb.pt", "imgsz": imgsz, "batch": batch},
            {
                "id": "yolo12",
                "family": "YOLOv12",
                "model": "yolo12n-obb.yaml",
                "init_from": "yolo12n.pt",
                "imgsz": imgsz,
                "batch": batch,
            },
            {"id": "yolov8", "family": "YOLOv8", "model": "yolov8s-obb.pt", "imgsz": imgsz, "batch": batch},
            {"id": "yolo11", "family": "YOLOv11", "model": "yolo11s-obb.pt", "imgsz": imgsz, "batch": batch},
        ],
        epochs=300,
        patience=40,
        skip_existing=skip_existing,
        map_tol=0.002,
        winner_weight_name="ocr-roi-v3.pt",
        include_confusion=True,
        optim=dict(optimizer="AdamW", lr0=0.001, lrf=0.01),
        loss=dict(box=10.0, dfl=2.0, angle=1.5, cls=0.75),
        augment=dict(
            degrees=10.0,
            flipud=0.0,
            fliplr=0.0,
            translate=0.08,
            scale=0.25,
            shear=1.0,
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
            "Receita do ocr-roi-v2: HSV como o detector de carta, geometria fraca, sem flip/mosaic (a imagem já é a carta 63×88 mm).",
            "RT-DETR fica de fora: não há OBB oficial e as faixas são polígonos rotacionados.",
            "Não renomeie `colection` no YAML — é o id do Roboflow e o pipeline usa esse nome.",
            "`colection` falta em muitas cartas (set não impresso). AP mais baixo nessa classe não é falha de OCR.",
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
        description="Treina o sweep ocr-roi-v3 e grava o relatório em train/roi/."
    )
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
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
        if cfg.data_yaml.is_file():
            try:
                ds = validate_dataset(cfg.dataset_dir, cfg.data_yaml)
                cfg.class_names = ds["names"]
            except Exception:
                pass

    ranking, winner = elect_winner(cfg, ranking)
    if not args.no_plots:
        generate_all_plots(cfg, device)
    write_yolo_readme(cfg, ranking, winner)
    if args.pdf:
        write_yolo_pdf(cfg, ranking, winner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
