"""Run the OBB detector and save each rectified card (63 mm × 88 mm)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

try:
    from .rectify import DEFAULT_DPI, card_size_px, crop_result
except ImportError:
    from rectify import DEFAULT_DPI, card_size_px, crop_result

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent


def _in_scanner_repo() -> bool:
    return (REPO_ROOT / "pipeline.py").is_file() and (REPO_ROOT / "train").is_dir()


def _detection_obb_runs() -> Path:
    new = REPO_ROOT / "runs" / "detection" / "obb"
    old = REPO_ROOT / "src" / "detection" / "runs" / "obb"
    return new if new.is_dir() else old


DEFAULT_OUTPUT = REPO_ROOT / "output" / "cropper" if _in_scanner_repo() else PACKAGE_DIR / "output"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def imgsz_for_weights(weights: Path) -> int:
    name = weights.name.lower()
    if "v3" in name or "v2" in name or "v1" in name:
        return 800
    return 960


def find_default_weights() -> Path:
    """Weights: env, then models/, then v6…v3 bundled/runs copies."""
    env = os.environ.get("PTCG_CROPPER_WEIGHTS") or os.environ.get("TCG_CROPPER_WEIGHTS")
    if env:
        return Path(env)
    export_dir = REPO_ROOT / "models"
    bundled = PACKAGE_DIR / "weights"
    runs = _detection_obb_runs()
    for name in ("obb-v6.pt", "obb-v5.pt", "obb-v4.pt", "obb-v3.pt"):
        for candidate in (
            export_dir / name,
            bundled / name,
            runs / name.replace(".pt", "") / "weights" / name,
        ):
            if candidate.is_file():
                return candidate
    if bundled.is_dir():
        pts = sorted(bundled.glob("*.pt"))
        if pts:
            return pts[0]
    return export_dir / "obb-v6.pt"


DEFAULT_WEIGHTS = find_default_weights()
DEFAULT_IMGSZ = imgsz_for_weights(DEFAULT_WEIGHTS)


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = [p for p in sorted(source.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"No images in {source}")
    return files


def write_preview(paths: list[Path], dest: Path, cols: int = 5, cell=(186, 260)) -> Path | None:
    if not paths:
        return None
    tiles = []
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        tiles.append(cv2.resize(img, cell))
    if not tiles:
        return None
    cols = max(1, min(cols, len(tiles)))
    rows = (len(tiles) + cols - 1) // cols
    pad = rows * cols - len(tiles)
    if pad:
        tiles.extend([np.zeros((cell[1], cell[0], 3), dtype=tiles[0].dtype)] * pad)
    grid_rows = []
    for r in range(rows):
        chunk = tiles[r * cols : (r + 1) * cols]
        grid_rows.append(cv2.hconcat(chunk))
    dest.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dest), cv2.vconcat(grid_rows))
    print(f"Preview: {dest}")
    return dest


def save_crops(stem: str, crops, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for card in crops:
        base = f"{stem}_card_{card.index:02d}_{card.conf:.2f}"
        color_path = out_dir / f"{base}.jpg"
        cv2.imwrite(str(color_path), card.image)
        saved.append(color_path)
    return saved


def run(
    source: Path,
    weights: Path = DEFAULT_WEIGHTS,
    out_dir: Path = DEFAULT_OUTPUT,
    conf: float = 0.8,
    imgsz: int = DEFAULT_IMGSZ,
    dpi: int = DEFAULT_DPI,
    device: str | int = 0,
    refine: bool = False,
    inset: float = 0.02,
    enhance: bool = True,
    frame: bool = False,
    model: YOLO | None = None,
) -> list[Path]:
    if model is None:
        if not weights.exists():
            raise FileNotFoundError(
                f"OBB weights not found: {weights}\n"
                "Pass --weights, set PTCG_CROPPER_WEIGHTS, "
                "export the cropper (python -m cropper export ...) "
                "or train with `python -m train detector`."
            )
        model = YOLO(str(weights))
        print(f"OBB weights: {weights}  imgsz={imgsz}")
    saved: list[Path] = []
    for image_path in collect_images(source):
        results = model.predict(
            source=str(image_path),
            conf=conf,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )
        crops = crop_result(
            results[0],
            dpi=dpi,
            refine=refine,
            inset=inset,
            enhance=enhance,
            frame=frame,
        )
        paths = save_crops(image_path.stem, crops, out_dir)
        saved.extend(paths)
        print(f"{image_path.name}: {len(crops)} card(s) -> {out_dir}")
    return saved


def parse_args() -> argparse.Namespace:
    width, height = card_size_px(DEFAULT_DPI)
    parser = argparse.ArgumentParser(
        description=(
            "Detect cards (YOLOv8 OBB) and warp each one to "
            f"{width}x{height} px (63 mm x 88 mm @ {DEFAULT_DPI} DPI)."
        )
    )
    parser.add_argument("source", type=Path, help="Image file or folder")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="YOLO OBB checkpoint (default: obb-v6, then v5, then v4, then v3).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--conf", type=float, default=0.8)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Legacy edge snap (do not use for OCR).",
    )
    parser.add_argument(
        "--inset",
        type=float,
        default=0.02,
        help="Shrink the OBB toward the center (0.02 = 2 percent).",
    )
    parser.add_argument("--no-enhance", action="store_true", help="Disable lighting equalization.")
    parser.add_argument(
        "--frame",
        action="store_true",
        help="Shrink the card and paint a light border (off by default).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Also write a contact-sheet JPEG in --out (replaces the old crop_cards notebook).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device: str | int = args.device
    if str(device).isdigit():
        device = int(device)
    saved = run(
        source=args.source,
        weights=args.weights,
        out_dir=args.out,
        conf=args.conf,
        imgsz=args.imgsz,
        dpi=args.dpi,
        device=device,
        refine=args.refine,
        inset=args.inset,
        enhance=not args.no_enhance,
        frame=args.frame,
    )
    if args.preview:
        write_preview(saved, args.out / "preview.jpg")
    return 0


if __name__ == "__main__":
    sys.exit(main())
