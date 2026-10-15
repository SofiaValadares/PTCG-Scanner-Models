"""Copy the cropper package plus .pt weights to another folder."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent
CODE_FILES = (
    "crop_from_obb.py",
    "rectify.py",
    "enhance.py",
    "__init__.py",
    "__main__.py",
    "export.py",
    "pyproject.toml",
    "README.md",
)
REQUIREMENTS = "ultralytics\nopencv-python\nnumpy\n"


def find_weights_to_copy(explicit: Path | None = None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    env = os.environ.get("PTCG_CROPPER_WEIGHTS") or os.environ.get("TCG_CROPPER_WEIGHTS")
    if env:
        return Path(env).expanduser().resolve()
    export_dir = REPO_ROOT / "models"
    bundled = PACKAGE_DIR / "weights"
    new_runs = REPO_ROOT / "runs" / "detection" / "obb"
    old_runs = REPO_ROOT / "src" / "detection" / "runs" / "obb"
    runs = new_runs if new_runs.is_dir() else old_runs
    for name, rel in (
        ("obb-v6.pt", "obb-v6/weights/obb-v6.pt"),
        ("obb-v5.pt", "obb-v5/weights/obb-v5.pt"),
        ("obb-v4.pt", "obb-v4/weights/obb-v4.pt"),
        ("obb-v3.pt", "obb-v3/weights/obb-v3.pt"),
    ):
        for candidate in (export_dir / name, bundled / name, runs / rel):
            if candidate.is_file():
                return candidate
    if bundled.is_dir():
        pts = sorted(bundled.glob("*.pt"))
        if pts:
            return pts[0]
    return export_dir / "obb-v6.pt"


def export_cropper(dest: Path, weights: Path | None = None) -> Path:
    dest = dest.expanduser().resolve()
    src_root = PACKAGE_DIR.resolve()
    if dest == src_root:
        raise ValueError("Destination cannot be this repository's cropper folder.")
    src_weights = find_weights_to_copy(weights)
    if not src_weights.is_file():
        raise FileNotFoundError(
            f"Weights not found: {src_weights}\n"
            "Missing models/obb-v6.pt. Train the detector or pass --weights."
        )

    dest.mkdir(parents=True, exist_ok=True)
    for name in CODE_FILES:
        src = src_root / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing cropper file: {src}")
        shutil.copy2(src, dest / name)

    (dest / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    for extra in ("LICENSE", "NOTICE"):
        src_extra = REPO_ROOT / extra
        if src_extra.is_file():
            shutil.copy2(src_extra, dest / extra)
    weights_dir = dest / "weights"
    weights_dir.mkdir(exist_ok=True)
    shutil.copy2(src_weights, weights_dir / src_weights.name)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the cropper (code + weights) for use outside this repo.",
    )
    parser.add_argument(
        "dest",
        type=Path,
        help="Destination folder (created if missing).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=".pt weights (default: the same file this cropper uses here).",
    )
    args = parser.parse_args()
    dest = export_cropper(args.dest, weights=args.weights)
    weights_path = next((dest / "weights").glob("*.pt"))
    script = dest / "crop_from_obb.py"
    print(f"Exported: {dest}")
    print(f"Weights : {weights_path}")
    print()
    print("In the other project:")
    print(f"  python {script} photo.jpg --out cards")
    print("or:")
    print(f"  python -m pip install -e {dest}")
    print("  ptcg-crop photo.jpg --out cards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
