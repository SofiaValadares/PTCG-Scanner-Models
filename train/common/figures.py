from __future__ import annotations

import io
from pathlib import Path

from PIL import Image

JPEG_QUALITY = 50
PNG_MAX_SIDE = 900
JPEG_MAX_SIDE = 900

PLOT_NAMES = [
    "results.png",
    "BoxPR_curve.png",
    "PR_curve.png",
    "BoxF1_curve.png",
    "F1_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
]
PLOT_ALIASES = {
    "BoxPR_curve.png": ["PR_curve.png"],
    "BoxF1_curve.png": ["F1_curve.png"],
}


def _resize(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)


def compress_image(src: Path, dest: Path, *, jpeg: bool | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)
    if img.mode in {"RGBA", "P"}:
        img = img.convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")
    use_jpeg = jpeg if jpeg is not None else src.suffix.lower() in {".jpg", ".jpeg", ".webp"}
    if use_jpeg:
        img = _resize(img, JPEG_MAX_SIDE)
        dest = dest.with_suffix(".jpg")
        img.save(dest, "JPEG", quality=JPEG_QUALITY, optimize=True)
    else:
        img = _resize(img, PNG_MAX_SIDE)
        dest = dest.with_suffix(".png")
        buf = io.BytesIO()
        img.save(buf, "PNG", optimize=True)
        dest.write_bytes(buf.getvalue())
    return dest


def find_plot(folders: list[Path], filename: str) -> Path | None:
    names = [filename, *PLOT_ALIASES.get(filename, [])]
    for folder in folders:
        if not folder.is_dir():
            continue
        for name in names:
            direct = folder / name
            if direct.is_file():
                return direct
        for name in names:
            matches = sorted(folder.rglob(name))
            if matches:
                return matches[0]
    return None


def collect_run_images(cand_dir: Path, *, include_confusion: bool = True) -> list[tuple[str, Path]]:
    report_dir = cand_dir / "report"
    folders = [report_dir, cand_dir]
    wanted = [
        ("Curvas de treino", "results.png"),
        ("Precision–Recall", "BoxPR_curve.png"),
        ("F1 vs confiança", "BoxF1_curve.png"),
        ("Precision vs confiança", "BoxP_curve.png"),
        ("Recall vs confiança", "BoxR_curve.png"),
    ]
    if include_confusion:
        wanted.extend(
            [
                ("Matriz de confusão", "confusion_matrix.png"),
                ("Matriz de confusão (normalizada)", "confusion_matrix_normalized.png"),
            ]
        )
    found: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for title, name in wanted:
        path = find_plot(folders, name)
        if path is None or path.name in seen:
            continue
        seen.add(path.name)
        found.append((title, path))
    val_images: list[Path] = []
    for folder in folders:
        if not folder.is_dir():
            continue
        val_images.extend(sorted(folder.glob("val_batch*_labels.jpg")))
        val_images.extend(sorted(folder.glob("val_batch*_pred.jpg")))
    if not val_images:
        for folder in folders:
            if folder.is_dir():
                val_images.extend(sorted(folder.rglob("val_batch*.jpg")))
    for path in val_images:
        if path.name in seen:
            continue
        seen.add(path.name)
        label = "Lote de validação (labels)" if "labels" in path.name else "Lote de validação (predições)"
        found.append((f"{label} — {path.name}", path))
    return found


def publish_images(
    items: list[tuple[str, Path]],
    dest_dir: Path,
    *,
    jpeg_photos: bool = True,
) -> list[tuple[str, Path]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    published: list[tuple[str, Path]] = []
    for title, src in items:
        as_jpeg = jpeg_photos and src.suffix.lower() in {".jpg", ".jpeg", ".webp"}
        dest = dest_dir / src.name
        out = compress_image(src, dest, jpeg=as_jpeg)
        published.append((title, out))
    return published
