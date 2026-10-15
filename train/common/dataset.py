from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path} não contém um mapeamento YAML válido.")
    return data


def resolve_class_names(cfg: dict) -> dict[int, str]:
    names = cfg.get("names")
    if names is None:
        raise ValueError("Campo obrigatório ausente em data.yaml: 'names'.")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    raise ValueError(f"Formato de 'names' não suportado: {type(names)}")


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def parse_obb_line(line: str, n_classes: int) -> tuple[bool, str]:
    parts = line.strip().split()
    if not parts:
        return True, "linha vazia ignorada"
    if len(parts) != 9:
        return False, f"esperado 9 valores (class + 8 coords), obtido {len(parts)}"
    try:
        cls = int(float(parts[0]))
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return False, "valores não numéricos"
    if cls < 0 or cls >= n_classes:
        return False, f"classe {cls} fora de [0, {n_classes - 1}]"
    if any(c < -0.05 or c > 1.05 for c in coords):
        return False, f"coordenadas fora de [0, 1]: {coords}"
    return True, "ok"


def validate_split(split_dir: Path, n_classes: int) -> dict:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    images = list_images(images_dir)
    labels = sorted(labels_dir.glob("*.txt")) if labels_dir.is_dir() else []
    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}
    n_ok, n_bad, n_objs = 0, 0, 0
    class_hist: Counter = Counter()
    errors: list[str] = []
    for lbl in labels:
        text = lbl.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            ok, msg = parse_obb_line(line, n_classes)
            if ok and msg == "ok":
                n_ok += 1
                n_objs += 1
                class_hist[int(float(line.split()[0]))] += 1
            elif not ok:
                n_bad += 1
                if len(errors) < 8:
                    errors.append(f"{lbl.name}:{i} -> {msg}")
    return {
        "images_dir": images_dir,
        "labels_dir": labels_dir,
        "n_images": len(images),
        "n_labels": len(labels),
        "missing_labels": sorted(img_stems - lbl_stems)[:8],
        "orphan_labels": sorted(lbl_stems - img_stems)[:8],
        "obb_ok": n_ok,
        "obb_bad": n_bad,
        "objects": n_objs,
        "class_hist": class_hist,
        "errors": errors,
    }


def validate_dataset(dataset_dir: Path, yaml_path: Path) -> dict:
    report: dict = {"ok": True, "issues": []}
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_dir}")
    if not yaml_path.is_file():
        raise FileNotFoundError(f"data.yaml não encontrado: {yaml_path}")
    cfg = load_yaml(yaml_path)
    names = resolve_class_names(cfg)
    n_classes = len(names)
    for key in ("train", "val"):
        if key not in cfg:
            report["ok"] = False
            report["issues"].append(f"Campo obrigatório ausente: '{key}'")
    yaml_root = Path(cfg["path"]).resolve() if cfg.get("path") else yaml_path.parent.resolve()

    def split_root(key: str, default_images: str) -> Path:
        rel = Path(str(cfg.get(key, default_images)))
        images_dir = rel if rel.is_absolute() else (yaml_root / rel)
        return images_dir.parent if images_dir.name == "images" else images_dir

    split_map = {
        "train": split_root("train", "train/images"),
        "valid": split_root("val", "valid/images"),
        "test": split_root("test", "test/images"),
    }
    split_reports = {name: validate_split(folder, n_classes) for name, folder in split_map.items()}
    report.update(
        {
            "cfg": cfg,
            "names": names,
            "n_classes": n_classes,
            "yaml_root": yaml_root,
            "has_test": "test" in cfg,
            "splits": split_reports,
        }
    )
    for name, info in split_reports.items():
        if name in ("train", "valid") and info["n_images"] == 0:
            report["ok"] = False
            report["issues"].append(f"Split '{name}' sem imagens em {info['images_dir']}")
        if info["obb_bad"] > 0:
            report["issues"].append(f"Split '{name}' contém {info['obb_bad']} label(s) OBB inválido(s).")
            report["issues"].extend(info["errors"])
            if info["obb_bad"] > max(8, int(0.01 * max(info["objects"], 1))):
                report["ok"] = False
        if info["missing_labels"]:
            report["issues"].append(f"Split '{name}' com imagens sem label: {info['missing_labels']}")
    return report


def print_dataset_report(ds: dict, yaml_path: Path) -> None:
    print("=" * 72)
    print("data.yaml")
    print("=" * 72)
    print(yaml_path.read_text(encoding="utf-8"))
    print(f"Raiz efetiva : {ds['yaml_root']}")
    print(f"Classes      : {ds['n_classes']} -> {ds['names']}")
    for split, info in ds["splits"].items():
        print(
            f"  {split:6s}  imgs={info['n_images']:4d}  labels={info['n_labels']:4d}  "
            f"objs={info['objects']:5d}  OBB inválidos={info['obb_bad']}"
        )
    if ds["issues"]:
        print("\nObservações:")
        for issue in ds["issues"]:
            print(f"  - {issue}")
    if not ds["ok"]:
        raise RuntimeError("Validação do dataset falhou. Corrija os problemas acima antes de treinar.")
    print("Dataset válido.")
