from __future__ import annotations

import argparse
import re
import shutil
import time
import unicodedata
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import yaml

from train.common.env import inspect_environment, write_environment
from train.common.figures import compress_image
from train.common.paths import chdir_repo, find_repo, ocr_runs, pretrained_dir
from train.common.progress import CountProgress
from train.common.ranking import markdown_table
from train.common.report import write_ocr_pdf, write_ocr_readme

ENGINES = ("easyocr", "paddleocr", "tesseract")
LABEL = {"easyocr": "EasyOCR", "paddleocr": "RapidOCR (Paddle)", "tesseract": "Tesseract"}
_NUM_RE = re.compile(r"^(?:[A-Z]{1,8})?\d{1,4}[-/](?:[A-Z]{1,8})?\d{1,4}$", re.I)
_ILLEGIBLE_RE = re.compile(r"^[-_.\s]*(?:v|ex|gx|vstar|vmax)?[-_.\s]*$", re.I)


def load_class_names(path: Path) -> dict[int, str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data["names"]
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {i: str(name) for i, name in enumerate(names)}


def infer_tipo(gt: str) -> str:
    s = (gt or "").strip()
    if not s:
        return "vazio"
    compact = re.sub(r"[^A-Za-z0-9]", "", s)
    if _ILLEGIBLE_RE.fullmatch(s):
        return "ilegivel"
    if not re.search(r"\d", s) and s.count("-") >= 2 and len(compact) <= 6:
        return "ilegivel"
    token = re.sub(r"\s+", "", s)
    if _NUM_RE.fullmatch(token):
        return "number"
    if re.fullmatch(r"\d{2,4}", compact) and "-" not in s and "/" not in s:
        return "collection"
    if re.fullmatch(r"S\d{1,2}[A-Za-z]{0,2}", compact, re.I):
        return "collection"
    if " " not in s and 2 <= len(compact) <= 7 and s.isupper() and re.search(r"[A-Za-z]", compact):
        return "collection"
    if " " not in s and 2 <= len(compact) <= 5 and re.fullmatch(r"[A-Z]{2,5}\d{0,2}[A-Za-z]{0,2}", compact):
        return "collection"
    return "name"


def edit_distance(left, right) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)
    prev = list(range(len(right) + 1))
    for i, ca in enumerate(left, start=1):
        curr = [i]
        for j, cb in enumerate(right, start=1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def cer(gt: str, pred: str) -> float:
    if not gt and not pred:
        return 0.0
    return edit_distance(gt, pred) / max(len(gt), 1)


def fold_name(text: str, fold_text) -> str:
    return fold_text(text)


def fold_collection(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "", text or "").upper()[:3]


def fold_number(text: str, number_translate) -> str:
    raw = (text or "").upper().replace(" ", "").replace("\\", "/").replace("⁄", "/").replace("-", "/")
    prefixed = re.search(r"([A-Z]{2,})(\d{1,4})/([A-Z]{2,})(\d{1,4})", raw)
    if prefixed:
        left, a, right, b = prefixed.groups()
        return f"{left}{int(a)}/{right}{int(b)}"
    translated = raw.translate(number_translate)
    plain = re.search(r"(\d{1,4})/(\d{1,4})", translated)
    if plain:
        return f"{int(plain.group(1))}/{int(plain.group(2))}"
    if "/" not in translated:
        digits = re.sub(r"\D", "", translated)
        letters = re.sub(r"[^A-Z]", "", translated)
        if digits and not letters:
            return str(int(digits))
    return re.sub(r"[^A-Z0-9/]", "", translated)


def same_collection(expected: str, predicted: str) -> bool:
    left, right = fold_collection(expected), fold_collection(predicted)
    if not left and not right:
        return True
    if not left or not right:
        return False
    return edit_distance(left, right) <= 1


def word_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return [tok.casefold() for tok in re.findall(r"[0-9A-Za-z]+", text)]


def wer(gt: str, pred: str) -> float:
    left, right = word_tokens(gt), word_tokens(pred)
    if not left and not right:
        return 0.0
    return edit_distance(left, right) / max(len(left), 1)


def load_engines():
    import torch
    from ocr.read_card import (
        _FIELD_ALLOW,
        _NUMBER_TRANSLATE,
        _clean,
        _parse_ocr_items,
        _readtext,
        _strip_variants,
        enhance_strip,
    )

    loaded = {}
    reader = paddle = tesseract_cmd = None
    try:
        import easyocr

        reader = easyocr.Reader(["en", "pt"], gpu=torch.cuda.is_available())
        loaded["easyocr"] = "ok"
        print("EasyOCR ok")
    except Exception as exc:
        loaded["easyocr"] = f"skip: {exc}"
        print(loaded["easyocr"])
    try:
        from rapidocr import RapidOCR

        paddle = RapidOCR(params={"Global.log_level": "error"})
        loaded["paddleocr"] = "ok"
        print("RapidOCR ok")
    except Exception as exc:
        loaded["paddleocr"] = f"skip: {exc}"
        print(loaded["paddleocr"])
    tesseract_cmd = shutil.which("tesseract")
    if not tesseract_cmd:
        for candidate in (
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ):
            if candidate.is_file():
                tesseract_cmd = str(candidate)
                break
    if tesseract_cmd:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = str(tesseract_cmd)
        loaded["tesseract"] = f"ok ({tesseract_cmd})"
        print("Tesseract ok")
    else:
        loaded["tesseract"] = "skip: binário tesseract não encontrado"
        print(loaded["tesseract"])
    active = [k for k in ENGINES if str(loaded.get(k, "")).startswith("ok")]
    if not active:
        raise RuntimeError("Nenhum motor de OCR carregou.")
    return loaded, active, reader, paddle, tesseract_cmd, {
        "allow": _FIELD_ALLOW,
        "translate": _NUMBER_TRANSLATE,
        "clean": _clean,
        "parse": _parse_ocr_items,
        "readtext": _readtext,
        "variants": _strip_variants,
        "enhance": enhance_strip,
    }


def _readers(reader, paddle, helpers):
    enhance = helpers["enhance"]
    allow = helpers["allow"]

    def easy_raw(image: np.ndarray, field: str) -> str:
        min_h = 80 if field == "collection" else 112
        enhanced = enhance(image, min_h=min_h)

        def _run(view: np.ndarray) -> tuple[str, float]:
            rgb = cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)
            return helpers["parse"](helpers["readtext"](reader, rgb, allow[field]))

        text, _conf = _run(enhanced)
        if text.strip():
            return text
        if field == "name":
            text, _conf = _run(255 - enhanced)
            return text
        if field == "number":
            best_text, best_conf = "", -1.0
            for view in helpers["variants"](enhanced)[1:]:
                cand, cand_conf = _run(view)
                if cand.strip() and cand_conf > best_conf:
                    best_text, best_conf = cand, cand_conf
            return best_text
        return text

    def paddle_raw(image: np.ndarray, field: str) -> str:
        min_h = 80 if field == "collection" else 112
        gray = enhance(image, min_h=min_h)
        view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
        try:
            result = paddle(view, use_det=False, use_cls=False)
        except TypeError:
            result = paddle(view)
        texts = [str(item).strip() for item in (getattr(result, "txts", None) or []) if str(item).strip()]
        if texts:
            return " ".join(texts)
        if isinstance(result, (list, tuple)):
            parts = []
            blob = result[0] if result and isinstance(result[0], list) else result
            for item in blob or []:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    payload = item[1]
                    if isinstance(payload, (list, tuple)):
                        parts.append(str(payload[0]).strip())
                    else:
                        parts.append(str(payload).strip())
            return " ".join(p for p in parts if p)
        return ""

    def tesseract_raw(image: np.ndarray, field: str) -> str:
        import pytesseract

        view = enhance(image, min_h=80 if field == "collection" else 112)
        config = "--psm 7"
        if field == "number":
            config += " -c tessedit_char_whitelist=0123456789/OoQqDdIiLlZzSsGgBb"
        elif field == "collection":
            config += " -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        data = pytesseract.image_to_data(view, lang="eng", config=config, output_type=pytesseract.Output.DICT)
        words = [str(word).strip() for word in data.get("text", []) if str(word).strip()]
        return " ".join(words)

    return {"easyocr": easy_raw, "paddleocr": paddle_raw, "tesseract": tesseract_raw}


def iter_boxes(dataset_dir: Path, split: str, class_names: dict[int, str]):
    img_dir = dataset_dir / split / "images"
    lab_dir = dataset_dir / split / "labels"
    if not img_dir.is_dir() or not lab_dir.is_dir():
        return
    images = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}}
    for lab_path in sorted(lab_dir.glob("*.txt")):
        img_path = images.get(lab_path.stem)
        if img_path is None:
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        for box_i, line in enumerate(lab_path.read_text(encoding="utf-8").splitlines()):
            parts = line.split()
            if len(parts) < 9:
                continue
            cls_id = int(float(parts[0]))
            gt = class_names.get(cls_id)
            if gt is None:
                continue
            pts = np.array(list(map(float, parts[1:9])), dtype=np.float32).reshape(4, 2)
            pts[:, 0] *= w
            pts[:, 1] *= h
            yield {
                "image": img_path.name,
                "path": img_path,
                "box_i": box_i,
                "gt": gt,
                "tipo": infer_tipo(gt),
                "quad": pts,
                "bgr": bgr,
            }


def pick_ocr_winner(summary_df: pd.DataFrame) -> str:
    work = summary_df.sort_values(["exact_clean", "CER_clean", "ms/caixa"], ascending=[False, True, True])
    winner = str(work.iloc[0]["motor"])
    if len(work) > 1:
        gap = float(work.iloc[0]["exact_clean"] - work.iloc[1]["exact_clean"])
        if gap < 1.0:
            tied = work[work["exact_clean"] >= work.iloc[0]["exact_clean"] - 1.0]
            winner = str(tied.sort_values(["CER_clean", "ms/caixa"]).iloc[0]["motor"])
    return winner


def _save_charts(docs_dir: Path, summary: pd.DataFrame, by_tipo: pd.DataFrame, samples_path: Path | None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = docs_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    published: list[tuple[str, Path]] = []

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(summary["motor"], summary["exact_clean"], color="#3b6ea5")
    ax.set_ylabel("exact_clean (%)")
    ax.set_title("Acerto normalizado por motor")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    raw = fig_dir / "exact_clean.png"
    fig.savefig(raw, dpi=110)
    plt.close(fig)
    published.append(("Acerto exact_clean", compress_image(raw, raw, jpeg=False)))

    if not by_tipo.empty:
        fig, ax = plt.subplots(figsize=(8.0, 3.8))
        motors = [c for c in by_tipo.columns if c != "tipo" and c != "n"]
        x = np.arange(len(by_tipo))
        width = 0.8 / max(len(motors), 1)
        for i, motor in enumerate(motors):
            ax.bar(x + i * width, by_tipo[motor], width=width, label=motor)
        ax.set_xticks(x + width * (len(motors) - 1) / 2)
        ax.set_xticklabels(by_tipo["tipo"])
        ax.set_ylabel("exact_clean (%)")
        ax.set_title("Acerto por tipo de faixa")
        ax.legend()
        fig.tight_layout()
        raw = fig_dir / "exact_by_tipo.png"
        fig.savefig(raw, dpi=110)
        plt.close(fig)
        published.append(("Acerto por tipo", compress_image(raw, raw, jpeg=False)))

    if samples_path is not None and samples_path.is_file():
        published.append(("Amostras", compress_image(samples_path, fig_dir / "samples.jpg", jpeg=True)))
    return published


def _draw_samples(dataset_dir, split, readable, active, max_show, helpers) -> Path | None:
    from ocr.read_card import _WARP_PAD, warp_quad
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)
    show = readable
    if len(show) > max_show:
        idx = rng.choice(show.index.to_numpy(), size=max_show, replace=False)
        show = show.loc[sorted(idx)]
    images = {p.name: p for p in (dataset_dir / split / "images").iterdir()}
    fig, axes = plt.subplots(len(show), 1, figsize=(10, 2.2 * max(len(show), 1)))
    if len(show) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, show.iterrows()):
        img_path = images.get(row["imagem"])
        ax.axis("off")
        bits = [f"GT: {row['gt']}  [{row['tipo']}]"]
        crop = None
        if img_path is not None:
            bgr = cv2.imread(str(img_path))
            lab = dataset_dir / split / "labels" / f"{img_path.stem}.txt"
            if bgr is not None and lab.is_file():
                lines = [ln for ln in lab.read_text(encoding="utf-8").splitlines() if ln.strip()]
                if row["caixa"] < len(lines):
                    parts = lines[int(row["caixa"])].split()
                    h, w = bgr.shape[:2]
                    pts = np.array(list(map(float, parts[1:9])), dtype=np.float32).reshape(4, 2)
                    pts[:, 0] *= w
                    pts[:, 1] *= h
                    field = row["tipo"] if row["tipo"] in {"name", "number", "collection"} else "name"
                    crop = warp_quad(bgr, pts, pad=_WARP_PAD.get(field, 0.12))
        if crop is not None and crop.size:
            ax.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        for engine in active:
            mark = "OK" if row[f"ok_clean_{engine}"] else "ERR"
            bits.append(f"{LABEL[engine]} {mark} {row[f'clean_{engine}']}")
        ax.set_title("  |  ".join(bits), fontsize=9, loc="left")
    fig.tight_layout()
    dest = Path("_ocr_samples_tmp.png")
    fig.savefig(dest, dpi=100)
    plt.close(fig)
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compara EasyOCR / RapidOCR / Tesseract e grava o relatório.")
    parser.add_argument("--report-only", action="store_true", help="Só regenera README a partir do CSV já gravado.")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--max-boxes", type=int, default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args(argv)

    repo = find_repo()
    chdir_repo(repo)
    from ocr.read_card import (
        _NUMBER_TRANSLATE,
        _WARP_PAD,
        _clean,
        fold_text,
        warp_quad,
    )

    experiment = "ocr-cmp-v1"
    dataset_dir = repo / "data" / "ocr"
    data_yaml = dataset_dir / "data.yaml"
    runs = ocr_runs(repo)
    docs_dir = repo / "train" / "ocr"
    runs.mkdir(parents=True, exist_ok=True)

    env_df, _device = inspect_environment(runs, pretrained_dir(repo))
    print(env_df.to_string(index=False))
    write_environment(runs / "environment.json", env_df)

    summary_path = runs / f"{experiment}_summary.csv"
    detail_path = runs / f"{experiment}_boxes.csv"
    docs_summary = docs_dir / "ranking.csv"

    notes = [
        "A classe YOLO é o texto impresso (não name/number/colection).",
        "exact_clean usa os limpadores de `read_card.py` (o que o pipeline veria).",
        "Caixas `--` / ilegíveis saem do ranking.",
    ]

    if args.report_only:
        if not summary_path.is_file() and docs_summary.is_file():
            summary_path = docs_summary
        if not summary_path.is_file():
            raise FileNotFoundError("Sem CSV de ranking. Rode sem --report-only.")
        summary = pd.read_csv(summary_path)
        winner = pick_ocr_winner(summary)
        by_tipo = pd.DataFrame()
        n_readable = int(summary["n"].iloc[0]) if "n" in summary.columns else 0
        figures = _save_charts(docs_dir, summary, by_tipo, None) if "exact_clean" in summary.columns else []
        write_ocr_readme(
            repo=repo,
            docs_dir=docs_dir,
            summary=summary,
            by_tipo=by_tipo,
            winner=winner,
            env={str(r["Item"]): str(r["Valor"]) for _, r in env_df.iterrows()},
            figures=figures,
            n_readable=n_readable,
            n_illegible=0,
            notes=notes,
        )
        if args.pdf:
            write_ocr_pdf(docs_dir, "ocr-cmp-v1", figures, [f"Vencedor: {winner}"])
        return 0

    if not data_yaml.is_file():
        raise FileNotFoundError(f"Unzip o dataset OCR em {dataset_dir}")
    class_names = load_class_names(data_yaml)
    print("classes:", len(class_names), dict(Counter(infer_tipo(n) for n in class_names.values())))

    loaded, active, reader, paddle, _tess, helpers = load_engines()
    readers = _readers(reader, paddle, helpers)

    def fold_for(tipo: str, text: str) -> str:
        if tipo == "number":
            return fold_number(text, _NUMBER_TRANSLATE)
        if tipo == "collection":
            return fold_collection(text)
        return fold_text(text)

    def same_field(tipo: str, expected: str, predicted: str) -> bool:
        if tipo == "number":
            return fold_number(expected, _NUMBER_TRANSLATE) == fold_number(predicted, _NUMBER_TRANSLATE)
        if tipo == "collection":
            return same_collection(expected, predicted)
        return fold_text(expected) == fold_text(predicted)

    lab_dir = dataset_dir / args.split / "labels"
    n_labels = 0
    if lab_dir.is_dir():
        for path in lab_dir.glob("*.txt"):
            n_labels += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    n_total = n_labels if args.max_boxes is None else min(n_labels, args.max_boxes)
    meter = CountProgress("OCR", n_total or None)

    rows = []
    n_seen = 0
    for sample in iter_boxes(dataset_dir, args.split, class_names):
        if args.max_boxes is not None and n_seen >= args.max_boxes:
            break
        n_seen += 1
        tipo = sample["tipo"]
        field = tipo if tipo in {"name", "number", "collection"} else "name"
        crop = warp_quad(sample["bgr"], sample["quad"], pad=_WARP_PAD.get(field, 0.12))
        row = {
            "imagem": sample["image"],
            "caixa": sample["box_i"],
            "tipo": tipo,
            "gt": sample["gt"],
            "legivel": tipo not in {"ilegivel", "vazio"},
        }
        meter.update(n_seen, extra=str(sample["gt"])[:24])
        for engine in active:
            started = time.perf_counter()
            try:
                raw = readers[engine](crop, field) if crop.size else ""
            except Exception as exc:
                raw = ""
                row[f"err_{engine}"] = str(exc)
            elapsed = 1000.0 * (time.perf_counter() - started)
            cleaned = _clean(field, raw) if row["legivel"] else (raw or "")
            row[f"raw_{engine}"] = raw
            row[f"clean_{engine}"] = cleaned
            row[f"ms_{engine}"] = round(elapsed, 1)
            row[f"ok_raw_{engine}"] = same_field(field, sample["gt"], raw) if row["legivel"] else False
            row[f"ok_clean_{engine}"] = same_field(field, sample["gt"], cleaned) if row["legivel"] else False
            row[f"cer_raw_{engine}"] = (
                cer(fold_for(field, sample["gt"]), fold_for(field, raw)) if row["legivel"] else np.nan
            )
            row[f"cer_clean_{engine}"] = (
                cer(fold_for(field, sample["gt"]), fold_for(field, cleaned)) if row["legivel"] else np.nan
            )
            row[f"wer_{engine}"] = (
                wer(sample["gt"], cleaned) if field == "name" and row["legivel"] else np.nan
            )
        rows.append(row)

    meter.done()
    detail = pd.DataFrame(rows)
    detail.to_csv(detail_path, index=False, encoding="utf-8")
    readable = detail[detail["legivel"]].copy()
    n = len(readable)
    if n == 0:
        raise RuntimeError("Nenhuma caixa legível.")

    summary_rows = []
    for engine in active:
        ok = readable[f"ok_clean_{engine}"]
        rec = {
            "motor": LABEL[engine],
            "exact_clean": round(100 * float(ok.mean()), 2),
            "exact_raw": round(100 * float(readable[f"ok_raw_{engine}"].mean()), 2),
            "CER_clean": round(float(readable[f"cer_clean_{engine}"].mean()), 4),
            "CER_raw": round(float(readable[f"cer_raw_{engine}"].mean()), 4),
            "vazio_%": round(
                100 * float((readable[f"clean_{engine}"].fillna("").astype(str).str.strip() == "").mean()), 2
            ),
            "ms/caixa": round(float(readable[f"ms_{engine}"].mean()), 1),
            "n": n,
        }
        name_mask = readable["tipo"] == "name"
        rec["WER_nome"] = (
            round(float(readable.loc[name_mask, f"wer_{engine}"].mean()), 4) if name_mask.any() else np.nan
        )
        summary_rows.append(rec)
    summary = pd.DataFrame(summary_rows)
    winner = pick_ocr_winner(summary)
    summary.insert(0, "winner", summary["motor"].eq(winner).map({True: "←", False: ""}))
    summary = summary.sort_values("exact_clean", ascending=False)
    print(summary.to_string(index=False))
    print("Vencedor:", winner)

    by_tipo_rows = []
    for tipo, chunk in readable.groupby("tipo"):
        rec = {"tipo": tipo, "n": len(chunk)}
        for engine in active:
            rec[LABEL[engine]] = round(100 * float(chunk[f"ok_clean_{engine}"].mean()), 1)
        by_tipo_rows.append(rec)
    by_tipo = pd.DataFrame(by_tipo_rows)
    print(by_tipo.to_string(index=False))

    summary.to_csv(summary_path, index=False, encoding="utf-8")
    docs_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(docs_dir / "ranking.csv", index=False, encoding="utf-8")

    samples = _draw_samples(dataset_dir, args.split, readable, active, 8, helpers)
    figures = _save_charts(docs_dir, summary, by_tipo, samples)
    if samples is not None and samples.exists():
        samples.unlink()

    env = {str(r["Item"]): str(r["Valor"]) for _, r in env_df.iterrows()}
    write_ocr_readme(
        repo=repo,
        docs_dir=docs_dir,
        summary=summary,
        by_tipo=by_tipo,
        winner=winner,
        env=env,
        figures=figures,
        n_readable=n,
        n_illegible=int((~detail["legivel"]).sum()),
        notes=notes,
    )
    if args.pdf:
        body = [f"Vencedor: {winner}", markdown_table(summary)]
        write_ocr_pdf(docs_dir, "ocr-cmp-v1", figures, body)
    _ = loaded
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
