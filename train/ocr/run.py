from __future__ import annotations

import argparse
import json
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
from train.ocr.preprocess import (
    ADD_OPS,
    TECHNIQUES,
    apply_pipeline,
    apply_treatment,
    mix_key,
    mix_label,
    technique_steps,
)

ENGINES = ("easyocr", "paddleocr", "tesseract")
MIN_GAIN_PP = 2.0
FIELD_REGRESS_PP = 1.0
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


def load_engines(wanted: tuple[str, ...] | None = None):
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

    wanted = wanted or ENGINES
    loaded = {}
    reader = paddle = tesseract_cmd = None
    if "easyocr" in wanted:
        try:
            import easyocr

            reader = easyocr.Reader(["en", "pt"], gpu=torch.cuda.is_available())
            loaded["easyocr"] = "ok"
            print("EasyOCR ok")
        except Exception as exc:
            loaded["easyocr"] = f"skip: {exc}"
            print(loaded["easyocr"])
    if "paddleocr" in wanted:
        try:
            from rapidocr import RapidOCR

            paddle = RapidOCR(params={"Global.log_level": "error"})
            loaded["paddleocr"] = "ok"
            print("RapidOCR ok")
        except Exception as exc:
            loaded["paddleocr"] = f"skip: {exc}"
            print(loaded["paddleocr"])
    if "tesseract" in wanted:
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
    active = [k for k in wanted if str(loaded.get(k, "")).startswith("ok")]
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


def _rapid_text(paddle, gray: np.ndarray) -> str:
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


def _treatment_readers(paddle):
    readers = {}
    for kind, _label in TECHNIQUES:

        def _read(image: np.ndarray, field: str, *, _kind=kind) -> str:
            view = apply_treatment(image, field, _kind)
            return _rapid_text(paddle, view)

        readers[kind] = _read
    return readers


def _mix_readers(paddle, candidates: list[tuple[str, ...]]):
    readers = {}
    labels = {}
    for steps in candidates:
        key = mix_key(steps)
        labels[key] = mix_label(steps)

        def _read(image: np.ndarray, field: str, *, _steps=steps) -> str:
            view = apply_pipeline(image, field, _steps)
            return _rapid_text(paddle, view)

        readers[key] = _read
    return readers, labels


def _kind_from_label(label: str) -> str:
    for kind, name in TECHNIQUES:
        if name == label:
            return kind
    raise KeyError(f"técnica não encontrada: {label}")


def _se_pp(p_percent: float, n: int) -> float:
    if n <= 0:
        return 0.0
    p = min(max(p_percent / 100.0, 1e-6), 1 - 1e-6)
    return 100.0 * float(np.sqrt(p * (1 - p) / n))


def _paired_counts(detail: pd.DataFrame, base_id: str, mix_id: str) -> tuple[int, int, int]:
    readable = detail[detail["legivel"]]
    base_ok = readable[f"ok_clean_{base_id}"].astype(bool)
    mix_ok = readable[f"ok_clean_{mix_id}"].astype(bool)
    only_mix = int((mix_ok & ~base_ok).sum())
    only_base = int((base_ok & ~mix_ok).sum())
    return only_mix, only_base, only_mix - only_base


def judge_mix_worth(
    *,
    mix_name: str,
    mix_score: float,
    base_score: float,
    n: int,
    by_tipo: pd.DataFrame | None = None,
    mix_tipo_col: str | None = None,
    base_tipo_col: str = "upscale",
    only_mix: int | None = None,
    only_base: int | None = None,
) -> dict:
    gain = float(mix_score) - float(base_score)
    se = round(_se_pp(base_score, n), 1)
    net_boxes = (
        int(only_mix - only_base)
        if only_mix is not None and only_base is not None
        else int(round(gain / 100.0 * n))
    )
    fields: list[dict] = []
    field_fail: list[str] = []
    if by_tipo is not None and not by_tipo.empty and mix_tipo_col and mix_tipo_col in by_tipo.columns:
        base_col = base_tipo_col if base_tipo_col in by_tipo.columns else None
        if base_col:
            for _, row in by_tipo.iterrows():
                tipo = str(row["tipo"])
                base_v = float(row[base_col])
                mix_v = float(row[mix_tipo_col])
                delta = mix_v - base_v
                rec = {"tipo": tipo, "base": base_v, "mix": mix_v, "gain": round(delta, 1)}
                fields.append(rec)
                if delta <= -FIELD_REGRESS_PP:
                    field_fail.append(f"{tipo} {delta:+.1f} p.p.")
    gain_ok = gain >= MIN_GAIN_PP
    fields_ok = not field_fail
    already = mix_name.strip().lower() == "upscale" or abs(gain) < 1e-9
    worth = (not already) and gain_ok and fields_ok
    checks = []
    if already:
        checks.append({"ok": True, "text": "a mistura vencedora já é o upscale de produção."})
    else:
        checks.append(
            {
                "ok": gain_ok,
                "text": (
                    f"ganho {gain:+.2f} p.p. ≥ {MIN_GAIN_PP:.0f} p.p."
                    if gain_ok
                    else f"ganho {gain:+.2f} p.p. < {MIN_GAIN_PP:.0f} p.p. (SE ≈ {se} p.p.)"
                ),
            }
        )
        checks.append(
            {
                "ok": fields_ok,
                "text": (
                    f"nenhum tipo caiu ≥ {FIELD_REGRESS_PP:.0f} p.p."
                    if fields_ok
                    else "piorou tipo: " + "; ".join(field_fail)
                ),
            }
        )
    if already:
        headline = "Já é a produção"
        verdict = "A mistura vencedora já é o upscale."
    elif worth:
        headline = "Vale"
        verdict = "Entra em `enhance_strip`."
    else:
        headline = "Não vale"
        failed = "; ".join(c["text"] for c in checks if not c["ok"])
        verdict = (failed[:1].upper() + failed[1:] if failed else "não passou no critério") + "."
    md = [
        f"**{headline}.** {verdict}",
        "",
        f"Critério: ganho de exact_clean ≥ **{MIN_GAIN_PP:.0f} p.p.** sobre o upscale "
        f"e nenhum tipo de faixa cair ≥ **{FIELD_REGRESS_PP:.0f} p.p.**",
        "",
        "| | upscale (produção) | mistura | Δ |",
        "| --- | --- | --- | --- |",
        f"| exact_clean (%) | {base_score:.2f} | {mix_score:.2f} | {gain:+.2f} |",
        f"| caixas (n={n}) | — | — | {net_boxes:+d} |",
    ]
    if only_mix is not None and only_base is not None:
        md.append(f"| só a mistura acerta | — | {only_mix} | — |")
        md.append(f"| só o upscale acerta | {only_base} | — | — |")
    md.append("")
    if fields:
        md.extend(
            [
                "| tipo | upscale | mistura | Δ |",
                "| --- | --- | --- | --- |",
            ]
        )
        for rec in fields:
            md.append(f"| {rec['tipo']} | {rec['base']:.1f} | {rec['mix']:.1f} | {rec['gain']:+.1f} |")
        md.append("")
    md.append("Checagens:")
    md.append("")
    for check in checks:
        mark = "ok" if check["ok"] else "falhou"
        md.append(f"- [{mark}] {check['text']}")
    return {
        "worth": worth,
        "verdict": verdict,
        "mix": mix_name,
        "baseline": "upscale",
        "mix_score": round(float(mix_score), 2),
        "base_score": round(float(base_score), 2),
        "gain_pp": round(float(gain), 2),
        "n": int(n),
        "net_boxes": int(net_boxes),
        "only_mix": only_mix,
        "only_base": only_base,
        "se_pp": se,
        "min_gain_pp": MIN_GAIN_PP,
        "field_regress_pp": FIELD_REGRESS_PP,
        "by_tipo": fields,
        "checks": checks,
        "md": md,
    }


def print_worth(worth: dict) -> None:
    print("\n======== Vale a pena? ========")
    print(f"Mistura: {worth['mix']} ({worth['mix_score']:.2f}%)")
    print(f"Produção: {worth['baseline']} ({worth['base_score']:.2f}%)")
    print(f"Ganho: {worth['gain_pp']:+.2f} p.p.  (≈{worth['net_boxes']:+d} caixas em {worth['n']})")
    print(f"SE ≈ {worth['se_pp']} p.p.   limiar: {worth['min_gain_pp']:.0f} p.p.")
    for rec in worth.get("by_tipo") or []:
        print(f"  {rec['tipo']}: {rec['gain']:+.1f} p.p.")
    print(f"Veredito: {'Vale' if worth['worth'] else 'Não vale'} — {worth['verdict']}")


def _mix_tipo_column(mix_name: str) -> str | None:
    raw = re.sub(r"\s*\([\d.]+%\)\s*$", "", mix_name or "").strip()
    if not raw or raw == "upscale":
        return "upscale"
    for kind, label in TECHNIQUES:
        if raw == mix_label(technique_steps(kind)) or raw == label:
            return label
    return None


def worth_from_tables(
    treat_summary: pd.DataFrame | None,
    treat_by_tipo: pd.DataFrame | None,
    mix_name: str,
    mix_score: float | None,
    detail: pd.DataFrame | None = None,
    mix_kind: str | None = None,
) -> dict | None:
    if treat_summary is None or treat_summary.empty:
        return None
    up = treat_summary.loc[treat_summary["motor"].astype(str).str.lower() == "upscale"]
    if up.empty:
        return None
    base_score = float(up.iloc[0]["exact_clean"])
    n = int(up.iloc[0]["n"]) if "n" in up.columns else 0
    score = mix_score
    if score is None:
        return None
    mix_col = _mix_tipo_column(mix_name)
    only_mix = only_base = None
    if detail is not None and mix_kind and f"ok_clean_{mix_kind}" in detail.columns:
        if "ok_clean_upscale" in detail.columns:
            only_mix, only_base, _net = _paired_counts(detail, "upscale", mix_kind)
    return judge_mix_worth(
        mix_name=re.sub(r"\s*\([\d.]+%\)\s*$", "", mix_name or "").strip() or "upscale",
        mix_score=float(score),
        base_score=base_score,
        n=n,
        by_tipo=treat_by_tipo,
        mix_tipo_col=mix_col,
        only_mix=only_mix,
        only_base=only_base,
    )


def run_worth_eval(samples, paddle, pack, mix_steps: tuple[str, ...], mix_name: str, mix_score: float) -> dict:
    """Confronta a mistura vencedora com o upscale de produção (passo final)."""
    if mix_steps == ():
        return judge_mix_worth(mix_name="upscale", mix_score=mix_score, base_score=mix_score, n=0)
    candidates = [(), mix_steps]
    readers, labels = _mix_readers(paddle, candidates)
    active = [mix_key(steps) for steps in candidates]
    detail = run_ocr_pass(samples, readers, active, title="OCR vale a pena", **pack)
    summary, by_tipo, _winner, n, _bad = summarize_ocr(detail, active, labels)
    print("\nConfronto final (produção vs mistura)")
    print(summary.to_string(index=False))
    print(by_tipo.to_string(index=False))
    base_name = mix_label(())
    win_name = labels[mix_key(mix_steps)]
    base_score = float(summary.loc[summary["motor"] == base_name, "exact_clean"].iloc[0])
    score = float(summary.loc[summary["motor"] == win_name, "exact_clean"].iloc[0])
    only_mix, only_base, _net = _paired_counts(detail, mix_key(()), mix_key(mix_steps))
    return judge_mix_worth(
        mix_name=mix_name,
        mix_score=score,
        base_score=base_score,
        n=n,
        by_tipo=by_tipo,
        mix_tipo_col=win_name,
        base_tipo_col=base_name,
        only_mix=only_mix,
        only_base=only_base,
    )


def run_greedy_mix(
    samples, paddle, pack, root_kind: str, root_score: float
) -> tuple[list[str], str, float, tuple[str, ...]]:
    """A partir da melhor técnica individual, mistura as restantes por ramo (para se piorar)."""
    root = technique_steps(root_kind)
    remaining = [op for op in ADD_OPS if op not in root]
    log = [
        f"- Raiz da busca: **{mix_label(root)}** ({root_score:.2f}%), eleita entre as cinco técnicas individuais.",
    ]
    best_label = mix_label(root)
    best_score = float(root_score)
    best_steps = root

    def _expand(base: tuple[str, ...], left: list[str], parent_score: float, indent: str) -> None:
        nonlocal best_label, best_score, best_steps
        if not left:
            return
        candidates = [base + (op,) for op in left]
        readers, labels = _mix_readers(paddle, candidates)
        active = [mix_key(steps) for steps in candidates]
        title = f"OCR mix {mix_label(base)}"
        detail = run_ocr_pass(samples, readers, active, title=title, **pack)
        summary, _by_tipo, _winner, _n, _bad = summarize_ocr(detail, active, labels)
        print(f"\nRamo {mix_label(base)}  ({parent_score:.2f}%)")
        print(summary.to_string(index=False))
        improved: list[tuple[str, tuple[str, ...], float]] = []
        for op, steps in zip(left, candidates):
            name = labels[mix_key(steps)]
            score = float(summary.loc[summary["motor"] == name, "exact_clean"].iloc[0])
            if score > parent_score:
                log.append(
                    f"{indent}- `{name}`: {score:.2f}% (melhorou vs {parent_score:.2f}%; continua o ramo)"
                )
                if score > best_score:
                    best_label, best_score, best_steps = name, score, steps
                improved.append((op, steps, score))
            else:
                log.append(
                    f"{indent}- `{name}`: {score:.2f}% (piorou vs {parent_score:.2f}%; ramo parado)"
                )
        for op, steps, score in improved:
            _expand(steps, [item for item in left if item != op], score, indent + "  ")

    if root_kind != "upscale":
        log.append(
            "- `upscale` já entra em todas as técnicas (redimensionar a faixa); "
            "não é um passo extra neste ramo."
        )
    if remaining:
        _expand(root, remaining, float(root_score), "  ")
    else:
        log.append("- Não restavam técnicas para misturar.")
    return log, best_label, best_score, best_steps


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


def _save_charts(
    docs_dir: Path,
    summary: pd.DataFrame,
    by_tipo: pd.DataFrame,
    samples_path: Path | None,
    *,
    prefix: str = "",
    title: str = "Acerto normalizado por motor",
    tipo_title: str = "Acerto por tipo de faixa",
    sample_caption: str = "Amostras",
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = docs_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    published: list[tuple[str, Path]] = []
    stem = prefix or ""
    rotate = len(summary) > 4
    width = 9.4 if rotate else 7.2

    fig, ax = plt.subplots(figsize=(width, 3.8 if rotate else 3.6))
    ax.bar(summary["motor"], summary["exact_clean"], color="#3b6ea5")
    ax.set_ylabel("exact_clean (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    if rotate:
        ax.tick_params(axis="x", rotation=32)
    fig.tight_layout()
    raw = fig_dir / f"{stem}exact_clean.png"
    fig.savefig(raw, dpi=110)
    plt.close(fig)
    published.append((title, compress_image(raw, raw, jpeg=False)))

    if not by_tipo.empty:
        fig, ax = plt.subplots(figsize=(max(width, 8.0), 4.0 if rotate else 3.8))
        motors = [c for c in by_tipo.columns if c != "tipo" and c != "n"]
        x = np.arange(len(by_tipo))
        bar_w = 0.8 / max(len(motors), 1)
        for i, motor in enumerate(motors):
            ax.bar(x + i * bar_w, by_tipo[motor], width=bar_w, label=motor)
        ax.set_xticks(x + bar_w * (len(motors) - 1) / 2)
        ax.set_xticklabels(by_tipo["tipo"])
        ax.set_ylabel("exact_clean (%)")
        ax.set_title(tipo_title)
        ax.legend(fontsize=8, ncol=2 if len(motors) > 4 else 1)
        fig.tight_layout()
        raw = fig_dir / f"{stem}exact_by_tipo.png"
        fig.savefig(raw, dpi=110)
        plt.close(fig)
        published.append((tipo_title, compress_image(raw, raw, jpeg=False)))

    if samples_path is not None and samples_path.is_file():
        dest = fig_dir / f"{stem}samples.jpg" if stem else fig_dir / "samples.jpg"
        published.append((sample_caption, compress_image(samples_path, dest, jpeg=True)))
    return published


def summarize_ocr(detail: pd.DataFrame, active: list[str], labels: dict[str, str]):
    readable = detail[detail["legivel"]].copy()
    n = len(readable)
    if n == 0:
        raise RuntimeError("Nenhuma caixa legível.")
    summary_rows = []
    for engine in active:
        ok = readable[f"ok_clean_{engine}"]
        rec = {
            "motor": labels[engine],
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
    by_tipo_rows = []
    for tipo, chunk in readable.groupby("tipo"):
        rec = {"tipo": tipo, "n": len(chunk)}
        for engine in active:
            rec[labels[engine]] = round(100 * float(chunk[f"ok_clean_{engine}"].mean()), 1)
        by_tipo_rows.append(rec)
    by_tipo = pd.DataFrame(by_tipo_rows)
    n_illegible = int((~detail["legivel"]).sum())
    return summary, by_tipo, winner, n, n_illegible


def _draw_samples(dataset_dir, split, readable, active, max_show, helpers, labels=None) -> Path | None:
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
        names = labels or LABEL
        for engine in active:
            mark = "OK" if row[f"ok_clean_{engine}"] else "ERR"
            bits.append(f"{names.get(engine, engine)} {mark} {row[f'clean_{engine}']}")
        ax.set_title("  |  ".join(bits), fontsize=9, loc="left")
    fig.tight_layout()
    dest = Path("_ocr_samples_tmp.png")
    fig.savefig(dest, dpi=100)
    plt.close(fig)
    return dest


def run_ocr_pass(samples, readers, active, *, clean, fold_for, same_field, title: str) -> pd.DataFrame:
    meter = CountProgress(title, len(samples) or None)
    rows = []
    for index, sample in enumerate(samples, start=1):
        meter.update(index, extra=str(sample["gt"])[:24])
        field = sample["field"]
        crop = sample["crop"]
        row = {
            "imagem": sample["imagem"],
            "caixa": sample["caixa"],
            "tipo": sample["tipo"],
            "gt": sample["gt"],
            "legivel": sample["legivel"],
        }
        for engine in active:
            started = time.perf_counter()
            try:
                raw = readers[engine](crop, field) if crop is not None and crop.size else ""
            except Exception as exc:
                raw = ""
                row[f"err_{engine}"] = str(exc)
            elapsed = 1000.0 * (time.perf_counter() - started)
            cleaned = clean(field, raw) if row["legivel"] else (raw or "")
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
            row[f"wer_{engine}"] = wer(sample["gt"], cleaned) if field == "name" and row["legivel"] else np.nan
        rows.append(row)
    meter.done()
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compara motores de OCR e pré-processamentos da faixa; grava o relatório único em train/ocr."
    )
    parser.add_argument("--report-only", action="store_true", help="Só regenera README a partir do CSV já gravado.")
    parser.add_argument("--pdf", action="store_true")
    parser.add_argument("--max-boxes", type=int, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--treatments",
        action="store_true",
        help="Só o sweep de tratamentos. O ranking de motores já gravado entra no mesmo README.",
    )
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

    treatments_only = bool(args.treatments)
    dataset_dir = repo / "data" / "ocr"
    data_yaml = dataset_dir / "data.yaml"
    runs = ocr_runs(repo)
    docs_dir = repo / "train" / "ocr"
    runs.mkdir(parents=True, exist_ok=True)
    docs_dir.mkdir(parents=True, exist_ok=True)

    env_df, _device = inspect_environment(runs, pretrained_dir(repo))
    print(env_df.to_string(index=False))
    write_environment(runs / "environment.json", env_df)

    engine_summary_path = runs / "ocr-cmp-v1_summary.csv"
    engine_tipo_path = runs / "ocr-cmp-v1_by_tipo.csv"
    treat_summary_path = runs / "ocr-treat-v1_summary.csv"
    treat_tipo_path = runs / "ocr-treat-v1_by_tipo.csv"

    notes = [
        "A classe YOLO é o texto impresso (não name/number/colection).",
        "exact_clean usa os limpadores de `read_card.py` (o que o pipeline veria).",
        "Caixas `--` / ilegíveis saem do ranking.",
        "Motores e produção usam `enhance_strip` (só upscale, sem CLAHE e sem invert).",
        "Cinco técnicas individuais (RapidOCR fixo): upscale, invert, borda branca, Otsu, adaptativo.",
        "A busca gulosa parte da melhor individual e mistura as outras; ramo que piora para.",
        "No fim, a mistura vencedora é confrontada com o upscale de produção.",
        "Só vale a pena (e entra em `enhance_strip`) com ganho ≥ 2 p.p. e nenhum tipo caindo ≥ 1 p.p.",
    ]
    mix_log_path = runs / "ocr-mix-v1_log.txt"
    mix_final_path = runs / "ocr-mix-v1_final.txt"
    worth_path = runs / "ocr-mix-v1_worth.json"

    def _write_full_readme(
        *,
        summary,
        by_tipo,
        winner,
        n_readable,
        n_illegible,
        figures,
        treat_summary=None,
        treat_by_tipo=None,
        treat_winner=None,
        treat_figures=None,
        mix_log=None,
        mix_winner=None,
        mix_score=None,
        worth=None,
    ):
        env = {str(r["Item"]): str(r["Valor"]) for _, r in env_df.iterrows()}
        treat_elect = None
        if treat_winner:
            treat_elect = (
                f"**{treat_winner}** foi a melhor das cinco técnicas sozinhas e virou a raiz da busca gulosa."
            )
        write_ocr_readme(
            repo=repo,
            docs_dir=docs_dir,
            summary=summary,
            by_tipo=by_tipo,
            winner=winner,
            env=env,
            figures=figures,
            n_readable=n_readable,
            n_illegible=n_illegible,
            notes=notes,
            heading="ocr-cmp-v1 — motores e tratamentos de faixa",
            elect_line=(
                f"**{winner}** entra em `src/ocr/read_card.py` e no pipeline. "
                "EasyOCR e Tesseract ficam como baseline. "
                "A faixa de produção só muda se a mistura vencedora valer a pena (seção no fim)."
            ),
            treat_summary=treat_summary,
            treat_by_tipo=treat_by_tipo,
            treat_winner=treat_winner,
            treat_figures=treat_figures,
            treat_elect_line=treat_elect,
            mix_log=mix_log,
            mix_winner=mix_winner,
            worth=worth,
        )

    if args.report_only:
        if not engine_summary_path.is_file() and (docs_dir / "ranking.csv").is_file():
            engine_summary_path = docs_dir / "ranking.csv"
        if not engine_summary_path.is_file() and not treat_summary_path.is_file():
            raise FileNotFoundError("Sem CSV de ranking. Rode sem --report-only.")
        summary = pd.read_csv(engine_summary_path) if engine_summary_path.is_file() else pd.DataFrame()
        if summary.empty and treat_summary_path.is_file():
            summary = pd.read_csv(treat_summary_path)
        winner = pick_ocr_winner(summary)
        by_tipo = pd.read_csv(engine_tipo_path) if engine_tipo_path.is_file() else pd.DataFrame()
        treat_summary = pd.read_csv(treat_summary_path) if treat_summary_path.is_file() else None
        treat_by_tipo = pd.read_csv(treat_tipo_path) if treat_tipo_path.is_file() else None
        treat_winner = pick_ocr_winner(treat_summary) if treat_summary is not None else None
        mix_log = (
            [ln for ln in mix_log_path.read_text(encoding="utf-8").splitlines() if ln]
            if mix_log_path.is_file()
            else None
        )
        mix_winner = mix_final_path.read_text(encoding="utf-8").strip() if mix_final_path.is_file() else None
        mix_score = None
        if mix_winner:
            found = re.search(r"\(([\d.]+)%\)", mix_winner)
            if found:
                mix_score = float(found.group(1))
        if mix_score is None and treat_summary is not None and treat_winner:
            hit = treat_summary.loc[treat_summary["motor"] == treat_winner, "exact_clean"]
            if not hit.empty:
                mix_score = float(hit.iloc[0])
        worth = json.loads(worth_path.read_text(encoding="utf-8")) if worth_path.is_file() else None
        if worth is None:
            treat_boxes = runs / "ocr-treat-v1_boxes.csv"
            detail = pd.read_csv(treat_boxes) if treat_boxes.is_file() else None
            mix_kind = None
            col = _mix_tipo_column(mix_winner or "")
            if col:
                mix_kind = next((kind for kind, label in TECHNIQUES if label == col), col)
            worth = worth_from_tables(
                treat_summary, treat_by_tipo, mix_winner or "upscale", mix_score, detail, mix_kind
            )
            if worth is not None:
                worth_path.write_text(json.dumps(worth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        n_readable = int(summary["n"].iloc[0]) if "n" in summary.columns else 0
        figures = _save_charts(docs_dir, summary, by_tipo, None) if "exact_clean" in summary.columns else []
        treat_figures = None
        if treat_summary is not None and "exact_clean" in treat_summary.columns:
            treat_figures = _save_charts(
                docs_dir,
                treat_summary,
                treat_by_tipo if treat_by_tipo is not None else pd.DataFrame(),
                None,
                prefix="treat_",
                title="Acerto por tratamento (RapidOCR)",
                tipo_title="Acerto por tipo (tratamentos)",
            )
        _write_full_readme(
            summary=summary,
            by_tipo=by_tipo,
            winner=winner,
            n_readable=n_readable,
            n_illegible=0,
            figures=figures,
            treat_summary=treat_summary,
            treat_by_tipo=treat_by_tipo,
            treat_winner=treat_winner,
            mix_log=mix_log,
            mix_winner=mix_winner,
            mix_score=mix_score,
            treat_figures=treat_figures,
            worth=worth,
        )
        if args.pdf:
            body = [f"Motor: {winner}", markdown_table(summary)]
            if treat_summary is not None:
                body += [f"Tratamento: {treat_winner}", markdown_table(treat_summary)]
            write_ocr_pdf(docs_dir, "ocr-cmp-v1", figures + (treat_figures or []), body)
        return 0

    if not data_yaml.is_file():
        raise FileNotFoundError(f"Unzip o dataset OCR em {dataset_dir}")
    class_names = load_class_names(data_yaml)
    print("classes:", len(class_names), dict(Counter(infer_tipo(n) for n in class_names.values())))

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

    pack = dict(clean=_clean, fold_for=fold_for, same_field=same_field)
    samples = []
    for sample in iter_boxes(dataset_dir, args.split, class_names):
        if args.max_boxes is not None and len(samples) >= args.max_boxes:
            break
        tipo = sample["tipo"]
        field = tipo if tipo in {"name", "number", "collection"} else "name"
        samples.append(
            {
                "imagem": sample["image"],
                "caixa": sample["box_i"],
                "tipo": tipo,
                "field": field,
                "gt": sample["gt"],
                "legivel": tipo not in {"ilegivel", "vazio"},
                "crop": warp_quad(sample["bgr"], sample["quad"], pad=_WARP_PAD.get(field, 0.12)),
            }
        )

    summary = by_tipo = pd.DataFrame()
    winner = "RapidOCR (Paddle)"
    figures: list[tuple[str, Path]] = []
    n_readable = n_illegible = 0
    helpers = {"enhance": None}
    paddle = None

    if not treatments_only:
        loaded, active, reader, paddle, _tess, helpers = load_engines()
        engine_readers = _readers(reader, paddle, helpers)
        engine_detail = run_ocr_pass(samples, engine_readers, active, title="OCR motores", **pack)
        engine_detail.to_csv(runs / "ocr-cmp-v1_boxes.csv", index=False, encoding="utf-8")
        summary, by_tipo, winner, n_readable, n_illegible = summarize_ocr(engine_detail, active, dict(LABEL))
        print(summary.to_string(index=False))
        print("Vencedor (motores):", winner)
        print(by_tipo.to_string(index=False))
        summary.to_csv(engine_summary_path, index=False, encoding="utf-8")
        by_tipo.to_csv(engine_tipo_path, index=False, encoding="utf-8")
        summary.to_csv(docs_dir / "ranking.csv", index=False, encoding="utf-8")
        samples_img = _draw_samples(
            dataset_dir, args.split, engine_detail[engine_detail["legivel"]], active, 8, helpers, labels=LABEL
        )
        figures = _save_charts(docs_dir, summary, by_tipo, samples_img)
        if samples_img is not None and samples_img.exists():
            samples_img.unlink()
        _ = loaded

    treat_summary = treat_by_tipo = treat_figures = None
    treat_winner = None
    treat_helpers = helpers
    if paddle is None:
        treat_loaded, _active, _reader, paddle, _tess, treat_helpers = load_engines(("paddleocr",))
        _ = treat_loaded
    if paddle is None:
        raise RuntimeError("RapidOCR não carregou.")
    treat_ids = [kind for kind, _label in TECHNIQUES]
    treat_labels = {kind: label for kind, label in TECHNIQUES}
    treat_detail = run_ocr_pass(
        samples, _treatment_readers(paddle), treat_ids, title="OCR tratamentos", **pack
    )
    treat_detail.to_csv(runs / "ocr-treat-v1_boxes.csv", index=False, encoding="utf-8")
    treat_summary, treat_by_tipo, treat_winner, n_treat, n_treat_bad = summarize_ocr(
        treat_detail, treat_ids, treat_labels
    )
    n_readable = n_readable or n_treat
    n_illegible = n_illegible or n_treat_bad
    print(treat_summary.to_string(index=False))
    print("Vencedor (técnicas individuais):", treat_winner)
    print(treat_by_tipo.to_string(index=False))
    treat_summary.to_csv(treat_summary_path, index=False, encoding="utf-8")
    treat_by_tipo.to_csv(treat_tipo_path, index=False, encoding="utf-8")
    treat_summary.to_csv(docs_dir / "ranking_treatments.csv", index=False, encoding="utf-8")
    if treatments_only:
        if engine_summary_path.is_file():
            summary = pd.read_csv(engine_summary_path)
            by_tipo = pd.read_csv(engine_tipo_path) if engine_tipo_path.is_file() else pd.DataFrame()
            winner = pick_ocr_winner(summary)
            n_readable = int(summary["n"].iloc[0]) if "n" in summary.columns else n_readable
        elif summary.empty:
            summary, by_tipo, winner = treat_summary, treat_by_tipo, treat_winner
    root_score = float(treat_summary.loc[treat_summary["motor"] == treat_winner, "exact_clean"].iloc[0])
    mix_log, mix_winner, mix_score, mix_steps = run_greedy_mix(
        samples, paddle, pack, _kind_from_label(treat_winner), root_score
    )
    print(f"\nMistura final: {mix_winner}  ({mix_score:.2f}%)")
    mix_log_path.write_text("\n".join(mix_log) + "\n", encoding="utf-8")
    mix_final_path.write_text(f"{mix_winner} ({mix_score:.2f}%)\n", encoding="utf-8")
    if len(mix_steps) <= 1:
        mix_kind = mix_steps[0] if mix_steps else "upscale"
        worth = worth_from_tables(
            treat_summary, treat_by_tipo, mix_winner, mix_score, treat_detail, mix_kind
        )
    else:
        worth = None
    if worth is None:
        worth = run_worth_eval(samples, paddle, pack, mix_steps, mix_winner, mix_score)
    print_worth(worth)
    worth_path.write_text(json.dumps(worth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    show_ids = [kind for kind in treat_ids]
    treat_samples = _draw_samples(
        dataset_dir,
        args.split,
        treat_detail[treat_detail["legivel"]],
        show_ids[:4],
        8,
        treat_helpers,
        labels=treat_labels,
    )
    treat_figures = _save_charts(
        docs_dir,
        treat_summary,
        treat_by_tipo,
        treat_samples,
        prefix="treat_",
        title="Acerto por tratamento (RapidOCR)",
        tipo_title="Acerto por tipo (tratamentos)",
        sample_caption="Amostras (tratamentos)",
    )
    if treat_samples is not None and treat_samples.exists():
        treat_samples.unlink()

    _write_full_readme(
        summary=summary,
        by_tipo=by_tipo,
        winner=winner,
        n_readable=n_readable,
        n_illegible=n_illegible,
        figures=figures,
        treat_summary=treat_summary,
        treat_by_tipo=treat_by_tipo,
        treat_winner=treat_winner,
        treat_figures=treat_figures,
        mix_log=mix_log,
        mix_winner=f"{mix_winner} ({mix_score:.2f}%)",
        mix_score=mix_score,
        worth=worth,
    )
    if args.pdf:
        body = [
            f"Motor: {winner}",
            markdown_table(summary),
            f"Técnica individual: {treat_winner}",
            markdown_table(treat_summary),
            f"Mistura final: {mix_winner} ({mix_score:.2f}%)",
        ]
        write_ocr_pdf(docs_dir, "ocr-cmp-v1", figures + (treat_figures or []), body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
