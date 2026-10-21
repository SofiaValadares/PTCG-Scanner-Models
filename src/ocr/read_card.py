"""Detect name / number / collection regions on a cropped card and read them."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# Roboflow: 0=colection (typo), 1=name, 2=number
CLASS_TO_FIELD = {0: "collection", 1: "name", 2: "number"}
FIELD_TO_CLASS = {v: k for k, v in CLASS_TO_FIELD.items()}

FALLBACK_TEMPLATE = {
    "name": (0.048, 0.128, 0.10, 0.68),
    "number": (0.915, 0.995, 0.14, 0.48),
    "collection": (0.915, 0.995, 0.48, 0.92),
}

# EasyOCR only: empty means do not restrict the charset (CJK / Hangul / Cyrillic).
_NAME_ALLOW = ""
_NAME_PUNCT = set("'’`-·・ー.☆★&:")
# Digits EasyOCR confuses with letters are mapped in clean_number.
_NUMBER_ALLOW = "0123456789/OoQqDdIiLl|ZzSsGgBb"
_NUMBER_TRANSLATE = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "Q": "0",
        "q": "0",
        "D": "0",
        "I": "1",
        "i": "1",
        "L": "1",
        "l": "1",
        "|": "1",
        "Z": "2",
        "z": "2",
        "S": "5",
        "s": "5",
        "G": "6",
        "b": "6",
        "B": "8",
        "g": "9",
    }
)
_COLLECTION_ALLOW = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "ÁÉÍÓÚáéíóúÃÕãõÂÊÔâêô '-"
)
_FIELD_ALLOW = {
    "name": _NAME_ALLOW,
    "number": _NUMBER_ALLOW,
    "collection": _COLLECTION_ALLOW,
}
_WARP_PAD = {"name": 0.18, "number": 0.12, "collection": 0.14}


@dataclass
class Region:
    field: str
    image: np.ndarray
    conf: float
    quad: np.ndarray
    text: str = ""
    source: str = "det"
    ocr_conf: float = 0.0


@dataclass
class CardRead:
    name: str = ""
    number: str = ""
    collection: str = ""
    regions: dict[str, Region] = field(default_factory=dict)


def order_corners(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    pts = np.roll(pts, -start, axis=0)
    edge = pts[1] - pts[0]
    to_last = pts[-1] - pts[0]
    if float(edge[0] * to_last[1] - edge[1] * to_last[0]) < 0:
        pts = np.array([pts[0], pts[3], pts[2], pts[1]], dtype=np.float32)
    return pts.astype(np.float32)


def warp_quad(image: np.ndarray, quad: np.ndarray, pad: float = 0.12) -> np.ndarray:
    src = order_corners(quad)
    center = src.mean(axis=0)
    src = center + (src - center) * (1.0 + pad)
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    w, h = max(w, 8), max(h, 8)
    if h > w * 1.4:
        src = np.array([src[1], src[2], src[3], src[0]], dtype=np.float32)
        w, h = h, w
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_CUBIC)


def crop_template(image: np.ndarray, field: str) -> np.ndarray:
    y0, y1, x0, x1 = FALLBACK_TEMPLATE[field]
    h, w = image.shape[:2]
    crop = image[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
    return crop if crop.size else image


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def enhance_strip(image: np.ndarray, min_h: int = 112) -> np.ndarray:
    """Amplia a faixa em cinza. Invert ganhou 1 p.p. no sweep; fica fora por não ser significativo."""
    gray = _gray(image)
    h, w = gray.shape[:2]
    if h < min_h:
        scale = min_h / max(h, 1)
        gray = cv2.resize(
            gray,
            (max(8, int(round(w * scale))), min_h),
            interpolation=cv2.INTER_CUBIC,
        )
    return gray


def _strip_variants(gray: np.ndarray) -> list[np.ndarray]:
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return [gray, 255 - gray, otsu, 255 - otsu]


def _readtext(reader, rgb: np.ndarray, allowlist: str):
    with_allow = bool(allowlist)
    attempts = (
        {"detail": 1, "paragraph": False, "allowlist": allowlist, "mag_ratio": 1.8},
        {"detail": 1, "paragraph": False, "allowlist": allowlist},
        {"detail": 1, "paragraph": False},
    )
    if not with_allow:
        attempts = (
            {"detail": 1, "paragraph": False, "mag_ratio": 1.8},
            {"detail": 1, "paragraph": False},
        )
    last_items = []
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            items = reader.readtext(rgb, **kwargs)
        except TypeError as exc:
            last_error = exc
            continue
        text, _conf = _parse_ocr_items(items)
        if text.strip():
            return items
        last_items = items
    if last_error and not last_items:
        return reader.readtext(rgb)
    return last_items


def _parse_ocr_items(items) -> tuple[str, float]:
    parts: list[str] = []
    scores: list[float] = []
    for item in items or []:
        if isinstance(item, str):
            if item.strip():
                parts.append(item.strip())
            continue
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        text = str(item[1]).strip()
        conf = float(item[2]) if len(item) > 2 else 0.0
        if text:
            parts.append(text)
            scores.append(conf)
    joined = " ".join(parts)
    mean = float(np.mean(scores)) if scores else 0.0
    return joined, mean


def _format_number(left: str, right: str) -> str:
    return f"{int(left):03d}/{int(right)}"


def _split_number_digits(digits: str) -> tuple[str, str] | None:
    """Remonta NNN/NNN quando a barra some ou é lida como 1."""
    if len(digits) == 7 and digits[3] == "1":
        return digits[:3], digits[4:]
    if len(digits) == 6:
        return digits[:3], digits[3:]
    if len(digits) == 5:
        return digits[:2], digits[2:]
    return None


def clean_number(text: str) -> str:
    raw = (text or "").translate(_NUMBER_TRANSLATE)
    raw = raw.replace(" ", "").replace("\\", "/").replace("⁄", "/").replace("-", "/")
    match = re.search(r"(\d{1,3})/(\d{2,3})", raw)
    if match:
        return _format_number(match.group(1), match.group(2))
    parts = _split_number_digits(re.sub(r"\D", "", raw))
    if parts:
        return _format_number(*parts)
    return ""


def _keep_name_char(ch: str) -> bool:
    return ch.isalnum() or ch.isspace() or ch in _NAME_PUNCT


def clean_name(text: str) -> str:
    text = "".join(ch if _keep_name_char(ch) else " " for ch in (text or ""))
    return re.sub(r"\s+", " ", text).strip(" -'`")


def fold_text(text: str) -> str:
    """Catalog key: Latin accents fold to ASCII; CJK / Hangul / Cyrillic stay."""
    out: list[str] = []
    for ch in text or "":
        name = unicodedata.name(ch, "")
        if ch.isascii() or name.startswith("LATIN"):
            for part in unicodedata.normalize("NFKD", ch):
                if not unicodedata.combining(part) and part.isalnum():
                    out.append(part.casefold())
            continue
        if ch.isalnum() or ch in "ー":
            out.append(ch.casefold())
    return "".join(out)


def catalog_language_codes(catalog_root: Path | None = None) -> list[str]:
    """Language folders under `data/catalog/{lang}/` that contain cards.csv."""
    root = catalog_root
    if root is None:
        here = Path(__file__).resolve()
        for parent in [here.parent, *here.parents]:
            candidate = parent / "data" / "catalog"
            if candidate.is_dir():
                root = candidate
                break
    if root is None or not Path(root).is_dir():
        return ["en"]
    return sorted(
        path.name
        for path in Path(root).iterdir()
        if path.is_dir() and (path / "cards.csv").is_file()
    )


def clean_collection(text: str) -> str:
    """Fica o código da coleção: 2 a 7 letras ou dígitos, como S9, MEW, OBF ou 151."""
    code = re.sub(r"[^A-Za-z0-9]+", "", text or "")
    if 2 <= len(code) <= 7:
        return code.upper()
    return ""


def _clean(field: str, text: str) -> str:
    if field == "number":
        return clean_number(text)
    if field == "collection":
        return clean_collection(text)
    return clean_name(text)


def _script_counts(text: str) -> tuple[int, int, int, int]:
    latin = cjk = hangul = cyrillic = 0
    for ch in text or "":
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7AF:
            hangul += 1
        elif 0x0400 <= code <= 0x04FF:
            cyrillic += 1
        elif (
            0x3040 <= code <= 0x30FF
            or 0x31F0 <= code <= 0x31FF
            or 0x3400 <= code <= 0x4DBF
            or 0x4E00 <= code <= 0x9FFF
            or 0xF900 <= code <= 0xFAFF
        ):
            cjk += 1
        elif ch.isalpha():
            latin += 1
    return latin, cjk, hangul, cyrillic


def script_family(text: str) -> str:
    """Dominant letter script: latin, cjk, hangul, cyrillic, or none."""
    latin, cjk, hangul, cyrillic = _script_counts(text)
    best = max(latin, cjk, hangul, cyrillic)
    if best == 0:
        return "none"
    if cjk == best:
        return "cjk"
    if hangul == best:
        return "hangul"
    if cyrillic == best:
        return "cyrillic"
    return "latin"


def _kana_count(text: str) -> int:
    return sum(1 for ch in text or "" if 0x3040 <= ord(ch) <= 0x30FF or ch == "ー")


def _pick_name_reading(primary: tuple[str, float], extras: list[tuple[str, str, float]]) -> tuple[str, float]:
    """Prefer a dedicated Japan rec when it reads more kana; Hangul/Cyrillic only if v6 read no letters."""
    text, conf = primary
    for kind, extra_text, extra_conf in extras:
        if kind == "japan" and _kana_count(extra_text) > _kana_count(text):
            text, conf = extra_text, extra_conf
    latin, cjk, _, _ = _script_counts(text)
    if latin + cjk >= 2:
        return text, conf
    for kind, extra_text, extra_conf in extras:
        e_latin, e_cjk, e_hangul, e_cyr = _script_counts(extra_text)
        if kind == "korean" and e_hangul >= 2 and e_hangul > e_latin + e_cjk:
            return extra_text, extra_conf
        if kind == "cyrillic" and e_cyr >= 2 and e_cyr > e_latin + e_cjk:
            return extra_text, extra_conf
    return text, conf


class CardOCR:
    """RapidOCR PP-OCRv6 (Latin + Japanese + Chinese) plus Hangul/Cyrillic rec when the catalog has those langs."""

    def __init__(self, langs: list[str] | None = None):
        from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR

        self.langs = list(langs or ["en"])
        self._RapidOCR = RapidOCR
        self._LangRec = LangRec
        self._ModelType = ModelType
        self._OCRVersion = OCRVersion
        self._primary = RapidOCR(params={"Global.log_level": "error"})
        self._korean = None
        self._cyrillic = None
        self._japan = None
        self._failed: set[str] = set()
        self._want_korean = "ko" in self.langs
        self._want_cyrillic = "ru" in self.langs
        import rapidocr as rapidocr_pkg

        japan_onnx = Path(rapidocr_pkg.__file__).resolve().parent / "models" / "japan_PP-OCRv4_rec_mobile.onnx"
        self._want_japan = "ja" in self.langs and japan_onnx.is_file()

    def __call__(self, *args, **kwargs):
        return self._primary(*args, **kwargs)

    def _load_extra(self, kind: str):
        if kind in self._failed:
            return None
        cached = {"korean": self._korean, "cyrillic": self._cyrillic, "japan": self._japan}[kind]
        if cached is not None:
            return cached
        try:
            if kind == "japan":
                engine = self._RapidOCR(
                    params={
                        "Global.log_level": "error",
                        "Rec.lang_type": self._LangRec.JAPAN,
                        "Rec.ocr_version": self._OCRVersion.PPOCRV4,
                        "Rec.model_type": self._ModelType.MOBILE,
                    }
                )
                self._japan = engine
                return engine
            lang = self._LangRec.KOREAN if kind == "korean" else self._LangRec.CYRILLIC
            engine = self._RapidOCR(
                params={
                    "Global.log_level": "error",
                    "Rec.lang_type": lang,
                    "Rec.ocr_version": self._OCRVersion.PPOCRV5,
                    "Rec.model_type": self._ModelType.MOBILE,
                }
            )
        except Exception:
            self._failed.add(kind)
            return None
        if kind == "korean":
            self._korean = engine
        else:
            self._cyrillic = engine
        return engine

    def read_field(self, gray: np.ndarray, field: str) -> tuple[str, float]:
        text, conf = _rapid_raw(self._primary, gray)
        if field != "name":
            return text, conf
        extras: list[tuple[str, str, float]] = []
        for kind, needed in (
            ("japan", self._want_japan),
            ("korean", self._want_korean),
            ("cyrillic", self._want_cyrillic),
        ):
            if not needed:
                continue
            engine = self._load_extra(kind)
            if engine is None:
                continue
            try:
                extra_text, extra_conf = _rapid_raw(engine, gray)
            except Exception:
                continue
            extras.append((kind, extra_text, extra_conf))
        return _pick_name_reading((text, conf), extras)


def create_reader(catalog_root: Path | str | None = None):
    """RapidOCR covering every script in `data/catalog/{lang}/`."""
    langs = catalog_language_codes(Path(catalog_root) if catalog_root else None)
    return CardOCR(langs)


def _rapid_raw(reader, gray: np.ndarray) -> tuple[str, float]:
    view = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray
    try:
        result = reader(view, use_det=False, use_cls=False)
    except TypeError:
        result = reader(view)
    txts = [str(item).strip() for item in (getattr(result, "txts", None) or []) if str(item).strip()]
    scores = [float(score) for score in (getattr(result, "scores", None) or [])]
    if not txts and isinstance(result, (list, tuple)):
        blob = result[0] if result and isinstance(result[0], list) else result
        for item in blob or []:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                payload = item[1]
                if isinstance(payload, (list, tuple)):
                    txts.append(str(payload[0]).strip())
                    if len(payload) > 1:
                        scores.append(float(payload[1]))
                else:
                    txts.append(str(payload).strip())
        txts = [part for part in txts if part]
    text = " ".join(txts)
    conf = float(np.mean(scores)) if scores else 0.0
    return text, conf


def _easy_ocr_image(reader, enhanced: np.ndarray, field: str) -> tuple[str, float, np.ndarray]:
    allow = _FIELD_ALLOW[field]

    def _run(view: np.ndarray) -> tuple[str, float]:
        rgb = cv2.cvtColor(view, cv2.COLOR_GRAY2RGB)
        text, conf = _parse_ocr_items(_readtext(reader, rgb, allow))
        return _clean(field, text), conf

    text, conf = _run(enhanced)
    if field == "collection" and conf < 0.35:
        text = ""
    if text:
        return text, conf, enhanced
    if field == "name":
        inv = 255 - enhanced
        text, conf = _run(inv)
        if text:
            return text, conf, inv
    if field == "number":
        best_text, best_conf, best_view = "", -1.0, enhanced
        for view in _strip_variants(enhanced)[1:]:
            cand, cconf = _run(view)
            if cand and cconf > best_conf:
                best_text, best_conf, best_view = cand, cconf, view
        return best_text, max(best_conf, 0.0), best_view
    return "", max(conf, 0.0), enhanced


def ocr_image(reader, image: np.ndarray, field: str = "name") -> tuple[str, float, np.ndarray]:
    min_h = 80 if field == "collection" else 112
    enhanced = enhance_strip(image, min_h=min_h)
    if hasattr(reader, "readtext"):
        return _easy_ocr_image(reader, enhanced, field)
    if hasattr(reader, "read_field"):
        raw, conf = reader.read_field(enhanced, field)
    else:
        raw, conf = _rapid_raw(reader, enhanced)
    return _clean(field, raw), conf, enhanced


def regions_from_obb(result, image: np.ndarray | None = None, conf_min: float = 0.25) -> dict[str, Region]:
    bgr = image if image is not None else result.orig_img
    found: dict[str, Region] = {}
    if result.obb is None or len(result.obb) == 0:
        return found
    xy = result.obb.xyxyxyxy.cpu().numpy()
    cls = result.obb.cls.cpu().numpy().astype(int)
    confs = result.obb.conf.cpu().numpy()
    for quad, c, cf in zip(xy, cls, confs):
        if cf < conf_min:
            continue
        field_name = CLASS_TO_FIELD.get(int(c))
        if field_name is None:
            continue
        if field_name in found and found[field_name].conf >= float(cf):
            continue
        found[field_name] = Region(
            field=field_name,
            image=warp_quad(bgr, quad.reshape(4, 2), pad=_WARP_PAD[field_name]),
            conf=float(cf),
            quad=quad.reshape(4, 2),
            source="det",
        )
    return found


def fill_missing(image: np.ndarray, regions: dict[str, Region]) -> dict[str, Region]:
    for field_name in ("name", "number", "collection"):
        if field_name in regions:
            continue
        crop = crop_template(image, field_name)
        h, w = image.shape[:2]
        y0, y1, x0, x1 = FALLBACK_TEMPLATE[field_name]
        quad = np.array(
            [[x0 * w, y0 * h], [x1 * w, y0 * h], [x1 * w, y1 * h], [x0 * w, y1 * h]],
            dtype=np.float32,
        )
        regions[field_name] = Region(
            field=field_name,
            image=crop,
            conf=0.0,
            quad=quad,
            source="template",
        )
    return regions


def read_card(result, reader, image: np.ndarray | None = None, conf_min: float = 0.25) -> CardRead:
    bgr = image if image is not None else result.orig_img
    regions = fill_missing(bgr, regions_from_obb(result, bgr, conf_min=conf_min))
    out = CardRead(regions=regions)
    for field_name in ("name", "number", "collection"):
        region = regions[field_name]
        if field_name == "collection" and region.source == "template":
            continue
        text, ocr_conf, view = ocr_image(reader, region.image, field=field_name)
        region.image = view
        region.text = text
        region.ocr_conf = ocr_conf
        setattr(out, field_name, text)
    return out
