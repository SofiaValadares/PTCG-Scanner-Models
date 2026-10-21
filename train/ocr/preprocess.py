"""Pré-processamento da faixa para o sweep de OCR. Não altera o pipeline de produção."""

from __future__ import annotations

import cv2
import numpy as np

# Cinco técnicas avaliadas sozinhas na rodada 1; as outras quatro entram na mistura gulosa.
TECHNIQUES: tuple[tuple[str, str], ...] = (
    ("upscale", "upscale"),
    ("invert", "invert"),
    ("pad", "borda branca"),
    ("otsu", "Otsu"),
    ("adaptive", "adaptativo"),
)

ADD_OPS: tuple[str, ...] = tuple(kind for kind, _ in TECHNIQUES if kind != "upscale")
_MIX_LABEL = {kind: label for kind, label in TECHNIQUES}


def technique_steps(kind: str) -> tuple[str, ...]:
    if kind == "upscale":
        return ()
    if kind not in _MIX_LABEL:
        raise KeyError(f"técnica desconhecida: {kind}")
    return (kind,)


def mix_key(steps: tuple[str, ...]) -> str:
    return "+".join(steps) if steps else "upscale"


def mix_label(steps: tuple[str, ...]) -> str:
    if not steps:
        return "upscale"
    return "upscale + " + " + ".join(_MIX_LABEL.get(step, step) for step in steps)


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _upscale(gray: np.ndarray, min_h: int) -> np.ndarray:
    h, w = gray.shape[:2]
    if h >= min_h:
        return gray
    scale = min_h / max(h, 1)
    return cv2.resize(
        gray,
        (max(8, int(round(w * scale))), min_h),
        interpolation=cv2.INTER_CUBIC,
    )


def _pad(gray: np.ndarray, value: int) -> np.ndarray:
    return cv2.copyMakeBorder(gray, 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=value)


def _adaptive(gray: np.ndarray) -> np.ndarray:
    block = 31 if min(gray.shape[:2]) >= 31 else 15
    if block % 2 == 0:
        block += 1
    return cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, 8
    )


def apply_pipeline(image: np.ndarray, field: str, steps: tuple[str, ...] = ()) -> np.ndarray:
    """Upscale e depois os passos da mistura (invert / borda / Otsu / adaptativo)."""
    min_h = 80 if field == "collection" else 112
    gray = _upscale(_gray(image), min_h)
    for step in steps:
        if step == "invert":
            gray = 255 - gray
        elif step == "pad":
            gray = _pad(gray, 255)
        elif step == "otsu":
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif step == "adaptive":
            gray = _adaptive(gray)
        else:
            raise KeyError(f"passo de mistura desconhecido: {step}")
    return gray


def apply_treatment(image: np.ndarray, field: str, kind: str) -> np.ndarray:
    return apply_pipeline(image, field, technique_steps(kind))
