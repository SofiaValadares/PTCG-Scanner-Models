"""Catálogo de tratamentos da faixa. Os pixels saem de `ocr.read_card.prepare_strip`."""

from __future__ import annotations

import numpy as np

from ocr.read_card import prepare_strip

# Rodada 1. `fallback` é uma segunda leitura, não um passo de pixel.
TECHNIQUES: tuple[tuple[str, str], ...] = (
    ("upscale", "upscale"),
    ("h160", "altura 160"),
    ("h192", "altura 192"),
    ("tall_number", "número alto"),
    ("invert", "invert"),
    ("invert_number", "inverte número"),
    ("sharpen", "nitidez"),
    ("stretch", "contraste"),
    ("pad", "borda branca"),
    ("otsu", "Otsu"),
    ("adaptive", "adaptativo"),
    ("fallback", "fallback número"),
)

STRATEGY_KINDS = frozenset({"fallback"})
HEIGHT_STEPS = frozenset({"h160", "h192", "tall_number"})
INVERT_STEPS = frozenset({"invert", "invert_number"})
ADD_OPS: tuple[str, ...] = tuple(
    kind for kind, _ in TECHNIQUES if kind != "upscale" and kind not in STRATEGY_KINDS
)
_MIX_LABEL = {kind: label for kind, label in TECHNIQUES}
FALLBACK_LABEL = _MIX_LABEL["fallback"]

# Grade altura × transformação. Cada campo escolhe a sua célula.
SCALE_HEIGHTS: tuple[int, ...] = (64, 80, 112, 144, 176, 208, 256)
SCALE_OPS: tuple[tuple[tuple[str, ...], str], ...] = (
    ((), "só ampliar"),
    (("invert",), "invert"),
    (("sharpen",), "nitidez"),
    (("stretch",), "contraste"),
)
SCALE_BASE = {"name": 112, "number": 112, "collection": 80}


def scale_key(height: int, steps: tuple[str, ...]) -> str:
    if not steps:
        return f"h{height}"
    return f"h{height}+" + "+".join(steps)


def technique_steps(kind: str) -> tuple[str, ...]:
    if kind == "upscale":
        return ()
    if kind not in _MIX_LABEL:
        raise KeyError(f"técnica desconhecida: {kind}")
    if kind in STRATEGY_KINDS:
        return ()
    return (kind,)


def mix_key(steps: tuple[str, ...]) -> str:
    return "+".join(steps) if steps else "upscale"


def mix_label(steps: tuple[str, ...]) -> str:
    if not steps:
        return "upscale"
    return "upscale + " + " + ".join(_MIX_LABEL.get(step, step) for step in steps)


def mixable(base: tuple[str, ...], op: str) -> bool:
    """Duas alturas, ou dois inverts, não entram no mesmo ramo."""
    if op in STRATEGY_KINDS:
        return False
    if op in HEIGHT_STEPS and any(step in HEIGHT_STEPS for step in base):
        return False
    if op in INVERT_STEPS and any(step in INVERT_STEPS for step in base):
        return False
    return True


def apply_pipeline(image: np.ndarray, field: str, steps: tuple[str, ...] = ()) -> np.ndarray:
    return prepare_strip(image, field, steps)


def apply_treatment(image: np.ndarray, field: str, kind: str) -> np.ndarray:
    return prepare_strip(image, field, technique_steps(kind))
