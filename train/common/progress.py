"""Progresso de treino no terminal (não no stdout redirecionado do Ultralytics)."""

from __future__ import annotations

import sys
import time


def stream():
    return sys.__stdout__


def write(msg: str, *, newline: bool = True) -> None:
    out = stream()
    out.write(msg + ("\n" if newline else ""))
    out.flush()


def _fmt_hms(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _bar(cur: int, total: int, width: int = 22) -> str:
    if total <= 0:
        return "░" * width
    frac = min(max(cur / total, 0.0), 1.0)
    filled = int(round(width * frac))
    return "█" * filled + "░" * (width - filled)


def _gpu() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return ""
        props = torch.cuda.get_device_properties(0)
        used = torch.cuda.memory_allocated(0) / 1024**3
        total = props.total_memory / 1024**3
        return f"  GPU {used:.1f}/{total:.1f}G"
    except Exception:
        return ""


class SweepProgress:
    """Uma linha que atualiza no lugar: família, época, mAP, ETA, VRAM."""

    def __init__(self, title: str, n_families: int):
        self.title = title
        self.n_families = n_families
        self.index = 0
        self.family = ""
        self.epoch = 0
        self.epochs = 0
        self.map50_95: float | None = None
        self.family_t0 = time.time()
        self.sweep_t0 = time.time()
        write(f"{title}  —  {n_families} família(s)")

    def start_family(self, index: int, family: str, epochs: int) -> None:
        self.index = index
        self.family = family
        self.epoch = 0
        self.epochs = epochs
        self.map50_95 = None
        self.family_t0 = time.time()
        write(
            f"\n[{index}/{self.n_families}] {family}  teto {epochs} épocas",
        )
        self._emit()

    def epoch_end(self, epoch: int, epochs: int, map50_95: float | None = None) -> None:
        self.epoch = epoch
        self.epochs = epochs
        if map50_95 is not None and map50_95 == map50_95:
            self.map50_95 = float(map50_95)
        self._emit()

    def note(self, text: str) -> None:
        write(f"  {text}")

    def family_done(self, text: str) -> None:
        self._clear()
        write(f"  ✓ {self.family}: {text}")

    def family_fail(self, text: str) -> None:
        self._clear()
        write(f"  ✗ {self.family}: {text}")

    def _clear(self) -> None:
        out = stream()
        out.write("\r" + " " * 120 + "\r")
        out.flush()

    def _emit(self) -> None:
        elapsed = time.time() - self.family_t0
        eta = ""
        if self.epoch >= 2 and self.epochs:
            pace = elapsed / self.epoch
            remain = pace * max(self.epochs - self.epoch, 0)
            eta = f"  {_fmt_hms(elapsed)} decorrido  ~{_fmt_hms(remain)} restante"
        elif self.epoch >= 1:
            eta = f"  {_fmt_hms(elapsed)} decorrido"
        map_txt = f"  mAP50-95={self.map50_95:.4f}" if self.map50_95 is not None else ""
        line = (
            f"  {self.family}  {self.epoch}/{self.epochs}  "
            f"{_bar(self.epoch, self.epochs)}{map_txt}{eta}{_gpu()}"
        )
        out = stream()
        out.write("\r" + line[:118].ljust(118))
        out.flush()


class CountProgress:
    """Contador [n/total] com taxa e ETA — OCR e extração da pipeline."""

    def __init__(self, label: str, total: int | None):
        self.label = label
        self.total = total
        self.n = 0
        self.t0 = time.time()

    def update(self, n: int, extra: str = "") -> None:
        self.n = n
        elapsed = time.time() - self.t0
        rate = n / elapsed if elapsed > 0 else 0.0
        if self.total:
            remain = (self.total - n) / rate if rate > 0 else 0.0
            bar = _bar(n, self.total, 18)
            tail = f"  {n}/{self.total}  {bar}  {rate:.1f}/s  ~{_fmt_hms(remain)}"
        else:
            tail = f"  {n}  {rate:.1f}/s  {_fmt_hms(elapsed)}"
        if extra:
            tail += f"  {extra}"
        out = stream()
        out.write("\r" + f"{self.label}{tail}"[:118].ljust(118))
        out.flush()

    def done(self, extra: str = "") -> None:
        elapsed = time.time() - self.t0
        write("")
        write(f"{self.label}  {self.n} em {_fmt_hms(elapsed)}" + (f"  {extra}" if extra else ""))
