from __future__ import annotations

import sys

JOBS = ("detector", "roi", "ocr")

USAGE = """\
Treinos do PTCG Scanner.

  python -m train all        [--report-only] [--pdf] [--no-plots]
  python -m train detector   [--report-only] [--pdf] [--no-plots]
  python -m train roi        [--report-only] [--pdf] [--no-plots]
  python -m train ocr        [--report-only] [--pdf] [--max-boxes N]

`all` roda detector -> roi -> ocr, nesta ordem (cada um pode levar horas).
Na raiz do repo, se `pip install -e .` não foi feito:

  $env:PYTHONPATH="src"

Cada comando grava o relatório na pasta do treino (train/detector, train/roi, train/ocr).
--pdf gera report.pdf no mesmo diretório (gitignored; use no TCC).
A barra de progresso (época, mAP, ETA, VRAM) atualiza na mesma linha do terminal.
"""

_OCR_SKIP = {"--no-plots", "--no-skip-existing"}


def _job_main(cmd: str):
    if cmd in {"detector", "obb", "obb-v6"}:
        from train.detector import main as run

        return "detector", run
    if cmd in {"roi", "ocr-roi", "ocr-roi-v3"}:
        from train.roi import main as run

        return "roi", run
    if cmd in {"ocr", "ocr-cmp"}:
        from train.ocr import main as run

        return "ocr", run
    return None, None


def _filter_argv(job: str, argv: list[str]) -> list[str]:
    if job != "ocr":
        return argv
    return [a for a in argv if a not in _OCR_SKIP]


def _run_all(argv: list[str]) -> int:
    if argv and argv[0] in {"-h", "--help"}:
        print(USAGE, end="")
        return 0
    for index, job in enumerate(JOBS, start=1):
        print(f"\n{'=' * 64}\n[{index}/{len(JOBS)}] {job}\n{'=' * 64}\n", flush=True)
        _, run = _job_main(job)
        code = run(_filter_argv(job, argv))
        if code:
            print(f"\nParou em `{job}` (código {code}).", flush=True)
            return code
    print("\nOs três treinos terminaram.", flush=True)
    return 0


def main() -> int:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print(USAGE, end="")
        return 0
    cmd, rest = sys.argv[1], sys.argv[2:]
    if cmd in {"all", "*"}:
        return _run_all(rest)
    job, run = _job_main(cmd)
    if run is None:
        print(f"Comando desconhecido: {cmd}\n")
        print(USAGE, end="")
        return 2
    return run(_filter_argv(job, rest))


if __name__ == "__main__":
    raise SystemExit(main())
