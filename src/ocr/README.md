# OCR — reading the bands

RapidOCR (Paddle ONNX) on strips whose boxes already exist.

- Pipeline: `read_card.py`. **0°/180°** is chosen in `pipeline.py` before this file runs.
- Engine ranking: `python -m train ocr` on [`data/ocr/`](../../data/ocr/).

Docs: [`docs/ocr.md`](../../docs/ocr.md)
