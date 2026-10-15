# OCR — reading the bands

The ROI detector ([`roi.md`](roi.md)) already boxed `name` / `number` / `colection` on a 63×88 mm crop. This stage **reads** those strips. Production uses **RapidOCR** (Paddle ONNX) — winner of `python -m train ocr`. We do not train a recognizer.

Comparison CLI: `python -m train ocr`. Report: [`train/ocr/README.md`](../train/ocr/README.md).

Code: `src/ocr/read_card.py`.

## Ground truth for engine comparison

[`data/ocr/`](../data/ocr/) (Roboflow [ptcg-scanner-ocr](https://universe.roboflow.com/pokemon-tcc/ptcg-scanner-ocr)) labels each OBB with a class whose **name is the printed text** (`Scatterbug`, `005-66`, `TEF`, `--`, …). That is not a 3-class detector problem: there are hundreds of distinct strings. The comparison therefore **does not** report mAP.

The evaluation warps each annotated quad (no ROI YOLO in the loop), infers the field type from the GT string only to pick allowlist/`_clean`, and scores three engines: EasyOCR, **RapidOCR (Paddle ONNX)**, and Tesseract. RapidOCR won (`exact_clean` 81% vs 54% / 30%) and is what `src/ocr/read_card.py` uses.

Printed names exist in every language folder under [`data/catalog/`](../data/catalog/README.md): Latin (en, fr, es, it, pt, pt-pt, de, nl, pl, id), Japanese (`ja`), Chinese (`zh-cn`, `zh-tw`), Korean (`ko`), and Russian (`ru`). `clean_name` / catalog `fold` keep those scripts. RapidOCR PP-OCRv6 covers Latin + Chinese; a **Japan PP-OCRv4** rec runs on the name band when `ja/` exists (Hangul/Cyrillic use PP-OCRv5). The pipeline loads **all** catalog languages so a kana name matches `ja/` instead of a Latin card. Report PDFs use Yu Gothic / Meiryo so Japanese is not tofu.

Illegible classes (`--`, `-----ex`, …) are reported separately and **do not** enter the ranking.

## How it reads (pipeline)

[`pipeline.py`](../pipeline.py) warps each card to 63×88 mm (the cropper already turns **90°** landscape into portrait). It then runs the ROI detector at **0° and 180°** and keeps the crop where the `name` band is above `number` (`orient_deg` in the extract CSV). RapidOCR runs on that crop (PP-OCRv6 for Latin/Japanese/Chinese; extra Hangul/Cyrillic rec on the **name** band when `ko/` or `ru/` exist in the catalog).

OCR uses detector boxes; if `name`/`number` is missing it falls back to a fixed template (which assumes printed-up layout). Missing `collection` stays empty (no OCR on a template collection band). Collector numbers accept `-` or `/` (`005-66` = `5/66`).

## Metrics (ranking)

Primary: **normalized exact match** after `read_card` cleaners (`exact_clean`) on readable boxes.

Tie-break (difference under 1 percentage point): mean **CER** on folded strings.

Then: latency (ms/box).

Also reported, not used to pick a winner: raw exact match, WER on names, accuracy by inferred field, empty-prediction rate, empty rate on illegible boxes.

Tables: `runs/ocr/ocr-cmp-v1_boxes.csv` and `ocr-cmp-v1_summary.csv` (gitignored).

The old 29-card spreadsheet (`data/legacy/cards_read.csv`) mixed ROI detection error with OCR. It is no longer the ranking protocol.

## Pipeline metrics

[`pipeline.py`](../pipeline.py) does not retrain. After the 63×88 mm crop it orients **0°/180°** with the ROI layout, then RapidOCR. Default eval is [`data/identify/`](../data/identify/) (class name = `SET number name`): OBB IoU matching, then OCR field accuracy / CER and catalog hit. `--source input` still uses sibling CSVs in `data/input/`.

Part of [PTCG Scanner](../README.md). License: [CC BY-NC 4.0](../LICENSE).
