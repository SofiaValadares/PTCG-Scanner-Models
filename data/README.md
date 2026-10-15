# Data

| Folder | Contents |
|---|---|
| [`catalog/`](catalog/) | Language CSVs for matching (`en` is the default eval catalog) |
| `detection/` | Roboflow card-detector dataset (local, gitignored) |
| `roi/` | Roboflow text-band dataset (local, gitignored) |
| `ocr/` | Roboflow OCR-box dataset (local, gitignored) |
| `identify/` | Pipeline eval photos + OBB labels (local, gitignored) |
| `input/` | Ad-hoc photos for `python pipeline.py --source input` |
| `legacy/` | Old spreadsheets kept for reference |

Unzip the Roboflow YOLOv8 OBB exports into `detection/` and `roi/`. Details: [`docs/detection.md`](../docs/detection.md), [`docs/roi.md`](../docs/roi.md), [`docs/ocr.md`](../docs/ocr.md).
