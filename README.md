# PTCG Scanner

End-to-end computer-vision pipeline that **finds Pokémon TCG cards in a photo or PDF**, **warps each one to real card size**, **reads name / number / set**, and **matches the result against a catalog**.

This repository is the experimental code of a bachelor’s thesis (TCC) by **Sofia Valadares**.

License: [CC BY-NC 4.0](LICENSE) — you may study, teach, and build on this work **with credit**. **Commercial use is reserved to the author** (or anyone she authorizes in writing: `sofiav.cav@gmail.com`). Trademarks and third-party libraries: [NOTICE](NOTICE). How to cite: [CITATION.cff](CITATION.cff).

---

## What problem this solves

Photos of binders, tables, or set checklists are messy: cards sit at an angle, several appear in one frame, and the text on the card is small. A catalog lookup only works if you know *which* card you are looking at.

PTCG Scanner splits that into three models plus a catalog step, instead of one opaque “read the card” network:

1. **Detection** — is there a card here, and at what rotation? (YOLO OBB, class `card`; pipeline **obb-v6**)
2. **Crop** — warp that quadrilateral to a **63 mm × 88 mm** portrait image (the physical TCG size), in color. 90° is fixed here; **180°** is chosen in the pipeline with the ROI layout.
3. **OCR** — on the cropped card, find the **name**, **collector number**, and **set name** (when printed) and read them with RapidOCR (Paddle ONNX), including Japanese, Chinese, Korean, and Russian prints.
4. **Catalog** — score the reading against `data/catalog/{lang}/` (default `en`; PkmnCards `setId`s). For evaluation PDFs, cards are paired **in spreadsheet order**.

English catalog names count as a hit for Portuguese prints (e.g. `N's Castle` ↔ `Castelo do N`).

---

## Pipeline (what actually runs)

```
set PDF or photo
    → rasterize pages (PyMuPDF, 200 DPI)
    → YOLO OBB (card boxes; **obb-v6**)
    → perspective warp 63×88 mm @ 300 DPI (744×1039 px; 90° landscape → portrait)
    → ROI OBB on the crop at **0° and 180°** (keep printed-up: name above number)
    → RapidOCR on those bands
    → nearest row in data/catalog/{lang}/ (name, then number, then collection)
    → compare to identify GT (OBB IoU) or the CSV next to the PDF
```

Nothing in the full pipeline **trains** a model. Training is `python -m train all` (or `detector` / `roi` / `ocr` one at a time). The cropper has no trainable weights of its own; it consumes the detector.

Typical settings used in the thesis experiments: card confidence **0.8**, OCR-ROI confidence **0.4**, about **20 seconds per PDF page** on a consumer GPU (detect + crop + OCR).

---

## Repository layout

The **pipeline** and **training CLIs** sit at the repo root. Library code is **`src/`**. Datasets, trained weights, and dumps stay out of `src/`.

```
PTCG Scanner/
├── pipeline.py                photos/PDF → catalog (start here)
├── train/
│   ├── detector/              python -m train detector
│   ├── roi/                   python -m train roi
│   └── ocr/                   python -m train ocr
├── src/                       cropper, roi, ocr (code only)
├── data/                      catalog + local datasets
├── models/                    export weights: obb-v6.pt, ocr-roi-v3.pt
├── pretrained/                official YOLO downloads (gitignored)
├── runs/                      training dumps (gitignored)
├── output/                    pipeline + cropper dumps
└── docs/                      guides + experiment reports
```

| Path | You use it for |
|---|---|
| [`pipeline.py`](pipeline.py) | Full pipeline on `data/identify/` (or `--source input`) |
| [`train/`](train/) | `python -m train all` or `detector` / `roi` / `ocr` |
| [`src/cropper/`](src/cropper/) | `python -m cropper` |
| [`src/roi/`](src/roi/) | Strip detector package; pipeline uses **ocr-roi-v3** |
| [`src/ocr/`](src/ocr/) | `read_card.py` |
| [`data/`](data/README.md) | Catalog, training datasets, and pipeline photos |
| [`models/`](models/README.md) | `obb-v6.pt` (detector) and `ocr-roi-v3.pt` (ROI) — the files to export |
| [`pretrained/`](pretrained/README.md) | Official Ultralytics downloads used when training |
| [`runs/`](runs/README.md) | Local training dumps (all families, plots, `last.pt`) |
| [`output/`](output/README.md) | Relatório: `pdfs/`, `tables/`, `figures/`, `cropper/` |
| [`docs/`](docs/README.md) | Detector metrics, cropper/OCR details, version comparisons |

Training images (`data/detection/{train,valid,test}/`, `data/roi/{train,valid,test}/`), `runs/`, and `pretrained/` stay out of Git. The two production checkpoints in [`models/`](models/README.md) are the ones meant to go on GitHub. Unzip the Roboflow YOLOv8 OBB exports locally. Photos you drop in `data/input/` are also local.

---

## Data

**Catalog.** `data/catalog/{lang}/` holds `cards.csv` (`Name`, `Number`, `Rarity`, `setId`, [PkmnCards](https://pkmncards.com/sets/) codes) and `sets.csv` (set id → display name). The pipeline defaults to `en`. Other folders overlay TCGdex names when that language has the card.

**Eval.** [`pipeline.py`](pipeline.py) defaults to `data/identify/` (class name = `SET number name`, boxes matched by IoU). `--source input` uses photos in `data/input/` and an optional sibling CSV in **row order**.

**Training (local).** Unzip the Roboflow YOLOv8 OBB export of [PTCG Scanner - Detection](https://universe.roboflow.com/pokemon-tcc/ptcg-scanner-detection) into `data/detection/`, and [PTCG Scanner - ROI](https://universe.roboflow.com/pokemon-tcc/ptcg-scanner-roi) into `data/roi/`. Details: [`docs/detection.md`](docs/detection.md), [`docs/roi.md`](docs/roi.md).

---

## How to run

### 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

The last line makes `python -m cropper` work from the repo root. If you skip it, set `$env:PYTHONPATH="src"` in the same PowerShell session.

GPU is strongly recommended (YOLO). RapidOCR runs on ONNX Runtime.

### 2. Weights

Production checkpoints live in [`models/`](models/README.md) (local only — **not** on GitHub):

- Card detector: `models/obb-v6.pt` (pipeline and cropper). Fallback: `runs/…/obb-v6.pt`, then `src/cropper/weights/obb-v5.pt`.
- Text bands: `models/ocr-roi-v3.pt` (pipeline). Fallback: `runs/roi/ocr-roi-v3/`.
- OCR: **no `.pt` of ours**. RapidOCR ships ONNX inside the Python package. `python -m train ocr` only ranks engines.

Official Ultralytics downloads used as training starts go to [`pretrained/`](pretrained/README.md). The cropper looks for **v6**, then v5, then v4, then v3, then `PTCG_CROPPER_WEIGHTS`. Inference size is **960** for v6/v5/v4 and **800** for v3.

### 3. Full pipeline

```powershell
python pipeline.py
python pipeline.py --source input
python pipeline.py --max-images 20
```

Default eval is `data/identify/`. For ad-hoc photos, `--source input` and drop files in `data/input/`.

Identify mode writes [`output/README.md`](output/README.md) (tables + figures, like the training reports), plus CSVs and three debug PDFs. Per-file stage dumps (`pdfs/<name>/00`–`05`) are only written when `SAVE_STAGES` is on (`EVAL_SOURCE = "input"`).

| Output | Meaning |
|---|---|
| `output/README.md` | Pipeline eval report (tables + figures) |
| `output/pdfs/` | `01_obb.pdf`, `03_acertos.pdf`, `03_erros.pdf` |
| `output/tables/` | `pipeline_extract.csv`, `eval`, `metrics`, `errors` |
| `output/figures/` | funnel / ROI / OCR / catalog plots |

The card detector already has YOLO mAP in [`docs/detection.md`](docs/detection.md). The pipeline also reports:

1. **ROI** — strip detector mAP on `data/roi` (if the test split is present), plus how often each band came from YOLO vs the template.
2. **OCR** — name / number / collection accuracy vs the spreadsheet (when present), and mean character error rate (CER).
3. **End-to-end** — catalog hit at that slot, and a funnel: detected → ROI box → OCR field → catalog.

A catalog hit at a given slot is **name (PT or EN) or number** matching the spreadsheet row.

### 4. Crop photos only

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m cropper path\to\photo.jpg --out output\cropper\cards --conf 0.8
```

Each card becomes `*_card_00_0.96.jpg` (color). Add `--preview` to also write a contact sheet `preview.jpg` in `--out`.

To ship the **cropper** (code + detector `.pt`) to another project:

```powershell
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe -m cropper export C:\other-project\ptcg_cropper
```

The weight it copies is `models/obb-v6.pt` (YOLOv26). The ROI model for the full pipeline is `models/ocr-roi-v3.pt` — copy that separately if the other project also reads text; the cropper export is only detect + warp.

### 5. Train models

Each command writes a **README in that training folder** (`train/detector/`, `train/roi/`, `train/ocr/`) with tables of every candidate, compressed graphs of every candidate, and a section on the elected model. `--pdf` adds a small `report.pdf` in the same folder (gitignored — attach it to the thesis; do not put it on GitHub). The terminal shows epoch, mAP, ETA and VRAM on one updating line.

```powershell
$env:PYTHONPATH="src"   # skip if you already ran pip install -e .
python -m train all                   # detector → roi → ocr (hours)
python -m train detector              # obb-v6: YOLOv26 / 12 / 8 / RT-DETR / 11
python -m train roi                   # ocr-roi-v3: YOLOv26 / 12 / 8 / 11
python -m train ocr                   # EasyOCR vs RapidOCR vs Tesseract
python -m train detector --report-only --pdf   # reuse ranking.csv + best.pt
```

1. Unzip the Roboflow YOLOv8 OBB zips into `data/detection/` and `data/roi/`.
2. `python -m train detector` — **obb-v6** keeps the best (YOLOv26s, val mAP50-95 ≈ 0.994). Pipeline and cropper use that winner. Write-up: [`train/detector/`](train/detector/README.md). Earlier: [`docs/experiments/obb-v3-v5`](docs/experiments/obb-v3-v5/README.md).
3. `python -m train roi` — **ocr-roi-v3** on the text bands. Pipeline uses **ocr-roi-v3**.
4. `python -m train ocr` — does **not** train a recognizer. Ranks EasyOCR, RapidOCR/Paddle, and Tesseract on `data/ocr/`.

Label **printed card edges**, not binder plastic, or the crop (and then OCR) will include the sleeve.

---

## Design notes (short)

- **OBB, not axis-aligned boxes**, because cards are rotated on the table and in photos.
- **Full card crop**, not pre-cut text strips: OCR regions stay a template (or a second OBB) on a stable 63×88 mm image.
- **Color crop** at 63×88 mm. OCR converts each text strip to grayscale internally.
- **Catalog restricted to the file’s `setId`** (from a sibling CSV, if present), then number, then name.
- **Order-based evaluation**, not greedy matching, when a ground-truth CSV follows the same order as the page.

Longer write-ups: [`docs/detection.md`](docs/detection.md), [`docs/cropper.md`](docs/cropper.md), [`docs/roi.md`](docs/roi.md), [`docs/ocr.md`](docs/ocr.md).

---

## License and citation

- Academic / educational / research use: **allowed**, with attribution.
- Commercial use (product, SaaS, paid identification, etc.): **not allowed** unless the author agrees in writing.

```
Valadares, Sofia. PTCG Scanner. 2026. CC BY-NC 4.0.
```

This project is **not** affiliated with Nintendo, The Pokémon Company, or Creatures Inc.
