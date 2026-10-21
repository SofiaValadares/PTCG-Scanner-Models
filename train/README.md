# Train

One folder per training job. The **report** (README + figures) is written in that same folder.

```
train/
├── detector/     python -m train detector     → train/detector/README.md
├── roi/          python -m train roi          → train/roi/README.md
├── ocr/          python -m train ocr          → train/ocr/README.md
└── common/       ranking, plots, progress bar
```

```powershell
python -m train all                 # detector → roi → ocr
python -m train detector
python -m train roi
python -m train ocr
python -m train detector --report-only --pdf
```

`all` stops if a job fails. `--pdf` adds `report.pdf` (gitignored). The elected `.pt` is copied to [`models/`](../models/README.md). Dumps of every family stay in [`runs/`](../runs/). Historical comparisons stay in [`docs/experiments/`](../docs/experiments/). OCR does not write a `.pt`.
