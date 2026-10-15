# Runs

Dumps locais do Ultralytics / da comparação OCR (gitignored): `best.pt` de cada família, `last.pt`, CSVs, plots.

Os pesos que você exporta ou aponta na pipeline **não ficam aqui**. Copie (ou deixe o treino copiar) para [`models/`](../models/README.md):

| Produção | Treino que elege |
|---|---|
| `models/obb-v6.pt` | `runs/detection/obb/obb-v6/` |
| `models/ocr-roi-v3.pt` | `runs/roi/ocr-roi-v3/` |
| — | `runs/ocr/` (só CSVs; RapidOCR não treina `.pt`) |

Relatórios: [`train/detector/`](../train/detector/README.md), [`train/roi/`](../train/roi/README.md), [`train/ocr/`](../train/ocr/README.md).
