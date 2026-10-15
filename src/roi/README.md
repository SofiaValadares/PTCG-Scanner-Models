# ROI — text bands on the cropped card

Three OBB classes on the 63×88 mm crop: `name`, `number`, `colection`. Sweep: **ocr-roi-v3**. Pipeline uses [`models/ocr-roi-v3.pt`](../../models/README.md).

Dataset: [`data/roi/`](../../data/roi/). Reading the pixels is [`../ocr/`](../ocr/).

`python -m train roi` → [`train/roi`](../../train/roi/README.md)
