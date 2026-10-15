# Source

Library code for the pipeline stages. Training CLIs are [`../train/`](../train/). The full run is [`../pipeline.py`](../pipeline.py).

| Folder | Stage |
|---|---|
| [`cropper/`](cropper/) | Perspective warp to 63×88 mm (`python -m cropper`) |
| [`roi/`](roi/) | Text-band detector package |
| [`ocr/`](ocr/) | RapidOCR on those bands (`read_card.py`) |

Weights: [`../models/`](../models/) (export). Training dumps: [`../runs/`](../runs/). Official YOLO downloads: [`../pretrained/`](../pretrained/). Guides: [`../docs/`](../docs/).
