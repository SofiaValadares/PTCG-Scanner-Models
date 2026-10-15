# Models (export)

Pesos treinados que a pipeline, o cropper e o `python -m cropper export` usam. Dois arquivos:

| Arquivo | Papel | Origem |
|---|---|---|
| `obb-v6.pt` | Detector de carta (YOLOv26s-OBB, imgsz 960) | `python -m train detector` |
| `ocr-roi-v3.pt` | Faixas de texto name / number / colection (YOLOv8s-OBB, imgsz 800) | `python -m train roi` |

Não há `.pt` de OCR: `python -m train ocr` só compara engines. A produção usa RapidOCR (ONNX vem com o pacote).

`runs/` continua sendo o dump do Ultralytics (todas as famílias, `last.pt`, gráficos). Não suba `runs/` no GitHub — só estes dois pesos (~21 MB + ~22 MB, abaixo do limite de 100 MB sem Git LFS).

Depois de um treino novo, a eleição copia o vencedor para cá de novo.
