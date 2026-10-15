# obb-v6 — varredura de arquiteturas

Relatório gerado por `python -m train detector`. Este arquivo é o artefato **versionado no GitHub**: tabelas de todas as famílias, gráficos comprimidos de todas elas, e uma seção do modelo eleito. O PDF opcional (`report.pdf`) serve para o TCC e **não entra no git**.

Papel: **detector de carta**. Runs (gitignored): `runs/detection/obb/obb-v6`. Peso de produção: `models/obb-v6.pt`.

## Ambiente

| Item | Valor |
| --- | --- |
| Data/hora | 2026-10-15 14:16:11 |
| SO | Windows 11 |
| Python | 3.14.7 |
| PyTorch | 2.11.0+cu128 |
| Ultralytics | 8.4.173 |
| CUDA disponível | True |
| CUDA (compilado) | 12.8 |
| GPU | NVIDIA GeForce RTX 5060 |
| VRAM | 8.0 GB |
| Dispositivo de treino | 0 |
| cuDNN | True |
| cwd | C:\Users\Pichau\Documents\Repositorios\TCGTradeSegmentacao |
| runs_dir | C:\Users\Pichau\Documents\Repositorios\TCGTradeSegmentacao\runs\detection |
| weights_dir | C:\Users\Pichau\Documents\Repositorios\TCGTradeSegmentacao\models |

## Receita

- Épocas (teto): `300` · early stop `patience=40` · seed `42`
- Otimizador: `{'optimizer': 'AdamW', 'lr0': 0.001, 'lrf': 0.01}`
- Loss: `{'box': 12.0, 'dfl': 2.5, 'angle': 2.0}`
- Augmentation: `{'degrees': 30.0, 'flipud': 0.5, 'fliplr': 0.5, 'translate': 0.1, 'scale': 0.4, 'shear': 2.0, 'perspective': 0.0, 'hsv_h': 0.02, 'hsv_s': 0.7, 'hsv_v': 0.5, 'bgr': 0.0, 'mosaic': 0.0, 'mixup': 0.0, 'copy_paste': 0.0, 'erasing': 0.0, 'close_mosaic': 0, 'cos_lr': True}`
- Critério do vencedor: maior **val mAP@0.50:0.95** (4 casas). Só se empatar nessa precisão vai para o menor `val_infer_ms` (o tempo de inferência varia entre `val()`).
- Receita de borda do obb-v5: mosaic/mixup/erasing desligados (o warp 63×88 mm precisa do retângulo da carta).
- RT-DETR-L não tem OBB oficial no Ultralytics: treina AABB convertido das mesmas anotações e não entra no cropper.
- YOLOv12 OBB usa o yaml + backbone `yolo12n.pt` (não há checkpoint OBB oficial).

## Candidatos

| Família | Checkpoint | Tarefa | imgsz | batch |
| --- | --- | --- | --- | --- |
| YOLOv26 | `yolo26s-obb.pt` | obb | 960 | 4 |
| YOLOv12 | `yolo12n-obb.yaml` + `yolo12n.pt` | obb | 960 | 4 |
| YOLOv8 | `yolov8s-obb.pt` | obb | 960 | 4 |
| RT-DETR-L | `rtdetr-l.pt` | detect | 640 | 4 |
| YOLOv11 | `yolo11s-obb.pt` | obb | 960 | 4 |

## Ranking — validação (todas as famílias)

| winner | family | model | task | imgsz | batch | epochs | hours | val_precision | val_recall | val_f1 | val_map50 | val_map75 | val_map50_95 | val_infer_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ← | YOLOv26 | yolo26s-obb.pt | obb | 960 | 4 | 103 | 0.0420 | 0.9972 | 1.0000 | 0.9986 | 0.9950 | 0.9950 | 0.9938 | 8.4771 |
|  | YOLOv8 | yolov8s-obb.pt | obb | 960 | 4 | 104 | 0.0350 | 0.9946 | 1.0000 | 0.9973 | 0.9950 | 0.9950 | 0.9937 | 8.1334 |
|  | YOLOv11 | yolo11s-obb.pt | obb | 960 | 4 | 79 | 0.0020 | 0.9985 | 1.0000 | 0.9993 | 0.9949 | 0.9949 | 0.9915 | 7.6103 |
|  | YOLOv12 | yolo12n-obb.yaml | obb | 960 | 4 | 120 | 0.0330 | 0.9885 | 0.9974 | 0.9929 | 0.9946 | 0.9946 | 0.9818 | 6.2776 |
|  | RT-DETR-L | rtdetr-l.pt | detect | 640 | 4 | 120 | 0.0810 | 0.9845 | 0.9969 | 0.9907 | 0.9938 | 0.9932 | 0.9390 | 16.17 |

## Ranking — teste (todas as famílias)

| winner | family | test_precision | test_recall | test_f1 | test_map50 | test_map75 | test_map50_95 | test_infer_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ← | YOLOv26 | 0.9999 | 1.0000 | 1.0000 | 0.9950 | 0.9950 | 0.9939 | 8.8302 |
|  | YOLOv8 | 1.0000 | 0.9945 | 0.9973 | 0.9950 | 0.9950 | 0.9926 | 8.4943 |
|  | YOLOv11 | 0.9999 | 0.9925 | 0.9962 | 0.9950 | 0.9950 | 0.9927 | 7.9800 |
|  | YOLOv12 | 0.9949 | 0.9950 | 0.9950 | 0.9947 | 0.9937 | 0.9791 | 6.8280 |
|  | RT-DETR-L | 0.9950 | 0.9924 | 0.9937 | 0.9904 | 0.9904 | 0.9373 | 15.63 |

## Modelo eleito

**YOLOv26** (`yolo26`, checkpoint `yolo26s-obb.pt`).

- val mAP50-95 = **0.9938** · mAP75 = 0.9950 · F1 = 0.999 · 8.5 ms
- test mAP50-95 = **0.9939** · mAP75 = 0.9950 · F1 = 1.000
- Peso copiado para `models/obb-v6.pt`.

A eleição usa só o val. O test confirma; não escolhe o modelo.

## Gráficos de todas as famílias

Cada família tem curvas de treino, PR/F1 e lotes de validação. As imagens abaixo foram redimensionadas (lado máximo 900 px) e as fotos JPEG comprimidas para o GitHub não repetir o problema dos notebooks pesados.

### YOLOv26 ← eleito

**Precision–Recall**

![Precision–Recall — YOLOv26](figures/yolo26/BoxPR_curve.png)

**F1 vs confiança**

![F1 vs confiança — YOLOv26](figures/yolo26/BoxF1_curve.png)

**Precision vs confiança**

![Precision vs confiança — YOLOv26](figures/yolo26/BoxP_curve.png)

**Recall vs confiança**

![Recall vs confiança — YOLOv26](figures/yolo26/BoxR_curve.png)

**Lote de validação (labels) — val_batch0_labels.jpg**

![Lote de validação (labels) — val_batch0_labels.jpg — YOLOv26](figures/yolo26/val_batch0_labels.jpg)

**Lote de validação (labels) — val_batch1_labels.jpg**

![Lote de validação (labels) — val_batch1_labels.jpg — YOLOv26](figures/yolo26/val_batch1_labels.jpg)

**Lote de validação (labels) — val_batch2_labels.jpg**

![Lote de validação (labels) — val_batch2_labels.jpg — YOLOv26](figures/yolo26/val_batch2_labels.jpg)

**Lote de validação (predições) — val_batch0_pred.jpg**

![Lote de validação (predições) — val_batch0_pred.jpg — YOLOv26](figures/yolo26/val_batch0_pred.jpg)

**Lote de validação (predições) — val_batch1_pred.jpg**

![Lote de validação (predições) — val_batch1_pred.jpg — YOLOv26](figures/yolo26/val_batch1_pred.jpg)

**Lote de validação (predições) — val_batch2_pred.jpg**

![Lote de validação (predições) — val_batch2_pred.jpg — YOLOv26](figures/yolo26/val_batch2_pred.jpg)

### YOLOv12

**Precision–Recall**

![Precision–Recall — YOLOv12](figures/yolo12/BoxPR_curve.png)

**F1 vs confiança**

![F1 vs confiança — YOLOv12](figures/yolo12/BoxF1_curve.png)

**Precision vs confiança**

![Precision vs confiança — YOLOv12](figures/yolo12/BoxP_curve.png)

**Recall vs confiança**

![Recall vs confiança — YOLOv12](figures/yolo12/BoxR_curve.png)

**Lote de validação (labels) — val_batch0_labels.jpg**

![Lote de validação (labels) — val_batch0_labels.jpg — YOLOv12](figures/yolo12/val_batch0_labels.jpg)

**Lote de validação (labels) — val_batch1_labels.jpg**

![Lote de validação (labels) — val_batch1_labels.jpg — YOLOv12](figures/yolo12/val_batch1_labels.jpg)

**Lote de validação (labels) — val_batch2_labels.jpg**

![Lote de validação (labels) — val_batch2_labels.jpg — YOLOv12](figures/yolo12/val_batch2_labels.jpg)

**Lote de validação (predições) — val_batch0_pred.jpg**

![Lote de validação (predições) — val_batch0_pred.jpg — YOLOv12](figures/yolo12/val_batch0_pred.jpg)

**Lote de validação (predições) — val_batch1_pred.jpg**

![Lote de validação (predições) — val_batch1_pred.jpg — YOLOv12](figures/yolo12/val_batch1_pred.jpg)

**Lote de validação (predições) — val_batch2_pred.jpg**

![Lote de validação (predições) — val_batch2_pred.jpg — YOLOv12](figures/yolo12/val_batch2_pred.jpg)

### YOLOv8

**Precision–Recall**

![Precision–Recall — YOLOv8](figures/yolov8/BoxPR_curve.png)

**F1 vs confiança**

![F1 vs confiança — YOLOv8](figures/yolov8/BoxF1_curve.png)

**Precision vs confiança**

![Precision vs confiança — YOLOv8](figures/yolov8/BoxP_curve.png)

**Recall vs confiança**

![Recall vs confiança — YOLOv8](figures/yolov8/BoxR_curve.png)

**Lote de validação (labels) — val_batch0_labels.jpg**

![Lote de validação (labels) — val_batch0_labels.jpg — YOLOv8](figures/yolov8/val_batch0_labels.jpg)

**Lote de validação (labels) — val_batch1_labels.jpg**

![Lote de validação (labels) — val_batch1_labels.jpg — YOLOv8](figures/yolov8/val_batch1_labels.jpg)

**Lote de validação (labels) — val_batch2_labels.jpg**

![Lote de validação (labels) — val_batch2_labels.jpg — YOLOv8](figures/yolov8/val_batch2_labels.jpg)

**Lote de validação (predições) — val_batch0_pred.jpg**

![Lote de validação (predições) — val_batch0_pred.jpg — YOLOv8](figures/yolov8/val_batch0_pred.jpg)

**Lote de validação (predições) — val_batch1_pred.jpg**

![Lote de validação (predições) — val_batch1_pred.jpg — YOLOv8](figures/yolov8/val_batch1_pred.jpg)

**Lote de validação (predições) — val_batch2_pred.jpg**

![Lote de validação (predições) — val_batch2_pred.jpg — YOLOv8](figures/yolov8/val_batch2_pred.jpg)

### RT-DETR-L

**Precision–Recall**

![Precision–Recall — RT-DETR-L](figures/rtdetrv2/BoxPR_curve.png)

**F1 vs confiança**

![F1 vs confiança — RT-DETR-L](figures/rtdetrv2/BoxF1_curve.png)

**Precision vs confiança**

![Precision vs confiança — RT-DETR-L](figures/rtdetrv2/BoxP_curve.png)

**Recall vs confiança**

![Recall vs confiança — RT-DETR-L](figures/rtdetrv2/BoxR_curve.png)

**Lote de validação (labels) — val_batch0_labels.jpg**

![Lote de validação (labels) — val_batch0_labels.jpg — RT-DETR-L](figures/rtdetrv2/val_batch0_labels.jpg)

**Lote de validação (labels) — val_batch1_labels.jpg**

![Lote de validação (labels) — val_batch1_labels.jpg — RT-DETR-L](figures/rtdetrv2/val_batch1_labels.jpg)

**Lote de validação (labels) — val_batch2_labels.jpg**

![Lote de validação (labels) — val_batch2_labels.jpg — RT-DETR-L](figures/rtdetrv2/val_batch2_labels.jpg)

**Lote de validação (predições) — val_batch0_pred.jpg**

![Lote de validação (predições) — val_batch0_pred.jpg — RT-DETR-L](figures/rtdetrv2/val_batch0_pred.jpg)

**Lote de validação (predições) — val_batch1_pred.jpg**

![Lote de validação (predições) — val_batch1_pred.jpg — RT-DETR-L](figures/rtdetrv2/val_batch1_pred.jpg)

**Lote de validação (predições) — val_batch2_pred.jpg**

![Lote de validação (predições) — val_batch2_pred.jpg — RT-DETR-L](figures/rtdetrv2/val_batch2_pred.jpg)

### YOLOv11

**Precision–Recall**

![Precision–Recall — YOLOv11](figures/yolo11/BoxPR_curve.png)

**F1 vs confiança**

![F1 vs confiança — YOLOv11](figures/yolo11/BoxF1_curve.png)

**Precision vs confiança**

![Precision vs confiança — YOLOv11](figures/yolo11/BoxP_curve.png)

**Recall vs confiança**

![Recall vs confiança — YOLOv11](figures/yolo11/BoxR_curve.png)

**Lote de validação (labels) — val_batch0_labels.jpg**

![Lote de validação (labels) — val_batch0_labels.jpg — YOLOv11](figures/yolo11/val_batch0_labels.jpg)

**Lote de validação (labels) — val_batch1_labels.jpg**

![Lote de validação (labels) — val_batch1_labels.jpg — YOLOv11](figures/yolo11/val_batch1_labels.jpg)

**Lote de validação (labels) — val_batch2_labels.jpg**

![Lote de validação (labels) — val_batch2_labels.jpg — YOLOv11](figures/yolo11/val_batch2_labels.jpg)

**Lote de validação (predições) — val_batch0_pred.jpg**

![Lote de validação (predições) — val_batch0_pred.jpg — YOLOv11](figures/yolo11/val_batch0_pred.jpg)

**Lote de validação (predições) — val_batch1_pred.jpg**

![Lote de validação (predições) — val_batch1_pred.jpg — YOLOv11](figures/yolo11/val_batch1_pred.jpg)

**Lote de validação (predições) — val_batch2_pred.jpg**

![Lote de validação (predições) — val_batch2_pred.jpg — YOLOv11](figures/yolo11/val_batch2_pred.jpg)
