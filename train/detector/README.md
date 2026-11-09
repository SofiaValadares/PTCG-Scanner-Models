# obb-v6 — varredura de arquiteturas

Relatório gerado por `python -m train detector`. Este arquivo é o artefato **versionado no GitHub**: tabelas de todas as famílias, gráficos comprimidos de todas elas, e uma seção do modelo eleito. O PDF opcional (`report.pdf`) serve para o TCC e **não entra no git**.

Papel: **detector de carta**. Os pesos de todas as redes ficam em `runs/detection/obb/obb-v6` e não entram no git. O peso eleito é copiado para `models/obb-v6.pt`, e é esse arquivo que a pipeline usa.

## O que este modelo faz

Este modelo olha uma foto e marca cada carta. A carta pode estar torta na mesa, então a caixa é um retângulo que gira junto com ela. O treino mostra fotos em que uma pessoa já desenhou essas caixas e ajusta a rede até as caixas dela coincidirem.

As fotos são divididas em três grupos. O treino ensina. A validação escolhe qual rede fica. O teste só confere, no fim, se o modelo não decorou as fotos em que aprendeu. O teste não participa da escolha.

O treino pode ir até 300 passadas pelas fotos. Se a validação ficar 40 passadas sem melhorar, ele para. As fotos de treino são giradas, deslocadas e têm a cor alterada, para o modelo não depender de uma iluminação só. Montar um mosaico com várias fotos, ou misturar duas fotos, fica desligado: a carta precisa continuar um retângulo inteiro.

## Como ler a nota

- **Precisão**: das caixas que o modelo desenhou, a fração que cobre um objeto anotado.
- **Recall**: dos objetos anotados, a fração que o modelo encontrou.
- **F1**: um meio-termo entre as duas. 1,00 seria achar todos e não inventar nenhum.
- **mAP50**: a caixa precisa cobrir pelo menos metade do objeto. É uma nota generosa.
- **mAP75**: a caixa precisa cobrir pelo menos 75%. Exige um encaixe melhor.
- **mAP50-95**: a mesma conta repetida de 50% até 95% de cobertura. É a nota que elege o modelo. Mais alta é melhor.
- **ms**: tempo médio para processar uma imagem na validação. Só desempata quando a mAP50-95 fica igual até a quarta casa, dentro da tolerância.

## Ambiente

| Item | Valor |
| --- | --- |
| Data/hora | 2026-11-08 20:31:38 |
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
| cwd | C:\Users\Pichau\OneDrive\Documentos\Repositorios\TCGTradeSegmentacao |
| runs_dir | C:\Users\Pichau\OneDrive\Documentos\Repositorios\TCGTradeSegmentacao\runs\detection |
| weights_dir | C:\Users\Pichau\OneDrive\Documentos\Repositorios\TCGTradeSegmentacao\pretrained |

## Receita

- Épocas (teto): `300` · early stop `patience=40` · seed `42`
- Otimizador: `{'optimizer': 'AdamW', 'lr0': 0.001, 'lrf': 0.01}`
- Loss: `{'box': 12.0, 'dfl': 2.5, 'angle': 2.0}`
- Augmentation: `{'degrees': 30.0, 'flipud': 0.5, 'fliplr': 0.5, 'translate': 0.1, 'scale': 0.4, 'shear': 2.0, 'perspective': 0.0, 'hsv_h': 0.02, 'hsv_s': 0.7, 'hsv_v': 0.5, 'bgr': 0.0, 'mosaic': 0.0, 'mixup': 0.0, 'copy_paste': 0.0, 'erasing': 0.0, 'close_mosaic': 0, 'cos_lr': True}`
- Critério do vencedor: maior **val mAP@0.50:0.95** (4 casas). Diferença menor que `0.002` conta como empate e fica com o menor `val_infer_ms`.
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
| ← | YOLOv26 | yolo26s-obb.pt | obb | 960 | 4 | 103 | 0.0620 | 0.9999 | 1.0000 | 1.0000 | 0.9950 | 0.9950 | 0.9950 | 11.27 |
|  | YOLOv11 | yolo11s-obb.pt | obb | 960 | 4 | 79 | 0.0020 | 0.9980 | 0.9943 | 0.9962 | 0.9950 | 0.9950 | 0.9940 | 11.34 |
|  | YOLOv8 | yolov8s-obb.pt | obb | 960 | 4 | 104 | 0.0390 | 0.9980 | 0.9962 | 0.9971 | 0.9950 | 0.9950 | 0.9939 | 11.14 |
|  | YOLOv12 | yolo12n-obb.yaml | obb | 960 | 4 | 120 | 0.0430 | 0.9991 | 0.9943 | 0.9967 | 0.9950 | 0.9948 | 0.9894 | 8.0855 |
|  | RT-DETR-L | rtdetr-l.pt | detect | 640 | 4 | 120 | 0.1000 | 0.9937 | 0.9849 | 0.9893 | 0.9921 | 0.9921 | 0.9595 | 22.89 |

## Ranking — teste (todas as famílias)

| winner | family | test_precision | test_recall | test_f1 | test_map50 | test_map75 | test_map50_95 | test_infer_ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ← | YOLOv26 | 1.0000 | 1.0000 | 1.0000 | 0.9950 | 0.9950 | 0.9949 | 13.75 |
|  | YOLOv11 | 1.0000 | 0.9999 | 1.0000 | 0.9950 | 0.9950 | 0.9939 | 11.98 |
|  | YOLOv8 | 0.9996 | 1.0000 | 0.9998 | 0.9950 | 0.9950 | 0.9931 | 12.11 |
|  | YOLOv12 | 0.9895 | 1.0000 | 0.9947 | 0.9949 | 0.9949 | 0.9860 | 10.28 |
|  | RT-DETR-L | 0.9941 | 0.9951 | 0.9946 | 0.9950 | 0.9950 | 0.9478 | 23.94 |

## Modelo eleito

**YOLOv26** (`yolo26`, checkpoint `yolo26s-obb.pt`).

A rede eleita é a que teve a maior mAP50-95 na validação. Ela é a que a pipeline carrega. As outras permanecem na tabela para mostrar a margem: uma diferença de poucos milésimos não troca o modelo.

- val mAP50-95 = **0.9950** · mAP75 = 0.9950 · F1 = 1.000 · 11.3 ms
- test mAP50-95 = **0.9949** · mAP75 = 0.9950 · F1 = 1.000
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
