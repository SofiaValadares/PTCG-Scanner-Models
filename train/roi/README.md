# ocr-roi-v3 — varredura de arquiteturas

Relatório gerado por `python -m train roi`. Este arquivo é o artefato **versionado no GitHub**: tabelas de todas as famílias, gráficos comprimidos de todas elas, e uma seção do modelo eleito. O PDF opcional (`report.pdf`) serve para o TCC e **não entra no git**.

Papel: **detector de ROI**. Os pesos de todas as redes ficam em `runs/roi/ocr-roi-v3` e não entram no git. O peso eleito é copiado para `models/ocr-roi-v3.pt`, e é esse arquivo que a pipeline usa.

## O que este modelo faz

Este modelo olha a carta já recortada no tamanho físico, 63×88 mm, e marca três faixas: o nome, o número e o símbolo da coleção. A coleção não vem impressa em toda carta antiga, então essa faixa erra mais e isso não é um defeito da leitura do texto.

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
| Data/hora | 2026-11-08 20:31:41 |
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
| runs_dir | C:\Users\Pichau\OneDrive\Documentos\Repositorios\TCGTradeSegmentacao\runs\roi |
| weights_dir | C:\Users\Pichau\OneDrive\Documentos\Repositorios\TCGTradeSegmentacao\pretrained |

## Receita

- Épocas (teto): `300` · early stop `patience=40` · seed `42`
- Otimizador: `{'optimizer': 'AdamW', 'lr0': 0.001, 'lrf': 0.01}`
- Loss: `{'box': 10.0, 'dfl': 2.0, 'angle': 1.5, 'cls': 0.75}`
- Augmentation: `{'degrees': 10.0, 'flipud': 0.0, 'fliplr': 0.0, 'translate': 0.08, 'scale': 0.25, 'shear': 1.0, 'perspective': 0.0, 'hsv_h': 0.02, 'hsv_s': 0.7, 'hsv_v': 0.5, 'bgr': 0.0, 'mosaic': 0.0, 'mixup': 0.0, 'copy_paste': 0.0, 'erasing': 0.0, 'close_mosaic': 0, 'cos_lr': True}`
- Critério do vencedor: maior **val mAP@0.50:0.95** (4 casas). Diferença menor que `0.002` conta como empate e fica com o menor `val_infer_ms`.
- Receita do ocr-roi-v2: HSV como o detector de carta, geometria fraca, sem flip/mosaic (a imagem já é a carta 63×88 mm).
- RT-DETR fica de fora: não há OBB oficial e as faixas são polígonos rotacionados.
- Não renomeie `colection` no YAML — é o id do Roboflow e o pipeline usa esse nome.
- `colection` falta em muitas cartas (set não impresso). AP mais baixo nessa classe não é falha de OCR.

## Candidatos

| Família | Checkpoint | Tarefa | imgsz | batch |
| --- | --- | --- | --- | --- |
| YOLOv26 | `yolo26s-obb.pt` | obb | 800 | 8 |
| YOLOv12 | `yolo12n-obb.yaml` + `yolo12n.pt` | obb | 800 | 8 |
| YOLOv8 | `yolov8s-obb.pt` | obb | 800 | 8 |
| YOLOv11 | `yolo11s-obb.pt` | obb | 800 | 8 |

## Ranking — validação (todas as famílias)

| winner | family | model | task | imgsz | batch | epochs | hours | val_precision | val_recall | val_f1 | val_map50 | val_map75 | val_map50_95 | val_infer_ms | val_colection_map50_95 | val_name_map50_95 | val_number_map50_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ← | YOLOv8 | yolov8s-obb.pt | obb | 800 | 8 | 123 | 0.0030 | 0.9973 | 0.9959 | 0.9966 | 0.9938 | 0.9930 | 0.9654 | 7.0800 | 0.9724 | 0.9653 | 0.9586 |
|  | YOLOv26 | yolo26s-obb.pt | obb | 800 | 8 | 80 | 0.0590 | 0.9961 | 0.9988 | 0.9975 | 0.9945 | 0.9894 | 0.9645 | 7.1282 | 0.9738 | 0.9607 | 0.9589 |
|  | YOLOv11 | yolo11s-obb.pt | obb | 800 | 8 | 159 | 0.0030 | 0.9953 | 0.9973 | 0.9963 | 0.9934 | 0.9886 | 0.9622 | 6.6530 | 0.9734 | 0.9589 | 0.9544 |
|  | YOLOv12 | yolo12n-obb.yaml | obb | 800 | 8 | 257 | 0.0030 | 0.9972 | 0.9989 | 0.9980 | 0.9944 | 0.9873 | 0.9068 | 5.0959 | 0.9320 | 0.8748 | 0.9137 |

## Ranking — teste (todas as famílias)

| winner | family | test_precision | test_recall | test_f1 | test_map50 | test_map75 | test_map50_95 | test_infer_ms | test_colection_map50_95 | test_name_map50_95 | test_number_map50_95 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ← | YOLOv8 | 0.9983 | 0.9995 | 0.9989 | 0.9950 | 0.9909 | 0.9680 | 6.9219 | 0.9799 | 0.9694 | 0.9546 |
|  | YOLOv26 | 0.9993 | 1.0000 | 0.9996 | 0.9950 | 0.9910 | 0.9719 | 6.7398 | 0.9816 | 0.9758 | 0.9582 |
|  | YOLOv11 | 0.9993 | 0.9977 | 0.9985 | 0.9950 | 0.9877 | 0.9640 | 6.7876 | 0.9835 | 0.9719 | 0.9365 |
|  | YOLOv12 | 1.0000 | 0.9977 | 0.9988 | 0.9950 | 0.9800 | 0.9068 | 4.8043 | 0.9390 | 0.8819 | 0.8993 |

## Modelo eleito

**YOLOv8** (`yolov8`, checkpoint `yolov8s-obb.pt`).

A rede eleita é a que teve a maior mAP50-95 na validação. Ela é a que a pipeline carrega. As outras permanecem na tabela para mostrar a margem: uma diferença de poucos milésimos não troca o modelo.

- val mAP50-95 = **0.9654** · mAP75 = 0.9930 · F1 = 0.997 · 7.1 ms
- test mAP50-95 = **0.9680** · mAP75 = 0.9909 · F1 = 0.999
- Peso copiado para `models/ocr-roi-v3.pt`.

A eleição usa só o val. O test confirma; não escolhe o modelo.

## Gráficos de todas as famílias

Cada família tem curvas de treino, PR/F1 e lotes de validação. As imagens abaixo foram redimensionadas (lado máximo 900 px) e as fotos JPEG comprimidas para o GitHub não repetir o problema dos notebooks pesados.

### YOLOv26

**Precision–Recall**

![Precision–Recall — YOLOv26](figures/yolo26/BoxPR_curve.png)

**F1 vs confiança**

![F1 vs confiança — YOLOv26](figures/yolo26/BoxF1_curve.png)

**Precision vs confiança**

![Precision vs confiança — YOLOv26](figures/yolo26/BoxP_curve.png)

**Recall vs confiança**

![Recall vs confiança — YOLOv26](figures/yolo26/BoxR_curve.png)

**Matriz de confusão**

![Matriz de confusão — YOLOv26](figures/yolo26/confusion_matrix.png)

**Matriz de confusão (normalizada)**

![Matriz de confusão (normalizada) — YOLOv26](figures/yolo26/confusion_matrix_normalized.png)

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

**Matriz de confusão**

![Matriz de confusão — YOLOv12](figures/yolo12/confusion_matrix.png)

**Matriz de confusão (normalizada)**

![Matriz de confusão (normalizada) — YOLOv12](figures/yolo12/confusion_matrix_normalized.png)

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

### YOLOv8 ← eleito

**Precision–Recall**

![Precision–Recall — YOLOv8](figures/yolov8/BoxPR_curve.png)

**F1 vs confiança**

![F1 vs confiança — YOLOv8](figures/yolov8/BoxF1_curve.png)

**Precision vs confiança**

![Precision vs confiança — YOLOv8](figures/yolov8/BoxP_curve.png)

**Recall vs confiança**

![Recall vs confiança — YOLOv8](figures/yolov8/BoxR_curve.png)

**Matriz de confusão**

![Matriz de confusão — YOLOv8](figures/yolov8/confusion_matrix.png)

**Matriz de confusão (normalizada)**

![Matriz de confusão (normalizada) — YOLOv8](figures/yolov8/confusion_matrix_normalized.png)

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

### YOLOv11

**Precision–Recall**

![Precision–Recall — YOLOv11](figures/yolo11/BoxPR_curve.png)

**F1 vs confiança**

![F1 vs confiança — YOLOv11](figures/yolo11/BoxF1_curve.png)

**Precision vs confiança**

![Precision vs confiança — YOLOv11](figures/yolo11/BoxP_curve.png)

**Recall vs confiança**

![Recall vs confiança — YOLOv11](figures/yolo11/BoxR_curve.png)

**Matriz de confusão**

![Matriz de confusão — YOLOv11](figures/yolo11/confusion_matrix.png)

**Matriz de confusão (normalizada)**

![Matriz de confusão (normalizada) — YOLOv11](figures/yolo11/confusion_matrix_normalized.png)

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
