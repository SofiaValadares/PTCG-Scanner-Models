# Catálogo — pesos da decisão

Relatório gerado por `python -m train catalog`. Não treina rede: compara pesos da escolha da carta no catálogo em cima das leituras já gravadas.

O peso eleito é o de `pipeline.py` e o da API do site (`ptcg_api/catalog.py`).

## Ambiente

| Item | Valor |
| --- | --- |
| Data/hora | 2026-11-06 15:51:10 |
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
| runs_dir | C:\Users\Pichau\Documents\Repositorios\TCGTradeSegmentacao\runs\catalog |
| weights_dir | C:\Users\Pichau\Documents\Repositorios\TCGTradeSegmentacao\pretrained |

## Protocolo

- Fonte das leituras reais: `output/tables/pipeline_eval.csv` · split `identify/test`.
- Cada linha junta o texto lido (`ocr_name`, `ocr_number`, `ocr_collection`) à carta do catálogo com o mesmo número e a mesma coleção.
- Acerto: o primeiro lugar tem o mesmo número e a mesma coleção da anotação. Nome em português ou inglês não entra.
- Sem número de colecionador na leitura, a carta conta como erro.
- O arquivo não guarda `sample_id` nem o nome da foto.
- A linha sintética é uma carta do catálogo que não estava na leitura. O erro copia o padrão real: nome quase igual, número exato na maior parte, e o resto sem número ou com o total ou o colecionador deslocado.
- `ocr_collection` sintético fica vazio antes de SVI. De Scarlet & Violet em diante o código é texto, e a leitura repete o OCR: vazio, o código, o código com PT ou EN, ou uma letra trocada.
- A grade cruza todos os eixos, do zero até um peso alto: name_w (0, 40, 100, 400, 1000); name_exact (0, 10, 40); number_w (0, 40, 110, 400); number_near (0, 20, 80); total_w (0, 30, 120); set_w (0, 40, 150, 400).
- Depois da grade, cada peso do eleito anda de 10 em 10, de 20 abaixo a 20 acima.
- `atual` é o peso da pipeline e do site. `anterior` é o peso que estava na pipeline antes desta eleição. `nome_alto` é a conta antiga, com nome até 1000.
- O veredito elege o maior acerto nas leituras reais. Empate prefere o maior peso de número e, depois, o de coleção. As sintéticas entram no relatório ao lado, e não decidem o peso.

## O arquivo

| | Reais | Sintéticas | Todas |
| --- | --- | --- | --- |
| Cartas | 2677 | 6058 | 8735 |
| No `cards.csv` | 2677 | 6058 | 8735 |
| De SVI em diante | 974 | 3700 | 4674 |
| Com coleção lida | 814 | 3122 | 3936 |
| Com número lido | 2230 | 5009 | 7239 |
| Carta certa entre os candidatos | 2492/2677 (93.09%) | 5486/6058 (90.56%) | 7978/8735 (91.33%) |

A última linha é o teto desta conta. A soma linear não passa disso: se a carta certa não está no grupo, nenhum peso a escolhe.

## Veredito

O peso eleito pelas fotos reais é name_w 1000, name_exact 30, number_w 120, number_near 30, total_w 130, set_w 170. Ele acerta 2291/2677 (85.58%) das leituras reais e 5085/6058 (83.94%) das sintéticas. Nas duas juntas, 7376/8735 (84.44%).

A variação de 10 em 10 sobe uma carta real (2290 para 2291) e uma sintética (5084 para 5085) em relação ao melhor da grade.

Nota igual fica com o colecionador mais perto do número lido. Nas fotos reais isso acerta mais 13 e erra 5 que a nota sozinha deixava empatadas. A grade acima não conta esse desempate.

A pipeline e o site usam esse peso. O peso anterior acertava 2202/2677 (82.26%) das reais e 4728/6058 (78.05%) das sintéticas. A conta antiga, com o nome até 1000, acerta 2264/2677 (84.57%) das reais e 5056/6058 (83.46%) das sintéticas.

Nas sintéticas o melhor seria name_w 980, name_exact 30, number_w 130, number_near 10, total_w 120, set_w 170, com 5094/6058 (84.09%) nelas e 2284/2677 (85.32%) nas reais. A eleição fica com as fotos, porque as sintéticas repetem um padrão de erro e têm mais cartas de SVI em diante.

## Gráficos

![Acerto por peso](figures/acerto_por_peso.png)

Melhor acerto em cada valor de peso. Os outros pesos variam juntos.

![Reais e sintéticas](figures/reais_vs_sinteticas.png)

Cada ponto é uma combinação de pesos.

![Veredito](figures/veredito.png)

Pesos de referência. A linha tracejada é o teto: a carta certa está entre as candidatas.

## Resultado

| label | reais | sinteticas | todas | name_w | name_exact | number_w | number_near | total_w | set_w |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| melhor | 85.58% | 83.94% | 84.44% | 1000.00 | 30.00 | 120.00 | 30.00 | 130.00 | 170.00 |
| local | 85.58% | 83.94% | 84.44% | 1000.00 | 30.00 | 120.00 | 30.00 | 140.00 | 170.00 |
| local | 85.58% | 83.94% | 84.44% | 1000.00 | 40.00 | 120.00 | 30.00 | 130.00 | 170.00 |
| local | 85.58% | 83.94% | 84.44% | 1000.00 | 40.00 | 120.00 | 30.00 | 140.00 | 170.00 |
| local | 85.58% | 83.94% | 84.44% | 1000.00 | 40.00 | 120.00 | 40.00 | 130.00 | 170.00 |
| local | 85.58% | 83.94% | 84.44% | 1000.00 | 40.00 | 120.00 | 40.00 | 140.00 | 170.00 |
| local | 85.58% | 83.94% | 84.44% | 1000.00 | 50.00 | 120.00 | 30.00 | 130.00 | 170.00 |
| local | 85.58% | 83.94% | 84.44% | 1000.00 | 50.00 | 120.00 | 30.00 | 140.00 | 170.00 |
| melhor_sintetica | 85.32% | 84.09% | 84.46% | 980.00 | 30.00 | 130.00 | 10.00 | 120.00 | 170.00 |
| nome_alto | 84.57% | 83.46% | 83.80% | 1000.00 | 40.00 | 50.00 | 15.00 | 25.00 | 10.00 |
| atual | 85.58% | 83.94% | 84.44% | 1000.00 | 30.00 | 120.00 | 30.00 | 130.00 | 170.00 |
| anterior | 82.26% | 78.05% | 79.34% | 100.00 | 10.00 | 110.00 | 20.00 | 30.00 | 150.00 |

Tabela completa: [`ranking.csv`](ranking.csv).
