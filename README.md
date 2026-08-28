# Framework para Interpretação Quantitativa do Grad-CAM em Modelos de Classificação de Arritmias Cardíacas

Projeto Transformador II — Bacharelado em Ciência da Computação, PUCPR (Turma A, Equipe 09).

O objetivo é medir **quantitativamente**,e não por inspeção visual,se os mapas de
calor do Grad-CAM 1D destacam estruturas eletrofisiológicas reais do ECG (ondas P, T e
complexo QRS). A validação é *cross-dataset*: a CNN 1D é treinada no **MIT-BIH
Arrhythmia Database** e a explicabilidade é avaliada no **LUDB**, que fornece o
*ground truth* das delimitações clínicas.

## Estrutura do repositório

```
framework-xai-arrhythmia/
├── requirements.txt              # Dependências (pip)
├── pyproject.toml               # Pacote instalável `xai_arrhythmia`
├── data/
│   ├── raw/                     # Registros WFDB baixados (não versionado)
│   ├── interim/                 # Artefatos intermediários (não versionado)
│   └── processed/               # Datasets .npz prontos (não versionado)
├── notebooks/
│   ├── 01_exploracao_mitbih.ipynb
│   ├── 02_exploracao_ludb.ipynb
│   ├── 03_analise_treino.ipynb  # Análises pós-treino (MIT-BIH)
│   ├── 04_teste_ludb.ipynb      # Uso do modelo treinado no LUDB
│   └── 05_gradcam_demo.ipynb    # Grad-CAM pré-extraído + anotações clínicas
├── outputs/figures/             # Figuras geradas para o artigo
├── outputs/models/              # Checkpoints treinados (versionados)
├── scripts/
│   ├── preprocess_mitbih.py     # CLI do pré-processamento do MIT-BIH
│   ├── preprocess_ludb.py       # CLI do pré-processamento do LUDB
│   ├── train_cnn.py             # CLI do treino/validação no MIT-BIH
│   ├── test_ludb.py             # CLI do teste no LUDB (carrega o modelo)
│   └── extract_gradcam.py       # CLI da extração do Grad-CAM no LUDB
└── src/xai_arrhythmia/
    ├── config.py      # Constantes: janela, símbolos, caminhos, DS1/DS2
    ├── signals.py     # Derivação, reamostragem, segmentação, máscaras
    ├── io.py          # Persistência .npz + metadados
    ├── splits.py      # Divisão inter-paciente
    ├── metrics.py     # Métricas de classificação binária
    ├── models.py      # Arquitetura CNN 1D
    ├── training.py    # Loop de treino, checkpoint e avaliação
    ├── gradcam.py     # Extração do Grad-CAM 1D (em lote ou 1 batimento)
    └── datasets/
        ├── mitbih.py            # Pipeline completo do MIT-BIH (treino)
        └── ludb.py              # Pipeline completo do LUDB (teste)
```

## Instalação

Crie o ambiente conda com a versão do Python:

```bash
conda create -n xai-arrhythmia python=3.11
```

```bash
conda activate xai-arrhythmia
```

Instale as dependências via pip (a partir da raiz do repositório):

```bash
pip install -r requirements.txt
```

O `requirements.txt` já instala o pacote local em modo editável (`-e .`), então
`import xai_arrhythmia` funciona em qualquer script ou notebook a partir daqui.

### Alternativa sem conda

```bash
python -m venv .venv && .venv\Scripts\activate && pip install -r requirements.txt
```

## Pré-processamento do MIT-BIH

Baixa os registros para `data/raw/mitdb` e gera o dataset processado:

```bash
python scripts/preprocess_mitbih.py --download
```

Para ler direto do PhysioNet, sem cache local (mais lento, útil no Colab):

```bash
python scripts/preprocess_mitbih.py --stream
```

Saída padrão: `data/processed/mitbih_beats.npz`.

### O que o pipeline faz

1. **Leitura via WFDB** — sinal (`.dat`/`.hea`) e anotações (`.atr`) de cada registro.
2. **Seleção da derivação II** — o canal é escolhido **pelo nome** (`MLII`), nunca pelo
   índice: nos registros 102 e 104 o canal 0 é V5, e usar `p_signal[:, 0]` misturaria
   derivações diferentes no mesmo dataset.
3. **Segmentação beat-to-beat** — janela assimétrica de 234 amostras (~650 ms @360 Hz)
   centrada no pico R: 90 amostras antes (~250 ms, cobre a onda P) e 144 depois
   (~400 ms, cobre a onda T). Janelas que extrapolam os limites do sinal são
   descartadas.
4. **Descarte de anotações non-beat** — `! " + [ ] x | ~` marcam mudança de ritmo,
   ruído e artefatos, não batimentos.
5. **Z-normalization por batimento** — cada janela é normalizada isoladamente
   (média 0, desvio 1). Batimentos constantes viram vetor de zeros, sem divisão por zero.
   Normalizar por batimento é o que torna possível aplicar no LUDB um modelo treinado
   no MIT-BIH, já que remove diferenças de ganho e offset entre bases.
6. **Rotulagem binária** — `0` = não-arritmia, `1` = arritmia.

### Decisões que divergem do notebook exploratório inicial

| Decisão | Padrão adotado | Como reverter |
| --- | --- | --- |
| Registros com marcapasso (102, 104, 107, 217) | Excluídos — prática usual na literatura de classificação de batimentos; 102 e 104 também não têm MLII | `--include-paced` |
| Classe 0 | Apenas `N` | `--normal-symbols AAMI` agrupa `N, L, R, e, j` |
| Seleção de canal | Por nome (`MLII`) | `--lead V5`, etc. |

O registro `108` (ausente da lista do notebook) está incluído.

### Formato do arquivo de saída

`.npz` compactado, todos os arrays alinhados por índice:

| Chave | Shape | Descrição |
| --- | --- | --- |
| `series` | `(n, 234)` `float32` | batimento segmentado e normalizado |
| `labels` | `(n,)` `int64` | 0 = não-arritmia, 1 = arritmia |
| `simbolos` | `(n,)` `<U2` | símbolo original do MIT-BIH (`N`, `V`, `A`, …) |
| `pacientes` | `(n,)` `<U4` | id do registro de origem |
| `r_peaks` | `(n,)` `int64` | posição do pico R no sinal original |
| `metadata_json` | escalar | JSON com todos os parâmetros usados |

Leitura:

```python
from xai_arrhythmia.io import load_beat_dataset

dados = load_beat_dataset("data/processed/mitbih_beats.npz")
X, y = dados["series"], dados["labels"]
print(dados["metadata"]["distribuicao_labels"])
```

Manter `pacientes` é o que permite a divisão *inter-patient* no treino (Etapa 3):
batimentos de um mesmo paciente nunca devem aparecer em treino e validação ao mesmo
tempo. `simbolos` preserva o tipo original da arritmia para a análise de falhas por
classe na Etapa 4.

## Pré-processamento do LUDB

```bash
python scripts/preprocess_ludb.py --download
```

Saída padrão: `data/processed/ludb_beats.npz`.

O download baixa apenas as anotações da derivação usada — o LUDB tem um arquivo de
anotação por derivação (12 por registro), e pegar as 12 traria ~2200 arquivos inúteis.
Os arquivos caem em `data/raw/ludb/data/` (o `RECORDS` do PhysioNet já traz o prefixo
`data/`); o pipeline resolve esse subdiretório sozinho.

### O que o pipeline faz

1. **Leitura via WFDB** — sinal das 12 derivações e a anotação de delineação, que no
   LUDB tem a derivação como extensão do arquivo (`1.ii`).
2. **Seleção da derivação II** — mesma função do MIT-BIH; os nomes de canal do LUDB
   são minúsculos (`ii`) e a comparação é case-insensitive.
3. **Extração das ondas** — a anotação segue o padrão `( → pico → )`, com `p` (onda P),
   `N` (complexo QRS) e `t` (onda T). Cada onda vira a tripla `(onset, pico, offset)`.
4. **Reamostragem 500 Hz → 360 Hz** — `resample_poly` (18/25) no sinal, e os índices
   das anotações reescalados pela mesma razão. Sem isso os batimentos do LUDB teriam
   duração diferente dos do MIT-BIH e o modelo veria morfologias distorcidas.
5. **Segmentação beat-to-beat** — mesma janela de 234 amostras, centrada no **pico R**
   (o pico anotado do complexo QRS), igual ao MIT-BIH.
6. **Filtragem por completude** — o batimento só é mantido se o próprio QRS e a onda T
   que o segue estiverem **integralmente** dentro da janela. A onda P precedente é
   exigida apenas em registros que têm ondas P delimitadas (ver abaixo).
7. **Z-normalization por batimento** — idêntica à do MIT-BIH.
8. **Rótulo propagado** — o LUDB diagnostica o **registro**, não o batimento: o campo
   `Rhythm:` do header é replicado em todos os batimentos daquele registro.

### Por que a onda P é condicional

Os 18 registros de **fibrilação e flutter atrial** do LUDB têm **zero ondas P
anotadas**. Isso não é lacuna de anotação: a ausência de onda P *é a definição
eletrofisiológica* da fibrilação atrial — o átrio não despolariza de forma organizada e
a onda P é substituída por ondas fibrilatórias.

Como esses são exatamente os registros da classe arritmia, exigir P de todo batimento
zerava a classe 1 do conjunto de teste (`classe 1: 0 batimentos`), tornando impossível
medir a fidelidade do Grad-CAM justamente na classe que o artigo investiga. O critério
condicional recupera ~162 batimentos arrítmicos e mantém o rigor das três estruturas
para os ritmos que de fato produzem onda P.

Cada batimento carrega a flag `tem_onda_p`, e `regiao_p` recebe o sentinela `(-1, -1)`
quando a onda não existe — assim a Etapa 4 pode excluir a P do denominador da métrica
nesses casos, em vez de penalizar o Grad-CAM por não destacar uma estrutura ausente.

### Rótulo binário

O ritmo do header (`#Rhythm: Sinus bradycardia.`) é normalizado (minúsculas, sem
pontuação final) e mapeado assim:

| Classe | Ritmos |
| --- | --- |
| `0` (não-arritmia) | `sinus rhythm` |
| `1` (arritmia) | qualquer outro |

É o análogo direto do critério do MIT-BIH, onde apenas o símbolo `N` recebe classe 0.
O resumo impresso lista a distribuição por ritmo com a classe atribuída, o que permite
conferir se algum ritmo inesperado caiu na classe 1.

`--normal-rhythms sinus` troca para o agrupamento alternativo, que coloca todos os
ritmos de origem sinusal (`sinus bradycardia`, `sinus tachycardia`, `sinus arrhythmia`,
`irregular sinus rhythm`) na classe 0 e restringe a classe 1 aos ritmos não-sinusais.
Sob esse agrupamento a classe 1 cai de 363 para 162 batimentos. A diferença está nos
201 batimentos de ritmo sinusal não-normal, cuja distinção depende do intervalo RR —
informação que a janela de um único batimento não carrega. Vale usá-lo como **análise
de sensibilidade** no artigo.

### Formato do arquivo de saída

Além de `series`, `labels`, `simbolos` (ritmo propagado), `pacientes` e `r_peaks`:

| Chave | Shape | Descrição |
| --- | --- | --- |
| `mascara_clinica` | `(n, 234)` `int8` | união P+QRS+T — o Ω_clin da métrica de fidelidade |
| `mascara_p` / `mascara_qrs` / `mascara_t` | `(n, 234)` `int8` | máscara por estrutura, para análise separada |
| `regiao_p` / `regiao_qrs` / `regiao_t` | `(n, 2)` `int64` | `(onset, offset)` das ondas **do próprio batimento**, relativos à janela; `(-1, -1)` quando a onda não existe |
| `tem_onda_p` | `(n,)` `int8` | 1 se o batimento tem onda P delimitada, 0 em FA/flutter |

As máscaras incluem toda onda anotada que cai na janela (inclusive de batimentos
vizinhos), porque descrevem tudo que os especialistas delimitaram naquele trecho. Já as
`regiao_*` guardam só as três ondas do batimento em questão — é o que permite, na
Etapa 4, medir a ativação do Grad-CAM por estrutura individual.

A cobertura média da `mascara_clinica` impressa no resumo é a **pontuação esperada ao
acaso** da métrica de fidelidade: qualquer mapa de Grad-CAM aleatório atinge esse valor,
então ele é o piso contra o qual o resultado do modelo deve ser comparado.

## Treino da CNN 1D (Etapa 3)

```bash
python scripts/train_cnn.py
```

Saída padrão: `outputs/models/cnn1d_mitbih/` com `model.pt`, `metrics.json` e
`history.csv`.

Treino e teste são **scripts separados**, e a separação é metodológica, não
apenas organizacional: `train_cnn.py` só enxerga o MIT-BIH, e o teste no LUDB
(`test_ludb.py`, seção seguinte) só consome um modelo já salvo. Nenhuma decisão
de treino pode ser tomada olhando o conjunto de teste.

### Divisão inter-paciente

O modelo é treinado **exclusivamente no MIT-BIH**. A base inteira é usada como
conjunto de desenvolvimento e dividida **por registro**, nunca por batimento:
batimentos do mesmo paciente compartilham morfologia, ruído de eletrodo e
características do indivíduo — se caíssem em treino e validação ao mesmo tempo,
a rede poderia acertar por reconhecer o paciente, e não a arritmia.

O padrão é a partição de **de Chazal et al. (2004)**: DS1 (22 registros) treina,
DS2 (22 registros) valida. Juntos são exatamente os 44 registros de
`MITBIH_RECORDS` — os 4 com marcapasso já ficam de fora. `--split random`
sorteia pacientes até atingir `--val-fraction` dos batimentos, útil como análise
de sensibilidade. Em qualquer esquema o código **verifica e falha** se a
interseção de registros entre as partições não for vazia.

O conjunto de **teste** do projeto é o LUDB, e ele não participa de nenhuma
decisão de treino. `--eval-ludb` apenas reporta, ao final, o desempenho
cross-dataset com os pesos já congelados.

### Arquitetura

Quatro blocos `Conv1d → BatchNorm → ReLU → MaxPool → Dropout` (32, 64, 128, 128
canais), Global Average Pooling e uma única camada linear com 1 logit de saída —
187.425 parâmetros. Duas decisões são ditadas pela Etapa 4:

- **`padding = kernel_size // 2`** em todas as convoluções, para que o passo `i`
  do mapa de ativação continue correspondendo à mesma região do sinal, sem
  deslocamento acumulado.
- **GAP + uma linear**, que é a formulação direta do CAM/Grad-CAM: o peso da
  classe multiplica o canal convolucional. Uma MLP densa no topo tornaria a
  atribuição indireta.

O pooling nos três primeiros blocos leva 234 → 117 → 58 → 29, então o Grad-CAM
sai com 29 passos (~8 amostras, ~22 ms cada), resolução suficiente para separar
P, QRS e T dentro do batimento. `model.last_conv` já aponta a camada alvo.

### Desbalanceamento

A classe arritmia é ~26% do MIT-BIH. A `BCEWithLogitsLoss` recebe
`pos_weight = negativos/positivos` do conjunto de treino; sem isso a rede
maximiza acurácia respondendo "normal" para quase tudo e derruba a
sensibilidade, justamente a métrica que importa em triagem. `--sem-balanceamento`
desativa.

### Seleção de época e limiar de decisão

A melhor época é a de maior acurácia balanceada na validação (`--criterio`
troca o alvo), com early stopping por `--patience`. As métricas finais usam
sempre os pesos dessa época, não os da última.

O **limiar de decisão não é ajustável**: é sempre `config.DEFAULT_THRESHOLD`
(0,5), o mesmo valor em treino, validação e teste no LUDB. O projeto não faz
varredura de limiar por conjunto — isso abriria uma forma indireta de encaixar
o modelo aos dados de teste, o mesmo problema que a divisão inter-paciente
existe para evitar em relação aos pacientes.

### Resultados no MIT-BIH (execução de referência)

Números do checkpoint versionado em `outputs/models/cnn1d_mitbih/`: split
DS1/DS2, semente 42, `--epochs 100 --patience 25`, limiar fixo 0,5.

| Conjunto | Acurácia | Acur. balanceada | Sensibilidade | Especificidade | AUC-ROC |
| --- | --- | --- | --- | --- | --- |
| Treino (DS1) | 0,9862 | 0,9834 | 0,9778 | 0,9890 | 0,9978 |
| Validação (DS2) | 0,8678 | 0,8534 | 0,8223 | 0,8844 | 0,9228 |

A melhor época foi a **4**, mesmo com patience alto o suficiente para rodar até
a 29 sem interromper antes. O treino continua reduzindo a perda a cada época
(0,107 na época 4 → 0,028 na 29), mas a acurácia balanceada na validação nunca
mais supera o pico da época 4 — oscila entre 0,70 e 0,82, sinal de overfitting
ao DS1 depois desse ponto. **Mais épocas não ajudam aqui**; o gargalo é a
capacidade de generalizar entre pacientes, não o tempo de treino. O notebook
`03_analise_treino.ipynb` plota essa curva completa.

A queda de DS1 para DS2 é o efeito esperado da divisão inter-paciente, e é o
motivo de a literatura *intra-patient* reportar >99%.

Os números variam ligeiramente entre execuções mesmo com a semente fixa: o
cuDNN não é determinístico por padrão e o projeto não ativa o modo
determinístico, cujo custo em velocidade não compensa aqui.

### Formato do checkpoint

`model.pt` é autocontido — pesos, hiperparâmetros da arquitetura, divisão usada
e todas as métricas:

```python
from xai_arrhythmia.training import load_checkpoint

modelo, ckpt = load_checkpoint("outputs/models/cnn1d_mitbih/model.pt")
print(ckpt["threshold"], ckpt["info"]["metrics"]["validacao"])
```

`metrics.json` traz o mesmo conteúdo em texto (inclusive a lista de registros de
cada partição) e `history.csv` a curva por época.

## Teste no LUDB (cross-dataset)

```bash
python scripts/test_ludb.py
```

O script **não treina nada**: carrega o checkpoint, congela os pesos e mede o
desempenho no LUDB. O limiar não é reajustado aqui — é o mesmo 0,5 fixo do
treino e da validação.

Saída em `outputs/models/<run>/teste_ludb/`:

| Arquivo | Conteúdo |
| --- | --- |
| `ludb_metrics.json` | métricas, quebra por ritmo e a proveniência do modelo |
| `ludb_predictions.npz` | `y_true`, `y_score`, `y_pred`, `pacientes`, `ritmos` batimento a batimento |

O `.npz` de predições é o que a Etapa 4 consome: alinhado por índice com as
máscaras clínicas do LUDB, permite cruzar o Grad-CAM com as anotações de P, QRS
e T beat a beat, inclusive nos batimentos em que o modelo erra.

### Resultado

| Conjunto | Acurácia | Acur. balanceada | Sensibilidade | Especificidade | AUC-ROC |
| --- | --- | --- | --- | --- | --- |
| LUDB @ 0,50 | 0,7085 | 0,6012 | 0,3388 | 0,8636 | 0,6051 |

A quebra por ritmo explica o número agregado:

| Ritmo | n | Classe | Acerto | Score médio |
| --- | --- | --- | --- | --- |
| sinus rhythm | 865 | 0 | 0,864 | 0,189 |
| atrial fibrillation | 126 | 1 | 0,516 | 0,544 |
| sinus bradycardia | 75 | 1 | 0,080 | 0,140 |
| sinus arrhythmia | 64 | 1 | 0,047 | 0,076 |
| sinus tachycardia | 49 | 1 | 0,265 | 0,268 |
| atrial flutter, typical | 36 | 1 | 0,944 | 0,889 |
| irregular sinus rhythm | 13 | 1 | 0,154 | 0,342 |

O modelo responde bem onde a arritmia **altera a morfologia** do batimento
(flutter 0,94; fibrilação 0,52) e falha onde ela é uma anomalia de
**frequência** (bradicardia, taquicardia e arritmia sinusal, 188 batimentos com
score médio abaixo de 0,27). Isso não é acaso de treino: esses batimentos são
morfologicamente normais, e distingui-los exige o intervalo RR — informação que
a janela de um único batimento, por construção, não carrega. É o mesmo ponto
que motiva a análise de sensibilidade com `--normal-rhythms sinus` no
pré-processamento do LUDB.

Vale registrar que o teste cross-dataset é o cenário adverso por definição: as
bases diferem em população, equipamento e — sobretudo — em critério de rótulo
(o MIT-BIH rotula o *batimento*, o LUDB rotula o *ritmo do registro inteiro*).
Para o objeto do artigo isso é matéria-prima, não defeito: a Etapa 4 investiga
se o Grad-CAM aponta estruturas eletrofisiológicas reais **mesmo quando a
predição erra** — um falso negativo cujo mapa destaca corretamente QRS e T
indica erro de decisão, não de foco.

## Extração do Grad-CAM (Etapa 4)

```bash
python scripts/extract_gradcam.py
```

Terceiro script da cadeia, depois de `train_cnn.py` e `test_ludb.py`, mas
**autocontido**: não depende dos outros dois terem rodado, só do checkpoint e
do dataset do LUDB. A predição usada para os campos `y_pred`/`y_score` é
recomputada aqui (mesmo checkpoint, mesmo limiar fixo, sem aleatoriedade em
`eval()`), então sai idêntica à de `test_ludb.py` — os dois arquivos ficam
alinhados por índice sem precisar um do outro.

A lógica do Grad-CAM mora em [`xai_arrhythmia/gradcam.py`](src/xai_arrhythmia/gradcam.py),
não no script nem no notebook: é o mesmo padrão do resto do projeto
(`metrics.py`, `training.py`, `signals.py`) — uma função por conceito,
reaproveitada por CLI, script e notebook em vez de reimplementada em cada um.

### Grad-CAM em lote

`gradcam.grad_cam_1d` extrai o mapa de ativação de `modelo.last_conv` via
hooks de forward/backward, e aceita tanto um único batimento quanto um lote
inteiro — o LUDB inteiro (1228 batimentos) processa em ~2 s na GPU, numa
única passada por lote (`--batch-size`, padrão 256), sem laço em Python por
batimento.

Isso funciona porque o modelo roda sempre em modo `eval()`: BatchNorm usa
estatísticas fixas (não as do lote) e Dropout vira identidade, então a saída
de uma amostra do lote não depende de nenhuma outra. Somar os logits do lote e
chamar um único `.backward()` produz, para cada amostra, exatamente o mesmo
gradiente que um `.backward()` individual — verificado no próprio módulo
(diferença de ~1e-7, ruído de ponto flutuante, entre as duas formas de
calcular).

### Formato da saída

Em `outputs/models/<run>/gradcam_ludb/`:

| Arquivo | Conteúdo |
| --- | --- |
| `ludb_gradcam.npz` | `cam_raw` `(n, 29)`, `cam_interp` `(n, 234)`, `y_true`, `y_pred`, `y_score`, `pacientes`, `ritmos`, `threshold` |
| `ludb_gradcam.json` | checkpoint, dataset, camada alvo, dimensões, método de interpolação e as métricas de referência |

`cam_raw` é o mapa no comprimento nativo da rede (`L' = 29`, a saída de
`modelo.last_conv`); `cam_interp` já vem reamostrado para o tamanho da janela
do batimento (234 amostras) via `gradcam.interpola_cam` — interpolação linear
simples sobre o eixo `[0, 1]`, suficiente dado o tamanho pequeno do batimento
(~650 ms). O arquivo é autocontido: alinhado por índice com o dataset do LUDB
e com `ludb_predictions.npz`, mas não depende deste para ser lido.

Essa matriz `cam_interp` é a matéria-prima que falta para fechar a Etapa 4:
cruzá-la com `ludb["mascara_clinica"]` — a máscara booleana `(n, 234)` que já
une P, QRS e T amostra a amostra — reduz a comparação visual do notebook
`05_gradcam_demo.ipynb` a um número por batimento (por exemplo, a fração da
energia do mapa dentro da região clinicamente anotada), o que ainda falta
implementar.

## Notebooks

| Notebook | Conteúdo |
| --- | --- |
| `01_exploracao_mitbih.ipynb` | Exploração da base de treino |
| `02_exploracao_ludb.ipynb` | Exploração da base de teste e das anotações clínicas |
| `03_analise_treino.ipynb` | Análise pós-treino: curvas por época, matriz de confusão, falhas por tipo de batimento e dispersão entre pacientes |
| `04_teste_ludb.ipynb` | Uso do modelo treinado: carrega o checkpoint, avalia o LUDB, quebra por ritmo e plota batimentos com as regiões P/QRS/T sombreadas |
| `05_gradcam_demo.ipynb` | Carrega o Grad-CAM já extraído (`ludb_gradcam.npz`) e plota, numa grade 2×2, o batimento mais confiante de cada quadrante da matriz de confusão (VP/VN/FP/FN) com o ECG colorido pela ativação e as regiões de P/QRS/T sombreadas |

Os notebooks 03 e 04 são a versão exploratória do que os dois scripts fazem em
lote — o 04, em particular, demonstra o caminho de quem recebe apenas o `.pt` e
quer rodar o modelo em dados novos. O 05 é a versão visual do que
`extract_gradcam.py` faz em lote.

## Próximas etapas

- [x] CNN 1D binária treinada no MIT-BIH (Etapa 3)
- [x] Extração do Grad-CAM 1D no LUDB (Etapa 4, parte 1)
- [ ] Métrica quantitativa de fidelidade: cruzar `cam_interp` com
      `mascara_clinica`/`mascara_p`/`mascara_qrs`/`mascara_t` num número por
      batimento (Etapa 4, parte 2)
- [ ] Análise de falhas da métrica por ritmo e por tipo de arritmia (Etapa 4)
- [ ] Análise crítica e redação do artigo (Etapa 5)