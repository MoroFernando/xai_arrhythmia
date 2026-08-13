# Framework para Interpretação Quantitativa do Grad-CAM em Modelos de Classificação de Arritmias Cardíacas

Projeto Transformador II — Bacharelado em Ciência da Computação, PUCPR (Turma A, Equipe 09).

O objetivo é medir **quantitativamente** — e não por inspeção visual — se os mapas de
calor do Grad-CAM 1D destacam estruturas eletrofisiológicas reais do ECG (ondas P, T e
complexo QRS). A validação é *cross-dataset*: a CNN 1D é treinada no **MIT-BIH
Arrhythmia Database** e a explicabilidade é avaliada no **LUDB**, que fornece o
*ground truth* das delimitações clínicas.

> **Status atual: Etapa 2 (parcial)** — pré-processamento do MIT-BIH (conjunto de treino).
> LUDB, CNN 1D e Grad-CAM ainda não implementados.

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
│   └── 01_exploracao_mitbih.ipynb
├── outputs/figures/             # Figuras geradas para o artigo
├── scripts/
│   └── preprocess_mitbih.py     # CLI do pré-processamento do MIT-BIH
└── src/xai_arrhythmia/
    ├── config.py                # Constantes: janela, símbolos, caminhos
    ├── signals.py               # Seleção de derivação, segmentação, normalização
    ├── io.py                    # Persistência .npz + metadados
    └── datasets/
        └── mitbih.py            # Pipeline completo do MIT-BIH
```

## Instalação

Crie o ambiente conda apenas com a versão do Python:

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

## Próximas etapas

- [ ] Pré-processamento do LUDB com máscaras de *ground truth* P/QRS/T (`datasets/ludb.py`)
- [ ] CNN 1D binária treinada no MIT-BIH (Etapa 3)
- [ ] Grad-CAM 1D + métrica quantitativa de fidelidade (Etapa 4)
