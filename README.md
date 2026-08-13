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
│   └── 01_exploracao_mitbih.ipynb
├── outputs/figures/             # Figuras geradas para o artigo
├── scripts/
│   ├── preprocess_mitbih.py     # CLI do pré-processamento do MIT-BIH
│   └── preprocess_ludb.py       # CLI do pré-processamento do LUDB
└── src/xai_arrhythmia/
    ├── config.py                # Constantes: janela, símbolos, caminhos
    ├── signals.py               # Derivação, reamostragem, segmentação, máscaras
    ├── io.py                    # Persistência .npz + metadados
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

## Próximas etapas

- [ ] CNN 1D binária treinada no MIT-BIH (Etapa 3)
- [ ] Grad-CAM 1D + métrica quantitativa de fidelidade (Etapa 4)
