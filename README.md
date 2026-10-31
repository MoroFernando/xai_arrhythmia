# Framework para interpretação quantitativa do Grad-CAM em ECG

<!-- legenda-interna-projeto -->
## Legenda de nomes e abreviações do projeto

| Nome / abreviação | Significado nesta documentação |
| --- | --- |
| **SEM** / **SEM_BLOQUEIO** | **Sem bloqueio como alvo de classificação**: modelo treinado para reconhecer as extrasístoles compartilhadas entre MIT e LUDB. Bloqueios não são alvos positivos dessa política. O nome não garante ausência de bloqueios como comorbidade em todo registro. |
| **COM** / **COM_BLOQUEIO** | **Com bloqueio como alvo de classificação**: modelo que inclui extrasístoles e bloqueios de ramo completos e incompletos entre seus alvos positivos. |
| **COM somente completos** | Recorte de avaliação do mesmo modelo COM: exclui apenas positivos cuja única justificativa seja bloqueio incompleto. Mantém extrasístoles associadas a incompletos; não é um terceiro modelo. |
| **BEST** / **LAST** | BEST é o checkpoint da melhor época pela Balanced Accuracy da validação MIT; LAST é o da última época. Checkpoint significa o arquivo com os pesos e metadados do modelo. |
| **BA** | Balanced Accuracy, ou acurácia balanceada: média entre sensibilidade e especificidade. |
| **Política / slug** | Regra de seleção dos alvos e seu identificador nos arquivos: shared-extrasystoles-v1 = SEM; shared-extrasystoles-bbb-v1 = COM. |
| **beat / beat_id / n** | Beat é a janela de um batimento; beat_id é seu identificador; n é a quantidade de exemplos do recorte. As classes 0 e 1 são, respectivamente, referência negativa e alvo positivo da política. |

Esta legenda esclarece a nomenclatura interna atual. Em documentos marcados **HISTÓRICO**, não substitui nem atualiza as regras antigas descritas no corpo. As siglas médicas mantêm a nomenclatura dos dados originais.
<!-- /legenda-interna-projeto -->


STATUS: ATUAL — revisão editorial de 30/10/2026. Comece pelo [relatório geral](RELATORIO_GERAL_PROJETO.md), [relatório visual](RELATORIO_VISUAL_RESULTADOS.md), [guia dos notebooks oficiais](GUIA_NOTEBOOKS_RESULTADOS.md) e [correções finais](RELATORIO_CORRECOES_FINAIS.md).

O projeto treina uma CNN 1D binária em beats do MIT-BIH, testa a generalização externa em beats do LUDB e compara o Grad-CAM com as delimitações P/QRS/T por IoU. O LUDB diagnostica principalmente o registro; o diagnóstico selecionado é propagado aos beats aceitos, limitação metodológica reconhecida pelo grupo.

## Políticas de classificação

Treino e teste usam famílias compartilhadas e uma política explícita em todas as etapas:

| Slug | Nome | MIT-BIH | LUDB positivo |
| --- | --- | --- | --- |
| `shared-extrasystoles-v1` | `SEM_BLOQUEIO` | `N=0`; `A/a/V=1` | extrasístoles atriais/ventriculares |
| `shared-extrasystoles-bbb-v1` | `COM_BLOQUEIO` | `N=0`; `A/a/V/L/R=1` | anteriores + LBBB/RBBB completos/incompletos |

Os demais símbolos e diagnósticos ficam fora da tarefa; exclusão não significa normalidade. Isquemia, por exemplo, não é convertida em 0. Pacing é excluído nas duas políticas. Consulte [as políticas atuais](RELATORIO_GERAL_PROJETO.md#classificacao-detalhada) para o contrato operacional e o mapeamento completo.

## Estrutura

```text
src/
  classification.py       políticas, valores controlados e compatibilidade
  config.py               caminhos e parâmetros comuns
  beat_io.py              persistência dos datasets NPZ
  datasets/mitbih.py      segmentação e seleção MIT-BIH
  datasets/ludb.py        segmentação, P/QRS/T e seleção LUDB
  datasets/ludb_clinical.py  parser do CSV oficial LUDB
  preprocess_mitbih.py
  preprocess_ludb.py
  train_cnn.py            split por sujeito e treino
  test_ludb.py            inferência e métricas
  model_gradcam.py        CNN 1D e Grad-CAM
  extract_gradcam.py
  fidelity_ludb.py        IoU
notebooks/                análise dos artefatos já gerados
data/                     bases e datasets processados
outputs/                  manifests, checkpoints e resultados
```

## Instalação

O ambiente registrado usou Python 3.11.17 e PyTorch 2.11.0+cu128. PyTorch/CUDA são instalados separadamente dos requisitos gerais. Consulte [o ambiente e os limites de reprodução](RELATORIO_GERAL_PROJETO.md#ambiente-reprodutibilidade); não há lockfile completo das demais bibliotecas.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements-pytorch-cu128.txt
python -m pip install -r requirements.txt
```

## Cadeia de execução

Execute futuramente cada política de ponta a ponta. O exemplo abaixo usa o modo sem bloqueio:

```bash
python src/preprocess_mitbih.py --download --classification-policy shared-extrasystoles-v1
python src/train_cnn.py --classification-policy shared-extrasystoles-v1
python src/preprocess_ludb.py --download --classification-policy shared-extrasystoles-v1
python src/test_ludb.py --classification-policy shared-extrasystoles-v1
python src/extract_gradcam.py --classification-policy shared-extrasystoles-v1
python src/fidelity_ludb.py --classification-policy shared-extrasystoles-v1
```

Repita com `shared-extrasystoles-bbb-v1`. O CSV oficial LUDB 1.0.1 deve estar em `data/raw/ludb/ludb.csv` ou ser informado por `--clinical-csv`.

Depois das duas cadeias, valide os artefatos e compare os modelos nos mesmos beats:

```bash
python src/compare_policies.py
```

Os caminhos padrão incluem o slug:

```text
data/processed/mitbih_<slug>.npz
data/processed/ludb_<slug>.npz
outputs/models/cnn1d_mitbih_<slug>/
  model.pt          melhor época (val. acurácia balanceada), usado no LUDB/Grad-CAM/IoU
  model_last.pt     última época, apenas rastreabilidade
outputs/comparacao_politicas/   validação dos artefatos e SEM x COM nos mesmos beats
```

Cada etapa rejeita política incompatível e evita sobrescrever uma saída existente. SHA-256 vincula resultados a datasets, checkpoints e Grad-CAM. Os resultados históricos em `outputs/models/cnn1d_mitbih/` pertencem à classificação anterior e não devem ser comparados como se fossem uma das políticas novas.

## Pré-processamento e identidade dos beats

As duas bases usam Lead II, janela de 234 amostras centrada no pico R e normalização Z por beat. O LUDB é reamostrado de 500 Hz para 360 Hz e mantém máscaras e regiões P/QRS/T.

Todo beat selecionado recebe um `beat_id` estável baseado em base, registro e pico R, além do índice de seleção no registro. O MIT-BIH preserva `pacientes` como id do registro e `subject_ids` como id do sujeito; os registros 201 e 202 usam o mesmo `subject_id`. O LUDB preserva famílias, subtipos, lado/completude dos bloqueios, motivos de inclusão, comorbidades e ritmos.

## Split e treinamento

O treino usa `StratifiedGroupKFold`, por padrão com cinco folds, fold 0 e seed 0. O agrupamento usa `subject_ids`, impedindo que 201 e 202 sejam separados. Um manifesto JSON em `outputs/splits/` registra a atribuição por sujeito e pode ser reutilizado pelos dois modos quando válido. O checkpoint guarda política, versão, hash do dataset e identidade do split.

A arquitetura `CNN1D_Arritmia` e o cálculo do Grad-CAM permanecem iguais. Cada política exige preprocessing e treinamento independentes.

O treino roda o número planejado de épocas (20 por padrão). A cada época a validação MIT-BIH é avaliada no limiar fixo 0.5; quando a acurácia balanceada supera estritamente a melhor anterior, aqueles pesos passam a ser o melhor modelo. `model.pt` guarda o melhor modelo e `model_last.pt` a última época. `metrics.json` registra `best_epoch`, `best_validation_balanced_accuracy`, `last_epoch`, `last_validation_balanced_accuracy`, métricas completas de treino/validação (inclusive por símbolo) e o SHA-256 dos dois checkpoints. Teste LUDB, Grad-CAM e IoU recusam checkpoint que não seja o melhor. O LUDB não participa da seleção. Os kernels cuDNN são determinísticos.

## Avaliação

O teste LUDB produz Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, F1, ROC-AUC e matriz de confusão. Também produz, para famílias e subtipos positivos, sensibilidade, taxa de falso negativo e score médio. Especificidade, precisão, F1 e AUC não são atribuídas a um recorte que contenha somente positivos.

No modo com bloqueio, a análise pode separar:

- extrasístoles atriais e ventriculares;
- LBBB e RBBB;
- LBBB/RBBB completos e incompletos;
- o subconjunto de extrasístoles compartilhado com o modo sem bloqueio.

O teste também grava baselines triviais (sempre 0, sempre 1, classe majoritária), a atribuição dos falsos negativos a extrasístoles e bloqueios completos/incompletos (multirrótulo e partição exclusiva) e, no modo com bloqueio, a avaliação "somente completos": as mesmas predições do modelo COM, excluindo apenas positivos cuja única justificativa é bloqueio incompleto. Ela não é um terceiro modelo.

O subconjunto comum contém os beats da tarefa SEM: referências negativas e registros com extrasístole atrial/ventricular, inclusive quando também têm LBBB/RBBB. `compare_policies.py` usa a interseção exata de `beat_id` e avalia os dois modelos sobre os mesmos beats e os mesmos `y_true`.

Grad-CAM e IoU preservam os mesmos metadados. O Grad-CAM confere que usa o mesmo checkpoint e reproduz as mesmas predições do teste LUDB. O IoU pode ser resumido por família e subtipo sem alterar sua definição: interseção sobre união entre o mapa binarizado e as máscaras clínicas P/QRS/T.

## Notebooks e apresentação oficial

Os **12 notebooks oficiais** estão em `outputs/notebooks_executados/<slug>/`. A pasta `notebooks/` preserva outputs históricos e está marcada como tal. Os notebooks 03/04 oficiais contêm código de inferência: para consulta, leia seus outputs já salvos. O notebook 06 lê IoUs persistidos e mostra as contagens das duas classes por ritmo. Veja o [guia completo](GUIA_NOTEBOOKS_RESULTADOS.md).

A cobertura clínica ≈ 0,576 não foi validada como baseline aleatório de IoU. Campos antigos `cobertura_acaso`/`acima_do_acaso` são deprecated e preservados nos artefatos por compatibilidade, sem interpretação estatística. Consulte [o contrato](RELATORIO_GERAL_PROJETO.md#campos-legados-iou).

O treinamento aplica seed 0 a Python/NumPy/PyTorch e configura cuDNN; não há evidência no pacote para comprovar igualdade bit a bit entre execuções independentes.

## Validação sem regenerar resultados

```text
python src/compare_policies.py --validate-only
```

Este modo apenas lê/agrega. A revisão editorial foi validada com a ferramenta independente em `relatorios/validar_resultados.cjs`; os artefatos numéricos originais permaneceram idênticos.

A documentação da entrega é composta por este README e pelos quatro documentos principais indicados no início.
