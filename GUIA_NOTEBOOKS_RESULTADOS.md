# Guia dos notebooks e resultados oficiais

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


STATUS: ATUAL — revisão editorial de 30/10/2026. Fonte exclusiva: pos3.

Os **12 notebooks oficiais** estão em `outputs/notebooks_executados/<slug>/`. Os seis de `notebooks/` são **HISTÓRICOS**, com outputs anteriores preservados. Os nomes das duas políticas oficiais são shared-extrasystoles-v1 (SEM) e shared-extrasystoles-bbb-v1 (COM).

| Notebook oficial | Objetivo | Política | Resultados principais | Artefatos utilizados | Status |
| --- | --- | --- | --- | --- | --- |
| [01_exploracao_mitbih.ipynb](outputs/notebooks_executados/shared-extrasystoles-v1/01_exploracao_mitbih.ipynb) | População, símbolos e exemplos MIT | SEM | 84115 beats | data/processed/mitbih_<slug>.npz | ATUAL |
| [02_exploracao_ludb.ipynb](outputs/notebooks_executados/shared-extrasystoles-v1/02_exploracao_ludb.ipynb) | População, ritmos e máscaras LUDB | SEM | 192 beats | data/processed/ludb_<slug>.npz | ATUAL |
| [03_analise_treino.ipynb](outputs/notebooks_executados/shared-extrasystoles-v1/03_analise_treino.ipynb) | Curvas, BEST e validação MIT | SEM | BEST época 2; BA 0.9194 | model.pt, metrics.json, history.csv, dataset MIT | ATUAL |
| [04_teste_ludb.ipynb](outputs/notebooks_executados/shared-extrasystoles-v1/04_teste_ludb.ipynb) | Teste LUDB, erros e distribuições de scores | SEM | BA 0.6235 | model.pt e dataset LUDB; outputs finais preservados | ATUAL |
| [05_gradcam_demo.ipynb](outputs/notebooks_executados/shared-extrasystoles-v1/05_gradcam_demo.ipynb) | Painéis de Grad-CAM já calculados | SEM | VP/VN/FP/FN; quadrantes vazios explicitados | gradcam_ludb/ludb_gradcam.npz e dataset LUDB | ATUAL |
| [06_fidelidade_iou.ipynb](outputs/notebooks_executados/shared-extrasystoles-v1/06_fidelidade_iou.ipynb) | IoU salvo e tabela corrigida de ritmos | SEM | IoU 0.1141; ritmos com contagens 0/1 | fidelidade_ludb/ludb_fidelity.{json,npz}, Grad-CAM e dataset LUDB | ATUAL |
| [01_exploracao_mitbih.ipynb](outputs/notebooks_executados/shared-extrasystoles-bbb-v1/01_exploracao_mitbih.ipynb) | População, símbolos e exemplos MIT | COM | 99442 beats | data/processed/mitbih_<slug>.npz | ATUAL |
| [02_exploracao_ludb.ipynb](outputs/notebooks_executados/shared-extrasystoles-bbb-v1/02_exploracao_ludb.ipynb) | População, ritmos e máscaras LUDB | COM | 390 beats | data/processed/ludb_<slug>.npz | ATUAL |
| [03_analise_treino.ipynb](outputs/notebooks_executados/shared-extrasystoles-bbb-v1/03_analise_treino.ipynb) | Curvas, BEST e validação MIT | COM | BEST época 3; BA 0.8881 | model.pt, metrics.json, history.csv, dataset MIT | ATUAL |
| [04_teste_ludb.ipynb](outputs/notebooks_executados/shared-extrasystoles-bbb-v1/04_teste_ludb.ipynb) | Teste LUDB, erros e distribuições de scores | COM | BA 0.5755 | model.pt e dataset LUDB; outputs finais preservados | ATUAL |
| [05_gradcam_demo.ipynb](outputs/notebooks_executados/shared-extrasystoles-bbb-v1/05_gradcam_demo.ipynb) | Painéis de Grad-CAM já calculados | COM | VP/VN/FP/FN; quadrantes vazios explicitados | gradcam_ludb/ludb_gradcam.npz e dataset LUDB | ATUAL |
| [06_fidelidade_iou.ipynb](outputs/notebooks_executados/shared-extrasystoles-bbb-v1/06_fidelidade_iou.ipynb) | IoU salvo e tabela corrigida de ritmos | COM | IoU 0.1186; ritmos com contagens 0/1 | fidelidade_ludb/ludb_fidelity.{json,npz}, Grad-CAM e dataset LUDB | ATUAL |

## Como ler sem reexecutar o pipeline

Abra os outputs já salvos. **03 e 04 contêm chamadas de inferência em seu código**; não são necessários para consultar os relatórios. O 06 foi convertido para carregar o IoU salvo, sem chamar a função de cálculo. A revisão do 06 atualizou a tabela, os textos e o histograma por leitura direta; não executou o notebook inteiro.

## Material histórico

| Notebook | Objetivo histórico | Política | Resultados | Status |
| --- | --- | --- | --- | --- |
| [01_exploracao_mitbih.ipynb](notebooks/01_exploracao_mitbih.ipynb) | População, símbolos e exemplos MIT | Anterior às políticas finais | Outputs antigos preservados; não usar como execução final | HISTÓRICO |
| [02_exploracao_ludb.ipynb](notebooks/02_exploracao_ludb.ipynb) | População, ritmos e máscaras LUDB | Anterior às políticas finais | Outputs antigos preservados; não usar como execução final | HISTÓRICO |
| [03_analise_treino.ipynb](notebooks/03_analise_treino.ipynb) | Curvas, BEST e validação MIT | Anterior às políticas finais | Outputs antigos preservados; não usar como execução final | HISTÓRICO |
| [04_teste_ludb.ipynb](notebooks/04_teste_ludb.ipynb) | Teste LUDB, erros e distribuições de scores | Anterior às políticas finais | Outputs antigos preservados; não usar como execução final | HISTÓRICO |
| [05_gradcam_demo.ipynb](notebooks/05_gradcam_demo.ipynb) | Painéis de Grad-CAM já calculados | Anterior às políticas finais | Outputs antigos preservados; não usar como execução final | HISTÓRICO |
| [06_fidelidade_iou.ipynb](notebooks/06_fidelidade_iou.ipynb) | IoU salvo e tabela corrigida de ritmos | Anterior às políticas finais | Outputs antigos preservados; não usar como execução final | HISTÓRICO |

## Outros resultados

- [Relatório geral](RELATORIO_GERAL_PROJETO.md).
- [Relatório visual](RELATORIO_VISUAL_RESULTADOS.md).
- [Comparação por beat_id](outputs/comparacao_politicas/comparacao_sem_vs_com.json).
- COM somente completos está no JSON de teste COM e no relatório geral: é o mesmo modelo COM em um recorte, não um terceiro notebook/modelo.
