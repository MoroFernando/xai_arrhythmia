# Relatório geral do projeto

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


STATUS: ATUAL — revisão de 30/10/2026. Fonte exclusiva: códigos e artefatos finais de pos3.

## 1. Objetivo do projeto

Classificar beats individuais de ECG em uma tarefa binária com CNN 1D, treinada/validada no MIT-BIH e avaliada externamente no LUDB. Grad-CAM e IoU descrevem a relação entre mapas de influência e regiões P/QRS/T. O estudo não constitui diagnóstico médico nem validação para uso clínico.

## 2. Bases utilizadas

MIT-BIH 1.0.0 fornece símbolos por beat. LUDB 1.0.1 fornece delineações de ondas e diagnósticos predominantemente por registro, propagados aos beats aceitos. Essa diferença de granularidade é uma limitação aceita: positivo LUDB não comprova uma extrasístole individual em cada janela.

| Base / modo | Beats | Classe 0 | Classe 1 | Registros |
| --- | --- | --- | --- | --- |
| SEM MIT | 84115 | 74517 | 9598 | 44 |
| SEM LUDB | 192 | 107 | 85 | 28 |
| COM MIT | 99442 | 74517 | 24925 | 44 |
| COM LUDB | 390 | 107 | 283 | 58 |

## 3. Pipeline

Preprocessing e normalização por beat → classificação conforme política → split por sujeito → CNN → BEST exclusivamente pela validação MIT → teste LUDB com limiar fixo → Grad-CAM → comparação com máscaras por IoU. Esta revisão não executou nenhuma dessas etapas. Os hashes ligam datasets, modelos, predições e mapas.

## 4. Políticas de classificação

| Política oficial | Nome | MIT 0 | MIT 1 | LUDB positivo |
| --- | --- | --- | --- | --- |
| shared-extrasystoles-v1 | SEM_BLOQUEIO | N | A/a/V | Extrasístoles atriais/ventriculares |
| shared-extrasystoles-bbb-v1 | COM_BLOQUEIO | N | A/a/V/L/R | Anteriores + LBBB/RBBB completos/incompletos |

COM somente completos é análise adicional do mesmo modelo COM, não uma política de treinamento ou terceiro modelo.

## 5. Exclusões

Pacing é excluído: registros MIT 102/104/107/217 e símbolos / e f, além de registros LUDB com pacing. Outros símbolos MIT (E/F/J/Q/S/e/j, e L/R no SEM) não criam positivos. No LUDB, Ischemia, Hypertrophies e outras condições sem alvo compartilhado levam à exclusão, não a 0. Com alvo válido, permanecem como comorbidades. Rhythms é metadado e não decide 0/1. A referência negativa exige eixo explicitamente normal e categorias clínicas não alvo vazias.

## 6. Split

StratifiedGroupKFold, cinco folds, fold 0, seed 0. Um manifesto atribui os mesmos sujeitos às duas políticas. São 32 sujeitos no treino e 11 na validação, sem interseção. Registros 201 e 202 compartilham subject_id=201 e permanecem na validação. O fold foi estratificado com labels SEM e reutilizado no COM; as prevalências diferem no COM (treino 18,37%, validação 44,87%).

## 7. Arquitetura e treinamento

Entrada (1,234); Conv1d 1→16, kernel 7, padding 3, ReLU e MaxPool1d(2); Conv1d 16→32, kernel 5, padding 2, ReLU e MaxPool1d(2); flatten (1856 atributos); Linear 1856→64, ReLU; Linear 64→1 e sigmoide. Total: 121.633 parâmetros. Sem dropout, reponderação de classes ou early stopping no código auditado. BCE sobre a sigmoide; Adam, LR 0,001, batch 64, 20 épocas, threshold 0,5. Janela 90 amostras antes e 144 depois, 360 Hz; normalização z-score por beat; LUDB reamostrado de 500 Hz.

A cada melhora estrita de BA MIT, uma cópia dos melhores pesos é mantida em memória. Ao final, model.pt recebe o BEST e model_last.pt recebe a última época. Empates mantêm a época anterior; LUDB não seleciona modelos.

| Modelo | Best epoch | Best BA MIT | Última época | Última BA MIT |
| --- | --- | --- | --- | --- |
| SEM | 2 | 0.9194 | 20 | 0.9036 |
| COM | 3 | 0.8881 | 20 | 0.8150 |

## 8. Resultados MIT

| Avaliação | n | Accuracy | BA | Precision | Recall | Specificity | F1 | ROC-AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEM treino BEST | 68445 | 0.9887 | 0.9627 | 0.9703 | 0.9291 | 0.9964 | 0.9493 | 0.9939 | 60453 | 221 | 551 | 7220 |
| SEM validação BEST | 15670 | 0.9797 | 0.9194 | 0.9827 | 0.8407 | 0.9980 | 0.9062 | 0.9666 | 13816 | 27 | 291 | 1536 |
| COM treino BEST | 74332 | 0.9928 | 0.9838 | 0.9911 | 0.9696 | 0.9980 | 0.9802 | 0.9990 | 60555 | 119 | 415 | 13243 |
| COM validação BEST | 25110 | 0.8995 | 0.8881 | 0.9986 | 0.7770 | 0.9991 | 0.8740 | 0.9601 | 13831 | 12 | 2512 | 8755 |

As métricas de treino BEST foram avaliadas com os pesos congelados. As colunas por época do history.csv são médias durante aquela época, e não necessariamente coincidem com essa avaliação.

| Modelo | Símbolo | n validação | Acertos | Erros | Recall da classe |
| --- | --- | --- | --- | --- | --- |
| SEM | N | 13843 | 13816 | 27 | 0.9980 |
| SEM | V | 1417 | 1348 | 69 | 0.9513 |
| SEM | A | 294 | 134 | 160 | 0.4558 |
| SEM | a | 116 | 54 | 62 | 0.4655 |
| COM | N | 13843 | 13831 | 12 | 0.9991 |
| COM | R | 4948 | 3067 | 1881 | 0.6198 |
| COM | L | 4492 | 4094 | 398 | 0.9114 |
| COM | V | 1417 | 1354 | 63 | 0.9555 |
| COM | A | 294 | 141 | 153 | 0.4796 |
| COM | a | 116 | 99 | 17 | 0.8534 |

A taxa de N é a especificidade binária. A/a permanecem difíceis no SEM e A/R no COM. Há overfitting: loss de treino continua caindo enquanto a validação oscila e termina pior. A queda best→last de BA é maior no COM. A escolha do BEST evita a última época pior no MIT, mas não existe teste LUDB do LAST deste treino para quantificar ganho externo.

## 9. Resultados LUDB

| Avaliação | n | Accuracy | BA | Precision | Recall | Specificity | F1 | ROC-AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEM, n=192 | 192 | 0.6667 | 0.6235 | 1.0000 | 0.2471 | 1.0000 | 0.3962 | 0.7664 | 107 | 0 | 64 | 21 |
| COM, n=390 | 390 | 0.3923 | 0.5755 | 0.9600 | 0.1696 | 0.9813 | 0.2883 | 0.7143 | 105 | 2 | 235 | 48 |
| COM completos, n=226 | 226 | 0.6283 | 0.6461 | 0.9487 | 0.3109 | 0.9813 | 0.4684 | 0.7958 | 105 | 2 | 82 | 37 |

| População | Sempre 0: Accuracy | Sempre 1: Accuracy | Majoritária | BA dos baselines |
| --- | --- | --- | --- | --- |
| SEM | 0.5573 | 0.4427 | 0 | 0,5000 |
| COM | 0.2744 | 0.7256 | 1 | 0,5000 |
| COM completos | 0.4735 | 0.5265 | 1 | 0,5000 |

A Accuracy COM fica abaixo do baseline majoritário porque há muitos positivos e baixo recall. A BA pondera igualmente sensibilidade/especificidade e supera 0,5; isso não torna o desempenho externo satisfatório. Não interpretar Accuracy isoladamente.

## 10. Comparação SEM × COM nos mesmos beats

Interseção exata por beat_id: 192 comuns, zero exclusivos SEM e 198 exclusivos COM; mesmos registros e y_true. Inclui os 31 beats de seis registros com extrasístole + bloqueio (22, 60, 66, 78, 83, 108).

| Avaliação | n | Accuracy | BA | Precision | Recall | Specificity | F1 | ROC-AUC | TN | FP | FN | TP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SEM — mesmos beats | 192 | 0.6667 | 0.6235 | 1.0000 | 0.2471 | 1.0000 | 0.3962 | 0.7664 | 107 | 0 | 64 | 21 |
| COM — mesmos beats | 192 | 0.6406 | 0.5965 | 0.9000 | 0.2118 | 0.9813 | 0.3429 | 0.7351 | 105 | 2 | 67 | 18 |

| Grupo | 0→0 | 0→1 | 1→0 | 1→1 |
| --- | --- | --- | --- | --- |
| todos | 169 | 2 | 3 | 18 |
| negativos | 105 | 2 | 0 | 0 |
| atrial_extrasystole | 27 | 0 | 3 | 6 |
| ventricular_extrasystole | 37 | 0 | 0 | 12 |

Só cinco decisões mudaram: três atriais corretas no SEM passaram a FN e dois negativos a FP.

## 11. Extrasístoles

| Família | Modelo | n | TP | FN | Recall | Score médio | Score mediano |
| --- | --- | --- | --- | --- | --- | --- | --- |
| atrial_extrasystole | SEM | 36 | 9 | 27 | 0.2500 | 0.2568 | 0.0273 |
| atrial_extrasystole | COM | 36 | 6 | 30 | 0.1667 | 0.1895 | 0.0213 |
| ventricular_extrasystole | SEM | 49 | 12 | 37 | 0.2449 | 0.2479 | 0.0314 |
| ventricular_extrasystole | COM | 49 | 12 | 37 | 0.2449 | 0.2917 | 0.0955 |

A adição dos bloqueios está associada à perda de três detecções atriais nesta execução; decisões ventriculares permanecem iguais. A variação de scores não é uniforme. As famílias do LUDB não são equivalentes a anotações individuais A/a/V no MIT.

## 12. Bloqueios

| Subtipo | Beats | Registros | TP | FN | Recall | FNR | Score médio | Score mediano |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Complete LBBB | 24 | 4 | 23 | 1 | 0.9583 | 0.0417 | 0.9018 | 0.9869 |
| Incomplete LBBB | 24 | 3 | 0 | 24 | 0.0000 | 1.0000 | 0.0395 | 0.0010 |
| Complete RBBB | 24 | 3 | 6 | 18 | 0.2500 | 0.7500 | 0.2934 | 0.1637 |
| Incomplete RBBB | 157 | 26 | 13 | 144 | 0.0828 | 0.9172 | 0.1252 | 0.0079 |

Os incompletos têm recall e scores baixos. Complete RBBB também tem recall baixo, apesar de superar incomplete RBBB. Poucos registros por subtipo e comorbidades impedem conclusões fortes sobre compatibilidade com MIT L/R.

## 13. Principais falsos negativos

| Grupo exclusivo | Positivos | FN |
| --- | --- | --- |
| Somente incompleto | 164 | 153 |
| Somente extrasístole | 54 | 48 |
| Somente completo | 34 | 15 |
| Extrasístole + incompleto | 17 | 15 |
| Extrasístole + completo | 14 | 4 |

Total exclusivo = 235 FN. Na atribuição multirrótulo, 168 FN estão associados a incompletos (71,49%), 67 a extrasístoles e 19 a completos. Não somar categorias multirrótulo como independentes. **Concentração não significa causalidade.**

## 14. COM somente completos

Mesmos checkpoint COM e scores; exclui 164 beats cuja única justificativa positiva é bloqueio incompleto. Mantém 226 beats (107 negativos/119 positivos), inclusive extrasístole + incompleto e qualquer alvo completo coexistente. Nenhum removido vira 0. A melhora de BA para 0.6461 decorre da população avaliada, não de outro treinamento.

## 15. Grad-CAM

O código usa a saída positiva (probabilidade de classe 1), gradientes médios temporais por canal em conv2, soma ponderada e ReLU, interpolação linear e normalização pelo máximo de cada beat. Mesmo em uma previsão 0, o mapa é relativo à saída positiva; não é uma explicação específica da classe negativa. O Grad-CAM indica regiões que mais influenciaram a saída do modelo; não localiza doença nem prova causalidade clínica.

Os hashes de checkpoint são os do BEST. Classes previstas coincidem com o teste; diferença máxima nos scores: SEM 0 e COM aproximadamente 2,116e-6.

## 16. IoU

IoU = interseção/união entre o mapa já binarizado em 0,5 e a máscara clínica. O valor médio geral compara à união P+QRS+T; há também medidas por estrutura. Esta revisão apenas agregou os vetores já calculados.

| Modelo/população | IoU geral | P | QRS | T |
| --- | --- | --- | --- | --- |
| SEM — n=192 | 0.1141 | 0.0027 | 0.3501 | 0.0259 |
| COM — n=390 | 0.1186 | 0.0114 | 0.3292 | 0.0477 |

| Modelo — mesmos 192 | IoU geral | P | QRS | T |
| --- | --- | --- | --- | --- |
| SEM | 0.1141 | 0.0027 | 0.3501 | 0.0259 |
| COM | 0.1186 | 0.0113 | 0.3560 | 0.0351 |

A cobertura média das regiões clínicas é aproximadamente 0,576. Esse valor não foi validado como baseline aleatório de IoU e, portanto, não deve ser usado para afirmar que um resultado está acima ou abaixo do acaso. Os campos cobertura_acaso e acima_do_acaso nos artefatos originais são **deprecated**: representam cobertura e diferença aritmética, sem inferência estatística. Foram preservados por compatibilidade; consulte [o contrato dos campos](RELATORIO_GERAL_PROJETO.md#campos-legados-iou).

## 17. Classificação versus explicabilidade

| Modelo | Grupo | n | IoU médio | IoU mediano |
| --- | --- | --- | --- | --- |
| SEM | Acertos | 128 | 0.1018 | 0.0863 |
| SEM | Erros | 64 | 0.1387 | 0.1190 |
| COM | Acertos | 153 | 0.1171 | 0.1103 |
| COM | Erros | 237 | 0.1196 | 0.1103 |

Classificação correta ≠ IoU alto. Incomplete LBBB tem recall 0%, mas IoU-QRS médio ≈ 0,4008; isso não demonstra explicação correta. Erros têm IoU geral médio maior que acertos no SEM e muito semelhante no COM. Não interpretar associação como causalidade.

## 18. Limitações

Diagnóstico LUDB por registro propagado para beats; um fold/seed; poucos registros em alguns subtipos; seleção conservadora de negativos; eixos vertical/horizontal isolados excluídos; split comum com prevalências diferentes no COM; famílias sobrepostas; mapas normalizados que não preservam magnitude absoluta; P ausente pode produzir IoU-P zero quando há mapa ativo; IoU sem baseline aleatório validado. Limiar 0,5 não foi otimizado no LUDB e não deve ser ajustado no teste externo.

## 19. Reprodutibilidade

Ambiente registrado: Python 3.11.17, PyTorch 2.11.0+cu128; RTX 5080 declarada no relatório de execução. Seed 0 para Python/NumPy/PyTorch, cudnn.deterministic=True e cudnn.benchmark=False. Não há torch.use_deterministic_algorithms(True) nem evidência suficiente no pacote para comprovar igualdade bit a bit entre execuções independentes.

Veja [o ambiente](RELATORIO_GERAL_PROJETO.md#ambiente-reprodutibilidade). requirements gerais e instalação exata de PyTorch cu128 foram separados; as versões executadas das demais bibliotecas não foram todas preservadas, logo não existe lockfile completo da execução. A alegação original de duas repetições idênticas foi qualificada.

Os 10 JSONs originais são válidos, sem NaN/Infinity e sem caminhos absolutos locais. Os artefatos originais estão protegidos por [manifesto de hashes](geracao/manifesto_artefatos_originais.json). O validador complementar trabalha sem importar código do pipeline.

## 20. Conclusões

O BEST é selecionado corretamente e usado em toda a cadeia. A comparação pareada e o recorte COM somente completos são coerentes. O desempenho externo é limitado, especialmente a sensibilidade e os bloqueios incompletos; esses achados não devem ser escondidos. Não há evidência para uso clínico ou atribuição causal dos erros. As correções editoriais não alteram os números dos experimentos.

## Onde encontrar os resultados

Treinamento/validação: notebook 03; teste externo e exemplos de erro: 04; Grad-CAM: 05; IoU e metadados por ritmo: 06. Populações MIT/LUDB: 01/02.

### SEM

- [01_exploracao_mitbih.ipynb](../outputs/notebooks_executados/shared-extrasystoles-v1/01_exploracao_mitbih.ipynb) — População, símbolos e exemplos MIT.
- [02_exploracao_ludb.ipynb](../outputs/notebooks_executados/shared-extrasystoles-v1/02_exploracao_ludb.ipynb) — População, ritmos e máscaras LUDB.
- [03_analise_treino.ipynb](../outputs/notebooks_executados/shared-extrasystoles-v1/03_analise_treino.ipynb) — Curvas, BEST e validação MIT.
- [04_teste_ludb.ipynb](../outputs/notebooks_executados/shared-extrasystoles-v1/04_teste_ludb.ipynb) — Teste LUDB, erros e distribuições de scores.
- [05_gradcam_demo.ipynb](../outputs/notebooks_executados/shared-extrasystoles-v1/05_gradcam_demo.ipynb) — Painéis de Grad-CAM já calculados.
- [06_fidelidade_iou.ipynb](../outputs/notebooks_executados/shared-extrasystoles-v1/06_fidelidade_iou.ipynb) — IoU salvo e tabela corrigida de ritmos.

### COM

- [01_exploracao_mitbih.ipynb](../outputs/notebooks_executados/shared-extrasystoles-bbb-v1/01_exploracao_mitbih.ipynb) — População, símbolos e exemplos MIT.
- [02_exploracao_ludb.ipynb](../outputs/notebooks_executados/shared-extrasystoles-bbb-v1/02_exploracao_ludb.ipynb) — População, ritmos e máscaras LUDB.
- [03_analise_treino.ipynb](../outputs/notebooks_executados/shared-extrasystoles-bbb-v1/03_analise_treino.ipynb) — Curvas, BEST e validação MIT.
- [04_teste_ludb.ipynb](../outputs/notebooks_executados/shared-extrasystoles-bbb-v1/04_teste_ludb.ipynb) — Teste LUDB, erros e distribuições de scores.
- [05_gradcam_demo.ipynb](../outputs/notebooks_executados/shared-extrasystoles-bbb-v1/05_gradcam_demo.ipynb) — Painéis de Grad-CAM já calculados.
- [06_fidelidade_iou.ipynb](../outputs/notebooks_executados/shared-extrasystoles-bbb-v1/06_fidelidade_iou.ipynb) — IoU salvo e tabela corrigida de ritmos.

Consulte também [o guia completo](GUIA_NOTEBOOKS_RESULTADOS.md), [o relatório visual](RELATORIO_VISUAL_RESULTADOS.md) e [o resumo desta revisão](RELATORIO_CORRECOES_FINAIS.md).


<a id="classificacao-detalhada"></a>

## Referência das políticas atuais

### MIT-BIH

| Símbolo | `SEM_BLOQUEIO` | `COM_BLOQUEIO` | Justificativa |
| --- | --- | --- | --- |
| `N` | 0 | 0 | referência negativa |
| `A`, `a` | 1 | 1 | extrasístoles atriais compartilhadas |
| `V` | 1 | 1 | extrasístole ventricular compartilhada |
| `L` | excluir | 1 | beat com LBBB; incluído somente no experimento com bloqueio |
| `R` | excluir | 1 | beat com RBBB; incluído somente no experimento com bloqueio |
| `J` | excluir | excluir | prematuro juncional sem contraparte LUDB direta |
| `S` | excluir | excluir | supraventricular prematuro sem origem necessariamente atrial |
| `F` | excluir | excluir | fusão de beat normal e ventricular |
| `e` | excluir | excluir | escape atrial não equivale a extrasístole atrial |
| `j` | excluir | excluir | escape juncional sem contraparte direta |
| `E` | excluir | excluir | escape ventricular não equivale a PVC |
| `Q` | excluir | excluir | beat não classificável |
| `/` | excluir | excluir | paced beat |
| `f` | excluir | excluir | fusão de paced e normal |

Os registros paced 102, 104, 107 e 217 são excluídos. A relação especial entre os registros 201 e 202 é preservada por `subject_ids`: os dois recebem o mesmo grupo no split. O id do registro continua disponível separadamente.

### LUDB

Extrasístoles atriais aceitas:

- `Atrial extrasystole, type: single pac`
- `Atrial extrasystole: sa-nodal extrasystole`
- `Atrial extrasystole: undefined`
- `Atrial extrasystole: left atrial`
- `Atrial extrasystole: low atrial`
- `Atrial extrasystole, type: quadrigemini`
- `Atrial extrasystole, type: allorhythmic pattern`
- `Atrial extrasystole, type: bigemini`

Extrasístoles ventriculares aceitas:

- `Ventricular extrasystole, localisation: rvot, anterior wall`
- `Ventricular extrasystole, localisation: rvot, antero-septal part`
- `Ventricular extrasystole, localisation: lvot, lvs`
- `Ventricular extrasystole, localisation: ivs, middle part`
- `Ventricular extrasystole, localisation: lv, undefined`
- `Ventricular extrasystole, type: intercalary pvc`
- `Ventricular extrasystole, type: single pvc`
- `Ventricular extrasystole, type: couplet`
- `Ventricular extrasystole, morphology: polymorphic`

Somente `COM_BLOQUEIO` também aceita:

- `Complete left bundle branch block`
- `Incomplete left bundle branch block`
- `Complete right bundle branch block`
- `Incomplete right bundle branch block`

Valores inesperados em `Extrasystolies` ou `Conduction abnormalities` causam erro para impedir fallback silencioso. Condições como isquemia, hipertrofias, alterações de eixo, alterações de repolarização, fibrilação/flutter atrial, bloqueios AV, bloqueio sinoatrial, hemibloqueio anterior esquerdo e atraso intraventricular inespecífico não viram classe 0. Sem um alvo compartilhado, elas deixam o registro fora da tarefa.

Comorbidades não compartilhadas não removem um registro que possua alvo válido. Elas são preservadas em `other_conditions`; `target_families` e `target_subtypes` registram o motivo da inclusão. Esta decisão permite a análise de sensibilidade futura com e sem comorbidades sem alterar silenciosamente a coorte principal.

### Limitação e comparação

O LUDB fornece o diagnóstico principalmente por registro e o projeto o propaga aos beats aceitos. A avaliação permanece beat-level, com essa limitação explícita. `L` e `R` no MIT-BIH não informam se o bloqueio é completo ou incompleto; por isso os modos são treinados e avaliados separadamente.

O subconjunto comum SEM x COM é a interseção exata de `beat_id`: referências negativas e registros com extrasístole atrial/ventricular, inclusive os que também têm LBBB/RBBB. Sobreposições extrasístole + bloqueio não são removidas. A avaliação "COM_BLOQUEIO — somente completos" reutiliza o checkpoint e as predições do COM e exclui apenas positivos cuja única justificativa é bloqueio incompleto; esses beats ficam fora do recorte, nunca viram 0.

Uma eventual queda em `COM_BLOQUEIO` não prova que bloqueios incompletos sejam a causa. A comparação deve considerar métricas globais, o subconjunto compartilhado de extrasístoles, LBBB/RBBB por lado e completude, distribuição de classes, pacientes e número de beats. Em recortes positivos por subtipo, sensibilidade, falsos negativos e scores são válidos; especificidade e AUC exigem um conjunto negativo definido e não são reportadas automaticamente.


A referência negativa LUDB requer exatamente `Electric axis of the heart: normal` no CSV oficial 1.0.1 e células vazias em `Conduction abnormalities`, `Extrasystolies`, `Hypertrophies`, `Cardiac pacing`, `Ischemia`, `Non-specific repolarization abnormalities` e `Other states`. O campo `Rhythms` não participa dessa regra. Símbolos ou valores clínicos inesperados interrompem o processamento, evitando fallback silencioso.

<a id="ambiente-reprodutibilidade"></a>

## Ambiente e reprodutibilidade da entrega

### O que foi preservado sobre a execução

| Item | Evidência em pos3 |
| --- | --- |
| Python | 3.11.17 nos metadados dos modelos |
| PyTorch | 2.11.0+cu128 nos metadados dos modelos |
| CUDA da distribuição PyTorch | cu128, distribuição para CUDA 12.8; não comprova um toolkit local instalado |
| GPU | NVIDIA RTX 5080 declarada no relatório de execução; sem inventário independente de hardware |
| Seed | 0, aplicada a random, NumPy e PyTorch no código de treino |
| cuDNN | deterministic=True, benchmark=False |

O pipeline foi configurado para favorecer reprodutibilidade/determinismo, mas não há no pacote evidência suficiente para comprovar igualdade bit a bit entre execuções independentes. O código não utiliza `torch.use_deterministic_algorithms(True)`. Não existe lockfile completo das versões executadas de NumPy, SciPy e demais bibliotecas, nem registro completo de driver/cuDNN.

### Instalação futura do ambiente GPU registrado

Use Python **3.11.17** em um ambiente isolado. A partir desta pasta (`xai_arrhythmia`):

```text
python -m pip install -r requirements-pytorch-cu128.txt
python -m pip install -r requirements.txt
```

O primeiro arquivo fixa `torch==2.11.0+cu128` no índice cu128; o segundo contém dependências gerais, sem selecionar outra build de torch. A instrução anterior com cu124 foi substituída. Não há promessa de compatibilidade universal com qualquer GPU/driver: é necessário driver compatível com a distribuição escolhida e verificar disponibilidade da build na plataforma de destino. Esta receita foi documentada, não instalada/testada nesta revisão.

Para consultar resultados, use os relatórios Markdown e os outputs salvos dos [notebooks oficiais](GUIA_NOTEBOOKS_RESULTADOS.md). As ferramentas editoriais em `relatorios/*.cjs` usam somente módulos nativos de Node.js e não dependem de CUDA/PyTorch.

### Limites da reprodução

- Históricos e checkpoints comprovam quais pesos foram entregues, não igualdade com uma execução independente futura.
- BEST é escolhido pela validação MIT, com threshold fixo 0,5; LUDB não participa.
- O split preservado mantém 201/202 no mesmo grupo.
- Reprodução integral futura exige os sinais brutos não incluídos e versões do ambiente compatíveis; não é necessária para ler os resultados atuais.
- Não se deve substituir datasets/checkpoints/predições existentes numa revisão editorial.

### Verificação somente leitura

Com Python disponível, `python src/compare_policies.py --validate-only` verifica e agrega os artefatos sem gravar resultados. Esse modo não treina, infere ou calcula mapas/IoU. O script recebeu dez verificações adicionais às 85 originais; o resultado original de 85/85 foi preservado.

Nesta revisão foi executado o validador complementar independente `node relatorios/validar_resultados.cjs --read-only` a partir de `pos3`, com resultado em `relatorios/verificacao_revisao_final.json`. O Python do pipeline não foi executado.


<a id="campos-legados-iou"></a>

## Compatibilidade dos campos persistidos de IoU

Os JSONs, NPZs e logs originais foram preservados byte a byte. Não houve novo cálculo de IoU. Alguns nomes permanecem históricos/deprecated:

| Campo original | Significado real | Interpretação permitida |
| --- | --- | --- |
| `cobertura_acaso` | Fração média da janela coberta pela máscara clínica | Cobertura descritiva; não é expected random IoU nem baseline aleatório validado |
| `acima_do_acaso` | IoU médio menos a cobertura acima | Diferença aritmética legada, sem significado de significância ou desempenho acima/abaixo do acaso |

A cobertura média das regiões clínicas é aproximadamente 0,576. Esse valor não foi validado como baseline aleatório de IoU e, portanto, não deve ser usado para afirmar que um resultado está acima ou abaixo do acaso.

Os nomes foram mantidos porque renomear campos persistidos quebraria consumidores e invalidaria vínculos SHA-256. A documentação, os gráficos atuais e os textos de apresentação foram corrigidos. O parâmetro `cobertura_acaso` nas funções públicas também foi mantido por compatibilidade e documentado como deprecated. Os valores reais de IoU não foram alterados.

Os logs são registros históricos imutáveis: seu uso antigo da palavra “acaso” não é interpretação científica atual. Notebooks históricos possuem aviso inicial; notebooks 06 oficiais passaram a ler IoUs persistidos, sem executar a função de cálculo.
