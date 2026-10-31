# Correções finais e organização da entrega

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

## A. Arquivos corrigidos

| Arquivo / conjunto | Problema | Correção |
| --- | --- | --- |
| [src/fidelity_ludb.py](src/fidelity_ludb.py) | Cobertura chamada de acaso | Comentários, docstrings e apresentação corrigidos; parâmetros e campos legados mantidos por compatibilidade |
| Dois notebooks oficiais 06 | Classe única para grupos mistos e baseline sem derivação | Contagens/percentuais 0/1; leitura de IoUs persistidos; histograma sem linha de acaso |
| Outros dez notebooks oficiais | Identificação insuficiente | Banner ATUAL, política, BEST e escopo da revisão; outputs numéricos preservados |
| Seis notebooks de notebooks/ | Outputs antigos podiam parecer atuais | Banner HISTÓRICO; notebooks e outputs preservados |
| [Relatório geral atual](RELATORIO_GERAL_PROJETO.md) | Interpretação de IoU e afirmação forte de repetibilidade | Texto qualificado; tabelas numéricas originais mantidas |
| [README atual](README.md) | Navegação e instalação ambíguas | Relatórios/guia oficiais e instalação separada de PyTorch |
| [Políticas atuais consolidadas](RELATORIO_GERAL_PROJETO.md#classificacao-detalhada) | Falta de orientação unificada | Status atual, duas políticas oficiais e natureza do recorte COM completos |
| Documentação histórica e duplicada | Regras antigas e navegação redundante | Removida da entrega; a metodologia atual está no relatório geral |
| requirements.txt | Índice cu124 divergente | Dependências gerais separadas de requirements-pytorch-cu128.txt |
| [src/compare_policies.py](src/compare_policies.py) | Lacunas na validação | Dez checks adicionais: histórico BEST/LAST e recorte completos derivado dos scores; opção --validate-only |
| relatorios/*.cjs | Faltava ferramenta editorial isolada | Leitor ZIP/NPY, gerador de apresentação, correções editoriais, HTML e validador independente; sem imports do pipeline |

## B. Problemas de interpretação corrigidos

A cobertura ≈ 0,576 não é baseline aleatório validado de IoU. Os campos `cobertura_acaso` e `acima_do_acaso` são históricos/deprecated; o segundo contém apenas uma diferença aritmética. Nenhum valor original de IoU foi alterado. Veja [o contrato](RELATORIO_GERAL_PROJETO.md#campos-legados-iou).

O notebook 06 agora mostra `n_total`, `n_classe_0`, `n_classe_1` e percentuais. Não usa o primeiro label como classe do ritmo. As tabelas e histogramas foram atualizados por leitura dos vetores salvos, sem reexecutar o notebook inteiro. A nova fonte do 06 carrega JSON/NPZ de fidelidade em vez de chamar seu cálculo.

Classificação e explicabilidade foram tratadas separadamente: recall zero em incomplete LBBB coexiste com IoU-QRS médio ≈ 0,401. Não se atribuiu prova clínica ou causalidade ao Grad-CAM.

## C. Documentação histórica

Os seis notebooks de `notebooks/` permanecem marcados HISTÓRICOS. Seus outputs não foram apagados ou reinterpretados como resultados finais. A documentação histórica e as exportações duplicadas em PDF/HTML foram removidas. As regras atuais, o ambiente e a compatibilidade dos campos de IoU foram consolidados no relatório geral. Os logs originais permanecem preservados como registro da execução, inclusive sua terminologia antiga.

## D. Ambiente

O ambiente registrado é Python 3.11.17, PyTorch 2.11.0+cu128; RTX 5080 declarada no relatório original. A instalação PyTorch cu128 está separada das dependências gerais. Nenhum pacote foi instalado nesta revisão.

O pacote não preserva todas as versões das demais dependências ou evidência suficiente de igualdade bit a bit entre repetições. A documentação não faz essa promessa. Consulte [o ambiente](RELATORIO_GERAL_PROJETO.md#ambiente-reprodutibilidade).

## E. Notebooks oficiais

Há seis nomes reais em cada uma das duas pastas de política, totalizando 12:

| Notebook | Conteúdo |
| --- | --- |
| 01_exploracao_mitbih.ipynb | População e símbolos MIT |
| 02_exploracao_ludb.ipynb | População, ritmos e máscaras LUDB |
| 03_analise_treino.ipynb | Histórico, BEST e validação MIT |
| 04_teste_ludb.ipynb | Teste externo e exemplos de erro |
| 05_gradcam_demo.ipynb | Painéis Grad-CAM existentes, inclusive quadrantes vazios |
| 06_fidelidade_iou.ipynb | IoU salvo e tabela corrigida por ritmo |

Pastas oficiais: `outputs/notebooks_executados/shared-extrasystoles-v1/` e `outputs/notebooks_executados/shared-extrasystoles-bbb-v1/`.

O [guia completo](GUIA_NOTEBOOKS_RESULTADOS.md) contém links individuais, políticas, artefatos e status. Os notebooks 03/04 ainda contêm código de inferência; seus outputs foram somente consultados nesta revisão.

## F. Relatório geral

[RELATORIO_GERAL_PROJETO.md](RELATORIO_GERAL_PROJETO.md), na raiz do repositório: 20 seções de metodologia/resultados e índice dos notebooks oficiais. Tabelas derivadas dos JSONs finais, sem copiar números históricos.

## G. Relatório visual

[RELATORIO_VISUAL_RESULTADOS.md](RELATORIO_VISUAL_RESULTADOS.md), na raiz do repositório. Usa **15 imagens**: 13 SVGs de apresentação e dois painéis PNG de Grad-CAM extraídos sem alteração dos notebooks. Outros dois histogramas SVG foram criados para os notebooks 06, totalizando 17 assets.

Os painéis originais selecionam extremos de score por quadrante; tabelas apresentam registro, beat_id, labels, score e subtipo. Nenhum exemplo de subtipo ausente do painel foi inventado. Links são relativos. A [versão Markdown atual](RELATORIO_VISUAL_RESULTADOS.md) é a referência visual da entrega.

## H. Verificações e pendências

Os 41 arquivos originais protegidos (checkpoints, datasets, scores, métricas, históricos, CSV clínico e logs) possuem hashes em [manifesto_artefatos_originais.json](relatorios/manifesto_artefatos_originais.json). A validação independente está em [verificacao_revisao_final.json](relatorios/verificacao_revisao_final.json).

Resultado final da ferramenta complementar: **314 verificações, 314 aprovadas, zero falhas**. Os 41 hashes originais permaneceram iguais. Os links locais dos documentos novos foram conferidos, os SVGs foram analisados como XML e o painel Grad-CAM COM extraído foi inspecionado visualmente.

As 85 verificações originais continuam como evidência histórica. O validador Python recebeu dez verificações adicionais e modo somente leitura; não foi executado nesta revisão. A ferramenta Node complementar foi executada sem carregar modelos nem importar código de inferência/Grad-CAM/IoU. Ela confere máximos dos históricos e recalcula apenas métricas a partir dos scores existentes.

| Classificação | Pendência / condição |
| --- | --- |
| CRÍTICO | Nenhum problema confirmado que invalide os resultados preservados |
| IMPORTANTE | Nenhuma correção editorial obrigatória conhecida ficou sem tratamento; a receita de instalação não foi testada num ambiente novo |
| DESEJÁVEL | Obter lockfile completo e evidência verificável de repetição independente em trabalho futuro; não recriar essa evidência retroativamente |
| DESEJÁVEL | Executar futuramente o modo Python --validate-only no ambiente documentado; nesta revisão foi usada validação Node independente |
| LIMITAÇÃO METODOLÓGICA | Um fold/seed; diagnóstico LUDB por registro propagado aos beats; poucos registros por subtipo; negativos conservadores; IoU sem baseline aleatório validado |

## I. Veredito final

**SIM COM RESSALVAS.** O pacote está organizado para entrega acadêmica e mantém explícitos os resultados externos fracos e as limitações. Não é validação clínica nem garantia de reprodutibilidade bit a bit em outro ambiente.

Nenhum treinamento, inferência, Grad-CAM ou cálculo novo de IoU foi executado. Os relatórios foram produzidos exclusivamente a partir dos códigos e artefatos já existentes em pos3.
