# Decisões de classificação

Regras definidas pelo grupo. `ONLY_ARRHYTHMIA=False` é o padrão; `True` ativa somente arritmia. O modo restrito não corresponde ao agrupamento AAMI.

## MIT-BIH

| Símbolo/grupo | Padrão | Somente arritmia | Motivo resumido |
| --- | --- | --- | --- |
| N | 0 | 0 | Batimento normal |
| L/R | 1 | 0 | Alteração de condução; permanece como negativo no modo restrito |
| A/a/J/S/V | 1 | 1 | Batimentos considerados arrítmicos |
| e/j/E | 1 | 1 | Batimentos de escape considerados positivos pelo projeto |
| F | 1 | 1 | Batimento de fusão considerado positivo pelo projeto |
| Q | Excluir | Excluir | Batimento não classificável |
| Pacing, sem `--include-paced` | Excluir | Excluir | Fora da população principal; registros 102/104/107/217 e símbolos `/`/`f` |
| `/` e `f`, com `--include-paced` | 0 | 0 | Decisão explícita do grupo para a população alternativa |

Non-beats mantêm o filtro existente. Símbolos desconhecidos causam erro, nunca um label automático.

## LUDB

| Categoria | Padrão | Somente arritmia | Explicação simples |
| --- | --- | --- | --- |
| Rhythms | Não interfere | Não interfere | Descreve o ritmo do registro, não identifica diretamente o beat individual |
| Electric axis | Não interfere | Não interfere | Não foi escolhido como critério do ground truth |
| Conduction abnormalities | Força 1 | Não interfere | Anormalidade elétrica incluída apenas no modo padrão |
| Extrasystolies | Força 1 | Força 1 | Principal critério arrítmico adotado para o LUDB |
| Hypertrophies | Não interfere | Não interfere | Diagnóstico preservado, mas não escolhido como critério |
| Cardiac pacing, sem `--include-paced` | Excluir | Excluir | Registro fora da população principal |
| Cardiac pacing, com `--include-paced` | Não interfere | Não interfere | Registro permitido; as demais categorias decidem o label |
| Ischemia | Força 1 | Não interfere | Alteração relevante apenas para o modo padrão |
| Non-specific repolarization abnormalities | Não interfere | Não interfere | Alteração inespecífica, não utilizada como critério |
| Other states | Não interfere | Não interfere | Categoria ampla, não utilizada para definir o label |

**Força 1:** basta uma categoria selecionada preenchida para o registro ser positivo (OU lógico). **Não interfere:** não participa da decisão; não significa normalidade clínica. Sem inclusão explícita, pacing exclui o registro antes dessa decisão.

O label do registro é propagado a todos os beats aceitos. Isso pode gerar falsos positivos por beat, limitação conhecida e aceita pelo grupo. A exclusão histórica por ausência de `Rhythm` no header permanece, por confirmação do grupo.

Fonte clínica: CSV oficial LUDB 1.0.1. Célula vazia significa ausência de diagnóstico. Arquivo, coluna obrigatória ou registro ausente interrompem o processamento. Conteúdo e motivos de positividade são preservados por registro.
