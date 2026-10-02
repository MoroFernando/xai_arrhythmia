# Classificação atualizada

Dois modos: padrão (`ONLY_ARRHYTHMIA=False`), arritmias + anormalidades selecionadas; e somente arritmia (`--only-arrhythmia`). `--include-paced` é independente e desativado por padrão.

## MIT-BIH

Padrão: N=0; L/R/A/a/J/S/V/F/e/j/E=1. Somente arritmia: N/L/R=0; A/a/J/S/V/F/e/j/E=1. Q é sempre excluído; símbolo desconhecido causa erro. Com `--include-paced`, `/` e `f`=0 em ambos os modos; registros sem a derivação escolhida continuam sujeitos ao filtro histórico.

## LUDB

CSV oficial 1.0.1: padrão usa Conduction abnormalities **OU** Extrasystolies **OU** Ischemia; modo restrito usa apenas Extrasystolies. Categoria selecionada preenchida força 1; nenhuma preenchida gera 0. Demais categorias não interferem.

Pacing exclui o registro, exceto com `--include-paced`: nesse caso não interfere no label. Ausência de Rhythm no header continua excluindo. Labels são propagados aos beats aceitos, sem mudar a segmentação.

## Resultados

Contagens de beats/classes nos dois modos, Q removidos, registros/beats excluídos por pacing, positivos por categoria, sobreposições e métricas: **pendentes de execução**. Nenhum resultado foi regenerado nesta implementação. Os metadados registrarão exclusões, motivos e combinações de categorias; Q removidos são contados após os filtros históricos de borda.

Datasets/resultados novos identificam as duas opções e a versão da política; consumidores rejeitam identificação ausente/incompatível. SHA-256 vincula resultados ao dataset/checkpoint. Saídas existentes não são sobrescritas. Os seis notebooks e resultados antigos são históricos, não validam a nova política.

## Como executar

Executar posteriormente, em ambiente apropriado, a partir da raiz do repositório, com as dependências instaladas. A branch usa scripts `python src/...`, sem configuração `uv` vigente.

Antes do LUDB, salvar o [CSV oficial 1.0.1](https://physionet.org/files/ludb/1.0.1/ludb.csv) em `data/raw/ludb/ludb.csv`, ou informar `--clinical-csv CAMINHO`. O CSV local é obrigatório mesmo com `--download`/`--stream`.

### Modo padrão

```bash
python src/preprocess_mitbih.py --download
python src/train_cnn.py
python src/preprocess_ludb.py --download
python src/test_ludb.py
python src/extract_gradcam.py
python src/fidelity_ludb.py
```

### Modo somente arritmia

```bash
python src/preprocess_mitbih.py --download --only-arrhythmia
python src/train_cnn.py --only-arrhythmia
python src/preprocess_ludb.py --download --only-arrhythmia
python src/test_ludb.py --only-arrhythmia
python src/extract_gradcam.py --only-arrhythmia
python src/fidelity_ludb.py --only-arrhythmia
```

Para incluir pacing, acrescentar `--include-paced` a **todos os seis comandos** do modo escolhido. As opções devem coincidir em todas as etapas. Após o primeiro download, pode-se omitir `--download`.

Os defaults são `data/processed/{mitbih,ludb}_{modo}.npz` e `outputs/models/cnn1d_mitbih_{modo}/`, com `{modo}` igual a `selected_abnormalities_without_pacing`, `only_arrhythmia_without_pacing` ou suas variantes `_with_pacing`. Resultados ficam nas subpastas `teste_ludb`, `gradcam_ludb` e `fidelidade_ludb`.

Para repetir uma execução, escolher destinos novos com `--output`, `--run-name`/`--models-dir` e `--output-dir`, passando as entradas correspondentes por `--dataset`, `--checkpoint` e `--gradcam`. Não usar os notebooks históricos para a nova execução. `--normal-symbols` e `--normal-rhythms` foram retirados.
