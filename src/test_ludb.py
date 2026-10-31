"""CLI de teste do modelo treinado no conjunto independente LUDB (Etapa 3/4).

Este script NAO treina nada: ele carrega um checkpoint gerado por
src/train_cnn.py, congela os pesos e mede o desempenho cross-dataset no
LUDB. A separacao em relacao ao script de treino e metodologica, nao apenas
organizacional — enquanto o teste vive em outro arquivo e consome um modelo ja
salvo, nenhuma decisao de treino pode ser tomada olhando o conjunto de teste.

O limiar de decisao NAO e ajustavel: e sempre ``model_gradcam.DEFAULT_THRESHOLD``
(0.5), o mesmo valor usado no treino e na validacao. Nao ha otimizacao de
limiar neste projeto — ela abriria uma forma indireta de encaixar o modelo aos
dados de teste.

Alem das metricas, o script grava as predicoes batimento a batimento — e esse
arquivo que a Etapa 4 consome para cruzar o Grad-CAM com as anotacoes clinicas.

Exemplos:
    # Testa o modelo padrao
    python src/test_ludb.py

    # Outro checkpoint / outra saida
    python src/test_ludb.py --checkpoint outputs/models/exp2/model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

# Permite executar o script de qualquer diretório e importar os módulos irmãos.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import beat_io  # noqa: E402
import config  # noqa: E402
import classification  # noqa: E402
import model_gradcam as mg  # noqa: E402

PREDICTIONS_NAME = "ludb_predictions.npz"
RESULTS_NAME = "ludb_metrics.json"


# --------------------------------------------------------------------------
# Metricas de classificacao binaria
#
# A classe positiva e sempre a 1 (arritmia). Em base desbalanceada — o MIT-BIH
# tem ~74% de batimentos normais — a acuracia sozinha engana: um classificador
# que responde "normal" para tudo ja acerta 74%. Por isso o projeto reporta
# tambem sensibilidade, especificidade e acuracia balanceada, como pede o
# artigo.
# --------------------------------------------------------------------------
def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Matriz de confusao binaria com a classe 1 (arritmia) como positiva."""
    tn, fp, fn, tp = confusion_matrix(
        np.asarray(y_true).astype(int).ravel(),
        np.asarray(y_pred).astype(int).ravel(),
        labels=[0, 1],
    ).ravel()
    return {"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC, ou ``nan`` quando so ha uma classe presente."""
    y_true = np.asarray(y_true).astype(int).ravel()
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(y_score, dtype=np.float64).ravel()))


def binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    """Conjunto completo de metricas a um dado limiar de decisao.

    Args:
        y_true: rotulos verdadeiros (0/1).
        y_score: probabilidade predita da classe 1.
        threshold: limiar de binarizacao do score.

    Returns:
        Dicionario com a matriz de confusao e acuracia, acuracia balanceada,
        sensibilidade, especificidade, precisao, F1 e AUC-ROC. As razoes sao
        calculadas a partir da matriz de confusao para que denominador zero
        vire 0.0 em vez de ``nan``.
    """
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    y_pred = (y_score >= threshold).astype(int)
    cm = confusion(y_true, y_pred)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]

    def razao(num: int, den: int) -> float:
        return float(num / den) if den else 0.0

    sensibilidade = razao(tp, tp + fn)
    especificidade = razao(tn, tn + fp)

    return {
        **cm,
        "n": int(tp + tn + fp + fn),
        "threshold": float(threshold),
        "acuracia": razao(tp + tn, tp + tn + fp + fn),
        "acuracia_balanceada": (sensibilidade + especificidade) / 2,
        "sensibilidade": sensibilidade,
        "especificidade": especificidade,
        "precisao": razao(tp, tp + fp),
        "f1": razao(2 * tp, 2 * tp + fp + fn),
        "auc_roc": roc_auc(y_true, y_score),
    }


def por_grupo(
    y_true: np.ndarray,
    y_score: np.ndarray,
    grupos: np.ndarray,
    threshold: float = 0.5,
) -> list[dict]:
    """Desempenho estratificado por um rotulo categorico.

    Serve para a analise de falhas da Etapa 4: no MIT-BIH o grupo e o simbolo
    original do batimento (``V``, ``A``, ``L``, ...), e no LUDB e o ritmo do
    registro. Como cada grupo costuma ter uma unica classe verdadeira, o que
    interessa aqui e a taxa de acerto e o score medio, nao a matriz completa.

    Returns:
        Lista de dicionarios ordenada por numero de amostras (decrescente).
    """
    y_true = np.asarray(y_true).astype(int).ravel()
    y_score = np.asarray(y_score, dtype=np.float64).ravel()
    grupos = np.asarray(grupos).ravel()
    y_pred = (y_score >= threshold).astype(int)

    linhas = []
    for grupo in np.unique(grupos):
        mask = grupos == grupo
        classes = np.unique(y_true[mask])
        linhas.append({
            "grupo": str(grupo),
            "n": int(mask.sum()),
            "classe_verdadeira": (
                int(classes[0]) if classes.size == 1 else "mista"
            ),
            "acertos": int((y_pred[mask] == y_true[mask]).sum()),
            "taxa_acerto": float((y_pred[mask] == y_true[mask]).mean()),
            "score_medio": float(y_score[mask].mean()),
        })

    return sorted(linhas, key=lambda linha: -linha["n"])


def distribuicao_scores(y_score) -> dict:
    """Quantis e histograma (10 faixas em [0, 1]) de um conjunto de scores."""
    y_score = np.asarray(y_score, dtype=float)
    if not y_score.size:
        return {"quantis": None, "histograma": None}
    contagens, bordas = np.histogram(y_score, bins=10, range=(0.0, 1.0))
    return {
        "quantis": {
            f"p{q}": float(np.percentile(y_score, q)) for q in (10, 25, 50, 75, 90)
        },
        "histograma": {"bordas": bordas.round(2).tolist(), "contagens": contagens.tolist()},
    }


def desempenho_positivo_multirrotulo(y_true, y_score, grupos_json, threshold=0.5, pacientes=None):
    """Recall/FNR em subgrupos positivos possivelmente sobrepostos.

    Especificidade, precisao e AUC nao sao calculadas aqui porque cada recorte
    contem apenas positivos e nao define negativos proprios do subtipo.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    decoded = [json.loads(str(value)) for value in np.asarray(grupos_json)]
    nomes = sorted({name for values in decoded for name in values})
    rows = []
    for name in nomes:
        mask = np.asarray([name in values for values in decoded]) & (y_true == 1)
        n = int(mask.sum())
        tp = int((y_score[mask] >= threshold).sum())
        fn = n - tp
        row = {
            "grupo": name, "n_positivos": n, "tp": tp, "fn": fn,
            "sensibilidade": float(tp / n) if n else float("nan"),
            "taxa_falso_negativo": float(fn / n) if n else float("nan"),
            "score_medio": float(y_score[mask].mean()) if n else float("nan"),
            "score_mediano": float(np.median(y_score[mask])) if n else float("nan"),
            "distribuicao_scores": distribuicao_scores(y_score[mask]),
            "metricas_nao_aplicaveis": ["especificidade", "precisao", "f1", "auc_roc"],
        }
        if pacientes is not None:
            row["n_registros"] = int(np.unique(np.asarray(pacientes)[mask]).size)
        rows.append(row)
    return rows


def baselines(y_true) -> dict:
    """Classificadores triviais na mesma populacao: sempre 0, sempre 1, majoritaria.

    A classe majoritaria e descritiva (contagem da propria populacao avaliada);
    nao participa de nenhuma decisao do modelo.
    """
    y_true = np.asarray(y_true).astype(int)
    majoritaria = int((y_true == 1).sum() > (y_true == 0).sum())
    return {
        nome: {"classe_prevista": classe,
               **binary_metrics(y_true, np.full(y_true.size, float(classe)))}
        for nome, classe in (("sempre_0", 0), ("sempre_1", 1), ("classe_majoritaria", majoritaria))
    }


def _decode(values) -> list[list[str]]:
    return [json.loads(str(value)) for value in np.asarray(values)]


def atribuicao_falsos_negativos(y_true, y_score, target_families, target_subtypes, threshold=0.5):
    """Associa cada falso negativo aos alvos do seu registro.

    Categorias: extrassistole (atrial/ventricular), bloqueio completo e bloqueio
    incompleto. Um registro pode ter mais de uma; por isso o resultado traz a
    contagem multirrotulo (somas podem exceder o total) e a particao exclusiva
    por combinacao (soma igual ao total de FN).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    familias, subtipos = _decode(target_families), _decode(target_subtypes)

    def categorias(i: int) -> tuple[str, ...]:
        cats = []
        if classification.SHARED_TARGET_FAMILIES & set(familias[i]):
            cats.append("extrassistole")
        if classification.LUDB_COMPLETE_BUNDLE_SUBTYPES & set(subtipos[i]):
            cats.append("bloqueio_completo")
        if classification.LUDB_INCOMPLETE_BUNDLE_SUBTYPES & set(subtipos[i]):
            cats.append("bloqueio_incompleto")
        return tuple(cats)

    positivos = np.flatnonzero(y_true == 1)
    fn_total = int(((y_true == 1) & (y_pred == 0)).sum())
    multirrotulo: dict[str, dict] = {}
    combinacoes: dict[str, dict] = {}
    for i in positivos:
        cats = categorias(i)
        fn = int(y_pred[i] == 0)
        for cat in cats:
            bloco = multirrotulo.setdefault(cat, {"n_positivos": 0, "n_fn": 0})
            bloco["n_positivos"] += 1
            bloco["n_fn"] += fn
        bloco = combinacoes.setdefault(" + ".join(cats), {"n_positivos": 0, "n_fn": 0})
        bloco["n_positivos"] += 1
        bloco["n_fn"] += fn

    def completa(bloco: dict) -> dict:
        return {
            **bloco,
            "fracao_do_total_fn": bloco["n_fn"] / fn_total if fn_total else float("nan"),
            "taxa_falso_negativo": bloco["n_fn"] / bloco["n_positivos"],
        }

    return {
        "n_falsos_negativos": fn_total,
        "multirrotulo": {cat: completa(b) for cat, b in sorted(multirrotulo.items())},
        "combinacao_exclusiva": {cat: completa(b) for cat, b in sorted(combinacoes.items())},
        "nota": (
            "multirrotulo: FN associados a cada categoria (um FN pode contar em "
            "mais de uma); combinacao_exclusiva: particao dos FN pela combinacao "
            "de alvos do registro."
        ),
    }


def avaliacao_somente_completos(y_true, y_score, target_families, target_subtypes,
                                pacientes, threshold=0.5) -> dict:
    """Recorte COM_BLOQUEIO sem bloqueios incompletos como alvos avaliados.

    Usa as mesmas predicoes do modelo COM (nao e um terceiro modelo). Saem do
    recorte somente os positivos cuja unica justificativa restante e bloqueio
    incompleto; eles ficam EXCLUIDOS, nunca viram 0.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    pacientes = np.asarray(pacientes)
    familias, subtipos = _decode(target_families), _decode(target_subtypes)
    mask = np.asarray([
        classification.is_complete_only_target(y_true[i], familias[i], subtipos[i])
        for i in range(y_true.size)
    ])
    sub_json = np.asarray(target_subtypes)[mask]
    return {
        "nome": "COM_BLOQUEIO — avaliacao somente completos",
        "descricao": (
            "Mesmo checkpoint e mesmas predicoes do modelo COM_BLOQUEIO; nao e um "
            "terceiro modelo. Exclui apenas positivos cuja unica justificativa "
            "restante e LBBB/RBBB incompleto."
        ),
        "n": int(mask.sum()),
        "classe_0": int((y_true[mask] == 0).sum()),
        "classe_1": int((y_true[mask] == 1).sum()),
        "n_registros": int(np.unique(pacientes[mask]).size),
        "n_excluidos": int((~mask).sum()),
        "registros_excluidos": sorted({str(p) for p in pacientes[~mask]}, key=int),
        "metrics": binary_metrics(y_true[mask], y_score[mask], threshold=threshold),
        "baselines": baselines(y_true[mask]),
        "positivos_por_subtipo": desempenho_positivo_multirrotulo(
            y_true[mask], y_score[mask], sub_json, threshold=threshold, pacientes=pacientes[mask]
        ),
        "mascara": mask,
    }


def formata_por_grupo(nome: str, linhas: list[dict]) -> str:
    """Tabela textual do resultado de :func:`por_grupo`."""
    saida = [
        f"[{nome}] desempenho por grupo",
        f"  {'grupo':<28} {'n':>6} {'classe':>7} {'acerto':>8} {'score':>7}",
    ]
    for linha in linhas:
        saida.append(
            f"  {linha['grupo'][:28]:<28} {linha['n']:>6} "
            f"{str(linha['classe_verdadeira']):>7} "
            f"{linha['taxa_acerto']:>8.4f} {linha['score_medio']:>7.4f}"
        )
    return "\n".join(saida)


def formata_metricas(nome: str, m: dict[str, float | int]) -> str:
    """Bloco de texto legivel para o log do treino e o resumo final."""
    return "\n".join([
        f"[{nome}] n={m['n']}  limiar={m['threshold']:.2f}",
        f"  acuracia            : {m['acuracia']:.4f}",
        f"  acuracia balanceada : {m['acuracia_balanceada']:.4f}",
        f"  sensibilidade (rec) : {m['sensibilidade']:.4f}",
        f"  especificidade      : {m['especificidade']:.4f}",
        f"  precisao            : {m['precisao']:.4f}",
        f"  f1                  : {m['f1']:.4f}",
        f"  auc-roc             : {m['auc_roc']:.4f}",
        f"  confusao            : tp={m['tp']} fn={m['fn']} tn={m['tn']} fp={m['fp']}",
    ])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia no LUDB um modelo ja treinado no MIT-BIH.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint .pt gerado por src/train_cnn.py.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset .npz gerado por src/preprocess_ludb.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Diretorio de saida. Por padrao, subpasta 'teste_ludb' ao lado do "
            "checkpoint, para que resultado e modelo fiquem juntos."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cuda' ou 'cpu'.",
    )
    classification.add_cli_argument(parser)
    return classification.prepare_args(parser.parse_args(), "test")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir or (args.checkpoint.parent / "teste_ludb")
    classification.require_new_output(out_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.checkpoint.exists():
        print(
            f"Checkpoint nao encontrado: {args.checkpoint}\n"
            f"Treine o modelo antes com: python src/train_cnn.py",
            file=sys.stderr,
        )
        return 1
    if not args.dataset.exists():
        print(
            f"Dataset nao encontrado: {args.dataset}\n"
            f"Gere-o antes com: python src/preprocess_ludb.py --download",
            file=sys.stderr,
        )
        return 1

    modelo, ckpt = mg.load_checkpoint(
        args.checkpoint, device=args.device,
        expected_policy=args.classification_policy,
    )
    info = ckpt.get("info", {})
    if "split_manifest_sha256" not in info:
        raise classification.ClassificationError(
            "Checkpoint sem identidade do split; checkpoint historico incompativel."
        )
    if info.get("checkpoint_role") != "best":
        raise classification.ClassificationError(
            f"{args.checkpoint}: checkpoint_role={info.get('checkpoint_role')!r}; "
            "o teste LUDB exige o melhor checkpoint por acuracia balanceada de validacao."
        )
    limiar = mg.DEFAULT_THRESHOLD

    dados = beat_io.load_beat_dataset(args.dataset)
    classification.require_policy(dados.get("metadata", {}), args.classification_policy, str(args.dataset))
    required_trace = {
        "beat_ids", "selection_indices", "target_families", "target_subtypes", "bundle_side",
        "bundle_completeness", "positive_reasons", "other_conditions", "ritmos",
        "is_shared_subset",
    }
    missing_trace = required_trace - dados.keys()
    if missing_trace:
        raise classification.ClassificationError(
            f"Dataset LUDB sem rastreabilidade nova: {sorted(missing_trace)}."
        )
    provenance = {
        **classification.artifact_identity(args.classification_policy),
        "dataset_sha256": classification.file_sha256(args.dataset),
        "checkpoint_sha256": classification.file_sha256(args.checkpoint),
        "checkpoint_role": info["checkpoint_role"],
        "checkpoint_epoch": int(info["checkpoint_epoch"]),
        "split_manifest_sha256": info["split_manifest_sha256"],
    }
    series = dados["series"]
    labels = np.asarray(dados["labels"]).astype(int)
    ritmos = np.asarray(dados["simbolos"])
    pacientes = np.asarray(dados["pacientes"])

    print("=" * 64)
    print("TESTE CROSS-DATASET NO LUDB")
    print("=" * 64)
    print(f"Checkpoint : {args.checkpoint}")
    print(f"  papel / epoca    : {info['checkpoint_role']} / {info['checkpoint_epoch']}")
    print(f"  treinado em      : {info.get('dataset_treino', 'n/d')}")
    print(f"  split de treino  : {info.get('train_config', {}).get('split_scheme')}")
    print(f"  parametros       : {info.get('n_parametros', 'n/d')}")
    print(f"Dataset    : {args.dataset}")
    print(
        f"  {series.shape[0]} batimentos de "
        f"{np.unique(pacientes).size} registros | "
        f"classe 0: {int((labels == 0).sum())} | "
        f"classe 1: {int((labels == 1).sum())}"
    )
    print(f"Limiar     : {limiar:.4f} (fixo, model_gradcam.DEFAULT_THRESHOLD)")
    print("-" * 64)

    scores = mg.predict_scores(modelo, series, device=args.device)
    m_ludb = binary_metrics(labels, scores, threshold=limiar)
    shared_mask = np.asarray(dados["is_shared_subset"]).astype(bool)
    m_shared = binary_metrics(labels[shared_mask], scores[shared_mask], threshold=limiar)
    print(formata_metricas("LUDB (cross-dataset)", m_ludb))
    print("-" * 64)
    print(formata_metricas("LUDB | subconjunto comum (tarefa SEM)", m_shared))
    print("-" * 64)

    grupos = por_grupo(labels, scores, ritmos, threshold=limiar)
    por_familia = desempenho_positivo_multirrotulo(
        labels, scores, dados["target_families"], threshold=limiar, pacientes=pacientes
    )
    por_subtipo = desempenho_positivo_multirrotulo(
        labels, scores, dados["target_subtypes"], threshold=limiar, pacientes=pacientes
    )
    fn_atribuicao = atribuicao_falsos_negativos(
        labels, scores, dados["target_families"], dados["target_subtypes"], threshold=limiar
    )
    somente_completos = avaliacao_somente_completos(
        labels, scores, dados["target_families"], dados["target_subtypes"], pacientes,
        threshold=limiar,
    )
    mascara_completos = somente_completos.pop("mascara")
    policy = classification.get_policy(args.classification_policy)
    print(formata_por_grupo("LUDB por ritmo", grupos))
    print("-" * 64)
    print("[LUDB por subtipo positivo] n  regs  TP  FN  recall  score medio/mediano")
    for linha in por_subtipo:
        print(f"  {linha['grupo'][:44]:<44} {linha['n_positivos']:>4} {linha['n_registros']:>3} "
              f"{linha['tp']:>4} {linha['fn']:>4} {linha['sensibilidade']:.4f} "
              f"{linha['score_medio']:.4f}/{linha['score_mediano']:.4f}")
    if policy.include_bundle_branch_block:
        print("-" * 64)
        print(formata_metricas(somente_completos["nome"], somente_completos["metrics"]))
    print("=" * 64)

    out_dir = args.output_dir or (args.checkpoint.parent / "teste_ludb")
    out_dir.mkdir(parents=True, exist_ok=True)

    predicoes = (scores >= limiar).astype(np.int64)
    np.savez_compressed(
        out_dir / PREDICTIONS_NAME,
        **provenance,
        y_true=labels,
        y_score=scores.astype(np.float32),
        y_pred=predicoes,
        pacientes=pacientes,
        ritmos=ritmos,
        beat_ids=dados["beat_ids"],
        selection_indices=dados["selection_indices"],
        target_families=dados["target_families"],
        target_subtypes=dados["target_subtypes"],
        bundle_side=dados["bundle_side"],
        bundle_completeness=dados["bundle_completeness"],
        positive_reasons=dados["positive_reasons"],
        other_conditions=dados["other_conditions"],
        is_shared_subset=dados["is_shared_subset"],
        is_complete_only_eval=mascara_completos.astype(np.int8),
        threshold=np.array(limiar),
    )

    resultado = {
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etapa": "3 - teste cross-dataset (LUDB)",
        "checkpoint": config.caminho_relativo(args.checkpoint),
        "dataset_teste": config.caminho_relativo(args.dataset),
        "threshold": limiar,
        **classification.metadata(args.classification_policy),
        **provenance,
        "dataset_metadata": dados.get("metadata", {}),
        "metrics": m_ludb,
        "baselines": baselines(labels),
        "metrics_shared_subset": m_shared,
        "shared_subset_population": {
            "n": int(shared_mask.sum()),
            "classe_0": int((labels[shared_mask] == 0).sum()),
            "classe_1": int((labels[shared_mask] == 1).sum()),
            "n_registros": int(np.unique(pacientes[shared_mask]).size),
        },
        "baselines_shared_subset": baselines(labels[shared_mask]),
        "por_ritmo": grupos,
        "positivos_por_familia": por_familia,
        "positivos_por_subtipo": por_subtipo,
        "falsos_negativos_por_alvo": fn_atribuicao,
        "shared_subset_definition": classification.SHARED_SUBSET_DEFINITION,
        "treino_info": {
            "split": info.get("split", {}),
            "train_config": info.get("train_config", {}),
            "metrics_mitbih": info.get("metrics", {}),
        },
        "observacao": (
            "Pesos e limiar congelados; nenhuma decisao de treino foi tomada "
            "com base neste conjunto."
        ),
    }
    if policy.include_bundle_branch_block:
        resultado["avaliacao_somente_completos"] = somente_completos
    (out_dir / RESULTS_NAME).write_text(
        classification.dumps_json(resultado), encoding="utf-8"
    )

    print(f"\nPredicoes : {out_dir / PREDICTIONS_NAME}")
    print(f"Metricas  : {out_dir / RESULTS_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
