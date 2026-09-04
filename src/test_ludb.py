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
        default=config.MODELS_DIR / "cnn1d_mitbih" / mg.CHECKPOINT_NAME,
        help="Checkpoint .pt gerado por src/train_cnn.py.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.LUDB_PROCESSED_FILE,
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    modelo, ckpt = mg.load_checkpoint(args.checkpoint, device=args.device)
    info = ckpt.get("info", {})
    limiar = mg.DEFAULT_THRESHOLD

    dados = beat_io.load_beat_dataset(args.dataset)
    series = dados["series"]
    labels = np.asarray(dados["labels"]).astype(int)
    ritmos = np.asarray(dados["simbolos"])
    pacientes = np.asarray(dados["pacientes"])

    print("=" * 64)
    print("TESTE CROSS-DATASET NO LUDB")
    print("=" * 64)
    print(f"Checkpoint : {args.checkpoint}")
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
    print(formata_metricas("LUDB (cross-dataset)", m_ludb))
    print("-" * 64)

    grupos = por_grupo(labels, scores, ritmos, threshold=limiar)
    print(formata_por_grupo("LUDB por ritmo", grupos))
    print("=" * 64)

    out_dir = args.output_dir or (args.checkpoint.parent / "teste_ludb")
    out_dir.mkdir(parents=True, exist_ok=True)

    predicoes = (scores >= limiar).astype(np.int64)
    np.savez_compressed(
        out_dir / PREDICTIONS_NAME,
        y_true=labels,
        y_score=scores.astype(np.float32),
        y_pred=predicoes,
        pacientes=pacientes,
        ritmos=ritmos,
        threshold=np.array(limiar),
    )

    resultado = {
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etapa": "3 - teste cross-dataset (LUDB)",
        "checkpoint": config.caminho_relativo(args.checkpoint),
        "dataset_teste": config.caminho_relativo(args.dataset),
        "threshold": limiar,
        "label_encoding": {"0": "nao-arritmia", "1": "arritmia"},
        "metrics": m_ludb,
        "por_ritmo": grupos,
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
    (out_dir / RESULTS_NAME).write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nPredicoes : {out_dir / PREDICTIONS_NAME}")
    print(f"Metricas  : {out_dir / RESULTS_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
