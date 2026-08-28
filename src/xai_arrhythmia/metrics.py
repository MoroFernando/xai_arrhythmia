"""Metricas de classificacao binaria (Etapa 3).

A classe positiva e sempre a 1 (arritmia). Em base desbalanceada — o MIT-BIH
tem ~74% de batimentos normais — a acuracia sozinha engana: um classificador
que responde "normal" para tudo ja acerta 74%. Por isso o projeto reporta
tambem sensibilidade, especificidade e acuracia balanceada, como pede o artigo.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


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
