"""Métrica quantitativa de fidelidade do Grad-CAM (Etapa 4, parte 2).

A Etapa 4 pergunta se o Grad-CAM 1D destaca estruturas eletrofisiológicas
reais do batimento (ondas P, T e complexo QRS) ou pontos arbitrários do sinal.
A parte 1 (``gradcam.py`` + ``scripts/extract_gradcam.py``) extrai os mapas; a
parte 2, aqui, reduz a comparação "mapa x anotação clínica" a **um número por
batimento**, no lugar da inspeção puramente visual do notebook
``05_gradcam_demo.ipynb``.

O número é a **Interseção sobre União (IoU)** entre o mapa do Grad-CAM
binarizado (``cam_interp >= limiar_cam``) e a máscara clínica do LUDB
(``mascara_clinica``, a união P+QRS+T amostra a amostra). Como o mapa já vem
normalizado em ``[0, 1]`` por ``gradcam.grad_cam_1d``, o limiar padrão de 0.5
seleciona a metade superior da ativação de cada batimento.

Referência ao acaso: qualquer mapa aleatório atinge, em média, um IoU próximo
da cobertura da máscara (fração da janela anotada). Esse valor —
``dataset["metadata"]["cobertura_media_mascara_clinica"]``, também devolvido
aqui como ``cobertura_acaso`` — é o piso contra o qual o resultado do modelo
deve ser lido: um IoU médio que apenas o iguala não indica foco nenhum.

Ressalva herdada da discussão da Etapa 4: o mapa é normalizado pelo próprio
máximo de cada batimento, então mesmo uma ativação fraca e ruidosa (típica de
um verdadeiro negativo) gera um pico "quente" em algum ponto. A quebra por
classe prevista (``iou_pred_0`` / ``iou_pred_1``) existe justamente para não
misturar esses dois regimes num único número.
"""

from __future__ import annotations

import numpy as np

#: Limiar de binarização do mapa do Grad-CAM. O mapa já está normalizado em
#: ``[0, 1]`` (máximo por batimento = 1), então 0.5 é a metade superior da
#: ativação. Fixo, pelo mesmo motivo de ``config.DEFAULT_THRESHOLD``: varrer o
#: limiar por conjunto abriria uma forma indireta de ajustar o resultado.
DEFAULT_CAM_THRESHOLD = 0.5


def binariza_cam(
    cam: np.ndarray,
    limiar: float = DEFAULT_CAM_THRESHOLD,
) -> np.ndarray:
    """Binariza o(s) mapa(s) do Grad-CAM em ``cam >= limiar``.

    Args:
        cam: ``(tamanho_janela,)`` ou ``(n, tamanho_janela)``, já normalizado
            em ``[0, 1]`` (ver ``gradcam.grad_cam_1d`` / ``interpola_cam``).
        limiar: ponto de corte da ativação.

    Returns:
        Máscara booleana com o mesmo shape da entrada.
    """
    return np.asarray(cam, dtype=np.float64) >= limiar


def iou_por_batimento(
    cam_bin: np.ndarray,
    mascara: np.ndarray,
    vazio_vazio: float = np.nan,
) -> np.ndarray:
    """IoU entre o mapa binarizado e a máscara clínica, batimento a batimento.

    Args:
        cam_bin: máscara booleana do Grad-CAM ``(n, tamanho_janela)`` (saída de
            :func:`binariza_cam`).
        mascara: máscara clínica ``(n, tamanho_janela)`` (0/1); aceita
            ``int8``, ``bool`` ou float.
        vazio_vazio: valor atribuído quando a união é vazia — nem o mapa nem a
            máscara marcam amostra alguma. O padrão ``nan`` faz esses
            batimentos serem ignorados por :func:`resumo_iou`
            (``np.nanmean``), o que importa nas máscaras por estrutura: em
            fibrilação/flutter atrial a onda P não existe, e um IoU de 1.0
            nesses casos premiaria o Grad-CAM por "acertar" uma estrutura
            ausente. Para a ``mascara_clinica`` a união nunca é vazia.

    Returns:
        Vetor ``(n,)`` de IoU em ``[0, 1]`` (com ``nan`` onde a união é vazia).
    """
    cam_bin = np.atleast_2d(np.asarray(cam_bin, dtype=bool))
    mascara = np.atleast_2d(np.asarray(mascara)).astype(bool)
    if cam_bin.shape != mascara.shape:
        raise ValueError(
            f"Shapes incompatíveis: cam_bin {cam_bin.shape} x mascara {mascara.shape}"
        )

    intersecao = np.logical_and(cam_bin, mascara).sum(axis=1)
    uniao = np.logical_or(cam_bin, mascara).sum(axis=1)

    iou = np.full(uniao.shape, float(vazio_vazio), dtype=np.float64)
    nao_vazio = uniao > 0
    iou[nao_vazio] = intersecao[nao_vazio] / uniao[nao_vazio]
    return iou


def _media_segura(valores: np.ndarray) -> float:
    """Média ignorando ``nan``; ``nan`` se não sobrar nenhum valor."""
    valores = np.asarray(valores, dtype=np.float64)
    if valores.size == 0 or np.all(np.isnan(valores)):
        return float("nan")
    return float(np.nanmean(valores))


def resumo_iou(
    iou: np.ndarray,
    y_pred: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    cobertura_acaso: float | None = None,
) -> dict[str, float | int]:
    """Agrega o IoU por batimento em médias — geral e estratificadas.

    Args:
        iou: vetor ``(n,)`` de :func:`iou_por_batimento`.
        y_pred: classe prevista por batimento (0/1); habilita ``iou_pred_*``.
        y_true: classe verdadeira por batimento (0/1); habilita ``iou_real_*``.
        cobertura_acaso: fração média da janela coberta pela máscara — a
            pontuação esperada de um mapa aleatório. Repassada como
            ``cobertura_acaso`` e usada para ``acima_do_acaso``.

    Returns:
        Dicionário com ``n``, ``n_avaliados`` (batimentos com IoU não-``nan``),
        ``iou_medio``, ``iou_mediana`` e, conforme os argumentos, a quebra por
        classe prevista/verdadeira e a comparação com o acaso.
    """
    iou = np.asarray(iou, dtype=np.float64)
    avaliados = ~np.isnan(iou)

    resultado: dict[str, float | int] = {
        "n": int(iou.size),
        "n_avaliados": int(avaliados.sum()),
        "iou_medio": _media_segura(iou),
        "iou_mediana": (
            float(np.nanmedian(iou)) if avaliados.any() else float("nan")
        ),
    }

    if y_pred is not None:
        y_pred = np.asarray(y_pred).astype(int).ravel()
        resultado["iou_pred_0"] = _media_segura(iou[y_pred == 0])
        resultado["iou_pred_1"] = _media_segura(iou[y_pred == 1])

    if y_true is not None:
        y_true = np.asarray(y_true).astype(int).ravel()
        resultado["iou_real_0"] = _media_segura(iou[y_true == 0])
        resultado["iou_real_1"] = _media_segura(iou[y_true == 1])

    if cobertura_acaso is not None:
        resultado["cobertura_acaso"] = float(cobertura_acaso)
        resultado["acima_do_acaso"] = resultado["iou_medio"] - float(cobertura_acaso)

    return resultado


def fidelidade(
    cam: np.ndarray,
    mascara_clinica: np.ndarray,
    y_pred: np.ndarray | None = None,
    y_true: np.ndarray | None = None,
    limiar_cam: float = DEFAULT_CAM_THRESHOLD,
    mascaras_por_estrutura: dict[str, np.ndarray] | None = None,
    cobertura_acaso: float | None = None,
) -> dict:
    """Métrica de fidelidade completa do Grad-CAM contra a anotação clínica.

    Args:
        cam: mapas do Grad-CAM ``(n, tamanho_janela)`` já interpolados para a
            janela do batimento e normalizados em ``[0, 1]`` (``cam_interp`` de
            ``scripts/extract_gradcam.py``).
        mascara_clinica: máscara ``(n, tamanho_janela)`` da união P+QRS+T.
        y_pred: classe prevista por batimento (0/1).
        y_true: classe verdadeira por batimento (0/1).
        limiar_cam: binarização do mapa (padrão :data:`DEFAULT_CAM_THRESHOLD`).
        mascaras_por_estrutura: opcional, ``{"P": ..., "QRS": ..., "T": ...}``
            com uma máscara ``(n, tamanho_janela)`` cada, para o IoU separado
            por onda (usa ``mascara_p`` / ``mascara_qrs`` / ``mascara_t``).
        cobertura_acaso: fração média da máscara clínica; normalmente
            ``dataset["metadata"]["cobertura_media_mascara_clinica"]``.

    Returns:
        Dicionário com ``limiar_cam``, o bloco ``geral`` (saída de
        :func:`resumo_iou` para a máscara clínica), ``iou_por_batimento`` (o
        vetor completo) e, se pedido, ``por_estrutura``.
    """
    cam = np.atleast_2d(np.asarray(cam, dtype=np.float64))
    cam_bin = binariza_cam(cam, limiar_cam)

    iou_clinica = iou_por_batimento(cam_bin, mascara_clinica, vazio_vazio=np.nan)

    resultado: dict = {
        "limiar_cam": float(limiar_cam),
        "geral": resumo_iou(iou_clinica, y_pred, y_true, cobertura_acaso),
        "iou_por_batimento": iou_clinica,
    }

    if mascaras_por_estrutura:
        resultado["por_estrutura"] = {
            nome: resumo_iou(
                iou_por_batimento(cam_bin, mascara, vazio_vazio=np.nan),
                y_pred,
                y_true,
            )
            for nome, mascara in mascaras_por_estrutura.items()
        }

    return resultado


def _fmt(valor: float) -> str:
    return "   n/d" if valor != valor else f"{valor:.4f}"  # valor != valor => nan


def formata_fidelidade(nome: str, resultado: dict) -> str:
    """Bloco de texto legível para o log do script e o resumo final."""
    g = resultado["geral"]
    linhas = [
        f"[{nome}] fidelidade do Grad-CAM  (limiar_cam={resultado['limiar_cam']:.2f})",
        f"  batimentos              : {g['n']}  (avaliados: {g['n_avaliados']})",
        f"  IoU medio (clinica)     : {_fmt(g['iou_medio'])}",
        f"  IoU mediana             : {_fmt(g['iou_mediana'])}",
    ]
    if "iou_pred_1" in g:
        linhas += [
            f"  IoU | previu arritmia   : {_fmt(g['iou_pred_1'])}",
            f"  IoU | previu normal     : {_fmt(g['iou_pred_0'])}",
        ]
    if "iou_real_1" in g:
        linhas += [
            f"  IoU | e arritmia (real) : {_fmt(g['iou_real_1'])}",
            f"  IoU | e normal (real)   : {_fmt(g['iou_real_0'])}",
        ]
    if "cobertura_acaso" in g:
        linhas += [
            f"  esperado ao acaso       : {_fmt(g['cobertura_acaso'])}",
            f"  acima do acaso          : {_fmt(g['acima_do_acaso'])}",
        ]

    por_estrutura = resultado.get("por_estrutura")
    if por_estrutura:
        linhas.append("  IoU por estrutura:")
        for estrutura, resumo in por_estrutura.items():
            linhas.append(
                f"    {estrutura:<20}: {_fmt(resumo['iou_medio'])} "
                f"(avaliados: {resumo['n_avaliados']})"
            )
    return "\n".join(linhas)
