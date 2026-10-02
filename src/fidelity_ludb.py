"""CLI da métrica quantitativa de fidelidade do Grad-CAM no LUDB (Etapa 4, parte 2).

Quarto script da cadeia, depois de ``train_cnn.py``, ``test_ludb.py`` e
``extract_gradcam.py``. Consome dois arquivos já prontos:

- ``ludb_gradcam.npz`` (de ``extract_gradcam.py``) — os mapas ``cam_interp``
  ``(n, 234)`` normalizados em ``[0, 1]`` e a predição de cada batimento;
- o dataset processado do LUDB (de ``preprocess_ludb.py``) — as máscaras
  clínicas ``mascara_clinica`` / ``mascara_p`` / ``mascara_qrs`` /
  ``mascara_t``, alinhadas por índice com os mapas.

e cruza os dois num IoU por batimento: o mapa binarizado em
``cam_interp >= --limiar-cam`` contra a máscara clínica.
Grava o IoU beat a beat e as médias (geral, por classe prevista, por classe
verdadeira e por estrutura), com a cobertura média da máscara como referência
ao acaso.

Exemplo:
    python src/fidelity_ludb.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Permite executar o script de qualquer diretório e importar os módulos irmãos.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import beat_io  # noqa: E402
import config  # noqa: E402
import classification  # noqa: E402

#: Nome do arquivo gravado por src/extract_gradcam.py.
GRADCAM_NAME = "ludb_gradcam.npz"

FIDELITY_NAME = "ludb_fidelity.npz"
FIDELITY_INFO_NAME = "ludb_fidelity.json"

#: Limiar de binarização do mapa do Grad-CAM. O mapa já está normalizado em
#: ``[0, 1]`` (máximo por batimento = 1), então 0.5 é a metade superior da
#: ativação. Fixo, pelo mesmo motivo de ``config.DEFAULT_THRESHOLD``: varrer o
#: limiar por conjunto abriria uma forma indireta de ajustar o resultado.
DEFAULT_CAM_THRESHOLD = 0.5


# --------------------------------------------------------------------------
# Métrica de fidelidade
#
# A Etapa 4 pergunta se o Grad-CAM 1D destaca estruturas eletrofisiológicas
# reais do batimento (ondas P, T e complexo QRS) ou pontos arbitrários do
# sinal. A parte 1 (``src/extract_gradcam.py``) extrai os mapas; a parte 2,
# aqui, reduz a comparação "mapa x anotação clínica" a **um número por
# batimento**, no lugar da inspeção puramente visual do notebook
# ``05_gradcam_demo.ipynb``.
#
# Referência ao acaso: qualquer mapa aleatório atinge, em média, um IoU próximo
# da cobertura da máscara (fração da janela anotada). Esse valor —
# ``dataset["metadata"]["cobertura_media_mascara_clinica"]``, também devolvido
# aqui como ``cobertura_acaso`` — é o piso contra o qual o resultado do modelo
# deve ser lido: um IoU médio que apenas o iguala não indica foco nenhum.
#
# Ressalva herdada da discussão da Etapa 4: o mapa é normalizado pelo próprio
# máximo de cada batimento, então mesmo uma ativação fraca e ruidosa (típica de
# um verdadeiro negativo) gera um pico "quente" em algum ponto. A quebra por
# classe prevista (``iou_pred_0`` / ``iou_pred_1``) existe justamente para não
# misturar esses dois regimes num único número.
# --------------------------------------------------------------------------
def binariza_cam(
    cam: np.ndarray,
    limiar: float = DEFAULT_CAM_THRESHOLD,
) -> np.ndarray:
    """Binariza o(s) mapa(s) do Grad-CAM em ``cam >= limiar``.

    Args:
        cam: ``(tamanho_janela,)`` ou ``(n, tamanho_janela)``, já normalizado
            em ``[0, 1]`` (ver ``model_gradcam.grad_cam_1d``).
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
            ``src/extract_gradcam.py``).
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula o IoU de fidelidade do Grad-CAM contra a anotação clínica do LUDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gradcam",
        type=Path,
        default=None,
        help="Arquivo .npz gerado por src/extract_gradcam.py.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset .npz gerado por src/preprocess_ludb.py (traz as máscaras).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Diretório de saída. Por padrão, subpasta 'fidelidade_ludb' ao lado "
            "do arquivo de Grad-CAM."
        ),
    )
    parser.add_argument(
        "--limiar-cam",
        type=float,
        default=DEFAULT_CAM_THRESHOLD,
        help="Limiar de binarização do mapa do Grad-CAM (já normalizado em [0, 1]).",
    )
    classification.add_cli_argument(parser)
    return classification.prepare_args(parser.parse_args(), "fidelity")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir or (args.gradcam.parent.parent / "fidelidade_ludb")
    classification.require_new_output(out_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.gradcam.exists():
        print(
            f"Grad-CAM nao encontrado: {args.gradcam}\n"
            f"Gere-o antes com: python src/extract_gradcam.py",
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

    gcam = np.load(args.gradcam, allow_pickle=False)
    classification.require_npz_policy(gcam, args.only_arrhythmia, args.include_paced, str(args.gradcam))
    cam_interp = gcam["cam_interp"]
    y_true = np.asarray(gcam["y_true"]).astype(int)
    y_pred = np.asarray(gcam["y_pred"]).astype(int)
    ritmos = np.asarray(gcam["ritmos"])

    dados = beat_io.load_beat_dataset(args.dataset)
    classification.require_policy(dados.get("metadata", {}), args.only_arrhythmia, args.include_paced, str(args.dataset))
    dataset_sha256 = classification.file_sha256(args.dataset)
    if "dataset_sha256" not in gcam or gcam["dataset_sha256"].item() != dataset_sha256:
        raise classification.ClassificationError("Grad-CAM e dataset LUDB nao tem a mesma origem (SHA-256).")
    if "checkpoint_sha256" not in gcam:
        raise classification.ClassificationError("Grad-CAM sem identificacao do checkpoint.")
    provenance = {
        **classification.artifact_identity(args.only_arrhythmia, args.include_paced),
        "dataset_sha256": dataset_sha256,
        "checkpoint_sha256": gcam["checkpoint_sha256"].item(),
        "gradcam_sha256": classification.file_sha256(args.gradcam),
    }
    if not np.array_equal(y_true, dados["labels"]) or not np.array_equal(gcam["pacientes"], dados["pacientes"]):
        raise classification.ClassificationError("Labels/pacientes do Grad-CAM nao estao alinhados ao dataset.")
    gcam.close()
    mascara_clinica = dados["mascara_clinica"]
    metadata = dados.get("metadata", {})
    cobertura_acaso = metadata.get("cobertura_media_mascara_clinica")
    if cobertura_acaso is None:
        cobertura_acaso = float(np.asarray(mascara_clinica).mean())

    if cam_interp.shape != mascara_clinica.shape:
        print(
            f"Grad-CAM {cam_interp.shape} e mascara clinica {mascara_clinica.shape} "
            f"nao estao alinhados — foram gerados do mesmo dataset do LUDB?",
            file=sys.stderr,
        )
        return 1

    mascaras_por_estrutura = {
        "P (mascara_p)": dados["mascara_p"],
        "QRS (mascara_qrs)": dados["mascara_qrs"],
        "T (mascara_t)": dados["mascara_t"],
    }

    print("=" * 64)
    print("FIDELIDADE DO GRAD-CAM NO LUDB (Etapa 4, parte 2)")
    print("=" * 64)
    print(f"Grad-CAM  : {args.gradcam}")
    print(f"Dataset   : {args.dataset}")
    print(f"  {cam_interp.shape[0]} batimentos | janela de {cam_interp.shape[1]} amostras")
    print(f"Limiar CAM : {args.limiar_cam:.2f} (mapa normalizado em [0, 1])")
    print("-" * 64)

    resultado = fidelidade(
        cam_interp,
        mascara_clinica,
        y_pred=y_pred,
        y_true=y_true,
        limiar_cam=args.limiar_cam,
        mascaras_por_estrutura=mascaras_por_estrutura,
        cobertura_acaso=cobertura_acaso,
    )
    print(formata_fidelidade("LUDB", resultado))
    print("=" * 64)

    out_dir = args.output_dir or (args.gradcam.parent.parent / "fidelidade_ludb")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / FIDELITY_NAME,
        **provenance,
        iou_por_batimento=resultado["iou_por_batimento"].astype(np.float32),
        y_true=y_true,
        y_pred=y_pred,
        ritmos=ritmos,
        limiar_cam=np.array(args.limiar_cam),
    )

    info = {
        **classification.metadata(args.only_arrhythmia, args.include_paced),
        **provenance,
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etapa": "4 - fidelidade quantitativa do Grad-CAM (LUDB)",
        "gradcam": config.caminho_relativo(args.gradcam),
        "dataset": config.caminho_relativo(args.dataset),
        "n_batimentos": int(cam_interp.shape[0]),
        "tamanho_janela": int(cam_interp.shape[1]),
        "limiar_cam": float(args.limiar_cam),
        "metrica": (
            "IoU entre (cam_interp >= limiar_cam) e a mascara clinica "
            "(uniao P+QRS+T), batimento a batimento; uniao vazia -> ignorado"
        ),
        "geral": {k: v for k, v in resultado["geral"].items()},
        "por_estrutura": resultado.get("por_estrutura", {}),
        "observacao": (
            "iou_por_batimento em ludb_fidelity.npz esta alinhado por indice "
            "com ludb_gradcam.npz e com o dataset do LUDB."
        ),
    }
    (out_dir / FIDELITY_INFO_NAME).write_text(
        json.dumps(info, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    print(f"\nIoU por batimento : {out_dir / FIDELITY_NAME}")
    print(f"Metadados         : {out_dir / FIDELITY_INFO_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
