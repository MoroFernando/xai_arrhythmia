"""CLI da métrica quantitativa de fidelidade do Grad-CAM no LUDB (Etapa 4, parte 2).

Quarto script da cadeia, depois de ``train_cnn.py``, ``test_ludb.py`` e
``extract_gradcam.py``. Consome dois arquivos já prontos:

- ``ludb_gradcam.npz`` (de ``extract_gradcam.py``) — os mapas ``cam_interp``
  ``(n, 234)`` normalizados em ``[0, 1]`` e a predição de cada batimento;
- o dataset processado do LUDB (de ``preprocess_ludb.py``) — as máscaras
  clínicas ``mascara_clinica`` / ``mascara_p`` / ``mascara_qrs`` /
  ``mascara_t``, alinhadas por índice com os mapas.

e cruza os dois num IoU por batimento (ver ``xai_arrhythmia/fidelity.py``): o
mapa binarizado em ``cam_interp >= --limiar-cam`` contra a máscara clínica.
Grava o IoU beat a beat e as médias (geral, por classe prevista, por classe
verdadeira e por estrutura), com a cobertura média da máscara como referência
ao acaso.

Exemplo:
    python scripts/fidelity_ludb.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Permite executar o script sem instalar o pacote (python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xai_arrhythmia import config, fidelity, io  # noqa: E402

#: Nome do arquivo gravado por scripts/extract_gradcam.py.
GRADCAM_NAME = "ludb_gradcam.npz"

FIDELITY_NAME = "ludb_fidelity.npz"
FIDELITY_INFO_NAME = "ludb_fidelity.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula o IoU de fidelidade do Grad-CAM contra a anotação clínica do LUDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gradcam",
        type=Path,
        default=config.MODELS_DIR / "cnn1d_mitbih" / "gradcam_ludb" / GRADCAM_NAME,
        help="Arquivo .npz gerado por scripts/extract_gradcam.py.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.LUDB_PROCESSED_FILE,
        help="Dataset .npz gerado por scripts/preprocess_ludb.py (traz as máscaras).",
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
        default=fidelity.DEFAULT_CAM_THRESHOLD,
        help="Limiar de binarização do mapa do Grad-CAM (já normalizado em [0, 1]).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.gradcam.exists():
        print(
            f"Grad-CAM nao encontrado: {args.gradcam}\n"
            f"Gere-o antes com: python scripts/extract_gradcam.py",
            file=sys.stderr,
        )
        return 1
    if not args.dataset.exists():
        print(
            f"Dataset nao encontrado: {args.dataset}\n"
            f"Gere-o antes com: python scripts/preprocess_ludb.py --download",
            file=sys.stderr,
        )
        return 1

    gcam = np.load(args.gradcam, allow_pickle=True)
    cam_interp = gcam["cam_interp"]
    y_true = np.asarray(gcam["y_true"]).astype(int)
    y_pred = np.asarray(gcam["y_pred"]).astype(int)
    ritmos = np.asarray(gcam["ritmos"])

    dados = io.load_beat_dataset(args.dataset)
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

    resultado = fidelity.fidelidade(
        cam_interp,
        mascara_clinica,
        y_pred=y_pred,
        y_true=y_true,
        limiar_cam=args.limiar_cam,
        mascaras_por_estrutura=mascaras_por_estrutura,
        cobertura_acaso=cobertura_acaso,
    )
    print(fidelity.formata_fidelidade("LUDB", resultado))
    print("=" * 64)

    out_dir = args.output_dir or (args.gradcam.parent.parent / "fidelidade_ludb")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / FIDELITY_NAME,
        iou_por_batimento=resultado["iou_por_batimento"].astype(np.float32),
        y_true=y_true,
        y_pred=y_pred,
        ritmos=ritmos,
        limiar_cam=np.array(args.limiar_cam),
    )

    info = {
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etapa": "4 - fidelidade quantitativa do Grad-CAM (LUDB)",
        "gradcam": str(args.gradcam),
        "dataset": str(args.dataset),
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
