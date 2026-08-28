"""CLI de extração do Grad-CAM 1D de todos os batimentos do LUDB (Etapa 4).

Carrega um checkpoint já treinado (scripts/train_cnn.py) e o dataset
processado do LUDB, calcula o Grad-CAM de cada batimento na última camada
convolucional — em lote, uma única passada de backward por lote — e grava os
mapas: no comprimento nativo da rede (``cam_raw``, L'=29) e já interpolados
para o tamanho da janela do batimento (``cam_interp``, 234), junto com a
predição de cada batimento.

O script é autocontido: não depende de scripts/test_ludb.py ter rodado antes.
A predição é recomputada aqui com o mesmo checkpoint e o mesmo limiar fixo
(config.DEFAULT_THRESHOLD) — determinística, sem aleatoriedade em eval() — e
por isso sai idêntica à de test_ludb.py.

Exemplo:
    python scripts/extract_gradcam.py
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

from xai_arrhythmia import config, gradcam, io, training  # noqa: E402

GRADCAM_NAME = "ludb_gradcam.npz"
GRADCAM_INFO_NAME = "ludb_gradcam.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai o Grad-CAM 1D de todos os batimentos do LUDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.MODELS_DIR / "cnn1d_mitbih" / training.CHECKPOINT_NAME,
        help="Checkpoint .pt gerado por scripts/train_cnn.py.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.LUDB_PROCESSED_FILE,
        help="Dataset .npz gerado por scripts/preprocess_ludb.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Diretorio de saida. Por padrao, subpasta 'gradcam_ludb' ao lado "
            "do checkpoint, ao lado de 'teste_ludb' (Etapa 3)."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batimentos por passada de forward/backward.",
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
            f"Treine o modelo antes com: python scripts/train_cnn.py",
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

    modelo, ckpt = training.load_checkpoint(args.checkpoint, device=args.device)
    limiar = config.DEFAULT_THRESHOLD

    dados = io.load_beat_dataset(args.dataset)
    X = dados["series"]
    y = np.asarray(dados["labels"]).astype(int)
    pacientes = np.asarray(dados["pacientes"])
    ritmos = np.asarray(dados["simbolos"])

    print(f"Checkpoint : {args.checkpoint}")
    print(f"  camada alvo : {modelo.last_conv}")
    print(f"Dataset    : {args.dataset}")
    print(f"  {X.shape[0]} batimentos | janela de {X.shape[1]} amostras")

    m, scores = training.evaluate_arrays(
        modelo, X, y, threshold=limiar, device=args.device
    )
    y_pred = (scores >= limiar).astype(np.int64)
    print(
        f"  predicao de referencia: acuracia {m['acuracia']:.4f} | "
        f"sensibilidade {m['sensibilidade']:.4f} | "
        f"especificidade {m['especificidade']:.4f}"
    )

    logging.info(
        "Extraindo Grad-CAM de %d batimentos (lote=%d)...",
        X.shape[0],
        args.batch_size,
    )
    cam_raw = gradcam.grad_cam_1d(modelo, X, batch_size=args.batch_size)
    cam_interp = gradcam.interpola_cam(cam_raw, X.shape[1])
    logging.info(
        "Concluido: cam_raw %s | cam_interp %s", cam_raw.shape, cam_interp.shape
    )

    out_dir = args.output_dir or (args.checkpoint.parent / "gradcam_ludb")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / GRADCAM_NAME,
        cam_raw=cam_raw.astype(np.float32),
        cam_interp=cam_interp.astype(np.float32),
        y_true=y,
        y_pred=y_pred,
        y_score=scores.astype(np.float32),
        pacientes=pacientes,
        ritmos=ritmos,
        threshold=np.array(limiar),
    )

    info = {
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etapa": "4 - extracao do Grad-CAM 1D (LUDB)",
        "checkpoint": str(args.checkpoint),
        "dataset": str(args.dataset),
        "camada_alvo": str(modelo.last_conv),
        "n_batimentos": int(X.shape[0]),
        "comprimento_cam_raw": int(cam_raw.shape[1]),
        "tamanho_janela": int(X.shape[1]),
        "interpolacao": "linear (np.interp sobre eixo normalizado [0, 1])",
        "threshold": limiar,
        "metrics_referencia": m,
        "observacao": (
            "cam_raw e cam_interp estao alinhados por indice com o dataset do "
            "LUDB e com outputs/models/<run>/teste_ludb/ludb_predictions.npz "
            "(mesma ordem, mesmo checkpoint, mesmo limiar fixo)."
        ),
    }
    (out_dir / GRADCAM_INFO_NAME).write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nGrad-CAM  : {out_dir / GRADCAM_NAME}")
    print(f"Metadados : {out_dir / GRADCAM_INFO_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
