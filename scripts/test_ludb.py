"""CLI de teste do modelo treinado no conjunto independente LUDB (Etapa 3/4).

Este script NAO treina nada: ele carrega um checkpoint gerado por
scripts/train_cnn.py, congela os pesos e mede o desempenho cross-dataset no
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
    python scripts/test_ludb.py

    # Outro checkpoint / outra saida
    python scripts/test_ludb.py --checkpoint outputs/models/exp2/model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Permite executar o script sem instalar o pacote e importar o módulo irmão.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import model_gradcam as mg  # noqa: E402
from xai_arrhythmia import config, io, metrics  # noqa: E402

PREDICTIONS_NAME = "ludb_predictions.npz"
RESULTS_NAME = "ludb_metrics.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avalia no LUDB um modelo ja treinado no MIT-BIH.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=config.MODELS_DIR / "cnn1d_mitbih" / mg.CHECKPOINT_NAME,
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

    modelo, ckpt = mg.load_checkpoint(args.checkpoint, device=args.device)
    info = ckpt.get("info", {})
    limiar = mg.DEFAULT_THRESHOLD

    dados = io.load_beat_dataset(args.dataset)
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
    m_ludb = metrics.binary_metrics(labels, scores, threshold=limiar)
    print(metrics.formata_metricas("LUDB (cross-dataset)", m_ludb))
    print("-" * 64)

    grupos = metrics.por_grupo(labels, scores, ritmos, threshold=limiar)
    print(metrics.formata_por_grupo("LUDB por ritmo", grupos))
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
        "checkpoint": str(args.checkpoint),
        "dataset_teste": str(args.dataset),
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
