"""CLI de treino da CNN 1D binaria no MIT-BIH (Etapa 3).

Este script cobre APENAS treino e validacao, ambos dentro do MIT-BIH e com
divisao inter-paciente (DS1 treina, DS2 valida). O teste acontece no LUDB, base
independente, e fica em scripts/test_ludb.py — que carrega o checkpoint gravado
aqui. Manter os dois separados e o que garante que nenhuma decisao de treino
possa ser tomada olhando o conjunto de teste.

Exemplos:
    # Treino padrao (split DS1/DS2 de de Chazal et al.)
    python scripts/train_cnn.py

    # Split inter-paciente aleatorio, 30% dos batimentos na validacao
    python scripts/train_cnn.py --split random --val-fraction 0.3 --seed 7

    # Em seguida, o teste independente:
    python scripts/test_ludb.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Permite executar o script sem instalar o pacote (python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xai_arrhythmia import config, io, metrics, training  # noqa: E402
from xai_arrhythmia.models import CNN1DConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina a CNN 1D binaria no MIT-BIH com split inter-paciente.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.MITBIH_PROCESSED_FILE,
        help="Dataset .npz gerado por scripts/preprocess_mitbih.py.",
    )
    parser.add_argument(
        "--run-name",
        default="cnn1d_mitbih",
        help="Nome do diretorio da execucao dentro de --models-dir.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=config.MODELS_DIR,
        help="Diretorio raiz onde as execucoes sao gravadas.",
    )

    grupo_split = parser.add_argument_group("divisao inter-paciente")
    grupo_split.add_argument(
        "--split",
        choices=["ds1ds2", "random"],
        default="ds1ds2",
        help=(
            "'ds1ds2' usa a particao de de Chazal et al. (2004); "
            "'random' sorteia pacientes ate atingir --val-fraction."
        ),
    )
    grupo_split.add_argument(
        "--val-fraction",
        type=float,
        default=0.3,
        help="Fracao de batimentos na validacao (apenas --split random).",
    )

    grupo_treino = parser.add_argument_group("treino")
    grupo_treino.add_argument("--epochs", type=int, default=40)
    grupo_treino.add_argument("--batch-size", type=int, default=256)
    grupo_treino.add_argument("--lr", type=float, default=1e-3)
    grupo_treino.add_argument("--weight-decay", type=float, default=1e-4)
    grupo_treino.add_argument(
        "--patience",
        type=int,
        default=8,
        help="Epocas sem melhora na validacao antes do early stopping.",
    )
    grupo_treino.add_argument(
        "--criterio",
        choices=["acuracia_balanceada", "f1", "auc_roc", "acuracia"],
        default="acuracia_balanceada",
        help=(
            "Metrica de validacao que seleciona a melhor epoca (early "
            "stopping). O limiar de decisao e sempre config.DEFAULT_THRESHOLD."
        ),
    )
    grupo_treino.add_argument(
        "--sem-balanceamento",
        action="store_true",
        help=(
            "Desativa o pos_weight da BCE. Por padrao a classe arritmia e "
            "reponderada pela razao negativos/positivos do conjunto de treino."
        ),
    )
    grupo_treino.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    grupo_treino.add_argument(
        "--device",
        default="auto",
        help="'auto', 'cuda' ou 'cpu'.",
    )
    grupo_treino.add_argument("--num-workers", type=int, default=0)

    grupo_modelo = parser.add_argument_group("arquitetura")
    grupo_modelo.add_argument(
        "--channels",
        type=int,
        nargs="+",
        default=[32, 64, 128, 128],
        help="Canais de cada bloco convolucional.",
    )
    grupo_modelo.add_argument("--kernel-size", type=int, default=7)
    grupo_modelo.add_argument("--dropout", type=float, default=0.3)
    grupo_modelo.add_argument("--dropout-conv", type=float, default=0.1)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.dataset.exists():
        print(
            f"Dataset nao encontrado: {args.dataset}\n"
            f"Gere-o antes com: python scripts/preprocess_mitbih.py --download",
            file=sys.stderr,
        )
        return 1

    dados = io.load_beat_dataset(args.dataset)
    print(
        f"Dataset: {args.dataset} | {dados['series'].shape[0]} batimentos "
        f"de {len(set(dados['pacientes'].tolist()))} registros"
    )

    model_cfg = CNN1DConfig(
        input_size=int(dados["series"].shape[1]),
        channels=list(args.channels),
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        dropout_conv=args.dropout_conv,
    )
    train_cfg = training.TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        criterio_selecao=args.criterio,
        balancear_classes=not args.sem_balanceamento,
        split_scheme=args.split,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        num_workers=args.num_workers,
    )

    resultado = training.train_model(dados, train_cfg, model_cfg)
    m = resultado["metrics"]

    print("\n" + "=" * 60)
    print("RESULTADO DO TREINO (Etapa 3)")
    print("=" * 60)
    print(f"Melhor epoca: {m['melhor_epoca']} de {m['epocas_executadas']} executadas")
    print(f"Duracao     : {m['duracao_s']:.1f} s")
    print("-" * 60)
    print(metrics.formata_metricas("TREINO (DS1)", m["treino"]))
    print("-" * 60)
    print(metrics.formata_metricas("VALIDACAO (DS2)", m["validacao"]))
    print("=" * 60)

    out_dir = training.save_run(
        resultado,
        Path(args.models_dir) / args.run_name,
        dataset_metadata=dados.get("metadata"),
    )
    print(f"\nCheckpoint : {out_dir / training.CHECKPOINT_NAME}")
    print(f"Metricas   : {out_dir / training.METRICS_NAME}")
    print(f"Historico  : {out_dir / training.HISTORY_NAME}")
    print(
        "\nProximo passo (teste independente):\n"
        f"  python scripts/test_ludb.py --checkpoint "
        f"{out_dir / training.CHECKPOINT_NAME}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
