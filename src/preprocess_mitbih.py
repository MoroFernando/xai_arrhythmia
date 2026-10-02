"""CLI de pre-processamento do MIT-BIH Arrhythmia Database.

Exemplos:
    # Baixa a base para data/raw/mitdb e processa localmente
    python src/preprocess_mitbih.py --download

    # Processa lendo direto do PhysioNet (sem cache local)
    python src/preprocess_mitbih.py --stream

    # Ativa a politica restrita do projeto (nao e AAMI)
    python src/preprocess_mitbih.py --only-arrhythmia
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Permite executar o script de qualquer diretorio (python src/...).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import beat_io  # noqa: E402
import config  # noqa: E402
import classification  # noqa: E402
from datasets import mitbih  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmenta e normaliza os batimentos do MIT-BIH.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Arquivo .npz de saida.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=config.MITBIH_RAW_DIR,
        help="Diretorio com os registros WFDB locais.",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Baixa os registros do PhysioNet para --data-dir antes de processar.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Le direto do PhysioNet a cada registro, ignorando --data-dir.",
    )
    parser.add_argument(
        "--lead",
        default=config.TARGET_LEAD,
        help="Derivacao a extrair (o projeto usa a derivacao II / MLII).",
    )
    parser.add_argument(
        "--window-left",
        type=int,
        default=config.WINDOW_LEFT,
        help="Amostras antes do pico R.",
    )
    parser.add_argument(
        "--window-right",
        type=int,
        default=config.WINDOW_RIGHT,
        help="Amostras a partir do pico R.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Desativa a Z-normalization por batimento.",
    )
    classification.add_cli_argument(parser)
    return classification.prepare_args(parser.parse_args(), "preprocess_mitbih")


def main() -> int:
    args = parse_args()
    classification.require_new_output(args.output)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    records = (
        config.MITBIH_RECORDS_ALL if args.include_paced else config.MITBIH_RECORDS
    )

    if args.download:
        mitbih.download_records(records=records, dest_dir=args.data_dir)

    cfg = mitbih.PreprocessConfig(
        records=records,
        lead=args.lead,
        window_left=args.window_left,
        window_right=args.window_right,
        only_arrhythmia=args.only_arrhythmia,
        include_paced=args.include_paced,
        normalize=not args.no_normalize,
        data_dir=None if args.stream else args.data_dir,
    )

    dataset = mitbih.build_dataset(cfg)
    print(mitbih.summarize(dataset))

    destino = beat_io.save_beat_dataset(
        args.output,
        series=dataset["series"],
        labels=dataset["labels"],
        simbolos=dataset["simbolos"],
        pacientes=dataset["pacientes"],
        metadata=dataset["metadata"],
        r_peaks=dataset["r_peaks"],
    )
    size_mb = destino.stat().st_size / (1024 * 1024)
    print(f"\nDataset salvo em: {destino} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
