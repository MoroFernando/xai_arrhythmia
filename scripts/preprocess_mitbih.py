"""CLI de pre-processamento do MIT-BIH Arrhythmia Database.

Exemplos:
    # Baixa a base para data/raw/mitdb e processa localmente
    python scripts/preprocess_mitbih.py --download

    # Processa lendo direto do PhysioNet (sem cache local)
    python scripts/preprocess_mitbih.py --stream

    # Reproduz exatamente o mapeamento AAMI da classe N
    python scripts/preprocess_mitbih.py --normal-symbols AAMI
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Permite executar o script sem instalar o pacote (python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xai_arrhythmia import config, io  # noqa: E402
from xai_arrhythmia.datasets import mitbih  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmenta e normaliza os batimentos do MIT-BIH.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.MITBIH_PROCESSED_FILE,
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
        "--include-paced",
        action="store_true",
        help=(
            "Inclui os registros 102, 104, 107 e 217 (pacientes com marcapasso), "
            "excluidos por padrao conforme pratica usual da literatura."
        ),
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
        "--normal-symbols",
        choices=["N", "AAMI"],
        default="N",
        help=(
            "Simbolos mapeados para a classe 0. 'N' usa apenas batimento normal; "
            "'AAMI' agrupa N, L, R, e, j."
        ),
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Desativa a Z-normalization por batimento.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
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

    normal_symbols = (
        config.NORMAL_SYMBOLS_AAMI
        if args.normal_symbols == "AAMI"
        else config.NORMAL_SYMBOLS
    )

    cfg = mitbih.PreprocessConfig(
        records=records,
        lead=args.lead,
        window_left=args.window_left,
        window_right=args.window_right,
        normal_symbols=normal_symbols,
        normalize=not args.no_normalize,
        data_dir=None if args.stream else args.data_dir,
    )

    dataset = mitbih.build_dataset(cfg)
    print(mitbih.summarize(dataset))

    destino = io.save_beat_dataset(
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
