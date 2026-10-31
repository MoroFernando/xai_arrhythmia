"""CLI de pre-processamento do LUDB (conjunto de teste / ground truth).

Exemplos:
    # Baixa a base para data/raw/ludb e processa localmente
    python src/preprocess_ludb.py --download

    # Processa lendo direto do PhysioNet (sem cache local)
    python src/preprocess_ludb.py --stream
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
from datasets import ludb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmenta, reamostra e normaliza os batimentos do LUDB.",
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
        default=config.LUDB_RAW_DIR,
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
        default=config.LUDB_LEAD,
        help="Derivacao a extrair (o projeto usa a derivacao II).",
    )
    parser.add_argument(
        "--fs-target",
        type=int,
        default=config.TARGET_FS,
        help="Frequencia alvo da reamostragem (deve bater com a do MIT-BIH).",
    )
    parser.add_argument(
        "--window-left",
        type=int,
        default=config.WINDOW_LEFT,
        help="Amostras antes do centro da janela.",
    )
    parser.add_argument(
        "--window-right",
        type=int,
        default=config.WINDOW_RIGHT,
        help="Amostras a partir do centro da janela.",
    )
    parser.add_argument(
        "--center",
        choices=["qrs-peak", "qrs-midpoint"],
        default="qrs-peak",
        help=(
            "Ponto de centragem da janela. 'qrs-peak' usa o pico R anotado, "
            "alinhando com a centragem do MIT-BIH."
        ),
    )
    parser.add_argument(
        "--clinical-csv", type=Path, default=config.LUDB_RAW_DIR / "ludb.csv",
        help="CSV oficial LUDB 1.0.1, obrigatorio inclusive com --stream/--download.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Desativa a Z-normalization por batimento.",
    )
    classification.add_cli_argument(parser)
    return classification.prepare_args(parser.parse_args(), "preprocess_ludb")


def main() -> int:
    args = parse_args()
    classification.require_new_output(args.output)
    # Validar a fonte clinica antes de qualquer download/processamento do sinal.
    ludb.read_clinical_csv(args.clinical_csv, config.LUDB_RECORDS)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.download:
        ludb.download_records(dest_dir=args.data_dir, lead=args.lead)

    cfg = ludb.PreprocessConfig(
        lead=args.lead,
        fs_target=args.fs_target,
        window_left=args.window_left,
        window_right=args.window_right,
        center=args.center,
        classification_policy=args.classification_policy,
        clinical_csv=args.clinical_csv,
        normalize=not args.no_normalize,
        data_dir=None if args.stream else args.data_dir,
    )

    dataset = ludb.build_dataset(cfg)
    print(ludb.summarize(dataset))

    destino = beat_io.save_beat_dataset(
        args.output,
        series=dataset["series"],
        labels=dataset["labels"],
        simbolos=dataset["simbolos"],
        pacientes=dataset["pacientes"],
        metadata=dataset["metadata"],
        mascara_clinica=dataset["mascara_clinica"],
        mascara_p=dataset["mascara_p"],
        mascara_qrs=dataset["mascara_qrs"],
        mascara_t=dataset["mascara_t"],
        regiao_p=dataset["regiao_p"],
        regiao_qrs=dataset["regiao_qrs"],
        regiao_t=dataset["regiao_t"],
        r_peaks=dataset["r_peaks"],
        tem_onda_p=dataset["tem_onda_p"],
        beat_ids=dataset["beat_ids"],
        selection_indices=dataset["selection_indices"],
        selection_status=dataset["selection_status"],
        target_families=dataset["target_families"],
        target_subtypes=dataset["target_subtypes"],
        bundle_side=dataset["bundle_side"],
        bundle_completeness=dataset["bundle_completeness"],
        positive_reasons=dataset["positive_reasons"],
        other_conditions=dataset["other_conditions"],
        ritmos=dataset["ritmos"],
        is_shared_subset=dataset["is_shared_subset"],
    )
    size_mb = destino.stat().st_size / (1024 * 1024)
    print(f"\nDataset salvo em: {destino} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
