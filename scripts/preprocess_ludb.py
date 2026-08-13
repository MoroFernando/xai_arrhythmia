"""CLI de pre-processamento do LUDB (conjunto de teste / ground truth).

Exemplos:
    # Baixa a base para data/raw/ludb e processa localmente
    python scripts/preprocess_ludb.py --download

    # Processa lendo direto do PhysioNet (sem cache local)
    python scripts/preprocess_ludb.py --stream
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Permite executar o script sem instalar o pacote (python scripts/...).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xai_arrhythmia import config, io  # noqa: E402
from xai_arrhythmia.datasets import ludb  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segmenta, reamostra e normaliza os batimentos do LUDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=config.LUDB_PROCESSED_FILE,
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
        "--normal-rhythms",
        choices=["strict", "sinus"],
        default="strict",
        help=(
            "Ritmos mapeados para a classe 0. 'strict' usa apenas "
            "'sinus rhythm'; 'sinus' agrupa todos os ritmos de origem sinusal."
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

    if args.download:
        ludb.download_records(dest_dir=args.data_dir, lead=args.lead)

    normal_rhythms = (
        config.LUDB_SINUS_RHYTHMS
        if args.normal_rhythms == "sinus"
        else config.LUDB_NORMAL_RHYTHMS
    )

    cfg = ludb.PreprocessConfig(
        lead=args.lead,
        fs_target=args.fs_target,
        window_left=args.window_left,
        window_right=args.window_right,
        center=args.center,
        normal_rhythms=normal_rhythms,
        normalize=not args.no_normalize,
        data_dir=None if args.stream else args.data_dir,
    )

    dataset = ludb.build_dataset(cfg)
    print(ludb.summarize(dataset))

    destino = io.save_beat_dataset(
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
    )
    size_mb = destino.stat().st_size / (1024 * 1024)
    print(f"\nDataset salvo em: {destino} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
