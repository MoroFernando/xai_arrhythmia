"""Carregamento e pre-processamento do MIT-BIH Arrhythmia Database.

Pipeline (Etapa 2 do projeto, conjunto de treino):
    download/leitura via WFDB -> selecao da derivacao II (MLII)
    -> segmentacao beat-to-beat centrada no pico R
    -> descarte de anotacoes non-beat
    -> Z-normalization por batimento
    -> rotulagem binaria (0 = nao-arritmia, 1 = arritmia)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import wfdb

import config
import classification

from . import signals

logger = logging.getLogger(__name__)


@dataclass
class PreprocessConfig:
    """Parametros do pre-processamento do MIT-BIH."""

    records: Sequence[str] = field(default_factory=lambda: list(config.MITBIH_RECORDS))
    lead: str = config.TARGET_LEAD
    window_left: int = config.WINDOW_LEFT
    window_right: int = config.WINDOW_RIGHT
    non_beat_symbols: frozenset[str] = config.NON_BEAT_SYMBOLS
    only_arrhythmia: bool = config.ONLY_ARRHYTHMIA
    include_paced: bool = False
    normalize: bool = True
    data_dir: Path | None = None  # None => leitura direta do PhysioNet

    @property
    def window_size(self) -> int:
        return self.window_left + self.window_right

    def to_metadata(self) -> dict:
        """Serializa os parametros para gravacao junto ao dataset."""
        return {
            "dataset": "mit-bih-arrhythmia",
            "fs_hz": config.MITBIH_FS,
            "records": list(self.records),
            "lead": self.lead,
            "window_left": self.window_left,
            "window_right": self.window_right,
            "window_size": self.window_size,
            "non_beat_symbols": sorted(self.non_beat_symbols),
            "normal_symbols": sorted(s for s in classification.MIT_KNOWN_SYMBOLS
                                     if classification.mit_label(s, self.only_arrhythmia, self.include_paced) == 0),
            "normalization": "z-score per beat" if self.normalize else "none",
            **classification.metadata(self.only_arrhythmia, self.include_paced),
        }


def download_records(
    records: Iterable[str] | None = None,
    dest_dir: Path = config.MITBIH_RAW_DIR,
) -> Path:
    """Baixa os registros do MIT-BIH do PhysioNet para uso offline.

    Args:
        records: ids a baixar; ``None`` baixa a base inteira.
        dest_dir: diretorio destino (``data/raw/mitdb`` por padrao).

    Returns:
        O diretorio onde os arquivos foram gravados.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    kwargs = {"db_dir": config.MITBIH_PN_DIR, "dl_dir": str(dest_dir)}
    if records is not None:
        kwargs["records"] = list(records)

    logger.info("Baixando MIT-BIH para %s", dest_dir)
    wfdb.dl_database(**kwargs)
    return dest_dir


def _read_record(record_id: str, data_dir: Path | None):
    """Le sinal e anotacoes de um registro, local ou direto do PhysioNet."""
    if data_dir is not None:
        base = str(Path(data_dir) / record_id)
        return wfdb.rdrecord(base), wfdb.rdann(base, "atr")

    return (
        wfdb.rdrecord(record_id, pn_dir=config.MITBIH_PN_DIR),
        wfdb.rdann(record_id, "atr", pn_dir=config.MITBIH_PN_DIR),
    )


def load_record_beats(
    record_id: str,
    cfg: PreprocessConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segmenta os batimentos de um unico registro (sem normalizar).

    Args:
        record_id: id do registro (ex.: ``"100"``).
        cfg: parametros do pre-processamento.

    Returns:
        Tupla ``(segmentos, simbolos, picos_r)``.

    Raises:
        ValueError: se a derivacao alvo nao existir no registro.
    """
    cfg = cfg or PreprocessConfig()
    if record_id in classification.MIT_PACED_RECORDS and not cfg.include_paced:
        raise classification.ClassificationError(f"Registro {record_id}: pacing fora da metodologia.")
    record, annotation = _read_record(record_id, cfg.data_dir)

    # Validar antes de segment_beats converter simbolos para strings <U2.
    unexpected = set(annotation.symbol) - config.NON_BEAT_SYMBOLS - classification.MIT_KNOWN_SYMBOLS
    if unexpected:
        raise classification.ClassificationError(
            f"Registro {record_id}: simbolos desconhecidos {sorted(unexpected)!r}."
        )

    signal = signals.select_lead(record.p_signal, record.sig_name, cfg.lead)
    if signal is None:
        raise ValueError(
            f"Registro {record_id} nao possui a derivacao {cfg.lead} "
            f"(canais disponiveis: {record.sig_name})"
        )

    return signals.segment_beats(
        signal,
        annotation.sample,
        annotation.symbol,
        window_left=cfg.window_left,
        window_right=cfg.window_right,
        non_beat_symbols=cfg.non_beat_symbols,
    )


def build_dataset(cfg: PreprocessConfig | None = None) -> dict:
    """Processa todos os registros e monta o dataset de batimentos.

    Registros sem a derivacao alvo sao ignorados com aviso, e nao interrompem
    o processamento.

    Returns:
        Dicionario com ``series``, ``labels``, ``simbolos``, ``pacientes``,
        ``r_peaks`` e ``metadata``.
    """
    cfg = cfg or PreprocessConfig()

    all_segments: list[np.ndarray] = []
    all_symbols: list[np.ndarray] = []
    all_peaks: list[np.ndarray] = []
    all_patients: list[np.ndarray] = []
    skipped: list[str] = []
    excluded_symbols = classification.MIT_EXCLUDED_SYMBOLS | (
        frozenset() if cfg.include_paced else classification.MIT_PACED_SYMBOLS
    )
    excluded_counts = {symbol: 0 for symbol in sorted(excluded_symbols)}
    excluded_by_record: dict[str, dict[str, int]] = {}
    paced_records = [r for r in cfg.records if r in classification.MIT_PACED_RECORDS and not cfg.include_paced]

    for record_id in cfg.records:
        if record_id in paced_records:
            continue
        try:
            segments, symbols, peaks = load_record_beats(record_id, cfg)
        except ValueError as exc:
            logger.warning("Registro %s ignorado: %s", record_id, exc)
            skipped.append(record_id)
            continue

        # Exclusao de classificacao apos os filtros existentes, mantendo X e
        # picos alinhados. As contagens se referem a janelas antes elegiveis.
        removed = {s: int((symbols == s).sum()) for s in excluded_counts}
        excluded_by_record[record_id] = removed
        for symbol, count in removed.items():
            excluded_counts[symbol] += count
        keep = ~np.isin(symbols, list(excluded_symbols))
        segments, symbols, peaks = segments[keep], symbols[keep], peaks[keep]

        if segments.shape[0] == 0:
            logger.warning("Registro %s nao gerou batimentos validos", record_id)
            skipped.append(record_id)
            continue

        logger.info("Registro %s: %d batimentos", record_id, segments.shape[0])
        all_segments.append(segments)
        all_symbols.append(symbols)
        all_peaks.append(peaks)
        all_patients.append(np.full(segments.shape[0], record_id, dtype="<U4"))

    if not all_segments:
        raise RuntimeError("Nenhum batimento extraido — verifique os registros.")

    series = np.concatenate(all_segments, axis=0)
    simbolos = np.concatenate(all_symbols, axis=0)
    r_peaks = np.concatenate(all_peaks, axis=0)
    pacientes = np.concatenate(all_patients, axis=0)

    if cfg.normalize:
        series = signals.z_normalize_per_beat(series)

    labels = signals.encode_binary_labels(simbolos, cfg.only_arrhythmia, cfg.include_paced)

    metadata = cfg.to_metadata()
    metadata["records_processados"] = [r for r in cfg.records if r not in skipped and r not in paced_records]
    metadata["records_excluidos_pacing"] = sorted(paced_records)
    metadata["records_pacing_fora_da_politica"] = [] if cfg.include_paced else sorted(classification.MIT_PACED_RECORDS)
    metadata["simbolos_excluidos_apos_bordas"] = excluded_counts
    metadata["simbolos_excluidos_por_registro"] = excluded_by_record
    metadata["records_ignorados"] = skipped
    metadata["n_batimentos"] = int(series.shape[0])
    metadata["distribuicao_simbolos"] = {
        str(symbol): int((simbolos == symbol).sum()) for symbol in np.unique(simbolos)
    }
    metadata["distribuicao_labels"] = {
        "0": int((labels == 0).sum()),
        "1": int((labels == 1).sum()),
    }

    return {
        "series": series,
        "labels": labels,
        "simbolos": simbolos,
        "pacientes": pacientes,
        "r_peaks": r_peaks,
        "metadata": metadata,
    }


def summarize(dataset: dict) -> str:
    """Gera um resumo textual do dataset processado."""
    series = dataset["series"]
    labels = dataset["labels"]
    simbolos = dataset["simbolos"]
    pacientes = dataset["pacientes"]

    n_normal = int((labels == 0).sum())
    n_arr = int((labels == 1).sum())
    total = int(labels.size)

    symbol_ids, symbol_counts = np.unique(simbolos, return_counts=True)
    ordem = np.argsort(-symbol_counts)

    linhas = [
        "=" * 60,
        "RESUMO DO DATASET MIT-BIH",
        "=" * 60,
        f"Shape das series      : {series.shape}",
        f"Registros processados : {len(np.unique(pacientes))}",
        f"Total de batimentos   : {total}",
        f"  classe 0 (normal)   : {n_normal} ({100 * n_normal / total:.2f}%)",
        f"  classe 1 (arritmia) : {n_arr} ({100 * n_arr / total:.2f}%)",
        "-" * 60,
        "Distribuicao por simbolo original:",
    ]
    for idx in ordem:
        simbolo = symbol_ids[idx]
        contagem = int(symbol_counts[idx])
        classe = 0 if labels[simbolos == simbolo][0] == 0 else 1
        linhas.append(
            f"  {simbolo:<3} -> classe {classe} : {contagem:>7} "
            f"({100 * contagem / total:.2f}%)"
        )
    linhas.append("=" * 60)
    return "\n".join(linhas)
