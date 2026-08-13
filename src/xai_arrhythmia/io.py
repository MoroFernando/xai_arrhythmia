"""Persistencia dos datasets processados em formato NPZ."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

#: Chaves obrigatorias de um dataset de batimentos processado.
DATASET_KEYS = ("series", "labels", "simbolos", "pacientes")


def save_beat_dataset(
    path: str | Path,
    series: np.ndarray,
    labels: np.ndarray,
    simbolos: np.ndarray,
    pacientes: np.ndarray,
    metadata: dict[str, Any] | None = None,
    **extra: np.ndarray,
) -> Path:
    """Salva o dataset de batimentos em ``.npz`` compactado.

    Args:
        path: caminho do arquivo de saida (``.npz``).
        series: batimentos segmentados e normalizados ``(n, window_size)``.
        labels: rotulos binarios ``(n,)`` — 0 nao-arritmia, 1 arritmia.
        simbolos: simbolo original do MIT-BIH de cada batimento ``(n,)``.
        pacientes: id do registro de origem de cada batimento ``(n,)``.
        metadata: dicionario serializavel em JSON com os parametros usados.
        **extra: arrays adicionais (ex.: ``r_peaks``).

    Returns:
        O caminho efetivamente escrito.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {
        "series": np.asarray(series),
        "labels": np.asarray(labels),
        "simbolos": np.asarray(simbolos),
        "pacientes": np.asarray(pacientes),
        **{key: np.asarray(value) for key, value in extra.items()},
    }

    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) > 1:
        raise ValueError(f"Arrays com tamanhos divergentes: {lengths}")

    if metadata is not None:
        # NPZ so guarda arrays; o JSON e embutido como string em array 0-d.
        arrays["metadata_json"] = np.array(
            json.dumps(metadata, ensure_ascii=False, indent=2)
        )

    np.savez_compressed(path, **arrays)
    return path


def load_beat_dataset(path: str | Path) -> dict[str, Any]:
    """Carrega um dataset salvo por :func:`save_beat_dataset`.

    Returns:
        Dicionario com os arrays e, se presente, a chave ``metadata`` ja
        desserializada a partir do JSON.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        data: dict[str, Any] = {key: npz[key] for key in npz.files}

    raw_metadata = data.pop("metadata_json", None)
    if raw_metadata is not None:
        data["metadata"] = json.loads(str(raw_metadata))

    missing = [key for key in DATASET_KEYS if key not in data]
    if missing:
        raise KeyError(f"Dataset {path} sem as chaves obrigatorias: {missing}")

    return data
