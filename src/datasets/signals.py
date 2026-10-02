"""Operacoes de sinal compartilhadas pelas duas bases (MIT-BIH e LUDB).

Manter segmentacao e normalizacao aqui garante que o conjunto de teste
(LUDB) receba exatamente o mesmo tratamento do conjunto de treino, requisito
da validacao cross-dataset descrita no artigo.
"""

from __future__ import annotations

from math import gcd
from typing import Iterable, Sequence

import numpy as np
from scipy.signal import resample_poly

import config
import classification


def select_lead(
    signals: np.ndarray,
    sig_names: Sequence[str],
    lead: str = config.TARGET_LEAD,
) -> np.ndarray | None:
    """Extrai o canal correspondente a derivacao desejada.

    Args:
        signals: matriz ``(n_amostras, n_canais)`` retornada pelo WFDB.
        sig_names: nomes dos canais, na ordem das colunas.
        lead: derivacao alvo (ex.: ``"MLII"``).

    Returns:
        Vetor 1D do canal, ou ``None`` se a derivacao nao existir no registro.

    Note:
        Nem todo registro do MIT-BIH possui MLII no canal 0 (102 e 104 usam
        V5), por isso a selecao e feita pelo nome do canal, nunca pelo indice.
    """
    aliases = config.LEAD_ALIASES.get(lead.upper(), (lead,))
    normalized = [str(name).strip().upper() for name in sig_names]

    for alias in aliases:
        alias_up = alias.strip().upper()
        if alias_up in normalized:
            return np.asarray(signals[:, normalized.index(alias_up)], dtype=np.float32)
    return None


def segment_beats(
    signal: np.ndarray,
    r_peaks: Sequence[int],
    symbols: Sequence[str],
    window_left: int = config.WINDOW_LEFT,
    window_right: int = config.WINDOW_RIGHT,
    non_beat_symbols: Iterable[str] = config.NON_BEAT_SYMBOLS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Segmenta o sinal em janelas beat-to-beat centradas no pico R.

    Descarta anotacoes que nao representam batimentos e janelas que
    extrapolariam os limites do sinal (primeiros e ultimos batimentos).

    Args:
        signal: sinal de ECG 1D de um unico registro.
        r_peaks: indices amostrais das anotacoes (``annotation.sample``).
        symbols: simbolos das anotacoes (``annotation.symbol``).
        window_left: amostras antes do pico R.
        window_right: amostras a partir do pico R (inclusive).
        non_beat_symbols: simbolos a descartar.

    Returns:
        Tupla ``(segmentos, simbolos, picos)`` onde ``segmentos`` tem shape
        ``(n_batimentos, window_left + window_right)``.
    """
    signal = np.asarray(signal, dtype=np.float32)
    n_samples = signal.shape[0]
    discard = frozenset(non_beat_symbols)
    window_size = window_left + window_right

    segments: list[np.ndarray] = []
    kept_symbols: list[str] = []
    kept_peaks: list[int] = []

    for peak, symbol in zip(r_peaks, symbols):
        if symbol in discard:
            continue

        start = int(peak) - window_left
        end = int(peak) + window_right
        if start < 0 or end > n_samples:
            continue

        segments.append(signal[start:end])
        kept_symbols.append(symbol)
        kept_peaks.append(int(peak))

    if not segments:
        return (
            np.empty((0, window_size), dtype=np.float32),
            np.empty((0,), dtype="<U2"),
            np.empty((0,), dtype=np.int64),
        )

    return (
        np.stack(segments).astype(np.float32),
        np.asarray(kept_symbols, dtype="<U2"),
        np.asarray(kept_peaks, dtype=np.int64),
    )


def resample_signal(
    signal: np.ndarray,
    fs_original: int,
    fs_target: int,
) -> tuple[np.ndarray, float]:
    """Reamostra o sinal para a frequencia alvo via filtro polifasico.

    Args:
        signal: sinal 1D.
        fs_original: frequencia de amostragem do sinal.
        fs_target: frequencia desejada.

    Returns:
        Tupla ``(sinal_reamostrado, razao)``, onde ``razao`` e o fator pelo
        qual os indices temporais devem ser multiplicados.

    Note:
        ``resample_poly`` aplica o filtro anti-aliasing necessario na
        decimacao 500 -> 360 Hz, o que uma interpolacao ingenua nao faria.
    """
    signal = np.asarray(signal, dtype=np.float32)
    if fs_original == fs_target:
        return signal, 1.0

    divisor = gcd(fs_target, fs_original)
    up = fs_target // divisor
    down = fs_original // divisor
    resampled = resample_poly(signal, up, down).astype(np.float32)
    return resampled, fs_target / fs_original


def rescale_positions(positions: Sequence[int], ratio: float) -> list[int]:
    """Converte indices amostrais para a escala de um sinal reamostrado."""
    return [int(round(position * ratio)) for position in positions]


def build_region_mask(
    regions: Iterable[tuple[int, int]],
    window_start: int,
    window_size: int,
) -> np.ndarray:
    """Monta a mascara binaria de regioes anotadas dentro de uma janela.

    Args:
        regions: pares ``(onset, offset)`` em coordenadas do sinal completo.
        window_start: indice inicial da janela no sinal completo.
        window_size: tamanho da janela em amostras.

    Returns:
        Vetor ``int8`` de tamanho ``window_size`` com 1 nas amostras cobertas.

    Note:
        Os limites do LUDB sao inclusivos nas duas pontas, entao o ``offset``
        entra na mascara (dai o ``+ 1`` no recorte).
    """
    mask = np.zeros(window_size, dtype=np.int8)

    for onset, offset in regions:
        start = max(0, int(onset) - window_start)
        end = min(window_size, int(offset) - window_start + 1)
        if start < end:
            mask[start:end] = 1

    return mask


def z_normalize_per_beat(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Aplica Z-normalization independente em cada batimento.

    Normalizar por batimento (e nao pelo registro inteiro) remove diferencas
    de ganho e offset entre pacientes e entre bases, condicao necessaria para
    aplicar no LUDB um modelo treinado no MIT-BIH.

    Args:
        X: matriz ``(n_batimentos, n_amostras)``.
        eps: limiar abaixo do qual o desvio padrao e tratado como zero.

    Returns:
        Matriz normalizada, mesmo shape da entrada. Batimentos constantes
        (desvio padrao nulo) resultam em vetor de zeros, sem divisao por zero.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.size == 0:
        return X

    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True)
    std_safe = np.where(std < eps, 1.0, std)
    return ((X - mean) / std_safe).astype(np.float32)


def encode_binary_labels(
    symbols: Sequence[str],
    only_arrhythmia: bool = config.ONLY_ARRHYTHMIA,
    include_paced: bool = False,
) -> np.ndarray:
    """Mapeia simbolos do MIT-BIH para rotulos binarios.

    Returns:
        Vetor ``int64`` com 0 para nao-arritmia e 1 para arritmia.
    """
    classification.metadata(only_arrhythmia, include_paced)
    labels = [classification.mit_label(symbol, only_arrhythmia, include_paced) for symbol in symbols]
    if any(label is None for label in labels):
        raise classification.ClassificationError("Simbolos excluidos pela politica chegaram ao encoder.")
    return np.asarray(labels, dtype=np.int64)
