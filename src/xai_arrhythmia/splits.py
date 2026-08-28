"""Divisao inter-paciente do MIT-BIH (Etapa 3).

A regra do projeto e uma so: um registro (paciente) pertence inteiramente a um
unico conjunto. Batimentos do mesmo paciente compartilham morfologia, ruido de
eletrodo e caracteristicas do proprio individuo — se caissem em treino e
validacao ao mesmo tempo, a rede poderia acertar por reconhecer o paciente, e
nao a arritmia, inflando as metricas em varios pontos percentuais.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from . import config


def indices_por_pacientes(
    pacientes: np.ndarray,
    selecionados: Sequence[str],
) -> np.ndarray:
    """Indices das amostras cujo paciente esta em ``selecionados``."""
    alvo = {str(p) for p in selecionados}
    mask = np.array([str(p) in alvo for p in pacientes], dtype=bool)
    return np.flatnonzero(mask)


def split_ds1_ds2(pacientes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Particao de de Chazal et al. (2004): DS1 para treino, DS2 para validacao."""
    return (
        indices_por_pacientes(pacientes, config.MITBIH_DS1),
        indices_por_pacientes(pacientes, config.MITBIH_DS2),
    )


def split_aleatorio_por_paciente(
    pacientes: np.ndarray,
    labels: np.ndarray,
    val_fraction: float = 0.3,
    seed: int = config.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Sorteia pacientes para a validacao ate atingir ``val_fraction`` das amostras.

    Os pacientes sao embaralhados e acumulados na validacao ate que a fracao
    alvo de BATIMENTOS seja atingida — sortear uma fracao dos pacientes daria
    conjuntos de tamanho imprevisivel, ja que os registros tem contagens de
    batimentos bem diferentes entre si.

    Raises:
        ValueError: se alguma das particoes ficar sem uma das classes, o que
            tornaria a metrica de validacao degenerada.
    """
    rng = np.random.default_rng(seed)
    unicos = np.unique(pacientes)
    ordem = rng.permutation(unicos)

    total = len(pacientes)
    alvo = val_fraction * total
    acumulado = 0
    val_pacientes: list[str] = []

    for paciente in ordem:
        if acumulado >= alvo:
            break
        val_pacientes.append(str(paciente))
        acumulado += int((pacientes == paciente).sum())

    val_idx = indices_por_pacientes(pacientes, val_pacientes)
    train_idx = np.setdiff1d(np.arange(total), val_idx)

    for nome, idx in (("treino", train_idx), ("validacao", val_idx)):
        classes = np.unique(labels[idx])
        if classes.size < 2:
            raise ValueError(
                f"Split aleatorio deixou o conjunto de {nome} com uma unica "
                f"classe ({classes.tolist()}). Tente outra --seed."
            )

    return train_idx, val_idx


def make_split(
    pacientes: np.ndarray,
    labels: np.ndarray,
    scheme: str = "ds1ds2",
    val_fraction: float = 0.3,
    seed: int = config.RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Aplica o esquema de divisao escolhido e valida o isolamento de pacientes.

    Args:
        pacientes: id do registro de origem de cada batimento.
        labels: rotulos binarios, usados apenas para conferir as particoes.
        scheme: ``"ds1ds2"`` (padrao) ou ``"random"``.
        val_fraction: fracao de batimentos na validacao (apenas ``"random"``).
        seed: semente do sorteio (apenas ``"random"``).

    Returns:
        Tupla ``(train_idx, val_idx)``.
    """
    if scheme == "ds1ds2":
        train_idx, val_idx = split_ds1_ds2(pacientes)
    elif scheme == "random":
        train_idx, val_idx = split_aleatorio_por_paciente(
            pacientes, labels, val_fraction=val_fraction, seed=seed
        )
    else:
        raise ValueError(f"Esquema de divisao desconhecido: {scheme!r}")

    if train_idx.size == 0 or val_idx.size == 0:
        raise ValueError(
            f"Esquema {scheme!r} gerou particao vazia "
            f"(treino={train_idx.size}, validacao={val_idx.size})."
        )

    # Garantia explicita do requisito inter-paciente: a intersecao dos
    # conjuntos de registros tem de ser vazia.
    vazamento = set(pacientes[train_idx]) & set(pacientes[val_idx])
    if vazamento:
        raise AssertionError(
            f"Vazamento inter-paciente: registros em ambos os conjuntos: "
            f"{sorted(vazamento)}"
        )

    return train_idx, val_idx


def resumo_split(
    pacientes: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
) -> dict:
    """Descreve a particao para o log e para o metadado do checkpoint."""

    def descreve(idx: np.ndarray) -> dict:
        y = labels[idx]
        n = int(y.size)
        n1 = int((y == 1).sum())
        return {
            "n_batimentos": n,
            "n_pacientes": int(np.unique(pacientes[idx]).size),
            "pacientes": sorted({str(p) for p in pacientes[idx]}),
            "classe_0": n - n1,
            "classe_1": n1,
            "prevalencia_classe_1": round(n1 / n, 4) if n else 0.0,
        }

    return {"treino": descreve(train_idx), "validacao": descreve(val_idx)}
