"""Treino da CNN 1D no MIT-BIH (Etapa 3).

O MIT-BIH e usado inteiro como base de desenvolvimento: DS1 treina, DS2 valida
e seleciona epoca e limiar. O conjunto de teste do projeto e o LUDB, avaliado
so na Etapa 4 — por isso nada aqui olha para ele.
"""

from __future__ import annotations

import json
import logging
import platform
import random
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import config, metrics, splits
from .models import CNN1D, CNN1DConfig

logger = logging.getLogger(__name__)

#: Nome dos arquivos gravados no diretorio da execucao.
CHECKPOINT_NAME = "model.pt"
METRICS_NAME = "metrics.json"
HISTORY_NAME = "history.csv"


@dataclass
class TrainConfig:
    """Parametros do treino."""

    epochs: int = 40
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 8
    criterio_selecao: str = "acuracia_balanceada"
    balancear_classes: bool = True
    split_scheme: str = "ds1ds2"
    val_fraction: float = 0.3
    seed: int = config.RANDOM_SEED
    device: str = "auto"
    num_workers: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def resolve_device(device: str = "auto") -> torch.device:
    """Escolhe a GPU quando disponivel, salvo pedido explicito de CPU."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    """Fixa as sementes de ``random``, NumPy e PyTorch.

    Nao ativa o modo deterministico do cuDNN: o custo em velocidade nao
    compensa, e a semente ja garante a mesma divisao de dados e a mesma
    inicializacao dos pesos.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tensores(series: np.ndarray, labels: np.ndarray) -> TensorDataset:
    x = torch.from_numpy(np.ascontiguousarray(series, dtype=np.float32)).unsqueeze(1)
    y = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))
    return TensorDataset(x, y)


def _pos_weight(labels: np.ndarray) -> float:
    """Peso da classe positiva na BCE: razao negativos/positivos.

    Sem isso a rede tende a maximizar acuracia prevendo "normal" para quase
    tudo (~74% das amostras), o que derruba a sensibilidade — justamente a
    metrica que importa em triagem de arritmia.
    """
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0:
        raise ValueError("Conjunto de treino sem amostras da classe 1.")
    return n_neg / n_pos


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Roda a rede em inferencia e devolve ``(y_true, probabilidades)``."""
    model.eval()
    scores: list[np.ndarray] = []
    alvos: list[np.ndarray] = []
    for xb, yb in loader:
        logits = model(xb.to(device, non_blocking=True))
        scores.append(torch.sigmoid(logits).float().cpu().numpy())
        alvos.append(yb.numpy())
    return np.concatenate(alvos), np.concatenate(scores)


def _epoca_treino(
    model: nn.Module,
    loader: DataLoader,
    criterio: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Uma passada completa de treino; retorna a perda media por amostra."""
    model.train()
    total = 0.0
    n = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = criterio(model(xb), yb)
        loss.backward()
        optimizer.step()

        total += float(loss.item()) * xb.size(0)
        n += xb.size(0)
    return total / max(n, 1)


def train_model(
    dataset: dict,
    train_cfg: TrainConfig | None = None,
    model_cfg: CNN1DConfig | None = None,
) -> dict[str, Any]:
    """Treina a CNN 1D com divisao inter-paciente e selecao por validacao.

    Args:
        dataset: dicionario carregado por
            :func:`xai_arrhythmia.io.load_beat_dataset`.
        train_cfg: parametros do treino.
        model_cfg: hiperparametros da arquitetura.

    Returns:
        Dicionario com ``model``, ``estado`` (state_dict da melhor epoca),
        ``history``, ``metrics`` e ``split``.
    """
    train_cfg = train_cfg or TrainConfig()
    series = np.asarray(dataset["series"], dtype=np.float32)
    labels = np.asarray(dataset["labels"]).astype(int)
    pacientes = np.asarray(dataset["pacientes"])

    model_cfg = model_cfg or CNN1DConfig(input_size=series.shape[1])

    set_seed(train_cfg.seed)
    device = resolve_device(train_cfg.device)

    train_idx, val_idx = splits.make_split(
        pacientes,
        labels,
        scheme=train_cfg.split_scheme,
        val_fraction=train_cfg.val_fraction,
        seed=train_cfg.seed,
    )
    resumo = splits.resumo_split(pacientes, labels, train_idx, val_idx)
    logger.info(
        "Split %s | treino: %d batimentos / %d pacientes | validacao: %d / %d",
        train_cfg.split_scheme,
        resumo["treino"]["n_batimentos"],
        resumo["treino"]["n_pacientes"],
        resumo["validacao"]["n_batimentos"],
        resumo["validacao"]["n_pacientes"],
    )

    ds_train = _tensores(series[train_idx], labels[train_idx])
    ds_val = _tensores(series[val_idx], labels[val_idx])

    gerador = torch.Generator().manual_seed(train_cfg.seed)
    loader_train = DataLoader(
        ds_train,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        generator=gerador,
        num_workers=train_cfg.num_workers,
    )
    loader_val = DataLoader(
        ds_val, batch_size=1024, shuffle=False, num_workers=train_cfg.num_workers
    )
    loader_train_eval = DataLoader(
        ds_train, batch_size=1024, shuffle=False, num_workers=train_cfg.num_workers
    )

    model = CNN1D(model_cfg).to(device)
    logger.info(
        "Modelo: %d parametros treinaveis | device=%s", model.n_parametros(), device
    )

    pos_weight = _pos_weight(labels[train_idx]) if train_cfg.balancear_classes else 1.0
    criterio = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=max(train_cfg.patience // 3, 1)
    )

    history: list[dict[str, float]] = []
    melhor_valor = -np.inf
    melhor_epoca = 0
    melhor_estado = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    epocas_sem_melhora = 0
    inicio = time.time()

    for epoca in range(1, train_cfg.epochs + 1):
        loss = _epoca_treino(model, loader_train, criterio, optimizer, device)
        y_val, s_val = predict_scores(model, loader_val, device)
        m_val = metrics.binary_metrics(y_val, s_val, threshold=config.DEFAULT_THRESHOLD)
        valor = float(m_val[train_cfg.criterio_selecao])
        scheduler.step(valor)

        history.append(
            {
                "epoca": epoca,
                "loss_treino": loss,
                "lr": optimizer.param_groups[0]["lr"],
                "val_acuracia": m_val["acuracia"],
                "val_acuracia_balanceada": m_val["acuracia_balanceada"],
                "val_sensibilidade": m_val["sensibilidade"],
                "val_especificidade": m_val["especificidade"],
                "val_f1": m_val["f1"],
                "val_auc_roc": m_val["auc_roc"],
            }
        )
        logger.info(
            "epoca %02d | loss %.4f | val acc %.4f | bal %.4f | sens %.4f | "
            "esp %.4f | auc %.4f",
            epoca,
            loss,
            m_val["acuracia"],
            m_val["acuracia_balanceada"],
            m_val["sensibilidade"],
            m_val["especificidade"],
            m_val["auc_roc"],
        )

        if valor > melhor_valor:
            melhor_valor = valor
            melhor_epoca = epoca
            melhor_estado = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epocas_sem_melhora = 0
        else:
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= train_cfg.patience:
                logger.info(
                    "Early stopping na epoca %d (melhor: epoca %d, %s=%.4f)",
                    epoca,
                    melhor_epoca,
                    train_cfg.criterio_selecao,
                    melhor_valor,
                )
                break

    duracao = time.time() - inicio

    # Metricas finais sempre com os pesos da MELHOR epoca, nao da ultima.
    model.load_state_dict(melhor_estado)
    model.to(device)

    y_train, s_train = predict_scores(model, loader_train_eval, device)
    y_val, s_val = predict_scores(model, loader_val, device)

    resultado_metricas = {
        "treino": metrics.binary_metrics(y_train, s_train, config.DEFAULT_THRESHOLD),
        "validacao": metrics.binary_metrics(y_val, s_val, config.DEFAULT_THRESHOLD),
        "threshold": config.DEFAULT_THRESHOLD,
        "criterio_selecao": train_cfg.criterio_selecao,
        "melhor_epoca": melhor_epoca,
        "epocas_executadas": len(history),
        "duracao_s": round(duracao, 1),
    }

    return {
        "model": model,
        "estado": melhor_estado,
        "model_cfg": model_cfg,
        "train_cfg": train_cfg,
        "history": history,
        "metrics": resultado_metricas,
        "split": resumo,
        "scores_validacao": {"y_true": y_val, "y_score": s_val},
    }


def save_run(
    resultado: dict[str, Any],
    out_dir: Path,
    dataset_metadata: dict | None = None,
    extras: dict | None = None,
) -> Path:
    """Grava checkpoint, metricas e historico do treino em ``out_dir``.

    O ``.pt`` e autocontido: carrega os pesos, os hiperparametros da
    arquitetura, o limiar escolhido na validacao e a divisao usada. E o que
    torna o modelo compartilhavel — quem receber o arquivo reconstroi a rede
    com :func:`load_checkpoint`, sem depender deste repositorio.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg: TrainConfig = resultado["train_cfg"]
    model_cfg: CNN1DConfig = resultado["model_cfg"]

    info = {
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etapa": "3 - modelagem preditiva (treino MIT-BIH)",
        "dataset_treino": "mit-bih-arrhythmia",
        "label_encoding": {"0": "nao-arritmia", "1": "arritmia"},
        "python": platform.python_version(),
        "torch": torch.__version__,
        "model_config": model_cfg.to_dict(),
        "train_config": train_cfg.to_dict(),
        "split": resultado["split"],
        "metrics": resultado["metrics"],
        "n_parametros": int(resultado["model"].n_parametros()),
    }
    if dataset_metadata is not None:
        info["dataset_metadata"] = dataset_metadata
    if extras:
        info.update(extras)

    torch.save(
        {
            "state_dict": resultado["estado"],
            "model_config": model_cfg.to_dict(),
            "threshold": config.DEFAULT_THRESHOLD,
            "info": info,
        },
        out_dir / CHECKPOINT_NAME,
    )

    (out_dir / METRICS_NAME).write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    history = resultado["history"]
    if history:
        colunas = list(history[0].keys())
        linhas = [",".join(colunas)]
        for linha in history:
            linhas.append(
                ",".join(
                    f"{linha[c]:.6f}" if isinstance(linha[c], float) else str(linha[c])
                    for c in colunas
                )
            )
        (out_dir / HISTORY_NAME).write_text("\n".join(linhas) + "\n", encoding="utf-8")

    return out_dir


def load_checkpoint(path: Path | str, device: str = "auto") -> tuple[CNN1D, dict]:
    """Reconstroi o modelo treinado a partir do ``.pt`` (usado na Etapa 4).

    Returns:
        Tupla ``(modelo_em_modo_eval, checkpoint)``.
    """
    dev = resolve_device(device)
    ckpt = torch.load(Path(path), map_location=dev, weights_only=False)
    model_cfg = CNN1DConfig(**ckpt["model_config"])
    model = CNN1D(model_cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()
    return model, ckpt


def evaluate_arrays(
    model: nn.Module,
    series: np.ndarray,
    labels: np.ndarray,
    threshold: float = config.DEFAULT_THRESHOLD,
    device: str = "auto",
    batch_size: int = 1024,
) -> tuple[dict[str, float | int], np.ndarray]:
    """Avalia o modelo sobre arrays soltos (ex.: o LUDB).

    O limiar padrao e ``config.DEFAULT_THRESHOLD`` (fixo, nao otimizado) — o
    parametro so existe para reuso interno do modulo, nao para escolha por
    conjunto.

    Returns:
        Tupla ``(metricas, probabilidades)``.
    """
    dev = resolve_device(device)
    model.to(dev).eval()
    loader = DataLoader(_tensores(series, labels), batch_size=batch_size, shuffle=False)
    y_true, y_score = predict_scores(model, loader, dev)
    return metrics.binary_metrics(y_true, y_score, threshold), y_score
