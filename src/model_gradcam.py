"""CNN 1D + Grad-CAM 1D.

Casa única da arquitetura e do Grad-CAM usados pelos scripts e notebooks de
CNN/Grad-CAM (`train_cnn.py`, `extract_gradcam.py`, `test_ludb.py`,
`notebooks/03`, `04`, `05`). O pré-processamento não passa por aqui: `series`,
`labels`, `pacientes` e as máscaras clínicas continuam vindo de
`beat_io.load_beat_dataset`.

- `CNN1D_Arritmia`: dois blocos `Conv1d + MaxPool1d` (16 e 32 canais, kernels
  7 e 5) e duas camadas lineares terminando em **sigmoide** (probabilidade de
  arritmia). O hook de Grad-CAM é registrado dentro do próprio `forward`, sobre
  as ativações de `conv2` (a última camada convolucional).
- Treino: `binary_cross_entropy` sobre a sigmoide, Adam, número fixo de épocas,
  sem reponderação de classe e sem early stopping. O checkpoint `model.pt` é o
  da época com maior acurácia balanceada na validação MIT-BIH.
- Grad-CAM: retropropagação a partir de `prob.sum()`, média temporal do
  gradiente por canal, `ReLU(Σ w·ativação)`, e `F.interpolate` linear já
  levando o mapa de volta às 234 amostras da janela.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, recall_score

#: Arquivos gravados no diretório da execução.
CHECKPOINT_NAME = "model.pt"  # melhor época (val. acurácia balanceada), usado no LUDB
LAST_CHECKPOINT_NAME = "model_last.pt"  # última época, apenas rastreabilidade
HISTORY_NAME = "history.csv"
METRICS_NAME = "metrics.json"

#: Limiar de decisão fixo — igual em treino, validação e teste no LUDB.
DEFAULT_THRESHOLD = 0.5


def resolve_device(device: str = "auto") -> torch.device:
    """`'auto'` escolhe a GPU quando disponível; senão respeita o pedido."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


class CNN1D_Arritmia(nn.Module):
    """CNN 1D binária: saída sigmoide, hook de Grad-CAM interno em `conv2`."""

    def __init__(self, input_length: int = 234) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        pooled_length = (input_length // 2) // 2
        self.fc1 = nn.Linear(32 * pooled_length, 64)
        self.fc2 = nn.Linear(64, 1)

        self.input_length = input_length
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None

    def activations_hook(self, grad: torch.Tensor) -> None:
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        # Registra o hook e guarda a ativação da última conv para o Grad-CAM.
        if x.requires_grad:
            x.register_hook(self.activations_hook)
        self.activations = x

        x = self.pool2(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

    def get_activations_gradient(self) -> torch.Tensor | None:
        return self.gradients

    def get_activations(self) -> torch.Tensor | None:
        return self.activations

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def run_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> tuple[float, float, float, float]:
    """Uma passada pelo `loader`; treina se `optimizer` for dado.

    Returns:
        `(perda_media, acuracia, sensibilidade, especificidade)` — a classe
        positiva é a 1 (arritmia).
    """
    device = device or next(model.parameters()).device
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, n = 0.0, 0
    all_probs, all_true = [], []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        with torch.set_grad_enabled(is_train):
            probs = model(xb)
            loss = F.binary_cross_entropy(probs, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * len(yb)
        n += len(yb)
        all_probs.append(probs.detach().cpu())
        all_true.append(yb.detach().cpu())

    preds = (torch.cat(all_probs) >= DEFAULT_THRESHOLD).float().numpy()
    true = torch.cat(all_true).numpy()

    acc = accuracy_score(true, preds)
    sensitivity = recall_score(true, preds, pos_label=1, zero_division=0)
    specificity = recall_score(true, preds, pos_label=0, zero_division=0)
    return total_loss / max(n, 1), acc, sensitivity, specificity


def grad_cam_1d(
    model: nn.Module,
    x: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grad-CAM 1D de `conv2` para um lote de batimentos.

    Args:
        model: rede já treinada.
        x: tensor `(B, 1, L)`.

    Returns:
        `(cam, prob)` — `cam` `(B, L)` já reamostrado para o comprimento da
        janela e normalizado em `[0, 1]` pelo máximo de cada batimento; `prob`
        `(B, 1)` a probabilidade de arritmia.
    """
    device = device or next(model.parameters()).device
    model.eval()
    x = x.clone().to(device).requires_grad_(True)
    prob = model(x)

    model.zero_grad()
    prob.sum().backward()

    gradients = model.get_activations_gradient()
    activations = model.get_activations()

    weights = gradients.mean(dim=2, keepdim=True)          # peso por canal
    cam = F.relu((weights * activations).sum(dim=1))       # (B, L')
    cam = F.interpolate(cam.unsqueeze(1), size=x.shape[2], mode="linear", align_corners=False)
    cam = cam.squeeze(1)
    cam = cam / (cam.amax(dim=1, keepdim=True) + 1e-8)
    return cam.detach().cpu(), prob.detach().cpu()


@torch.no_grad()
def predict_scores(
    model: nn.Module,
    series: np.ndarray,
    batch_size: int = 256,
    device: str | torch.device = "auto",
) -> np.ndarray:
    """Probabilidade de arritmia para cada batimento de `series` `(n, L)`.

    O modelo já devolve a sigmoide, então o score é a saída direta — sem
    aplicar sigmoide de novo.
    """
    dev = resolve_device(device) if isinstance(device, str) else device
    model.to(dev).eval()
    series = np.asarray(series, dtype=np.float32)

    scores: list[np.ndarray] = []
    for inicio in range(0, len(series), batch_size):
        lote = torch.from_numpy(series[inicio : inicio + batch_size]).unsqueeze(1).to(dev)
        scores.append(model(lote).squeeze(1).cpu().numpy())
    return np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)


def save_checkpoint(
    path: Path | str,
    model: CNN1D_Arritmia,
    info: dict[str, Any] | None = None,
) -> Path:
    """Grava um `.pt` autocontido: pesos, `input_length` e metadados.

    Guardar `input_length` é o que permite `load_checkpoint` reconstruir a
    rede sem conhecer o código de treino.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "arquitetura": "CNN1D_Arritmia",
            "state_dict": model.state_dict(),
            "input_length": model.input_length,
            "threshold": DEFAULT_THRESHOLD,
            "info": info or {},
        },
        path,
    )
    return path


def load_checkpoint(
    path: Path | str,
    device: str | torch.device = "auto",
    expected_policy: str | None = None,
) -> tuple[CNN1D_Arritmia, dict]:
    """Reconstrói a `CNN1D_Arritmia` treinada a partir do `.pt`.

    Returns:
        `(modelo_em_eval, checkpoint)`.
    """
    dev = resolve_device(device) if isinstance(device, str) else device
    ckpt = torch.load(Path(path), map_location=dev, weights_only=False)
    if expected_policy is not None:
        import classification

        classification.require_policy(ckpt.get("info", {}), expected_policy, str(path))
    model = CNN1D_Arritmia(input_length=int(ckpt.get("input_length", 234)))
    model.load_state_dict(ckpt["state_dict"])
    model.to(dev).eval()
    return model, ckpt
