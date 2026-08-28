"""Arquitetura CNN 1D binaria (Etapa 3).

O desenho da rede e condicionado pela Etapa 4: o Grad-CAM sera extraido da
ultima camada convolucional, entao a arquitetura precisa preservar o
alinhamento temporal entre o mapa de ativacao e o batimento de entrada. Duas
decisoes garantem isso:

1. ``padding = kernel_size // 2`` em todas as convolucoes ("same"), para que o
   passo ``i`` do mapa continue correspondendo a mesma regiao do sinal e nao
   haja deslocamento acumulado a cada bloco.
2. Global Average Pooling seguido de UMA camada linear. Nesse formato o peso
   da classe multiplica diretamente o canal convolucional, que e exatamente a
   formulacao do CAM/Grad-CAM — qualquer MLP densa no topo tornaria a
   atribuicao menos direta.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Sequence

import torch
from torch import nn


@dataclass
class CNN1DConfig:
    """Hiperparametros da arquitetura, gravados junto ao checkpoint.

    Guardar isto no ``.pt`` e o que permite a Etapa 4 (e qualquer pessoa que
    receba o modelo) reconstruir a rede sem precisar adivinhar a configuracao.
    """

    input_size: int = 234
    channels: Sequence[int] = field(default_factory=lambda: [32, 64, 128, 128])
    kernel_size: int = 7
    dropout: float = 0.3
    dropout_conv: float = 0.1

    def to_dict(self) -> dict:
        d = asdict(self)
        d["channels"] = list(self.channels)
        return d


class ConvBlock(nn.Module):
    """Conv1d -> BatchNorm -> ReLU (-> MaxPool -> Dropout)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,  # redundante com o beta do BatchNorm
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(2) if pool else nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.pool(self.act(self.bn(self.conv(x)))))


class CNN1D(nn.Module):
    """CNN 1D para classificacao binaria de batimentos (1 logit de saida).

    Com a configuracao padrao e janela de 234 amostras, o empilhamento aplica
    pooling nos tres primeiros blocos (234 -> 117 -> 58 -> 29) e mantem a
    resolucao no ultimo, de modo que o mapa do Grad-CAM tem 29 passos — cerca
    de 8 amostras (~22 ms @360 Hz) cada, resolucao suficiente para separar P,
    QRS e T dentro do batimento.
    """

    def __init__(self, cfg: CNN1DConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or CNN1DConfig()

        blocos: list[nn.Module] = []
        in_ch = 1
        n_blocos = len(self.cfg.channels)
        for i, out_ch in enumerate(self.cfg.channels):
            blocos.append(
                ConvBlock(
                    in_ch,
                    out_ch,
                    kernel_size=self.cfg.kernel_size,
                    pool=i < n_blocos - 1,
                    dropout=self.cfg.dropout_conv,
                )
            )
            in_ch = out_ch

        self.features = nn.Sequential(*blocos)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(self.cfg.dropout)
        self.classifier = nn.Linear(in_ch, 1)

    @property
    def last_conv(self) -> nn.Conv1d:
        """Camada alvo do Grad-CAM (Etapa 4)."""
        return self.features[-1].conv

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Mapa de ativacao da ultima camada convolucional ``(B, C, L')``."""
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: batimentos ``(B, 234)`` ou ``(B, 1, 234)``.

        Returns:
            Logits ``(B,)``. A probabilidade da classe 1 e ``sigmoid(logit)``;
            a perda usada no treino (``BCEWithLogitsLoss``) recebe o logit.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        z = self.forward_features(x)
        z = self.gap(z).flatten(1)
        return self.classifier(self.dropout(z)).squeeze(1)

    def n_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: CNN1DConfig | None = None) -> CNN1D:
    """Instancia a rede a partir da configuracao."""
    return CNN1D(cfg)
