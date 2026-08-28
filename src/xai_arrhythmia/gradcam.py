"""Grad-CAM 1D para a CNN do projeto (Etapa 4).

Extrai o mapa de ativação da última camada convolucional (``modelo.last_conv``,
ver :class:`xai_arrhythmia.models.CNN1D`) via hooks de forward/backward. A
mesma função atende um único batimento (usada interativamente, ex.: no
notebook de demonstração) ou o dataset inteiro em lote (usada por
``scripts/extract_gradcam.py``).

Nota de correção do processamento em lote: o modelo é sempre avaliado em modo
``eval()``. Nesse modo, BatchNorm usa estatísticas fixas (não as do lote) e
Dropout vira identidade — ou seja, a saída de cada amostra do lote não depende
de nenhuma outra amostra do mesmo lote. Por isso, somar os logits do lote e
chamar um único ``.backward()`` produz, para cada amostra, exatamente o mesmo
gradiente que um ``.backward()`` individual daquela amostra — e o Grad-CAM de
um lote inteiro sai de uma única passada, sem laço em Python por batimento.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def grad_cam_1d(
    modelo: nn.Module,
    batimentos: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Grad-CAM 1D de um ou mais batimentos.

    Args:
        modelo: rede já treinada (ver ``training.load_checkpoint``); é
            colocada em modo ``eval()`` internamente.
        batimentos: ``(tamanho_janela,)`` para um único batimento, ou
            ``(n, tamanho_janela)`` para vários.
        batch_size: batimentos processados por passada. Só afeta o pico de
            memória, não o resultado (ver nota de correção no docstring do
            módulo).

    Returns:
        Mapa de ativação normalizado em ``[0, 1]``: ``(L',)`` se a entrada foi
        um único batimento, ``(n, L')`` caso contrário. ``L'`` é o comprimento
        de saída de ``modelo.last_conv`` (29 na arquitetura padrão do
        projeto, para uma janela de 234 amostras).
    """
    entrada_unica = batimentos.ndim == 1
    lote_completo = np.atleast_2d(batimentos)

    modelo.eval()
    device = next(modelo.parameters()).device

    mapas = [
        _grad_cam_1d_lote(modelo, lote_completo[inicio : inicio + batch_size], device)
        for inicio in range(0, len(lote_completo), batch_size)
    ]
    cam = np.concatenate(mapas, axis=0)
    return cam[0] if entrada_unica else cam


def _grad_cam_1d_lote(
    modelo: nn.Module,
    lote: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    ativacao: dict[str, torch.Tensor] = {}
    gradiente: dict[str, torch.Tensor] = {}

    def hook_forward(_modulo, _entrada, saida):
        ativacao["valor"] = saida

    def hook_backward(_modulo, _grad_entrada, grad_saida):
        gradiente["valor"] = grad_saida[0]

    h1 = modelo.last_conv.register_forward_hook(hook_forward)
    h2 = modelo.last_conv.register_full_backward_hook(hook_backward)

    x = torch.tensor(lote, dtype=torch.float32, device=device).unsqueeze(1)
    x.requires_grad_(True)

    logits = modelo(x)  # (B,) — único logit por amostra (classe "arritmia")
    modelo.zero_grad()
    logits.sum().backward()  # ver nota de correção no docstring do módulo

    h1.remove()
    h2.remove()

    ativ = ativacao["valor"]  # (B, canais, L')
    grad = gradiente["valor"]  # (B, canais, L')
    pesos = grad.mean(dim=2, keepdim=True)  # media temporal -> peso por canal
    cam = torch.relu((pesos * ativ).sum(dim=1))  # (B, L')

    cam = cam.detach().cpu().numpy()
    maximos = cam.max(axis=1, keepdims=True)
    maximos[maximos == 0] = 1.0  # batimento sem ativação nenhuma: mapa fica todo zero
    return cam / maximos


def interpola_cam(cam: np.ndarray, tamanho_janela: int) -> np.ndarray:
    """Reamostra o(s) mapa(s) do Grad-CAM para ``tamanho_janela`` amostras.

    Interpolação linear simples sobre o eixo normalizado ``[0, 1]``. Alinhar
    pelos centros exatos do campo receptivo de cada posição é um refinamento
    possível, mas dado o tamanho pequeno do batimento (234 amostras, ~650 ms)
    essa aproximação já é suficiente para a métrica de fidelidade da Etapa 4.

    Args:
        cam: ``(L',)`` ou ``(n, L')``.
        tamanho_janela: tamanho de saída (234 na arquitetura padrão).

    Returns:
        Mesma forma de entrada, com o último eixo reamostrado para
        ``tamanho_janela``.
    """
    entrada_unica = cam.ndim == 1
    matriz = np.atleast_2d(cam)

    posicoes_cam = np.linspace(0, 1, num=matriz.shape[1])
    posicoes_janela = np.linspace(0, 1, num=tamanho_janela)
    interpolado = np.stack(
        [np.interp(posicoes_janela, posicoes_cam, linha) for linha in matriz]
    )
    return interpolado[0] if entrada_unica else interpolado
