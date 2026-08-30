"""CLI de treino da CNN 1D binária no MIT-BIH (Etapa 3).

A arquitetura e o loop de treino ficam em `scripts/model_gradcam.py`. O
pré-processamento não muda: o dataset vem de `scripts/preprocess_mitbih.py`
via `xai_arrhythmia.io`.

Treino e teste são scripts separados: aqui só se enxerga o MIT-BIH; o teste no
LUDB (`scripts/test_ludb.py`) apenas carrega o checkpoint gravado aqui.

Divisão inter-paciente: `StratifiedGroupKFold` agrupa por `pacientes` (nenhum
paciente aparece em treino e validação) e estratifica por `labels`. O fold 0 é
usado como partição treino/validação.

Exemplos:
    python scripts/train_cnn.py
    python scripts/train_cnn.py --epochs 30 --fold 1 --seed 7
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

# Permite executar o script sem instalar o pacote e importar o módulo irmão.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import model_gradcam as mg  # noqa: E402
from xai_arrhythmia import config, io  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina a CNN 1D binária no MIT-BIH com split inter-paciente.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=config.MITBIH_PROCESSED_FILE,
        help="Dataset .npz gerado por scripts/preprocess_mitbih.py.",
    )
    parser.add_argument(
        "--run-name",
        default="cnn1d_mitbih",
        help="Nome do diretório da execução dentro de --models-dir.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=config.MODELS_DIR,
        help="Diretório raiz onde as execuções são gravadas.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--n-splits", type=int, default=5,
        help="Número de folds do StratifiedGroupKFold.",
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Fold usado como partição treino/validação.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="'auto', 'cuda' ou 'cpu'.")
    return parser.parse_args()


def _split_inter_paciente(series, labels, pacientes, n_splits, fold, seed):
    """`train_idx, val_idx` do fold pedido do StratifiedGroupKFold."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(series, labels, groups=pacientes))
    train_idx, val_idx = folds[fold]

    vazamento = set(pacientes[train_idx]) & set(pacientes[val_idx])
    if vazamento:
        raise AssertionError(f"Vazamento inter-paciente: {sorted(vazamento)}")
    return train_idx, val_idx


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.dataset.exists():
        print(
            f"Dataset nao encontrado: {args.dataset}\n"
            f"Gere-o antes com: python scripts/preprocess_mitbih.py --download",
            file=sys.stderr,
        )
        return 1

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = mg.resolve_device(args.device)

    dados = io.load_beat_dataset(args.dataset)
    series = np.asarray(dados["series"], dtype=np.float32)
    labels = np.asarray(dados["labels"]).astype(int)
    pacientes = np.asarray(dados["pacientes"])
    print(
        f"Dataset: {args.dataset} | {series.shape[0]} batimentos "
        f"de {np.unique(pacientes).size} registros | device={device}"
    )

    train_idx, val_idx = _split_inter_paciente(
        series, labels, pacientes, args.n_splits, args.fold, args.seed
    )

    X = torch.tensor(series, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    train_loader = DataLoader(
        TensorDataset(X[train_idx], y[train_idx]), batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(TensorDataset(X[val_idx], y[val_idx]), batch_size=256)

    model = mg.CNN1D_Arritmia(input_length=series.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    logging.info("Modelo: %d parâmetros treináveis", model.n_parametros())

    history: list[dict] = []
    for epoca in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = mg.run_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc, va_sens, va_spec = mg.run_epoch(model, val_loader, None, device)
        history.append(
            {
                "epoca": epoca,
                "loss_treino": tr_loss,
                "acc_treino": tr_acc,
                "val_loss": va_loss,
                "val_acuracia": va_acc,
                "val_sensibilidade": va_sens,
                "val_especificidade": va_spec,
                "val_acuracia_balanceada": (va_sens + va_spec) / 2,
            }
        )
        logging.info(
            "epoca %02d | loss %.4f acc %.3f | val_loss %.4f acc %.3f sens %.3f spec %.3f",
            epoca, tr_loss, tr_acc, va_loss, va_acc, va_sens, va_spec,
        )

    # Métricas finais = pesos da última época (não há seleção de época).
    tr = history[-1]
    metricas = {
        "treino": {
            "acuracia": tr["acc_treino"], "loss": tr["loss_treino"],
        },
        "validacao": {
            "acuracia": tr["val_acuracia"],
            "acuracia_balanceada": tr["val_acuracia_balanceada"],
            "sensibilidade": tr["val_sensibilidade"],
            "especificidade": tr["val_especificidade"],
            "loss": tr["val_loss"],
        },
        "epocas_executadas": len(history),
        "melhor_epoca": len(history),  # sempre a última: sem seleção de época
    }

    print("\n" + "=" * 60)
    print("RESULTADO DO TREINO (Etapa 3)")
    print("=" * 60)
    print(f"Épocas executadas : {len(history)}")
    for nome, m in metricas.items():
        if isinstance(m, dict):
            print(f"[{nome}] " + "  ".join(f"{k}={v:.4f}" for k, v in m.items()))
    print("=" * 60)

    out_dir = Path(args.models_dir) / args.run_name
    info = {
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etapa": "3 - modelagem preditiva (treino MIT-BIH)",
        "dataset_treino": "mit-bih-arrhythmia",
        "label_encoding": {"0": "nao-arritmia", "1": "arritmia"},
        "arquitetura": "CNN1D_Arritmia",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "n_parametros": int(model.n_parametros()),
        "train_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "split_scheme": f"StratifiedGroupKFold(n_splits={args.n_splits}) fold {args.fold}",
            "seed": args.seed,
        },
        "split": {
            nome: {
                "n_batimentos": int(len(idx)),
                "n_pacientes": int(np.unique(pacientes[idx]).size),
                "pacientes": sorted({str(p) for p in pacientes[idx]}),
                "prevalencia_classe_1": round(float((labels[idx] == 1).mean()), 4),
            }
            for nome, idx in (("treino", train_idx), ("validacao", val_idx))
        },
        "val_idx": val_idx.tolist(),
        "metrics": metricas,
        "dataset_metadata": dados.get("metadata"),
    }

    ckpt_path = mg.save_checkpoint(out_dir / mg.CHECKPOINT_NAME, model, info)
    # metrics.json é a versão legível: fora o val_idx (fica só no .pt, é o que
    # os notebooks recarregam) e a lista de pacientes por partição.
    info_legivel = {k: v for k, v in info.items() if k != "val_idx"}
    info_legivel["split"] = {
        nome: {k: v for k, v in bloco.items() if k != "pacientes"}
        for nome, bloco in info["split"].items()
    }
    (out_dir / mg.METRICS_NAME).write_text(
        json.dumps(info_legivel, ensure_ascii=False, indent=2, default=float),
        encoding="utf-8",
    )

    colunas = list(history[0].keys())
    linhas = [",".join(colunas)] + [
        ",".join(
            f"{linha[c]:.6f}" if isinstance(linha[c], float) else str(linha[c])
            for c in colunas
        )
        for linha in history
    ]
    (out_dir / mg.HISTORY_NAME).write_text("\n".join(linhas) + "\n", encoding="utf-8")

    print(f"\nCheckpoint : {ckpt_path}")
    print(f"Métricas   : {out_dir / mg.METRICS_NAME}")
    print(f"Histórico  : {out_dir / mg.HISTORY_NAME}")
    print(
        "\nPróximo passo (teste independente):\n"
        f"  python scripts/test_ludb.py --checkpoint {ckpt_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
