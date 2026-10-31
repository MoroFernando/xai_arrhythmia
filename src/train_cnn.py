"""CLI de treino da CNN 1D binária no MIT-BIH (Etapa 3).

A arquitetura e o loop de treino ficam em `src/model_gradcam.py`. O
pré-processamento não muda: o dataset vem de `src/preprocess_mitbih.py`
via `beat_io`.

Treino e teste são scripts separados: aqui só se enxerga o MIT-BIH; o teste no
LUDB (`src/test_ludb.py`) apenas carrega o checkpoint gravado aqui.

Divisão inter-paciente: `StratifiedGroupKFold` agrupa por `subject_ids` (201 e
202 pertencem ao mesmo sujeito) e estratifica por `labels`. O manifesto permite
reutilizar a mesma atribuição de sujeitos nos dois modos.

Seleção do checkpoint: a cada época a validação MIT-BIH é avaliada no limiar
fixo; quando a acurácia balanceada supera estritamente a melhor anterior, os
pesos daquela época passam a ser o melhor modelo. `model.pt` guarda o melhor
modelo (consumido por teste LUDB, Grad-CAM e IoU) e `model_last.pt` a última
época, apenas para rastreabilidade. O LUDB nunca participa dessa escolha.

Exemplos:
    python src/train_cnn.py
    python src/train_cnn.py --epochs 30 --fold 1 --seed 7
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, TensorDataset

# Permite executar o script de qualquer diretório e importar os módulos irmãos.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import beat_io  # noqa: E402
import config  # noqa: E402
import classification  # noqa: E402
import model_gradcam as mg  # noqa: E402
import test_ludb as metrics  # noqa: E402  (apenas funcoes de metrica; nao le o LUDB)


def _avaliar(model, series, labels, simbolos, idx, device) -> dict:
    """Metricas completas no limiar fixo e desempenho por simbolo MIT-BIH."""
    scores = mg.predict_scores(model, series[idx], device=device)
    return {
        **metrics.binary_metrics(labels[idx], scores, threshold=mg.DEFAULT_THRESHOLD),
        "por_simbolo": metrics.por_grupo(
            labels[idx], scores, simbolos[idx], threshold=mg.DEFAULT_THRESHOLD
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina a CNN 1D binária no MIT-BIH com split inter-paciente.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset .npz gerado por src/preprocess_mitbih.py.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
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
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument(
        "--split-manifest", type=Path, default=None,
        help=("Manifesto JSON de atribuicao por sujeito. Se ausente, usa/cria "
              "um manifesto comum aos dois modos para n-splits/fold/seed."),
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cuda' ou 'cpu'.")
    classification.add_cli_argument(parser)
    return classification.prepare_args(parser.parse_args(), "train")


def _split_inter_paciente(series, labels, subject_ids, n_splits, fold, seed):
    """`train_idx, val_idx` agrupado por sujeito clinico."""
    if n_splits < 2 or fold < 0 or fold >= n_splits:
        raise classification.ClassificationError(
            f"Split invalido: n_splits={n_splits}, fold={fold}."
        )
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = list(sgkf.split(series, labels, groups=subject_ids))
    train_idx, val_idx = folds[fold]

    vazamento = set(subject_ids[train_idx]) & set(subject_ids[val_idx])
    if vazamento:
        raise AssertionError(f"Vazamento inter-paciente: {sorted(vazamento)}")
    return train_idx, val_idx


def _split_from_manifest_or_create(
    path, series, labels, records, subject_ids, n_splits, fold, seed, policy
):
    """Reutiliza atribuicoes por sujeito ou cria um manifesto versionado."""
    if n_splits < 2 or fold < 0 or fold >= n_splits:
        raise classification.ClassificationError(
            f"Split invalido: n_splits={n_splits}, fold={fold}."
        )
    path = Path(path)
    if path.exists():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if (manifest.get("schema_version") != 1
                or manifest.get("dataset") != "mit-bih-arrhythmia"
                or manifest.get("dataset_version") != config.MITBIH_VERSION):
            raise classification.ClassificationError(
                f"Manifesto {path} tem schema/dataset/version incompativel."
            )
        if policy not in manifest.get("compatible_classification_policies", []):
            raise classification.ClassificationError(
                f"Manifesto {path} nao declara compatibilidade com {policy}."
            )
        for key, expected in {"n_splits": n_splits, "fold": fold, "seed": seed}.items():
            if manifest.get(key) != expected:
                raise classification.ClassificationError(
                    f"Manifesto {path}: {key}={manifest.get(key)!r}; esperado {expected!r}."
                )
        assignments = manifest.get("subject_partition", {})
        missing = set(map(str, np.unique(subject_ids))) - assignments.keys()
        if missing:
            raise classification.ClassificationError(
                f"Manifesto {path} sem sujeitos do dataset: {sorted(missing)}."
            )
        invalid_partitions = set(assignments.values()) - {"train", "validation"}
        if invalid_partitions:
            raise classification.ClassificationError(
                f"Manifesto {path} contem particoes invalidas: {sorted(invalid_partitions)}."
            )
        expected_record_subject = {
            str(record): str(subject_ids[np.flatnonzero(records == record)[0]])
            for record in np.unique(records)
        }
        if manifest.get("record_subject") != expected_record_subject:
            raise classification.ClassificationError(
                f"Manifesto {path} nao corresponde ao mapa record/sujeito do dataset."
            )
        train_idx = np.flatnonzero([assignments[str(s)] == "train" for s in subject_ids])
        val_idx = np.flatnonzero([assignments[str(s)] == "validation" for s in subject_ids])
    else:
        train_idx, val_idx = _split_inter_paciente(
            series, labels, subject_ids, n_splits, fold, seed
        )
        subject_partition = {
            str(subject): ("validation" if subject in set(subject_ids[val_idx]) else "train")
            for subject in np.unique(subject_ids)
        }
        record_subject = {
            str(record): str(subject_ids[np.flatnonzero(records == record)[0]])
            for record in np.unique(records)
        }
        manifest = {
            "schema_version": 1,
            "dataset": "mit-bih-arrhythmia",
            "dataset_version": config.MITBIH_VERSION,
            "created_with_classification_policy": policy,
            "compatible_classification_policies": sorted(classification.POLICIES),
            "splitter": "StratifiedGroupKFold",
            "n_splits": n_splits,
            "fold": fold,
            "seed": seed,
            "subject_partition": subject_partition,
            "record_subject": record_subject,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if not len(train_idx) or not len(val_idx):
        raise classification.ClassificationError("Manifesto produziu particao vazia.")
    if set(subject_ids[train_idx]) & set(subject_ids[val_idx]):
        raise classification.ClassificationError("Manifesto causa vazamento entre sujeitos.")
    for partition, indices in (("train", train_idx), ("validation", val_idx)):
        if np.unique(labels[indices]).size != 2:
            raise classification.ClassificationError(
                f"Particao {partition} nao contem as duas classes na politica {policy}."
            )
    identity = classification.stable_json_sha256(manifest)
    return train_idx, val_idx, manifest, identity


def main() -> int:
    args = parse_args()
    out_dir = Path(args.models_dir) / args.run_name
    classification.require_new_output(out_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.dataset.exists():
        print(
            f"Dataset nao encontrado: {args.dataset}\n"
            f"Gere-o antes com: python src/preprocess_mitbih.py --download",
            file=sys.stderr,
        )
        return 1

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Reprodutibilidade em GPU: kernels cuDNN deterministicos, sem autotuning.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = mg.resolve_device(args.device)

    dados = beat_io.load_beat_dataset(args.dataset)
    classification.require_policy(dados.get("metadata", {}), args.classification_policy, str(args.dataset))
    dataset_sha256 = classification.file_sha256(args.dataset)
    series = np.asarray(dados["series"], dtype=np.float32)
    labels = np.asarray(dados["labels"]).astype(int)
    simbolos = np.asarray(dados["simbolos"])
    pacientes = np.asarray(dados["pacientes"])
    if "subject_ids" not in dados:
        raise classification.ClassificationError(
            "Dataset MIT-BIH sem subject_ids; regenere-o com a politica nova."
        )
    subject_ids = np.asarray(dados["subject_ids"])
    print(
        f"Dataset: {args.dataset} | {series.shape[0]} batimentos "
        f"de {np.unique(pacientes).size} registros | device={device}"
    )

    manifest_path = args.split_manifest or (
        config.SPLITS_DIR / f"mitbih-subject-n{args.n_splits}-fold{args.fold}-seed{args.seed}.json"
    )
    train_idx, val_idx, split_manifest, split_manifest_sha256 = _split_from_manifest_or_create(
        manifest_path, series, labels, pacientes, subject_ids,
        args.n_splits, args.fold, args.seed, args.classification_policy,
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
    best_ba, best_epoch, best_state = float("-inf"), None, None

    for epoca in range(1, args.epochs + 1):
        tr_loss, tr_acc, _, _ = mg.run_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc, va_sens, va_spec = mg.run_epoch(model, val_loader, None, device)
        va_ba = (va_sens + va_spec) / 2
        # Selecao exclusivamente pela validacao MIT-BIH; empate mantem a anterior.
        melhorou = va_ba > best_ba
        if melhorou:
            best_ba, best_epoch = va_ba, epoca
            best_state = copy.deepcopy(model.state_dict())
        history.append(
            {
                **classification.artifact_identity(args.classification_policy),
                "epoca": epoca,
                "loss_treino": tr_loss,
                "acc_treino": tr_acc,
                "val_loss": va_loss,
                "val_acuracia": va_acc,
                "val_sensibilidade": va_sens,
                "val_especificidade": va_spec,
                "val_acuracia_balanceada": va_ba,
                "melhor_ate_agora": int(melhorou),
            }
        )
        logging.info(
            "epoca %02d | loss %.4f acc %.3f | val_loss %.4f acc %.3f sens %.3f spec %.3f ba %.4f%s",
            epoca, tr_loss, tr_acc, va_loss, va_acc, va_sens, va_spec, va_ba,
            "  <- melhor" if melhorou else "",
        )

    # Avaliacao final em modo eval: ultima epoca (rastreabilidade) e melhor epoca.
    last_epoch = len(history)
    last_state = copy.deepcopy(model.state_dict())
    m_val_last = _avaliar(model, series, labels, simbolos, val_idx, device)
    model.load_state_dict(best_state)
    m_val_best = _avaliar(model, series, labels, simbolos, val_idx, device)
    m_treino_best = _avaliar(model, series, labels, simbolos, train_idx, device)

    selecao = {
        "criterio": (
            "maior acuracia balanceada na validacao MIT-BIH (limiar fixo "
            f"{mg.DEFAULT_THRESHOLD}); empate mantem a epoca anterior"
        ),
        "best_epoch": best_epoch,
        "best_validation_balanced_accuracy": best_ba,
        "last_epoch": last_epoch,
        "last_validation_balanced_accuracy": history[-1]["val_acuracia_balanceada"],
        "checkpoint_usado_nas_etapas_seguintes": f"{mg.CHECKPOINT_NAME} (best)",
        "checkpoint_ultima_epoca": f"{mg.LAST_CHECKPOINT_NAME} (apenas rastreabilidade)",
        "ludb_usado_na_selecao": False,
    }
    metricas = {
        "treino": {**m_treino_best, "loss_media_epoca": history[best_epoch - 1]["loss_treino"]},
        "validacao": {**m_val_best, "loss": history[best_epoch - 1]["val_loss"]},
        "validacao_ultima_epoca": {**m_val_last, "loss": history[-1]["val_loss"]},
        "selecao_checkpoint": selecao,
        "epocas_executadas": last_epoch,
        "melhor_epoca": best_epoch,
    }

    print("\n" + "=" * 60)
    print("RESULTADO DO TREINO (Etapa 3)")
    print("=" * 60)
    print(f"Épocas executadas : {last_epoch}")
    print(f"Melhor época      : {best_epoch} (val. acurácia balanceada {best_ba:.4f})")
    print(f"Última época      : {last_epoch} (val. acurácia balanceada "
          f"{selecao['last_validation_balanced_accuracy']:.4f})")
    print(metrics.formata_metricas("treino | best", m_treino_best))
    print(metrics.formata_metricas("validacao | best", m_val_best))
    print(metrics.formata_metricas("validacao | ultima epoca", m_val_last))
    print(metrics.formata_por_grupo("validacao | best | por simbolo", m_val_best["por_simbolo"]))
    print("=" * 60)

    out_dir = Path(args.models_dir) / args.run_name
    info = {
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etapa": "3 - modelagem preditiva (treino MIT-BIH)",
        "dataset_treino": "mit-bih-arrhythmia",
        **classification.metadata(args.classification_policy),
        "dataset_sha256": dataset_sha256,
        "dataset_path": config.caminho_relativo(args.dataset),
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
            "split_manifest": config.caminho_relativo(manifest_path),
            "split_manifest_sha256": split_manifest_sha256,
            "threshold": mg.DEFAULT_THRESHOLD,
            "selecao_checkpoint": "val_acuracia_balanceada (MIT-BIH)",
            "cudnn_deterministic": True,
            "device": str(device),
        },
        "checkpoint_role": "best",
        "checkpoint_epoch": best_epoch,
        "split": {
            nome: {
                "n_batimentos": int(len(idx)),
                "n_registros": int(np.unique(pacientes[idx]).size),
                "n_sujeitos": int(np.unique(subject_ids[idx]).size),
                "classe_0": int((labels[idx] == 0).sum()),
                "classe_1": int((labels[idx] == 1).sum()),
                "pacientes": sorted({str(p) for p in pacientes[idx]}),
                "sujeitos": sorted({str(p) for p in subject_ids[idx]}),
                "prevalencia_classe_1": round(float((labels[idx] == 1).mean()), 4),
            }
            for nome, idx in (("treino", train_idx), ("validacao", val_idx))
        },
        "val_idx": val_idx.tolist(),
        "split_manifest": split_manifest,
        "split_manifest_sha256": split_manifest_sha256,
        "metrics": metricas,
        "dataset_metadata": dados.get("metadata"),
    }

    # model.pt recebe os pesos da melhor epoca (ja carregados em `model`).
    ckpt_path = mg.save_checkpoint(out_dir / mg.CHECKPOINT_NAME, model, info)
    model.load_state_dict(last_state)
    last_path = mg.save_checkpoint(
        out_dir / mg.LAST_CHECKPOINT_NAME, model,
        {**info, "checkpoint_role": "last", "checkpoint_epoch": last_epoch},
    )
    # metrics.json é a versão legível: fora o val_idx (fica só no .pt, é o que
    # os notebooks recarregam) e a lista de pacientes por partição.
    info_legivel = {k: v for k, v in info.items() if k != "val_idx"}
    info_legivel["split"] = {
        nome: {k: v for k, v in bloco.items() if k != "pacientes"}
        for nome, bloco in info["split"].items()
    }
    info_legivel["checkpoints"] = {
        mg.CHECKPOINT_NAME: {
            "role": "best", "epoch": best_epoch,
            "sha256": classification.file_sha256(ckpt_path),
        },
        mg.LAST_CHECKPOINT_NAME: {
            "role": "last", "epoch": last_epoch,
            "sha256": classification.file_sha256(last_path),
        },
    }
    (out_dir / mg.METRICS_NAME).write_text(
        classification.dumps_json(info_legivel), encoding="utf-8"
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

    print(f"\nCheckpoint : {ckpt_path} (best, época {best_epoch})")
    print(f"Última     : {last_path} (época {last_epoch}, não usado no LUDB)")
    print(f"Métricas   : {out_dir / mg.METRICS_NAME}")
    print(f"Histórico  : {out_dir / mg.HISTORY_NAME}")
    print(
        "\nPróximo passo (teste independente):\n"
        f"  python src/test_ludb.py --checkpoint {ckpt_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
