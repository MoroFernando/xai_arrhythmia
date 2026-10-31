"""CLI de extração do Grad-CAM 1D de todos os batimentos do LUDB (Etapa 4).

Carrega o checkpoint gerado por `src/train_cnn.py` e o dataset processado
do LUDB, calcula o Grad-CAM de cada batimento na última camada convolucional
(`conv2`) — ver `src/model_gradcam.py` — e grava os mapas já interpolados
para as 234 amostras da janela, junto com a predição de cada batimento.

O script é autocontido: não depende de `src/test_ludb.py` ter rodado. A
predição é recomputada aqui com o mesmo checkpoint e o mesmo limiar fixo
(`model_gradcam.DEFAULT_THRESHOLD`), então sai idêntica à de `test_ludb.py`.

Exemplo:
    python src/extract_gradcam.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Permite executar o script de qualquer diretório e importar os módulos irmãos.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import beat_io  # noqa: E402
import config  # noqa: E402
import classification  # noqa: E402
import model_gradcam as mg  # noqa: E402

GRADCAM_NAME = "ludb_gradcam.npz"
GRADCAM_INFO_NAME = "ludb_gradcam.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extrai o Grad-CAM 1D de todos os batimentos do LUDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint .pt gerado por src/train_cnn.py.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset .npz gerado por src/preprocess_ludb.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Diretório de saída (padrão: subpasta 'gradcam_ludb' ao lado do checkpoint).",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=None,
        help=("ludb_predictions.npz do teste (padrão: teste_ludb/ ao lado do checkpoint); "
              "o Grad-CAM exige o mesmo checkpoint e as mesmas predições."),
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Batimentos por passada de forward/backward.",
    )
    parser.add_argument("--device", default="auto", help="'auto', 'cuda' ou 'cpu'.")
    classification.add_cli_argument(parser)
    return classification.prepare_args(parser.parse_args(), "gradcam")


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir or (args.checkpoint.parent / "gradcam_ludb")
    classification.require_new_output(out_dir)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.checkpoint.exists():
        print(
            f"Checkpoint nao encontrado: {args.checkpoint}\n"
            f"Treine o modelo antes com: python src/train_cnn.py",
            file=sys.stderr,
        )
        return 1
    if not args.dataset.exists():
        print(
            f"Dataset nao encontrado: {args.dataset}\n"
            f"Gere-o antes com: python src/preprocess_ludb.py --download",
            file=sys.stderr,
        )
        return 1

    device = mg.resolve_device(args.device)
    modelo, ckpt = mg.load_checkpoint(
        args.checkpoint, device=device,
        expected_policy=args.classification_policy,
    )
    ckpt_info = ckpt.get("info", {})
    if "split_manifest_sha256" not in ckpt_info:
        raise classification.ClassificationError(
            "Checkpoint sem identidade do split; checkpoint historico incompativel."
        )
    if ckpt_info.get("checkpoint_role") != "best":
        raise classification.ClassificationError(
            f"{args.checkpoint}: checkpoint_role={ckpt_info.get('checkpoint_role')!r}; "
            "o Grad-CAM exige o melhor checkpoint por acuracia balanceada de validacao."
        )
    limiar = mg.DEFAULT_THRESHOLD

    dados = beat_io.load_beat_dataset(args.dataset)
    classification.require_policy(dados.get("metadata", {}), args.classification_policy, str(args.dataset))
    trace_keys = (
        "beat_ids", "selection_indices", "target_families", "target_subtypes",
        "bundle_side", "bundle_completeness", "positive_reasons", "other_conditions",
        "is_shared_subset",
    )
    missing_trace = set(trace_keys) - dados.keys()
    if missing_trace:
        raise classification.ClassificationError(
            f"Dataset LUDB sem rastreabilidade nova: {sorted(missing_trace)}."
        )
    provenance = {
        **classification.artifact_identity(args.classification_policy),
        "dataset_sha256": classification.file_sha256(args.dataset),
        "checkpoint_sha256": classification.file_sha256(args.checkpoint),
        "checkpoint_role": ckpt_info["checkpoint_role"],
        "checkpoint_epoch": int(ckpt_info["checkpoint_epoch"]),
        "split_manifest_sha256": ckpt_info["split_manifest_sha256"],
    }
    predictions_path = args.predictions or (
        args.checkpoint.parent / "teste_ludb" / "ludb_predictions.npz"
    )
    if not predictions_path.exists():
        raise classification.ClassificationError(
            f"Predicoes LUDB ausentes: {predictions_path}. Rode src/test_ludb.py antes."
        )
    with np.load(predictions_path, allow_pickle=False) as pred:
        classification.require_npz_policy(pred, args.classification_policy, str(predictions_path))
        for key in ("checkpoint_sha256", "dataset_sha256"):
            if key not in pred or pred[key].item() != provenance[key]:
                raise classification.ClassificationError(
                    f"Predicoes {predictions_path} com {key} diferente do Grad-CAM."
                )
        if not np.array_equal(pred["beat_ids"], dados["beat_ids"]):
            raise classification.ClassificationError("Predicoes e dataset com beat_ids diferentes.")
        pred_scores = np.asarray(pred["y_score"], dtype=np.float32)
        pred_y_pred = np.asarray(pred["y_pred"]).astype(np.int64)
    provenance["predictions_sha256"] = classification.file_sha256(predictions_path)
    X = np.asarray(dados["series"], dtype=np.float32)
    y = np.asarray(dados["labels"]).astype(np.int64)
    pacientes = np.asarray(dados["pacientes"])
    ritmos = np.asarray(dados["simbolos"])

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Dataset    : {args.dataset}")
    print(f"  {X.shape[0]} batimentos | janela de {X.shape[1]} amostras")

    logging.info("Extraindo Grad-CAM de %d batimentos (lote=%d)...", X.shape[0], args.batch_size)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    cams, probs = [], []
    for inicio in range(0, len(X_tensor), args.batch_size):
        cam, prob = mg.grad_cam_1d(modelo, X_tensor[inicio : inicio + args.batch_size], device)
        cams.append(cam)
        probs.append(prob)

    cam_interp = torch.cat(cams).numpy().astype(np.float32)      # (n, 234)
    scores = torch.cat(probs).squeeze(1).numpy().astype(np.float32)
    y_pred = (scores >= limiar).astype(np.int64)
    logging.info("Concluido: cam_interp %s", cam_interp.shape)

    acerto = float((y_pred == y).mean())
    print(f"  predicao de referencia: acuracia {acerto:.4f}")
    # Mesmo checkpoint => mesmos scores (tolerancia so para arredondamento de GPU).
    max_diff = float(np.abs(scores - pred_scores).max())
    if max_diff > 1e-4 or not np.array_equal(y_pred, pred_y_pred):
        raise classification.ClassificationError(
            f"Scores/predicoes do Grad-CAM divergem do teste LUDB (max |diff|={max_diff:.2e})."
        )
    print(f"  conferencia com {predictions_path.name}: max |diff score| = {max_diff:.2e}, y_pred identico")

    out_dir = args.output_dir or (args.checkpoint.parent / "gradcam_ludb")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out_dir / GRADCAM_NAME,
        **provenance,
        cam_interp=cam_interp,
        y_true=y,
        y_pred=y_pred,
        y_score=scores,
        pacientes=pacientes,
        ritmos=ritmos,
        **{key: dados[key] for key in trace_keys},
        threshold=np.array(limiar),
    )

    info = {
        **classification.metadata(args.classification_policy),
        **provenance,
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etapa": "4 - extracao do Grad-CAM 1D (LUDB)",
        "checkpoint": config.caminho_relativo(args.checkpoint),
        "dataset": config.caminho_relativo(args.dataset),
        "camada_alvo": "conv2 (ultima camada convolucional)",
        "n_batimentos": int(X.shape[0]),
        "tamanho_janela": int(X.shape[1]),
        "interpolacao": "linear (F.interpolate) direto para o tamanho da janela",
        "threshold": limiar,
        "acuracia_referencia": acerto,
        "predictions": config.caminho_relativo(predictions_path),
        "max_abs_diff_score_vs_predictions": max_diff,
        "observacao": (
            "cam_interp esta alinhado por indice com o dataset do LUDB e com "
            "outputs/models/<run>/teste_ludb/ludb_predictions.npz."
        ),
    }
    (out_dir / GRADCAM_INFO_NAME).write_text(
        classification.dumps_json(info), encoding="utf-8"
    )

    print(f"\nGrad-CAM  : {out_dir / GRADCAM_NAME}")
    print(f"Metadados : {out_dir / GRADCAM_INFO_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
