"""CLI de validacao e comparacao SEM_BLOQUEIO x COM_BLOQUEIO (apos as duas cadeias).

Nao treina nem infere: consome os artefatos ja gerados por cada politica
(datasets, checkpoint, predicoes LUDB, Grad-CAM e IoU) e

1. confere a cadeia de identidade de cada politica (politica, SHA-256 de
   dataset/checkpoint/predicoes/Grad-CAM, beat_ids, split) e invariantes da
   classificacao (pacing, 201/202, Rhythms, referencia negativa);
2. compara os dois modelos nos MESMOS beats: intersecao exata de ``beat_id``
   entre as populacoes LUDB SEM e COM, com os mesmos ``y_true``;
3. reune as tres avaliacoes (SEM, COM, COM somente completos). A terceira usa
   o mesmo checkpoint e as mesmas predicoes do COM; nao e um terceiro modelo.

Exemplo:
    python src/compare_policies.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

# Permite executar o script de qualquer diretório e importar os módulos irmãos.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import beat_io  # noqa: E402
import config  # noqa: E402
import classification  # noqa: E402
import fidelity_ludb as fidelity  # noqa: E402
import model_gradcam as mg  # noqa: E402
import test_ludb as metrics  # noqa: E402

COMPARISON_DIR = config.OUTPUTS_DIR / "comparacao_politicas"
RESULTS_NAME = "comparacao_sem_vs_com.json"
BEATS_NAME = "comparacao_sem_vs_com_beats.npz"
SEM, COM = classification.SEM_BLOQUEIO, classification.COM_BLOQUEIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida os artefatos das duas politicas e compara SEM x COM nos mesmos beats.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-dir", type=Path, default=COMPARISON_DIR)
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Somente le/agrega os artefatos; nao grava nem substitui resultados.",
    )
    return parser.parse_args()


def _npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _scalar(arrays: dict, key: str):
    return arrays[key].item() if key in arrays else None


def carregar(policy: str) -> dict:
    """Le todos os artefatos padrao de uma politica."""
    run = config.MODELS_DIR / classification.run_name(policy)
    paths = {
        "mitbih": classification.dataset_path("mitbih", policy),
        "ludb": classification.dataset_path("ludb", policy),
        "checkpoint": run / mg.CHECKPOINT_NAME,
        "checkpoint_last": run / mg.LAST_CHECKPOINT_NAME,
        "metrics": run / mg.METRICS_NAME,
        "predictions": run / "teste_ludb" / metrics.PREDICTIONS_NAME,
        "test_json": run / "teste_ludb" / metrics.RESULTS_NAME,
        "gradcam": run / "gradcam_ludb" / fidelity.GRADCAM_NAME,
        "fidelity": run / "fidelidade_ludb" / fidelity.FIDELITY_NAME,
        "fidelity_json": run / "fidelidade_ludb" / fidelity.FIDELITY_INFO_NAME,
    }
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise classification.ClassificationError(f"Artefatos ausentes ({policy}): {missing}")
    _, ckpt = mg.load_checkpoint(paths["checkpoint"], device="cpu", expected_policy=policy)
    _, ckpt_last = mg.load_checkpoint(paths["checkpoint_last"], device="cpu", expected_policy=policy)
    return {
        "policy": policy,
        "run_dir": run,
        "paths": paths,
        "sha": {key: classification.file_sha256(p) for key, p in paths.items()},
        "info": ckpt["info"],
        "info_last": ckpt_last["info"],
        "mit": beat_io.load_beat_dataset(paths["mitbih"]),
        "ludb": beat_io.load_beat_dataset(paths["ludb"]),
        "pred": _npz(paths["predictions"]),
        "gcam": _npz(paths["gradcam"]),
        "fid": _npz(paths["fidelity"]),
        "metrics_json": json.loads(paths["metrics"].read_text(encoding="utf-8")),
        "test_json": json.loads(paths["test_json"].read_text(encoding="utf-8")),
        "fidelity_json": json.loads(paths["fidelity_json"].read_text(encoding="utf-8")),
        "history": list(csv.DictReader((run / mg.HISTORY_NAME).read_text(encoding="utf-8").splitlines())),
    }


# --------------------------------------------------------------------------
# Verificacoes de identidade e invariantes
# --------------------------------------------------------------------------
def verificar_politica(art: dict) -> list[dict]:
    policy, sha, info = art["policy"], art["sha"], art["info"]
    pred, gcam, fid, ludb, mit = art["pred"], art["gcam"], art["fid"], art["ludb"], art["mit"]
    selecao = art["metrics_json"]["metrics"]["selecao_checkpoint"]
    checks: list[tuple[str, bool]] = []

    def politica_ok(fn, *a) -> bool:
        try:
            fn(*a)
            return True
        except classification.ClassificationError:
            return False

    history = art["history"]
    # CSV publica seis casas decimais. A primeira epoca de BA maxima vence.
    best_history = max(history, key=lambda row: float(row["val_acuracia_balanceada"]))
    checks += [
        ("best_epoch maximiza BA do historico MIT",
         int(best_history["epoca"]) == selecao["best_epoch"]),
        ("BA best confere com historico (tolerancia de serializacao)",
         abs(float(best_history["val_acuracia_balanceada"])
             - selecao["best_validation_balanced_accuracy"]) <= 5.1e-7),
        ("LAST confere com ultima epoca do historico",
         int(history[-1]["epoca"]) == selecao["last_epoch"]
         and art["info_last"].get("checkpoint_epoch") == selecao["last_epoch"]
         and abs(float(history[-1]["val_acuracia_balanceada"])
                 - selecao["last_validation_balanced_accuracy"]) <= 5.1e-7),
    ]
    checks += [
        ("politica dataset MIT", politica_ok(classification.require_policy, mit["metadata"], policy, "mit")),
        ("politica dataset LUDB", politica_ok(classification.require_policy, ludb["metadata"], policy, "ludb")),
        ("politica checkpoint", politica_ok(classification.require_policy, info, policy, "ckpt")),
        ("politica predicoes", politica_ok(classification.require_npz_policy, pred, policy, "pred")),
        ("politica Grad-CAM", politica_ok(classification.require_npz_policy, gcam, policy, "gcam")),
        ("politica IoU", politica_ok(classification.require_npz_policy, fid, policy, "fid")),
        ("model.pt e o BEST", info.get("checkpoint_role") == "best"),
        ("model_last.pt e o LAST", art["info_last"].get("checkpoint_role") == "last"),
        ("epoca do model.pt = best_epoch", info.get("checkpoint_epoch") == selecao["best_epoch"]),
        ("SHA model.pt registrado em metrics.json",
         art["metrics_json"]["checkpoints"][mg.CHECKPOINT_NAME]["sha256"] == sha["checkpoint"]),
        ("checkpoint treinado no dataset MIT atual", info.get("dataset_sha256") == sha["mitbih"]),
        ("predicoes: dataset LUDB atual", _scalar(pred, "dataset_sha256") == sha["ludb"]),
        ("predicoes: checkpoint BEST", _scalar(pred, "checkpoint_sha256") == sha["checkpoint"]),
        ("predicoes: papel best", _scalar(pred, "checkpoint_role") == "best"),
        ("Grad-CAM: mesmo checkpoint das predicoes",
         _scalar(gcam, "checkpoint_sha256") == _scalar(pred, "checkpoint_sha256")),
        ("Grad-CAM: vinculado as predicoes", _scalar(gcam, "predictions_sha256") == sha["predictions"]),
        ("Grad-CAM: dataset LUDB atual", _scalar(gcam, "dataset_sha256") == sha["ludb"]),
        ("IoU: Grad-CAM atual", _scalar(fid, "gradcam_sha256") == sha["gradcam"]),
        ("IoU: mesmo checkpoint", _scalar(fid, "checkpoint_sha256") == sha["checkpoint"]),
        ("IoU: dataset LUDB atual", _scalar(fid, "dataset_sha256") == sha["ludb"]),
        ("JSON do teste: checkpoint BEST",
         art["test_json"].get("checkpoint_sha256") == sha["checkpoint"]
         and art["test_json"].get("checkpoint_role") == "best"),
        ("split identico em ckpt/predicoes/Grad-CAM/IoU",
         len({info["split_manifest_sha256"], _scalar(pred, "split_manifest_sha256"),
              _scalar(gcam, "split_manifest_sha256"), _scalar(fid, "split_manifest_sha256")}) == 1),
        ("beat_ids alinhados dataset/predicoes/Grad-CAM/IoU",
         all(np.array_equal(ludb["beat_ids"], a["beat_ids"]) for a in (pred, gcam, fid))),
        ("y_true alinhado ao dataset",
         all(np.array_equal(ludb["labels"], a["y_true"]) for a in (pred, gcam, fid))),
        ("y_pred Grad-CAM = y_pred predicoes", np.array_equal(gcam["y_pred"], pred["y_pred"])),
        ("scores Grad-CAM ~ scores predicoes",
         float(np.abs(gcam["y_score"] - pred["y_score"]).max()) <= 1e-4),
        ("beat_ids LUDB unicos", np.unique(ludb["beat_ids"]).size == ludb["beat_ids"].size),
        ("beat_ids MIT unicos", np.unique(mit["beat_ids"]).size == mit["beat_ids"].size),
    ]
    return [{"verificacao": nome, "ok": bool(ok)} for nome, ok in checks]


def verificar_invariantes(sem: dict, com: dict) -> list[dict]:
    checks: list[tuple[str, bool]] = []
    for art in (sem, com):
        tag = classification.get_policy(art["policy"]).display_name
        mit, ludb, info = art["mit"], art["ludb"], art["info"]
        policy = classification.get_policy(art["policy"])
        records = set(np.unique(mit["pacientes"]))
        checks += [
            (f"{tag}: registros MIT paced ausentes", not (records & classification.MIT_PACED_RECORDS)),
            (f"{tag}: simbolos de pacing ausentes", not (set(np.unique(mit["simbolos"])) & {"/", "f"})),
            (f"{tag}: simbolos MIT so da politica",
             set(np.unique(mit["simbolos"])) <= policy.mit_negative_symbols | policy.mit_positive_symbols),
            (f"{tag}: 201 e 202 com o mesmo subject_id",
             set(mit["subject_ids"][np.isin(mit["pacientes"], ["201", "202"])]) == {"201"}),
            (f"{tag}: nenhum sujeito em treino e validacao",
             not (set(info["split"]["treino"]["sujeitos"]) & set(info["split"]["validacao"]["sujeitos"]))),
            (f"{tag}: 201/202 na mesma particao",
             info["split_manifest"]["record_subject"].get("202") == "201"
             and ("201" in info["split"]["validacao"]["pacientes"])
             == ("202" in info["split"]["validacao"]["pacientes"])),
            (f"{tag}: classes 0 e 1 no treino e na validacao",
             all(info["split"][p]["classe_0"] > 0 and info["split"][p]["classe_1"] > 0
                 for p in ("treino", "validacao"))),
        ]
        clinical = ludb["metadata"]["clinical_by_record"]
        incluidos = set(np.unique(ludb["pacientes"]))
        rhythm_ok = True
        negativos_ok = True
        for record_id, entry in clinical.items():
            cats = entry["categories"]
            sem_ritmo = classification.ludb_decision({**cats, "rhythms": ""}, art["policy"])
            if sem_ritmo["selection_status"] != entry["selection_status"]:
                rhythm_ok = False
            if entry["label"] == 0 and (
                any(classification.split_ludb_values(cats[key])
                    for key in policy.ludb_negative_empty_categories)
                or classification.split_ludb_values(cats["electric_axis"])
                != policy.ludb_negative_axis
            ):
                negativos_ok = False
        stored = {r: int(ludb["labels"][ludb["pacientes"] == r][0]) for r in incluidos}
        checks += [
            (f"{tag}: nenhum registro LUDB com pacing incluido",
             not (incluidos & set(ludb["metadata"]["records_excluidos_pacing"]))),
            (f"{tag}: Rhythms nao altera nenhuma decisao LUDB", rhythm_ok),
            (f"{tag}: label 0 somente sem condicoes fora da tarefa (ex.: Ischemia)", negativos_ok),
            (f"{tag}: labels LUDB = decisao da politica por registro",
             all(stored[r] == clinical[r]["label"] for r in incluidos)),
        ]
    checks += [
        ("mesmo manifesto de split SEM e COM",
         sem["info"]["split_manifest_sha256"] == com["info"]["split_manifest_sha256"]),
        ("checkpoints SEM e COM distintos", sem["sha"]["checkpoint"] != com["sha"]["checkpoint"]),
        ("diretorios de saida distintos", sem["run_dir"] != com["run_dir"]),
    ]
    # Recalcula apenas metricas a partir dos scores existentes, sem inferencia.
    pc = com["pred"]
    complete = metrics.avaliacao_somente_completos(
        pc["y_true"], pc["y_score"], pc["target_families"],
        pc["target_subtypes"], pc["pacientes"], threshold=mg.DEFAULT_THRESHOLD,
    )
    saved_complete = com["test_json"]["avaliacao_somente_completos"]
    checks += [
        ("somente completos: mascara salva = regra aplicada aos alvos",
         np.array_equal(complete["mascara"], pc["is_complete_only_eval"].astype(bool))),
        ("somente completos: metricas recalculadas dos scores salvos",
         all(np.isclose(complete["metrics"][key], saved_complete["metrics"][key], atol=1e-10, rtol=0)
             for key in ("n", "tn", "fp", "fn", "tp", "acuracia", "acuracia_balanceada",
                         "precisao", "sensibilidade", "especificidade", "f1", "auc_roc"))),
        ("somente completos: populacao confere com JSON",
         all(complete[key] == saved_complete[key]
             for key in ("n", "classe_0", "classe_1", "n_registros", "n_excluidos"))),
        ("somente completos: negativos e extrassistoles preservados",
         np.all(complete["mascara"][pc["is_shared_subset"].astype(bool)])),
    ]
    return [{"verificacao": nome, "ok": bool(ok)} for nome, ok in checks]


# --------------------------------------------------------------------------
# Comparacao nos mesmos beats
# --------------------------------------------------------------------------
def _resumo_scores(score: np.ndarray) -> dict:
    return {
        "score_medio": float(score.mean()) if score.size else float("nan"),
        "score_mediano": float(np.median(score)) if score.size else float("nan"),
    }


def _grupo(y, score, mask, pacientes) -> dict:
    """Desempenho num recorte de classe unica (positivos ou negativos)."""
    y, score, pac = y[mask], score[mask], pacientes[mask]
    pred = (score >= mg.DEFAULT_THRESHOLD).astype(int)
    out = {"n": int(mask.sum()), "n_registros": int(np.unique(pac).size), **_resumo_scores(score)}
    if out["n"] and np.all(y == 1):
        tp = int(pred.sum())
        out.update(tp=tp, fn=out["n"] - tp, sensibilidade=tp / out["n"],
                   taxa_falso_negativo=(out["n"] - tp) / out["n"])
    elif out["n"] and np.all(y == 0):
        tn = int((pred == 0).sum())
        out.update(tn=tn, fp=out["n"] - tn, especificidade=tn / out["n"])
    return out


def _transicoes(pred_sem, pred_com, mask) -> dict:
    return {
        f"{a}->{b}": int(((pred_sem == a) & (pred_com == b) & mask).sum())
        for a in (0, 1) for b in (0, 1)
    }


def comparacao_comum(sem: dict, com: dict) -> tuple[dict, dict, list[dict]]:
    ps, pc = sem["pred"], com["pred"]
    comum = np.intersect1d(ps["beat_ids"], pc["beat_ids"])
    pos_sem = {b: i for i, b in enumerate(ps["beat_ids"])}
    pos_com = {b: i for i, b in enumerate(pc["beat_ids"])}
    i_sem = np.asarray([pos_sem[b] for b in comum])
    i_com = np.asarray([pos_com[b] for b in comum])

    y = ps["y_true"][i_sem].astype(int)
    s_sem = ps["y_score"][i_sem].astype(float)
    s_com = pc["y_score"][i_com].astype(float)
    pred_sem = (s_sem >= mg.DEFAULT_THRESHOLD).astype(int)
    pred_com = (s_com >= mg.DEFAULT_THRESHOLD).astype(int)
    pacientes = ps["pacientes"][i_sem]
    fam_com = [set(json.loads(str(v))) for v in pc["target_families"][i_com]]

    checks = [
        {"verificacao": "intersecao: y_true identico nos dois modelos",
         "ok": bool(np.array_equal(y, pc["y_true"][i_com]))},
        {"verificacao": "intersecao = populacao LUDB SEM inteira",
         "ok": bool(comum.size == ps["beat_ids"].size)},
        {"verificacao": "is_shared_subset do COM = intersecao por beat_id",
         "ok": bool(np.array_equal(pc["is_shared_subset"].astype(bool),
                                   np.isin(pc["beat_ids"], comum)))},
        {"verificacao": "pacientes identicos na intersecao",
         "ok": bool(np.array_equal(pacientes, pc["pacientes"][i_com]))},
    ]

    grupos = {
        "todos": np.ones(y.size, dtype=bool),
        "negativos": y == 0,
        "atrial_extrasystole": np.asarray(["atrial_extrasystole" in f for f in fam_com]),
        "ventricular_extrasystole": np.asarray(["ventricular_extrasystole" in f for f in fam_com]),
        "extrassistole_com_bloqueio_no_COM": np.asarray([
            bool(f & classification.SHARED_TARGET_FAMILIES) and bool(f & {"lbbb", "rbbb"})
            for f in fam_com
        ]),
    }
    por_grupo = {}
    for nome, mask in grupos.items():
        if nome == "todos" or not mask.any():
            continue
        diff = s_com[mask] - s_sem[mask]
        por_grupo[nome] = {
            "SEM": _grupo(y, s_sem, mask, pacientes),
            "COM": _grupo(y, s_com, mask, pacientes),
            "transicoes_pred_SEM_para_COM": _transicoes(pred_sem, pred_com, mask),
            "diferenca_score_COM_menos_SEM": {
                "media": float(diff.mean()), "mediana": float(np.median(diff)),
                "fracao_COM_maior": float((diff > 0).mean()),
            },
        }

    def correlacao(fn, a, b):
        return float(fn(a, b)[0]) if a.size > 2 and a.std() and b.std() else float("nan")

    diff = s_com - s_sem
    resultado = {
        "definicao": classification.SHARED_SUBSET_DEFINITION,
        "n_beats": int(comum.size),
        "n_registros": int(np.unique(pacientes).size),
        "classe_0": int((y == 0).sum()),
        "classe_1": int((y == 1).sum()),
        "metrics": {
            "SEM": metrics.binary_metrics(y, s_sem, mg.DEFAULT_THRESHOLD),
            "COM": metrics.binary_metrics(y, s_com, mg.DEFAULT_THRESHOLD),
        },
        "baselines": metrics.baselines(y),
        "por_grupo": por_grupo,
        "transicoes_pred_SEM_para_COM": {
            "todos": _transicoes(pred_sem, pred_com, grupos["todos"]),
            "y_true_0": _transicoes(pred_sem, pred_com, y == 0),
            "y_true_1": _transicoes(pred_sem, pred_com, y == 1),
        },
        "scores": {
            "SEM": _resumo_scores(s_sem),
            "COM": _resumo_scores(s_com),
            "diferenca_COM_menos_SEM": {"media": float(diff.mean()), "mediana": float(np.median(diff))},
            "pearson": correlacao(pearsonr, s_sem, s_com),
            "spearman": correlacao(spearmanr, s_sem, s_com),
        },
    }

    # IoU nos mesmos beats (explicabilidade; nao e metrica de classificacao).
    fs, fc = sem["fid"], com["fid"]
    iou = {}
    for chave, nome in (("iou_por_batimento", "clinica_PQRST"), ("iou_p", "P"),
                        ("iou_qrs", "QRS"), ("iou_t", "T")):
        a = fs[chave][i_sem].astype(float)
        b = fc[chave][i_com].astype(float)
        valid = ~np.isnan(a) & ~np.isnan(b)
        bloco = {
            "SEM": fidelity.resumo_iou(a, pred_sem, y),
            "COM": fidelity.resumo_iou(b, pred_com, y),
            "diferenca_media_pareada_COM_menos_SEM": (
                float((b[valid] - a[valid]).mean()) if valid.any() else float("nan")
            ),
            "por_grupo": {
                g: {"SEM": fidelity.resumo_iou(a[m]), "COM": fidelity.resumo_iou(b[m])}
                for g, m in grupos.items() if g != "todos" and m.any()
            },
        }
        iou[nome] = bloco
    resultado["iou_mesmos_beats"] = iou

    tabela = {
        "beat_ids": comum,
        "pacientes": pacientes,
        "y_true": y,
        "score_sem": s_sem.astype(np.float32),
        "score_com": s_com.astype(np.float32),
        "pred_sem": pred_sem,
        "pred_com": pred_com,
        "iou_sem": fs["iou_por_batimento"][i_sem],
        "iou_com": fc["iou_por_batimento"][i_com],
        "target_families_com": pc["target_families"][i_com],
        "target_subtypes_com": pc["target_subtypes"][i_com],
    }
    return resultado, tabela, checks


def tres_avaliacoes(sem: dict, com: dict) -> dict:
    def bloco(nome, populacao, m, b, extra=None):
        return {"avaliacao": nome, **populacao, "metrics": m, "baselines": b, **(extra or {})}

    def pop(test_json):
        md = test_json["dataset_metadata"]
        return {"n": md["n_batimentos"], "classe_0": md["distribuicao_labels"]["0"],
                "classe_1": md["distribuicao_labels"]["1"],
                "n_registros": len(md["records_processados"])}

    ts, tc = sem["test_json"], com["test_json"]
    sc = tc["avaliacao_somente_completos"]
    return {
        "SEM_BLOQUEIO": bloco("modelo SEM", pop(ts), ts["metrics"], ts["baselines"]),
        "COM_BLOQUEIO": bloco("modelo COM (completos + incompletos)", pop(tc), tc["metrics"], tc["baselines"]),
        "COM_BLOQUEIO_somente_completos": bloco(
            sc["nome"],
            {k: sc[k] for k in ("n", "classe_0", "classe_1", "n_registros")},
            sc["metrics"], sc["baselines"],
            {"nota": "mesmo checkpoint e mesmas predicoes do modelo COM; nao e um terceiro modelo",
             "n_excluidos": sc["n_excluidos"]},
        ),
    }


def main() -> int:
    args = parse_args()
    if not args.validate_only:
        classification.require_new_output(args.output_dir)
    sem, com = carregar(SEM), carregar(COM)

    verificacoes = {
        "SEM_BLOQUEIO": verificar_politica(sem),
        "COM_BLOQUEIO": verificar_politica(com),
        "invariantes": verificar_invariantes(sem, com),
    }
    comum, tabela, checks_comum = comparacao_comum(sem, com)
    verificacoes["subconjunto_comum"] = checks_comum
    falhas = [c["verificacao"] for bloco in verificacoes.values() for c in bloco if not c["ok"]]

    resultado = {
        "criado_em": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "etapa": "comparacao SEM_BLOQUEIO x COM_BLOQUEIO no LUDB",
        "threshold": mg.DEFAULT_THRESHOLD,
        "artefatos": {
            classification.get_policy(a["policy"]).display_name: {
                "classification_policy": a["policy"],
                "run_dir": config.caminho_relativo(a["run_dir"]),
                "checkpoint_usado": config.caminho_relativo(a["paths"]["checkpoint"]),
                "checkpoint_epoch": a["info"]["checkpoint_epoch"],
                "sha256": a["sha"],
            }
            for a in (sem, com)
        },
        "verificacoes": verificacoes,
        "verificacoes_falhas": falhas,
        "tres_avaliacoes": tres_avaliacoes(sem, com),
        "mesmos_beats_sem_vs_com": comum,
        "nota_iou": (
            "IoU mede sobreposicao entre a atencao do Grad-CAM e as regioes P/QRS/T; "
            "nao mede acerto de classificacao."
        ),
    }

    if not args.validate_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / RESULTS_NAME).write_text(classification.dumps_json(resultado), encoding="utf-8")
        np.savez_compressed(args.output_dir / BEATS_NAME, **tabela)

    print("=" * 64)
    print("VALIDACAO DE ARTEFATOS")
    print("=" * 64)
    for bloco, itens in verificacoes.items():
        ok = sum(c["ok"] for c in itens)
        print(f"  {bloco:<20} {ok}/{len(itens)} ok")
    for nome in falhas:
        print(f"  FALHA: {nome}")
    print("=" * 64)
    print(f"MESMOS BEATS SEM x COM: n={comum['n_beats']} "
          f"(classe 0: {comum['classe_0']}, classe 1: {comum['classe_1']})")
    for nome in ("SEM", "COM"):
        print(metrics.formata_metricas(f"modelo {nome} | mesmos beats", comum["metrics"][nome]))
    print(f"  transicoes y_pred SEM->COM: {comum['transicoes_pred_SEM_para_COM']['todos']}")
    print("=" * 64)
    if args.validate_only:
        print("Validacao somente leitura: nenhum artefato foi gravado.")
    else:
        print(f"Resultados : {args.output_dir / RESULTS_NAME}")
        print(f"Por beat   : {args.output_dir / BEATS_NAME}")
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
