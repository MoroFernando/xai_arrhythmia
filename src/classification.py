"""Politica binaria compartilhada; nao executa processamento de sinal."""

from __future__ import annotations

import hashlib
from pathlib import Path

import config

POLICY_VERSION = "selected-abnormalities-v1"
MIT_PACED_RECORDS = frozenset({"102", "104", "107", "217"})
MIT_EXCLUDED_SYMBOLS = frozenset({"Q"})
MIT_PACED_SYMBOLS = frozenset({"/", "f"})
MIT_POSITIVE_SYMBOLS = frozenset({"A", "a", "J", "S", "V", "F", "e", "j", "E"})
MIT_KNOWN_SYMBOLS = MIT_POSITIVE_SYMBOLS | {"N", "L", "R"} | MIT_EXCLUDED_SYMBOLS | MIT_PACED_SYMBOLS
LUDB_CATEGORIES = {
    "rhythms": "Rhythms",
    "electric_axis": "Electric axis of the heart",
    "conduction_abnormalities": "Conduction abnormalities",
    "extrasystolies": "Extrasystolies",
    "hypertrophies": "Hypertrophies",
    "cardiac_pacing": "Cardiac pacing",
    "ischemia": "Ischemia",
    "repolarization": "Non-specific repolarization abnormalities",
    "other_states": "Other states",
}


def ludb_decision(categories: dict[str, str], only_arrhythmia: bool, include_paced: bool = False) -> dict:
    """OU entre categorias escolhidas; pacing exclui somente sem inclusao explicita."""
    metadata(only_arrhythmia, include_paced)
    missing = set(LUDB_CATEGORIES) - categories.keys()
    if missing:
        raise ClassificationError(f"Categorias LUDB ausentes: {sorted(missing)}")
    if any(not isinstance(categories[key], str) for key in LUDB_CATEGORIES):
        raise ClassificationError("Categorias LUDB devem conter texto, inclusive vazio.")
    if categories["cardiac_pacing"].strip():
        if not include_paced:
            return {"label": None, "excluded_reason": "cardiac_pacing", "positive_reasons": []}
        # Inclusao explicita: pacing nao interfere; as outras categorias decidem.
    critical = ("extrasystolies",) if only_arrhythmia else (
        "conduction_abnormalities", "extrasystolies", "ischemia"
    )
    reasons = [key for key in critical if categories[key].strip()]
    return {"label": int(bool(reasons)), "excluded_reason": None, "positive_reasons": reasons}


class ClassificationError(RuntimeError):
    """Interrompe o pipeline em vez de rotular uma entrada nao prevista."""


def metadata(only_arrhythmia: bool, include_paced: bool = False) -> dict:
    if type(only_arrhythmia) is not bool or type(include_paced) is not bool:
        raise ClassificationError("only_arrhythmia e include_paced devem ser booleanos.")
    return {
        "only_arrhythmia": only_arrhythmia,
        "include_paced": include_paced,
        "classification_policy": POLICY_VERSION,
        "label_encoding": {
            "0": "negativo pelos criterios do projeto",
            "1": "arritmia" if only_arrhythmia else "arritmia ou anormalidade selecionada",
        },
    }


def add_cli_argument(parser) -> None:
    parser.add_argument(
        "--only-arrhythmia", action="store_true", default=config.ONLY_ARRHYTHMIA,
        help="Ativa a politica somente arritmia em ambas as bases.",
    )
    parser.add_argument(
        "--include-paced", action="store_true", default=False,
        help="Permite pacing conforme a politica explicita de cada dataset.",
    )


def mode_name(only_arrhythmia: bool, include_paced: bool = False) -> str:
    mode = "only_arrhythmia" if only_arrhythmia else "selected_abnormalities"
    return mode + ("_with_pacing" if include_paced else "_without_pacing")


def dataset_path(dataset: str, only_arrhythmia: bool, include_paced: bool = False):
    return config.PROCESSED_DIR / f"{dataset}_{mode_name(only_arrhythmia, include_paced)}.npz"


def run_name(only_arrhythmia: bool, include_paced: bool = False) -> str:
    return f"cnn1d_mitbih_{mode_name(only_arrhythmia, include_paced)}"


def mit_label(symbol: str, only_arrhythmia: bool, include_paced: bool = False) -> int | None:
    if symbol in MIT_EXCLUDED_SYMBOLS:
        return None
    if symbol in MIT_PACED_SYMBOLS:
        return 0 if include_paced else None
    if symbol == "N":
        return 0
    if symbol in {"L", "R"}:
        return 0 if only_arrhythmia else 1
    if symbol in MIT_POSITIVE_SYMBOLS:
        return 1
    raise ClassificationError(f"Simbolo MIT-BIH desconhecido: {symbol!r}; nenhum label atribuido.")


def require_policy(info: dict, only_arrhythmia: bool, include_paced: bool, source: str) -> None:
    if not isinstance(info, dict):
        raise ClassificationError(f"{source}: metadados de politica ausentes ou invalidos.")
    expected = metadata(only_arrhythmia, include_paced)
    for key in ("classification_policy", "only_arrhythmia", "include_paced"):
        value = info.get(key)
        if type(value) is not type(expected[key]) or value != expected[key]:
            raise ClassificationError(
                f"{source}: {key}={value!r}; esperado {expected[key]!r}. "
                "Artefato historico, incompleto ou de politica incompatível; gere uma nova execucao."
            )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_new_output(path: Path) -> None:
    if Path(path).exists():
        raise FileExistsError(f"Saida ja existe: {path}. Escolha outro destino; artefatos nao serao sobrescritos.")


def prepare_args(args, stage: str):
    """Defaults identificados por politica; caminhos explicitos continuam aceitos."""
    mode = (args.only_arrhythmia, args.include_paced)
    run = config.MODELS_DIR / run_name(*mode)
    if hasattr(args, "dataset") and args.dataset is None:
        args.dataset = dataset_path("mitbih" if stage == "train" else "ludb", *mode)
    if hasattr(args, "checkpoint") and args.checkpoint is None:
        args.checkpoint = run / "model.pt"
    if stage.startswith("preprocess_") and args.output is None:
        args.output = dataset_path(stage.removeprefix("preprocess_"), *mode)
    if stage == "train" and args.run_name is None:
        args.run_name = run_name(*mode)
    if stage == "fidelity" and args.gradcam is None:
        args.gradcam = run / "gradcam_ludb" / "ludb_gradcam.npz"
    return args


def require_npz_policy(archive, only_arrhythmia: bool, include_paced: bool, source: str) -> None:
    keys = ("classification_policy", "only_arrhythmia", "include_paced")
    info = {key: archive[key].item() for key in keys if key in archive}
    require_policy(info, only_arrhythmia, include_paced, source)


def artifact_identity(only_arrhythmia: bool, include_paced: bool) -> dict:
    info = metadata(only_arrhythmia, include_paced)
    return {key: info[key] for key in ("classification_policy", "only_arrhythmia", "include_paced")}
