"""Politicas binarias compartilhadas MIT-BIH/LUDB.

Este modulo contem apenas regras declarativas e validacao de metadados. Ele
nao le sinais nem executa etapas do pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path

import config

POLICY_SCHEMA_VERSION = 1
SEM_BLOQUEIO = "shared-extrasystoles-v1"
COM_BLOQUEIO = "shared-extrasystoles-bbb-v1"
MIT_PACED_RECORDS = frozenset({"102", "104", "107", "217"})
MIT_PACED_SYMBOLS = frozenset({"/", "f"})
MIT_ALL_BEAT_SYMBOLS = frozenset(
    {"N", "L", "R", "A", "a", "J", "S", "V", "F", "e", "j", "E", "Q", "/", "f"}
)
MIT_SUBJECT_BY_RECORD = {"202": "201"}

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

LUDB_ATRIAL_EXTRASYSTOLES = frozenset({
    "Atrial extrasystole, type: single pac",
    "Atrial extrasystole: sa-nodal extrasystole",
    "Atrial extrasystole: undefined",
    "Atrial extrasystole: left atrial",
    "Atrial extrasystole: low atrial",
    "Atrial extrasystole, type: quadrigemini",
    "Atrial extrasystole, type: allorhythmic pattern",
    "Atrial extrasystole, type: bigemini",
})
LUDB_VENTRICULAR_EXTRASYSTOLES = frozenset({
    "Ventricular extrasystole, localisation: rvot, anterior wall",
    "Ventricular extrasystole, localisation: rvot, antero-septal part",
    "Ventricular extrasystole, localisation: lvot, lvs",
    "Ventricular extrasystole, localisation: ivs, middle part",
    "Ventricular extrasystole, localisation: lv, undefined",
    "Ventricular extrasystole, type: intercalary pvc",
    "Ventricular extrasystole, type: single pvc",
    "Ventricular extrasystole, type: couplet",
    "Ventricular extrasystole, morphology: polymorphic",
})
LUDB_BUNDLE_BRANCH_BLOCKS = {
    "Complete left bundle branch block": ("lbbb", "left", "complete"),
    "Incomplete left bundle branch block": ("lbbb", "left", "incomplete"),
    "Complete right bundle branch block": ("rbbb", "right", "complete"),
    "Incomplete right bundle branch block": ("rbbb", "right", "incomplete"),
}
LUDB_KNOWN_CONDUCTION = frozenset({
    *LUDB_BUNDLE_BRANCH_BLOCKS,
    "I degree av block",
    "Iii degree av-block",
    "Left anterior hemiblock",
    "Non-specific intravintricular conduction delay",
    "Sinoatrial blockade, undefined",
})
#: Texto literal do CSV oficial LUDB 1.0.1 para o eixo eletrico normal.
LUDB_NORMAL_ELECTRIC_AXIS = "Electric axis of the heart: normal"

#: Familias-alvo da politica SEM_BLOQUEIO. Definem a populacao comum SEM x COM:
#: um beat pertence a ela se e referencia negativa ou se tem ao menos uma
#: destas familias, mesmo que o registro tambem tenha LBBB/RBBB no COM.
SHARED_TARGET_FAMILIES = frozenset({"atrial_extrasystole", "ventricular_extrasystole"})
SHARED_SUBSET_DEFINITION = (
    "label 0 (referencia negativa) ou registro com extrassistole atrial/ventricular, "
    "inclusive quando tambem possui LBBB/RBBB; a comparacao SEM x COM usa a "
    "intersecao exata de beat_id"
)

#: Subtipos de bloqueio que permanecem alvos no recorte "somente completos".
LUDB_COMPLETE_BUNDLE_SUBTYPES = frozenset(
    subtype for subtype, (_, _, completeness) in LUDB_BUNDLE_BRANCH_BLOCKS.items()
    if completeness == "complete"
)
LUDB_INCOMPLETE_BUNDLE_SUBTYPES = frozenset(
    subtype for subtype, (_, _, completeness) in LUDB_BUNDLE_BRANCH_BLOCKS.items()
    if completeness == "incomplete"
)


class ClassificationError(RuntimeError):
    """Interrompe o pipeline em vez de rotular uma entrada nao prevista."""


class SelectionStatus(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class ClassificationPolicy:
    slug: str
    display_name: str
    version: int
    mit_negative_symbols: frozenset[str]
    mit_positive_symbols: frozenset[str]
    ludb_target_families: tuple[str, ...]
    include_bundle_branch_block: bool
    ludb_negative_axis: tuple[str, ...] = (LUDB_NORMAL_ELECTRIC_AXIS,)
    ludb_negative_empty_categories: tuple[str, ...] = (
        "conduction_abnormalities", "extrasystolies", "hypertrophies",
        "cardiac_pacing", "ischemia", "repolarization", "other_states",
    )
    pacing_policy: str = "excluded"

    @property
    def mit_excluded_symbols(self) -> frozenset[str]:
        return MIT_ALL_BEAT_SYMBOLS - self.mit_negative_symbols - self.mit_positive_symbols


POLICIES = {
    SEM_BLOQUEIO: ClassificationPolicy(
        slug=SEM_BLOQUEIO,
        display_name="SEM_BLOQUEIO",
        version=1,
        mit_negative_symbols=frozenset({"N"}),
        mit_positive_symbols=frozenset({"A", "a", "V"}),
        ludb_target_families=("atrial_extrasystole", "ventricular_extrasystole"),
        include_bundle_branch_block=False,
    ),
    COM_BLOQUEIO: ClassificationPolicy(
        slug=COM_BLOQUEIO,
        display_name="COM_BLOQUEIO",
        version=1,
        mit_negative_symbols=frozenset({"N"}),
        mit_positive_symbols=frozenset({"A", "a", "V", "L", "R"}),
        ludb_target_families=(
            "atrial_extrasystole", "ventricular_extrasystole", "lbbb", "rbbb"
        ),
        include_bundle_branch_block=True,
    ),
}


def get_policy(policy: str | ClassificationPolicy) -> ClassificationPolicy:
    if isinstance(policy, ClassificationPolicy):
        return policy
    try:
        return POLICIES[policy]
    except (KeyError, TypeError) as exc:
        raise ClassificationError(
            f"Politica desconhecida: {policy!r}; use uma de {sorted(POLICIES)}."
        ) from exc


def split_ludb_values(text: str) -> tuple[str, ...]:
    if not isinstance(text, str):
        raise ClassificationError("Categorias LUDB devem conter texto, inclusive vazio.")
    # O CSV oficial separa diagnosticos por quebra de linha dentro da celula.
    return tuple(
        value.strip() for value in text.replace(";", "\n").splitlines() if value.strip()
    )


def validate_ludb_categories(categories: dict[str, str]) -> None:
    missing = set(LUDB_CATEGORIES) - categories.keys()
    if missing:
        raise ClassificationError(f"Categorias LUDB ausentes: {sorted(missing)}")
    for key in LUDB_CATEGORIES:
        if not isinstance(categories[key], str):
            raise ClassificationError(f"Categoria LUDB {key!r} deve conter texto.")

    extras = set(split_ludb_values(categories["extrasystolies"]))
    unknown_extras = extras - LUDB_ATRIAL_EXTRASYSTOLES - LUDB_VENTRICULAR_EXTRASYSTOLES
    if unknown_extras:
        raise ClassificationError(
            "Subtipos de extrasistolia LUDB sem tratamento explicito: "
            f"{sorted(unknown_extras)}"
        )
    conduction = set(split_ludb_values(categories["conduction_abnormalities"]))
    unknown_conduction = conduction - LUDB_KNOWN_CONDUCTION
    if unknown_conduction:
        raise ClassificationError(
            "Anormalidades de conducao LUDB sem tratamento explicito: "
            f"{sorted(unknown_conduction)}"
        )


def _other_conditions(categories: dict[str, str], used_targets: set[str]) -> list[str]:
    result: list[str] = []
    for key, column in LUDB_CATEGORIES.items():
        if key in {"rhythms", "cardiac_pacing"}:
            continue
        for value in split_ludb_values(categories[key]):
            if value in used_targets or (key == "electric_axis" and value == LUDB_NORMAL_ELECTRIC_AXIS):
                continue
            result.append(f"{column}: {value}")
    return result


def ludb_decision(categories: dict[str, str], policy: str | ClassificationPolicy) -> dict:
    """Classifica um registro LUDB em positivo, negativo ou fora da tarefa."""
    selected = get_policy(policy)
    validate_ludb_categories(categories)
    rhythms = list(split_ludb_values(categories["rhythms"]))

    if split_ludb_values(categories["cardiac_pacing"]):
        return {
            "selection_status": SelectionStatus.EXCLUDED.value,
            "label": None,
            "excluded_reason": "cardiac_pacing",
            "target_families": [],
            "target_subtypes": [],
            "positive_reasons": [],
            "bundle_branch": [],
            "rhythms": rhythms,
            "other_conditions": _other_conditions(categories, set()),
        }

    extras = set(split_ludb_values(categories["extrasystolies"]))
    conduction = set(split_ludb_values(categories["conduction_abnormalities"]))
    target_subtypes = sorted(
        extras & (LUDB_ATRIAL_EXTRASYSTOLES | LUDB_VENTRICULAR_EXTRASYSTOLES)
    )
    target_families: set[str] = set()
    if extras & LUDB_ATRIAL_EXTRASYSTOLES:
        target_families.add("atrial_extrasystole")
    if extras & LUDB_VENTRICULAR_EXTRASYSTOLES:
        target_families.add("ventricular_extrasystole")

    bundle_branch: list[dict[str, str]] = []
    if selected.include_bundle_branch_block:
        for subtype in sorted(conduction & LUDB_BUNDLE_BRANCH_BLOCKS.keys()):
            family, side, completeness = LUDB_BUNDLE_BRANCH_BLOCKS[subtype]
            target_families.add(family)
            target_subtypes.append(subtype)
            bundle_branch.append(
                {"side": side, "completeness": completeness, "subtype": subtype}
            )

    used_targets = set(target_subtypes)
    other_conditions = _other_conditions(categories, used_targets)
    if target_families:
        return {
            "selection_status": SelectionStatus.POSITIVE.value,
            "label": 1,
            "excluded_reason": None,
            "target_families": sorted(target_families),
            "target_subtypes": sorted(target_subtypes),
            "positive_reasons": sorted(target_families),
            "bundle_branch": bundle_branch,
            "rhythms": rhythms,
            "other_conditions": other_conditions,
        }

    axis = split_ludb_values(categories["electric_axis"])
    is_negative_reference = (
        axis == selected.ludb_negative_axis
        and all(
            not split_ludb_values(categories[key])
            for key in selected.ludb_negative_empty_categories
        )
    )
    if is_negative_reference:
        return {
            "selection_status": SelectionStatus.NEGATIVE.value,
            "label": 0,
            "excluded_reason": None,
            "target_families": [],
            "target_subtypes": [],
            "positive_reasons": [],
            "bundle_branch": [],
            "rhythms": rhythms,
            "other_conditions": [],
        }
    return {
        "selection_status": SelectionStatus.EXCLUDED.value,
        "label": None,
        "excluded_reason": "outside_shared_task",
        "target_families": [],
        "target_subtypes": [],
        "positive_reasons": [],
        "bundle_branch": [],
        "rhythms": rhythms,
        "other_conditions": other_conditions,
    }


def metadata(policy: str | ClassificationPolicy) -> dict:
    selected = get_policy(policy)
    return {
        "classification_policy": selected.slug,
        "classification_policy_display_name": selected.display_name,
        "classification_policy_version": selected.version,
        "classification_policy_schema_version": POLICY_SCHEMA_VERSION,
        "mit_negative_symbols": sorted(selected.mit_negative_symbols),
        "mit_positive_symbols": sorted(selected.mit_positive_symbols),
        "mit_excluded_symbols": sorted(selected.mit_excluded_symbols),
        "ludb_target_families": list(selected.ludb_target_families),
        "ludb_negative_reference": {
            "electric_axis": list(selected.ludb_negative_axis),
            "empty_categories": list(selected.ludb_negative_empty_categories),
            "rhythms_affect_label": False,
        },
        "pacing_policy": selected.pacing_policy,
        "label_encoding": {
            "0": "referencia negativa selecionada para a tarefa",
            "1": "familia compartilhada definida pela politica",
            "excluded": "fora da tarefa compartilhada",
        },
    }


def add_cli_argument(parser) -> None:
    parser.add_argument(
        "--classification-policy",
        choices=sorted(POLICIES),
        default=config.CLASSIFICATION_POLICY,
        help="Politica compartilhada MIT-BIH/LUDB usada em toda a execucao.",
    )


def dataset_path(dataset: str, policy: str | ClassificationPolicy) -> Path:
    return config.PROCESSED_DIR / f"{dataset}_{get_policy(policy).slug}.npz"


def run_name(policy: str | ClassificationPolicy) -> str:
    return f"cnn1d_mitbih_{get_policy(policy).slug}"


def mit_label(symbol: str, policy: str | ClassificationPolicy) -> int | None:
    selected = get_policy(policy)
    if symbol not in MIT_ALL_BEAT_SYMBOLS:
        raise ClassificationError(
            f"Simbolo MIT-BIH desconhecido: {symbol!r}; nenhum label atribuido."
        )
    if symbol in selected.mit_negative_symbols:
        return 0
    if symbol in selected.mit_positive_symbols:
        return 1
    return None


def mit_subject_id(record_id: str) -> str:
    return MIT_SUBJECT_BY_RECORD.get(str(record_id), str(record_id))


def require_policy(info: dict, policy: str | ClassificationPolicy, source: str) -> None:
    if not isinstance(info, dict):
        raise ClassificationError(f"{source}: metadados de politica ausentes ou invalidos.")
    expected = metadata(policy)
    for key in (
        "classification_policy", "classification_policy_version",
        "classification_policy_schema_version", "pacing_policy",
    ):
        value = info.get(key)
        if type(value) is not type(expected[key]) or value != expected[key]:
            raise ClassificationError(
                f"{source}: {key}={value!r}; esperado {expected[key]!r}. "
                "Artefato historico, incompleto ou de politica incompativel; gere nova execucao."
            )


def is_shared_subset(label: int, target_families) -> bool:
    """Beat pertencente a tarefa SEM_BLOQUEIO (populacao comum SEM x COM)."""
    return int(label) == 0 or bool(SHARED_TARGET_FAMILIES & set(target_families))


def is_complete_only_target(label: int, target_families, target_subtypes) -> bool:
    """Beat avaliado no recorte COM "somente completos".

    Exclui apenas positivos cuja unica justificativa restante e bloqueio
    incompleto; esses beats ficam fora do recorte, nunca viram 0.
    """
    return (
        is_shared_subset(label, target_families)
        or bool(LUDB_COMPLETE_BUNDLE_SUBTYPES & set(target_subtypes))
    )


def json_ready(value):
    """Converte NaN/inf em None e escalares numpy em tipos nativos para JSON valido."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if hasattr(value, "item") and getattr(value, "ndim", 1) == 0:
        value = value.item()
    if isinstance(value, float) and (value != value or value in (float("inf"), float("-inf"))):
        return None
    return value


def dumps_json(value) -> str:
    """JSON estrito (sem NaN) usado por todos os artefatos legiveis."""
    return json.dumps(json_ready(value), ensure_ascii=False, indent=2, allow_nan=False)


def stable_json_sha256(value) -> str:
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_new_output(path: Path) -> None:
    if Path(path).exists():
        raise FileExistsError(
            f"Saida ja existe: {path}. Escolha outro destino; artefatos nao serao sobrescritos."
        )


def prepare_args(args, stage: str):
    """Preenche caminhos default identificados pela politica explicita."""
    policy = get_policy(args.classification_policy)
    run = config.MODELS_DIR / run_name(policy)
    if hasattr(args, "dataset") and args.dataset is None:
        args.dataset = dataset_path("mitbih" if stage == "train" else "ludb", policy)
    if hasattr(args, "checkpoint") and args.checkpoint is None:
        args.checkpoint = run / "model.pt"
    if stage.startswith("preprocess_") and args.output is None:
        args.output = dataset_path(stage.removeprefix("preprocess_"), policy)
    if stage == "train" and args.run_name is None:
        args.run_name = run_name(policy)
    if stage == "fidelity" and args.gradcam is None:
        args.gradcam = run / "gradcam_ludb" / "ludb_gradcam.npz"
    return args


def require_npz_policy(archive, policy: str | ClassificationPolicy, source: str) -> None:
    keys = (
        "classification_policy", "classification_policy_version",
        "classification_policy_schema_version", "pacing_policy",
    )
    info = {key: archive[key].item() for key in keys if key in archive}
    require_policy(info, policy, source)


def artifact_identity(policy: str | ClassificationPolicy) -> dict:
    info = metadata(policy)
    return {key: info[key] for key in (
        "classification_policy", "classification_policy_display_name",
        "classification_policy_version", "classification_policy_schema_version",
        "pacing_policy",
    )}
