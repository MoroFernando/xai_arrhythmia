"""Carregamento e pre-processamento do LUDB (conjunto de teste).

Pipeline (Etapa 2 do projeto, conjunto de teste / ground truth):
    download/leitura via WFDB -> selecao da derivacao II
    -> extracao das triplas (onset, pico, offset) das ondas P, QRS e T
    -> reamostragem 500 Hz -> 360 Hz (sinal e anotacoes)
    -> segmentacao beat-to-beat centrada no pico R
    -> filtragem: so batimentos com P, QRS e T completos dentro da janela
    -> Z-normalization por batimento
    -> propagacao do ritmo do registro para cada batimento + rotulo binario

O LUDB nao rotula batimento a batimento: o diagnostico de ritmo vale para o
registro inteiro e e replicado em todos os batimentos dele.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import wfdb

from .. import config, signals

logger = logging.getLogger(__name__)

#: Casa linhas como "Rhythm: Sinus bradycardia." nos comentarios do header.
_RHYTHM_PATTERN = re.compile(
    rf"^\s*{config.LUDB_RHYTHM_FIELD}\s*:\s*(.+?)\s*$", re.IGNORECASE
)

#: Ordem canonica das estruturas, usada nas mascaras e nos relatorios.
WAVE_TYPES = ("p", "N", "t")


@dataclass
class PreprocessConfig:
    """Parametros do pre-processamento do LUDB."""

    records: Sequence[str] = field(default_factory=lambda: list(config.LUDB_RECORDS))
    lead: str = config.LUDB_LEAD
    fs_original: int = config.LUDB_FS
    fs_target: int = config.TARGET_FS
    window_left: int = config.WINDOW_LEFT
    window_right: int = config.WINDOW_RIGHT
    sinus_rhythms: frozenset[str] = config.LUDB_SINUS_RHYTHMS
    center: str = "qrs-peak"  # ou "qrs-midpoint"
    normalize: bool = True
    data_dir: Path | None = None  # None => leitura direta do PhysioNet

    @property
    def window_size(self) -> int:
        return self.window_left + self.window_right

    def to_metadata(self) -> dict:
        """Serializa os parametros para gravacao junto ao dataset."""
        return {
            "dataset": "ludb",
            "fs_original_hz": self.fs_original,
            "fs_hz": self.fs_target,
            "lead": self.lead,
            "window_left": self.window_left,
            "window_right": self.window_right,
            "window_size": self.window_size,
            "centro_da_janela": self.center,
            "criterio_inclusao": (
                "batimento mantido se o proprio complexo QRS e a onda T "
                "seguinte estiverem integralmente dentro da janela; a onda P "
                "precedente e exigida apenas em registros que possuem ondas P "
                "delimitadas (em fibrilacao/flutter atrial a onda P nao existe)"
            ),
            "ritmos_classe_0": sorted(self.sinus_rhythms),
            "rotulo_por_batimento": (
                "ritmo do registro propagado para todos os seus batimentos"
            ),
            "normalization": "z-score per beat" if self.normalize else "none",
            "label_encoding": {"0": "nao-arritmia", "1": "arritmia"},
        }


# --------------------------------------------------------------------------
# Leitura e parsing
# --------------------------------------------------------------------------
def download_records(
    dest_dir: Path = config.LUDB_RAW_DIR,
    lead: str = config.LUDB_LEAD,
) -> Path:
    """Baixa os registros do LUDB do PhysioNet para uso offline.

    Args:
        dest_dir: diretorio destino (``data/raw/ludb`` por padrao).
        lead: derivacao cujas anotacoes serao baixadas. O LUDB tem um arquivo
            de anotacao por derivacao (12 no total); baixar so a que o projeto
            usa evita ~2200 arquivos desnecessarios.

    Returns:
        O diretorio onde os arquivos foram gravados.

    Note:
        ``db_dir`` aponta para a raiz da versao (``ludb/1.0.1``) porque e la
        que ficam RECORDS e ANNOTATORS. Como as entradas do RECORDS vem com o
        prefixo ``data/``, os arquivos caem em ``<dest_dir>/data/``.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Baixando LUDB (derivacao %s) para %s", lead, dest_dir)
    wfdb.dl_database(
        db_dir=config.LUDB_DB_DIR,
        dl_dir=str(dest_dir),
        annotators=[lead],
    )
    return dest_dir


def resolve_records_dir(data_dir: Path) -> Path:
    """Localiza o diretorio que contem de fato os arquivos WFDB.

    O download do PhysioNet preserva o subdiretorio ``data/`` do RECORDS, mas
    um diretorio montado manualmente pode ter os registros na raiz.
    """
    data_dir = Path(data_dir)
    subdir = data_dir / "data"
    return subdir if subdir.is_dir() else data_dir


def _read_record(record_id: str, lead: str, data_dir: Path | None):
    """Le sinal e anotacoes de delineacao, local ou direto do PhysioNet.

    No LUDB o arquivo de anotacoes tem a derivacao como extensao (ex.: "1.ii").
    """
    if data_dir is not None:
        base = str(resolve_records_dir(data_dir) / record_id)
        return wfdb.rdrecord(base), wfdb.rdann(base, lead)

    return (
        wfdb.rdrecord(record_id, pn_dir=config.LUDB_PN_DIR),
        wfdb.rdann(record_id, lead, pn_dir=config.LUDB_PN_DIR),
    )


def normalize_rhythm(rhythm: str) -> str:
    """Padroniza a string de ritmo para agrupamento e comparacao.

    Converte para minusculas, colapsa espacos e remove a pontuacao final,
    de modo que "Sinus bradycardia." e "sinus  bradycardia" virem a mesma
    chave.
    """
    texto = re.sub(r"\s+", " ", str(rhythm)).strip()
    return texto.rstrip(".").strip().lower()


def parse_rhythm(comments: Iterable[str]) -> str | None:
    """Extrai o ritmo dos comentarios do header do registro.

    Returns:
        O ritmo normalizado, ou ``None`` se o campo nao existir no header.
    """
    for linha in comments:
        casamento = _RHYTHM_PATTERN.match(str(linha))
        if casamento:
            return normalize_rhythm(casamento.group(1))
    return None


def extract_waves(
    annotation,
) -> tuple[dict[str, list[tuple[int, int, int]]], Counter]:
    """Agrupa a anotacao de delineacao em triplas (onset, pico, offset).

    O LUDB marca cada onda com o padrao ``"(" -> <simbolo do pico> -> ")"``,
    onde o simbolo e ``p`` (onda P), ``N`` (complexo QRS) ou ``t`` (onda T).

    Returns:
        Tupla ``(ondas, estatisticas)``, com ``ondas`` no formato
        ``{"p": [(onset, pico, offset), ...], "N": [...], "t": [...]}``.
        As estatisticas contam triplas incompletas e picos com simbolo
        inesperado (``pico_<simbolo>``), que ficariam silenciosamente de fora.

    Note:
        O pico do complexo QRS e a posicao do pico R — e o que permite
        centrar a janela do LUDB no mesmo ponto usado no MIT-BIH.
    """
    ondas: dict[str, list[tuple[int, int, int]]] = {tipo: [] for tipo in WAVE_TYPES}
    estatisticas: Counter = Counter()

    amostras = annotation.sample
    simbolos = annotation.symbol
    total = len(simbolos)

    i = 0
    while i < total:
        if simbolos[i] != "(":
            i += 1
            continue

        tem_tripla = i + 2 < total and simbolos[i + 2] == ")"

        if tem_tripla and simbolos[i + 1] in ondas:
            ondas[simbolos[i + 1]].append(
                (int(amostras[i]), int(amostras[i + 1]), int(amostras[i + 2]))
            )
            i += 3
            continue

        if tem_tripla:
            # Tripla bem formada, mas com um simbolo de pico fora de p/N/t.
            estatisticas[f"pico_{simbolos[i + 1]}"] += 1
            i += 3
            continue

        estatisticas["triplas_incompletas"] += 1
        i += 1

    return ondas, estatisticas


# --------------------------------------------------------------------------
# Segmentacao
# --------------------------------------------------------------------------
def _preceding_wave(
    waves: Sequence[tuple[int, int, int]], reference: int
) -> tuple[int, int, int] | None:
    """Ultima onda que termina antes (ou em) ``reference``."""
    candidatas = [onda for onda in waves if onda[2] <= reference]
    return candidatas[-1] if candidatas else None


def _following_wave(
    waves: Sequence[tuple[int, int, int]], reference: int
) -> tuple[int, int, int] | None:
    """Primeira onda que comeca em (ou depois de) ``reference``."""
    for onda in waves:
        if onda[0] >= reference:
            return onda
    return None


def _fully_inside(onset: int, offset: int, start: int, end: int) -> bool:
    """Verifica se ``[onset, offset]`` cabe inteiro na janela ``[start, end)``."""
    return onset >= start and offset < end


def requires_p_wave(waves: dict[str, list[tuple[int, int, int]]]) -> bool:
    """Decide se a onda P deve ser exigida no criterio de inclusao do registro.

    A ausencia de onda P nao e uma lacuna de anotacao em fibrilacao e flutter
    atrial: e a propria definicao eletrofisiologica desses ritmos — o atrio
    nao despolariza de forma organizada e a onda P e substituida por ondas
    fibrilatorias. Exigi-la excluiria integralmente a classe arritmica do
    conjunto de teste.

    Returns:
        ``True`` se o registro tem ao menos uma onda P delimitada (e portanto
        a P deve estar presente e completa na janela), ``False`` caso o ritmo
        simplesmente nao produza ondas P.
    """
    return len(waves["p"]) > 0


def segment_record_beats(
    signal: np.ndarray,
    waves: dict[str, list[tuple[int, int, int]]],
    cfg: PreprocessConfig,
) -> tuple[list[dict], Counter]:
    """Segmenta os batimentos de um registro ja reamostrado.

    Cada complexo QRS anotado e um candidato a batimento. O candidato so e
    aceito se o proprio QRS e a onda T que o segue estiverem integralmente
    contidos na janela; a onda P precedente e exigida apenas quando o registro
    tem ondas P delimitadas (ver ``requires_p_wave``).

    Returns:
        Tupla ``(batimentos, estatisticas)``. Cada batimento e um dicionario
        com o segmento, as mascaras por estrutura e as posicoes relativas das
        ondas do proprio batimento.
    """
    n_amostras = len(signal)
    window_size = cfg.window_size
    batimentos: list[dict] = []
    estatisticas: Counter = Counter()

    # Em fibrilacao e flutter atrial nao existe onda P: o atrio nao despolariza
    # de forma organizada e a onda e substituida por ondas fibrilatorias. Exigir
    # P nesses registros descartaria 100% da classe arritmica.
    exigir_p = requires_p_wave(waves)

    # Para as mascaras usamos todas as ondas anotadas que caem na janela —
    # inclusive as de batimentos vizinhos — porque a mascara descreve tudo o
    # que os especialistas delimitaram naquele trecho.
    regioes_por_tipo = {
        tipo: [(onset, offset) for onset, _, offset in waves[tipo]]
        for tipo in WAVE_TYPES
    }

    for qrs_onset, qrs_peak, qrs_offset in waves["N"]:
        if cfg.center == "qrs-midpoint":
            centro = (qrs_onset + qrs_offset) // 2
        else:
            centro = qrs_peak

        inicio = centro - cfg.window_left
        fim = centro + cfg.window_right

        if inicio < 0 or fim > n_amostras:
            estatisticas["descartado_borda"] += 1
            continue

        if not _fully_inside(qrs_onset, qrs_offset, inicio, fim):
            estatisticas["descartado_qrs_incompleto"] += 1
            continue

        onda_p = _preceding_wave(waves["p"], qrs_onset)
        tem_p = onda_p is not None and _fully_inside(
            onda_p[0], onda_p[2], inicio, fim
        )
        if exigir_p and not tem_p:
            estatisticas["descartado_sem_p"] += 1
            continue

        onda_t = _following_wave(waves["t"], qrs_offset)
        if onda_t is None or not _fully_inside(onda_t[0], onda_t[2], inicio, fim):
            estatisticas["descartado_sem_t"] += 1
            continue

        mascaras = {
            tipo: signals.build_region_mask(regioes_por_tipo[tipo], inicio, window_size)
            for tipo in WAVE_TYPES
        }

        if not tem_p:
            estatisticas["aceito_sem_onda_p"] += 1

        batimentos.append(
            {
                "segmento": signal[inicio:fim],
                "mascara_p": mascaras["p"],
                "mascara_qrs": mascaras["N"],
                "mascara_t": mascaras["t"],
                # Batimentos sem onda P recebem (-1, -1): a estrutura nao existe
                # naquele ritmo, e o valor sentinela evita confundir com uma
                # regiao degenerada em 0.
                "regiao_p": (
                    (onda_p[0] - inicio, onda_p[2] - inicio) if tem_p else (-1, -1)
                ),
                "regiao_qrs": (qrs_onset - inicio, qrs_offset - inicio),
                "regiao_t": (onda_t[0] - inicio, onda_t[2] - inicio),
                "tem_onda_p": tem_p,
                "r_peak": qrs_peak,
            }
        )

    estatisticas["aceitos"] = len(batimentos)
    estatisticas["candidatos"] = len(waves["N"])
    return batimentos, estatisticas


def load_record_beats(
    record_id: str,
    cfg: PreprocessConfig | None = None,
) -> tuple[list[dict], str | None, Counter]:
    """Processa um unico registro do LUDB, do download a segmentacao.

    Returns:
        Tupla ``(batimentos, ritmo, estatisticas)``.

    Raises:
        ValueError: se a derivacao alvo nao existir no registro.
    """
    cfg = cfg or PreprocessConfig()
    record, annotation = _read_record(record_id, cfg.lead, cfg.data_dir)

    sinal = signals.select_lead(record.p_signal, record.sig_name, cfg.lead)
    if sinal is None:
        raise ValueError(
            f"Registro {record_id} nao possui a derivacao {cfg.lead} "
            f"(canais disponiveis: {record.sig_name})"
        )

    ritmo = parse_rhythm(record.comments or [])
    ondas, estatisticas_anotacao = extract_waves(annotation)

    sinal_reamostrado, razao = signals.resample_signal(
        sinal, cfg.fs_original, cfg.fs_target
    )
    ondas_reamostradas = {
        tipo: [
            tuple(signals.rescale_positions(onda, razao)) for onda in ondas[tipo]
        ]
        for tipo in WAVE_TYPES
    }

    batimentos, estatisticas = segment_record_beats(
        sinal_reamostrado, ondas_reamostradas, cfg
    )
    estatisticas.update(estatisticas_anotacao)
    return batimentos, ritmo, estatisticas


# --------------------------------------------------------------------------
# Dataset completo
# --------------------------------------------------------------------------
def build_dataset(cfg: PreprocessConfig | None = None) -> dict:
    """Processa todos os registros e monta o dataset de batimentos do LUDB.

    Returns:
        Dicionario com ``series``, ``labels``, ``simbolos``, ``pacientes``,
        as mascaras e regioes de P/QRS/T, ``r_peaks`` e ``metadata``.
    """
    cfg = cfg or PreprocessConfig()

    segmentos: list[np.ndarray] = []
    mascaras: dict[str, list[np.ndarray]] = {"p": [], "qrs": [], "t": []}
    regioes: dict[str, list[tuple[int, int]]] = {"p": [], "qrs": [], "t": []}
    r_peaks: list[int] = []
    pacientes: list[str] = []
    ritmos: list[str] = []
    tem_onda_p: list[bool] = []

    ignorados: list[str] = []
    sem_ritmo: list[str] = []
    estatisticas_totais: Counter = Counter()
    ritmos_por_registro: dict[str, str] = {}

    for record_id in cfg.records:
        try:
            batimentos, ritmo, estatisticas = load_record_beats(record_id, cfg)
        except Exception as exc:  # registro ausente, sem derivacao, etc.
            logger.warning("Registro %s ignorado: %s", record_id, exc)
            ignorados.append(record_id)
            continue

        estatisticas_totais.update(estatisticas)

        if ritmo is None:
            logger.warning("Registro %s sem campo de ritmo no header", record_id)
            sem_ritmo.append(record_id)
            ignorados.append(record_id)
            continue

        ritmos_por_registro[record_id] = ritmo

        if not batimentos:
            logger.warning("Registro %s nao gerou batimentos validos", record_id)
            continue

        logger.info(
            "Registro %s: %d/%d batimentos validos | ritmo: %s",
            record_id,
            len(batimentos),
            estatisticas["candidatos"],
            ritmo,
        )

        for batimento in batimentos:
            segmentos.append(batimento["segmento"])
            mascaras["p"].append(batimento["mascara_p"])
            mascaras["qrs"].append(batimento["mascara_qrs"])
            mascaras["t"].append(batimento["mascara_t"])
            regioes["p"].append(batimento["regiao_p"])
            regioes["qrs"].append(batimento["regiao_qrs"])
            regioes["t"].append(batimento["regiao_t"])
            r_peaks.append(batimento["r_peak"])
            tem_onda_p.append(batimento["tem_onda_p"])
            pacientes.append(record_id)
            ritmos.append(ritmo)

    if not segmentos:
        raise RuntimeError("Nenhum batimento extraido — verifique os registros.")

    series = np.stack(segmentos).astype(np.float32)
    if cfg.normalize:
        series = signals.z_normalize_per_beat(series)

    simbolos = np.asarray(ritmos)
    labels = np.asarray(
        [0 if ritmo in cfg.sinus_rhythms else 1 for ritmo in ritmos], dtype=np.int64
    )

    mascara_p = np.stack(mascaras["p"]).astype(np.int8)
    mascara_qrs = np.stack(mascaras["qrs"]).astype(np.int8)
    mascara_t = np.stack(mascaras["t"]).astype(np.int8)
    mascara_clinica = np.clip(mascara_p + mascara_qrs + mascara_t, 0, 1).astype(np.int8)

    ritmos_nao_mapeados = sorted(
        {ritmo for ritmo in ritmos_por_registro.values() if ritmo not in cfg.sinus_rhythms}
    )

    metadata = cfg.to_metadata()
    metadata["records_processados"] = sorted(set(pacientes), key=int)
    metadata["records_ignorados"] = ignorados
    metadata["records_sem_ritmo"] = sem_ritmo
    metadata["n_batimentos"] = int(series.shape[0])
    metadata["n_candidatos"] = int(estatisticas_totais["candidatos"])
    metadata["descartes"] = {
        chave: int(valor)
        for chave, valor in sorted(estatisticas_totais.items())
        if chave.startswith("descartado_")
    }
    anomalias_anotacao = {
        chave: int(valor)
        for chave, valor in sorted(estatisticas_totais.items())
        if chave.startswith("pico_") or chave == "triplas_incompletas"
    }
    if anomalias_anotacao:
        logger.warning(
            "Anotacoes fora do padrao '( pico )' encontradas: %s", anomalias_anotacao
        )
    metadata["anomalias_anotacao"] = anomalias_anotacao
    metadata["distribuicao_labels"] = {
        "0": int((labels == 0).sum()),
        "1": int((labels == 1).sum()),
    }
    metadata["ritmos_por_registro"] = ritmos_por_registro
    metadata["ritmos_classe_1"] = ritmos_nao_mapeados
    metadata["cobertura_media_mascara_clinica"] = float(mascara_clinica.mean())
    metadata["n_batimentos_sem_onda_p"] = int(len(tem_onda_p) - sum(tem_onda_p))

    return {
        "series": series,
        "labels": labels,
        "simbolos": simbolos,
        "pacientes": np.asarray(pacientes),
        "mascara_clinica": mascara_clinica,
        "mascara_p": mascara_p,
        "mascara_qrs": mascara_qrs,
        "mascara_t": mascara_t,
        "regiao_p": np.asarray(regioes["p"], dtype=np.int64),
        "regiao_qrs": np.asarray(regioes["qrs"], dtype=np.int64),
        "regiao_t": np.asarray(regioes["t"], dtype=np.int64),
        "r_peaks": np.asarray(r_peaks, dtype=np.int64),
        "tem_onda_p": np.asarray(tem_onda_p, dtype=np.int8),
        "metadata": metadata,
    }


def summarize(dataset: dict) -> str:
    """Gera um resumo textual do dataset processado."""
    series = dataset["series"]
    labels = dataset["labels"]
    simbolos = dataset["simbolos"]
    pacientes = dataset["pacientes"]
    metadata = dataset["metadata"]

    total = int(labels.size)
    n_normal = int((labels == 0).sum())
    n_arritmia = int((labels == 1).sum())
    candidatos = metadata.get("n_candidatos", 0)

    linhas = [
        "=" * 68,
        "RESUMO DO DATASET LUDB",
        "=" * 68,
        f"Shape das series      : {series.shape}",
        f"Registros processados : {len(np.unique(pacientes))}",
        f"Candidatos (QRS)      : {candidatos}",
        f"Batimentos validos    : {total}"
        + (f" ({100 * total / candidatos:.1f}% de aproveitamento)" if candidatos else ""),
        f"  classe 0 (normal)   : {n_normal} ({100 * n_normal / total:.2f}%)",
        f"  classe 1 (arritmia) : {n_arritmia} ({100 * n_arritmia / total:.2f}%)",
        "-" * 68,
        "Descartes por criterio de inclusao:",
    ]
    for motivo, contagem in metadata.get("descartes", {}).items():
        rotulo = motivo.replace("descartado_", "").replace("_", " ")
        linhas.append(f"  {rotulo:<22}: {contagem:>6}")

    sem_p = metadata.get("n_batimentos_sem_onda_p", 0)
    if sem_p:
        linhas += [
            "-" * 68,
            f"Batimentos sem onda P (ritmos que nao produzem P): {sem_p} "
            f"({100 * sem_p / total:.1f}%)",
        ]

    linhas += [
        "-" * 68,
        "Distribuicao por ritmo (propagado do registro para o batimento):",
    ]
    ritmo_ids, ritmo_counts = np.unique(simbolos, return_counts=True)
    for idx in np.argsort(-ritmo_counts):
        ritmo = ritmo_ids[idx]
        contagem = int(ritmo_counts[idx])
        classe = int(labels[simbolos == ritmo][0])
        linhas.append(
            f"  {ritmo:<28} -> classe {classe} : {contagem:>6} "
            f"({100 * contagem / total:.2f}%)"
        )

    linhas += [
        "-" * 68,
        "Cobertura media das mascaras (fracao da janela anotada):",
        f"  P                     : {float(dataset['mascara_p'].mean()):.2%}",
        f"  QRS                   : {float(dataset['mascara_qrs'].mean()):.2%}",
        f"  T                     : {float(dataset['mascara_t'].mean()):.2%}",
        f"  P+QRS+T (uniao)       : {float(dataset['mascara_clinica'].mean()):.2%}",
        "  ^ e a pontuacao esperada ao acaso para a metrica de fidelidade",
        "=" * 68,
    ]
    return "\n".join(linhas)
