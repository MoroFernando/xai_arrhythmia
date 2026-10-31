"""Constantes e caminhos padrao do projeto.

Centraliza os parametros do pre-processamento para que treino (MIT-BIH) e
teste (LUDB) compartilhem exatamente a mesma definicao de janela de batimento.
"""

from __future__ import annotations

from pathlib import Path

# Politica explicita e versionada compartilhada por MIT-BIH e LUDB.
CLASSIFICATION_POLICY = "shared-extrasystoles-v1"

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
SPLITS_DIR = OUTPUTS_DIR / "splits"


def caminho_relativo(p: Path | str) -> str:
    """Caminho relativo a raiz do projeto, com barras normais.

    Os JSONs de saida (metricas, Grad-CAM, fidelidade) registram de onde vieram
    o checkpoint e o dataset. Gravar o caminho absoluto da maquina que rodou o
    script torna esses artefatos ilegiveis em qualquer outro computador — e
    ainda versiona o nome de usuario no repositorio publico. Aqui o caminho e
    normalizado para algo como ``outputs/models/cnn1d_mitbih/model.pt``.

    Se o caminho estiver fora do repositorio (um dataset num disco externo, por
    exemplo), devolve o absoluto: um caminho longo e melhor do que uma
    referencia relativa que nao resolve.
    """
    p = Path(p).resolve()
    try:
        return p.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return p.as_posix()

MITBIH_RAW_DIR = RAW_DIR / "mitdb"
MITBIH_PROCESSED_FILE = PROCESSED_DIR / f"mitbih_{CLASSIFICATION_POLICY}.npz"

LUDB_RAW_DIR = RAW_DIR / "ludb"
LUDB_PROCESSED_FILE = PROCESSED_DIR / f"ludb_{CLASSIFICATION_POLICY}.npz"

# --------------------------------------------------------------------------
# MIT-BIH Arrhythmia Database
# --------------------------------------------------------------------------
MITBIH_PN_DIR = "mitdb"
MITBIH_VERSION = "1.0.0"
MITBIH_FS = 360  # Hz

#: Os 48 registros do MIT-BIH excluindo 102, 104, 107 e 217 (pacientes com
#: marcapasso). Convencao amplamente adotada na literatura de classificacao de
#: batimentos. Os registros 102 e 104 tambem nao possuem a derivacao MLII.
MITBIH_RECORDS = [
    "100", "101", "103", "105", "106", "108", "109", "111", "112", "113",
    "114", "115", "116", "117", "118", "119", "121", "122", "123", "124",
    "200", "201", "202", "203", "205", "207", "208", "209", "210", "212",
    "213", "214", "215", "219", "220", "221", "222", "223", "228", "230",
    "231", "232", "233", "234",
]

#: Todos os 48 registros, para experimentos que nao aplicam o filtro acima.
MITBIH_RECORDS_ALL = sorted(
    MITBIH_RECORDS + ["102", "104", "107", "217"]
)

#: Simbolos de anotacao que NAO representam batimentos (mudanca de ritmo,
#: ruido, inicio/fim de segmento, artefatos). Devem ser descartados.
#: Ref.: https://archive.physionet.org/physiobank/database/html/mitdbdir/intro.htm#symbols
NON_BEAT_SYMBOLS = frozenset(["!", '"', "+", "[", "]", "x", "|", "~"])

# --------------------------------------------------------------------------
# Lobachevsky University Electrocardiography Database (LUDB)
# --------------------------------------------------------------------------
#: Diretorio de leitura direta (rdrecord/rdann): os registros ficam em data/.
LUDB_PN_DIR = "ludb/1.0.1/data"

#: Slug da base para wfdb.dl_database. Sem a versao: o wfdb resolve a versao
#: mais recente sozinho e a acrescenta a URL (passar "ludb/1.0.1" faria ele
#: montar ".../1.0.1/1.0.1/" e falhar com 404). O RECORDS dessa raiz lista os
#: registros como "data/1", entao os arquivos caem em "<dest_dir>/data/".
LUDB_DB_DIR = "ludb"
LUDB_VERSION = "1.0.1"

LUDB_FS = 500  # Hz

#: Os 200 registros do LUDB (um por paciente).
LUDB_RECORDS = [str(i) for i in range(1, 201)]

#: Derivacao usada no LUDB. Os nomes de canal da base sao minusculos ("ii"),
#: e o arquivo de anotacoes tem a derivacao como extensao (ex.: "1.ii").
LUDB_LEAD = "ii"

#: Simbolos de pico das tres estruturas delimitadas pelos especialistas.
#: Cada onda aparece como a tripla "(" -> <pico> -> ")" na anotacao.
LUDB_WAVE_SYMBOLS = {"p": "P", "N": "QRS", "t": "T"}

#: O LUDB nao rotula batimento a batimento: o ritmo e um diagnostico do
#: registro inteiro, gravado nos comentarios do header como
#: "#Rhythm: Sinus bradycardia.". Esse rotulo e propagado para todos os
#: batimentos segmentados daquele registro.
LUDB_RHYTHM_FIELD = "Rhythm"

#: O ritmo e preservado como metadado e nao participa da decisao do label.

#: Frequencia de amostragem alvo do projeto. O LUDB (500 Hz) e reamostrado
#: para esta taxa para que os batimentos fiquem na mesma escala temporal dos
#: do MIT-BIH, requisito da validacao cross-dataset.
TARGET_FS = MITBIH_FS

# --------------------------------------------------------------------------
# Segmentacao beat-to-beat
# --------------------------------------------------------------------------
#: Janela assimetrica centrada no pico R, em amostras (@360 Hz).
#: 90 amostras antes (~250 ms, cobre a onda P) e 144 depois (~400 ms, cobre a
#: onda T), totalizando 234 amostras (~650 ms).
WINDOW_LEFT = 90
WINDOW_RIGHT = 144
WINDOW_SIZE = WINDOW_LEFT + WINDOW_RIGHT

#: Derivacao usada em todo o projeto (o artigo trabalha somente com Lead II).
#: No MIT-BIH ela aparece rotulada como "MLII".
TARGET_LEAD = "MLII"

#: Nomes aceitos como equivalentes da derivacao II entre as bases.
LEAD_ALIASES = {
    "MLII": ("MLII", "II", "ML II"),
    "II": ("II", "MLII", "ML II"),
}

# --------------------------------------------------------------------------
# Etapa 3 — divisao inter-paciente e treino
# --------------------------------------------------------------------------
MODELS_DIR = OUTPUTS_DIR / "models"

#: Semente padrao para todo o pipeline de treino (numpy, random e torch).
RANDOM_SEED = 0

#: Limiar de decisao fixo do projeto. Nao e otimizado na validacao: o score
#: (sigmoid do logit) e binarizado sempre neste valor, em treino, validacao e
#: teste no LUDB. Ajustar o limiar por conjunto abriria uma forma indireta de
#: encaixar o modelo aos dados de teste.
DEFAULT_THRESHOLD = 0.5
