"""Constantes e caminhos padrao do projeto.

Centraliza os parametros do pre-processamento para que treino (MIT-BIH) e
teste (LUDB) compartilhem exatamente a mesma definicao de janela de batimento.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Caminhos
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MITBIH_RAW_DIR = RAW_DIR / "mitdb"
MITBIH_PROCESSED_FILE = PROCESSED_DIR / "mitbih_beats.npz"

LUDB_RAW_DIR = RAW_DIR / "ludb"
LUDB_PROCESSED_FILE = PROCESSED_DIR / "ludb_beats.npz"

# --------------------------------------------------------------------------
# MIT-BIH Arrhythmia Database
# --------------------------------------------------------------------------
MITBIH_PN_DIR = "mitdb"
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

#: Simbolos mapeados para a classe 0 (nao-arritmia). Todo simbolo de batimento
#: que nao estiver aqui recebe rotulo 1 (arritmia).
#:
#: Por padrao usamos apenas 'N' (batimento normal), conforme a definicao do
#: projeto. Para o agrupamento AAMI da classe N, use NORMAL_SYMBOLS_AAMI.
NORMAL_SYMBOLS = frozenset(["N"])

#: Agrupamento AAMI EC57 da classe "N": normal, bloqueios de ramo esquerdo e
#: direito, escape atrial e escape nodal.
NORMAL_SYMBOLS_AAMI = frozenset(["N", "L", "R", "e", "j"])

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

#: Ritmos de origem sinusal, mapeados para a classe 0 (nao-arritmia).
#: Bradicardia e taquicardia sinusais sao anomalias de FREQUENCIA, nao de
#: morfologia do batimento: uma CNN treinada no MIT-BIH para morfologia
#: beat-to-beat classificaria esses batimentos como normais. Manter todos os
#: ritmos sinusais na classe 0 alinha a semantica das duas bases.
LUDB_SINUS_RHYTHMS = frozenset([
    "sinus rhythm",
    "sinus bradycardia",
    "sinus tachycardia",
    "sinus arrhythmia",
    "irregular sinus rhythm",
])

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
