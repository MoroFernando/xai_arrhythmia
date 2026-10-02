"""Leitura das categorias do CSV oficial LUDB 1.0.1, sem inferir diagnosticos."""

from __future__ import annotations

import csv
from pathlib import Path

import classification

OFFICIAL_CSV = "https://physionet.org/files/ludb/1.0.1/ludb.csv"


def read_clinical_csv(path: Path, records) -> tuple[dict, dict]:
    path = Path(path)
    if not path.is_file():
        raise classification.ClassificationError(
            f"CSV clinico ausente: {path}. Obtenha ludb.csv oficial 1.0.1: {OFFICIAL_CSV}"
        )
    rows = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, strict=True)
        required = {"ID", *classification.LUDB_CATEGORIES.values()}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise classification.ClassificationError(f"CSV LUDB sem colunas obrigatorias: {sorted(missing)}")
        if len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise classification.ClassificationError("CSV LUDB com colunas duplicadas.")
        for row in reader:
            if None in row or any(row.get(key) is None for key in required):
                raise classification.ClassificationError(f"Linha CSV incompleta/invalida perto da linha {reader.line_num}.")
            record_id = row["ID"].strip()
            if not record_id or record_id in rows:
                raise classification.ClassificationError(f"ID LUDB vazio ou duplicado: {record_id!r}")
            rows[record_id] = {
                "categories": {key: row[column].strip() for key, column in classification.LUDB_CATEGORIES.items()},
                "csv_fields": row,
            }
    missing_records = set(records) - rows.keys()
    if missing_records:
        raise classification.ClassificationError(f"Registros ausentes no CSV LUDB: {sorted(missing_records)}")
    return rows, {
        "official_url": OFFICIAL_CSV,
        "version": "1.0.1",
        "path": str(path),
        "sha256": classification.file_sha256(path),
        "empty_cell": "ausencia de diagnostico",
    }
