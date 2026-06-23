"""
Task functions for the traffic pipeline.

Requisitos:
- Lee Excel .xlsx/.xls desde /data/raw/traffic con cabecera en fila 3 (header=2)
- Consume mapping preferentemente en PARQUET (materializado por el DAG mensual)
- Devuelve outputs en PARQUET (enriched + ML-ready)
- Mantener incremental, pero INCREMENTAL_ENABLED=False al primer arranque
"""

from __future__ import annotations

import glob
import logging
import os
from typing import List, Set

import pandas as pd

from pipeline.config import (
    TRAFFIC_FOLDER, MAPPING_FOLDER,
    MASTER_FILE_CURRENT, CRITICAL_CHANNELS_FILE,
    EXCEL_READ_KWARGS, CSV_READ_KWARGS,
    INCREMENTAL_ENABLED, PROCESS_ONLY_LATEST,
    STAGING_RAW_PARQUET,
    WRITE_PARQUET, WRITE_EXCEL,
    PARQUET_OUTPUT, EXCEL_OUTPUT,
    ML_READY_PARQUET_OUTPUT, ML_READY_EXCEL_OUTPUT,
    STATE_FILE, AIRFLOW_STATE_VARIABLE,
    MASTER_PARQUET_CURRENT, CRITICAL_CHANNELS_PARQUET,
)

from pipeline.state import (
    load_disk_state, save_disk_state,
    load_airflow_state, save_airflow_state,
)

from pipeline.utils import extraer_fecha_datetime
from pipeline.transformations import pipeline_append_and_merge, make_ml_ready

log = logging.getLogger(__name__)


def list_input_files(folder: str = TRAFFIC_FOLDER) -> List[str]:
    patterns = [
        os.path.join(folder, "*.xlsx"),
        os.path.join(folder, "*.xls"),
        os.path.join(folder, "*.csv"),
    ]
    files: List[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(files)


def _file_signature(path: str) -> str:
    st = os.stat(path)
    return f"{os.path.basename(path)}\n{int(st.st_mtime)}\n{st.st_size}"


def compute_new_files(all_files: List[str]) -> List[str]:
    if not INCREMENTAL_ENABLED:
        return all_files

    disk = load_disk_state(STATE_FILE)
    airflow = load_airflow_state(AIRFLOW_STATE_VARIABLE)
    processed: Set[str] = set(disk) | set(airflow)

    new_files = [f for f in all_files if _file_signature(f) not in processed]

    if PROCESS_ONLY_LATEST and new_files:
        new_files = [max(new_files, key=lambda p: os.stat(p).st_mtime)]

    return new_files


def _read_input_file(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path, **CSV_READ_KWARGS)
    return pd.read_excel(path, **EXCEL_READ_KWARGS)


def stage_raw_traffic_parquet(files: List[str], output_path: str = STAGING_RAW_PARQUET) -> str:
    if not files:
        raise ValueError("No hay archivos nuevos para procesar.")

    dfs = []
    for path in files:
        df = _read_input_file(path)
        fname = os.path.basename(path)
        df["SOURCE_FILE"] = fname
        df["FECHA_ARCHIVO"] = extraer_fecha_datetime(fname)
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_all.to_parquet(output_path, index=False)
    log.info("Staged raw traffic: %s rows=%s cols=%s", output_path, len(df_all), len(df_all.columns))
    return output_path


def _read_mapping_master() -> pd.DataFrame:
    """Lee SOLO el Parquet V2 de la maestra. No usa Excel como fallback."""
    if not os.path.exists(MASTER_PARQUET_CURRENT):
        raise FileNotFoundError(
            f"No se encuentra el Parquet V2 de la maestra: {MASTER_PARQUET_CURRENT}\n"
            "Ejecuta primero el DAG mensual (tfm_master_monthly_pipeline) para generarlo."
        )
    
    return pd.read_parquet(MASTER_PARQUET_CURRENT)

def _read_critical_channels() -> pd.DataFrame:
    """Lee SOLO el Parquet de canales críticos. No usa Excel como fallback."""
    if not os.path.exists(CRITICAL_CHANNELS_PARQUET):
        raise FileNotFoundError(
            f"No se encuentra el Parquet de canales críticos: {CRITICAL_CHANNELS_PARQUET}\n"
            "Ejecuta el script de conversión para generarlo."
        )
    
    return pd.read_parquet(CRITICAL_CHANNELS_PARQUET)

def transform_merge_and_write(staged_raw_path: str) -> dict:
    df_traffic = pd.read_parquet(staged_raw_path)

    df_maestra = _read_mapping_master()
    df_criticos = _read_critical_channels()

    df_enriched = pipeline_append_and_merge(df=df_traffic, df_maestra=df_maestra, df_criticos=df_criticos)

    df_ml = make_ml_ready(df_enriched, timestamp_cols=[], drop_cols=[])

    os.makedirs(os.path.dirname(PARQUET_OUTPUT), exist_ok=True)

    outputs = {
        "rows_enriched": int(len(df_enriched)),
        "rows_ml_ready": int(len(df_ml)),
        "parquet_enriched": None,
        "excel_enriched": None,
        "parquet_ml_ready": None,
        "excel_ml_ready": None,
    }

    if WRITE_PARQUET:
        df_enriched.to_parquet(PARQUET_OUTPUT, index=False)
        df_ml.to_parquet(ML_READY_PARQUET_OUTPUT, index=False)
        outputs["parquet_enriched"] = PARQUET_OUTPUT
        outputs["parquet_ml_ready"] = ML_READY_PARQUET_OUTPUT

    if WRITE_EXCEL:
        CHUNK_SIZE = 50000
        for out_path, df_to_write in [(EXCEL_OUTPUT, df_enriched), (ML_READY_EXCEL_OUTPUT, df_ml)]:
            with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
                for i in range(0, len(df_to_write), CHUNK_SIZE):
                    chunk = df_to_write.iloc[i: i + CHUNK_SIZE]
                    chunk.to_excel(
                        writer,
                        index=False,
                        header=(i == 0),
                        startrow=i,
                        sheet_name="Sheet1",
                    )
        outputs["excel_enriched"] = EXCEL_OUTPUT
        outputs["excel_ml_ready"] = ML_READY_EXCEL_OUTPUT

    log.info("Outputs written: %s", outputs)
    return outputs


def update_processed_state(files: List[str]) -> int:
    if not INCREMENTAL_ENABLED:
        return 0

    disk = load_disk_state(STATE_FILE)
    airflow = load_airflow_state(AIRFLOW_STATE_VARIABLE)
    processed: Set[str] = set(disk) | set(airflow)

    for f in files:
        processed.add(_file_signature(f))

    save_disk_state(STATE_FILE, processed)
    save_airflow_state(AIRFLOW_STATE_VARIABLE, processed)

    log.info("Updated processed state: %s entries", len(processed))
    return len(files)