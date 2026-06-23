"""
Central configuration for the TFM pipelines.

Requisitos de este TFM:
- Entradas Excel en /data/raw/traffic y /data/raw/mensual
- Cabeceras en la fila 3 (header=2)
- Mapping Excel en /data/mapping: Maestra_Unificada_TFM, Canales_HC, Nombres_Rescatados
- Materializar mapping en Parquet y consumir preferentemente Parquet
- Salidas de ambos DAGs en Parquet
"""

from __future__ import annotations
import os

# Base paths inside the container
BASE_DATA_PATH = "/opt/airflow/data"

# Input folders
TRAFFIC_FOLDER = f"{BASE_DATA_PATH}/raw/traffic"
MENSUAL_FOLDER = f"{BASE_DATA_PATH}/raw/mensual"
MAPPING_FOLDER = f"{BASE_DATA_PATH}/mapping"

# Output folders
PROCESSED_FOLDER = f"{BASE_DATA_PATH}/processed"
OUTPUT_FOLDER = f"{BASE_DATA_PATH}/output"
STATE_FOLDER = f"{BASE_DATA_PATH}/state"

# -------------------------
# Excel reading convention
# -------------------------
# Cabecera en fila 3 => header row index = 2
EXCEL_HEADER_LINE = 3
EXCEL_HEADER_ROW = EXCEL_HEADER_LINE - 1
EXCEL_SHEET_NAME = 0

EXCEL_READ_KWARGS = {
    "sheet_name": EXCEL_SHEET_NAME,
    "header": EXCEL_HEADER_ROW,
}

# CSV (si vuelven a aparecer)
CSV_READ_KWARGS = {"sep": ",", "encoding": "utf-8", "low_memory": False}

# -------------------------
# Incremental processing
# -------------------------
# Mantener incremental pero False al primer arranque (lo modificarás luego)
INCREMENTAL_ENABLED = False

AIRFLOW_STATE_VARIABLE = "tfm_traffic_processed_files"
STATE_FILE = f"{STATE_FOLDER}/traffic_processed_files.json"

PROCESS_ONLY_LATEST = False
TRAFFIC_INPUT_EXTENSIONS = (".xlsx", ".xls")
MENSUAL_INPUT_EXTENSIONS = (".xlsx", ".xls")

# -------------------------
# Transformations (traffic)
# -------------------------
FACTOR_MBPS = 1.0

# Mantengo el nombre original por compatibilidad (había un typo COLS_MPBS)
COLS_MPBS = []
COLS_MBPS = COLS_MPBS  # alias

COLS_PCT = []
COL_TIMESTAMP = []

# Keys que se intentan usar para construir KEY_ID en tráfico
MERGE_KEY_CANDIDATES = [
    "DS_INSTALACION",
    "DS",
    "KEY_ID",
    "CIRCUITO",
    "CIRCUIT",
    "NODE_NAME",
    "NODE",
    "SERVICIO",
]

# -------------------------
# Master mapping files
# -------------------------
MASTER_FILE_BASE = "Maestra_Unificada_TFM.xlsx"
MASTER_FILE_CURRENT = "Maestra_Unificada_TFM_V2.xlsx"
CRITICAL_CHANNELS_FILE = "Canales_HC.xlsx"
RESCUED_NAMES_FILE = "Nombres_Rescatados.xlsx"

MASTER_KEY_COLUMN = "DS"  # key column inside master file

# -------------------------
# Mapping Parquet outputs (canonical)
# -------------------------
MAPPING_PARQUET_FOLDER = f"{MAPPING_FOLDER}"

MASTER_PARQUET_BASE = f"{MAPPING_PARQUET_FOLDER}/Maestra_Unificada_TFM.parquet"
MASTER_PARQUET_CURRENT = f"{MAPPING_PARQUET_FOLDER}/Maestra_Unificada_TFM_V2.parquet"
CRITICAL_CHANNELS_PARQUET = f"{MAPPING_PARQUET_FOLDER}/Canales_HC.parquet"
RESCUED_NAMES_PARQUET = f"{MAPPING_PARQUET_FOLDER}/Nombres_Rescatados.parquet"

# Master derivado mensual
IDENTIFIERS_PARQUET_PATTERN = f"{MAPPING_PARQUET_FOLDER}/Tabla_Maestra_Identificadores_{{run_tag}}.parquet"
MASTER_V2_MONTHLY_PARQUET_PATTERN = f"{MAPPING_PARQUET_FOLDER}/Maestra_Unificada_TFM_V2_{{run_tag}}.parquet"

# -------------------------
# Outputs (Traffic)
# -------------------------
WRITE_PARQUET = True
WRITE_EXCEL = False  # En este TFM el output requerido es Parquet

PARQUET_OUTPUT = f"{PROCESSED_FOLDER}/traffic_enriched.parquet"
ML_READY_PARQUET_OUTPUT = f"{PROCESSED_FOLDER}/traffic_ml_ready.parquet"

# (Opcional) Excel outputs si se reactiva WRITE_EXCEL
EXCEL_OUTPUT = f"{PROCESSED_FOLDER}/traffic_enriched.xlsx"
ML_READY_EXCEL_OUTPUT = f"{PROCESSED_FOLDER}/traffic_ml_ready.xlsx"

# Intermediate staging
STAGING_RAW_PARQUET = f"{PROCESSED_FOLDER}/_staging_raw_traffic.parquet"

# -------------------------
# Airflow tuning
# -------------------------
POKE_INTERVAL_SECONDS = 60
SENSOR_TIMEOUT_SECONDS = 7 * 24 * 60 * 60
RETRIES = 2
RETRY_DELAY_MINUTES = 5
TASK_EXECUTION_TIMEOUT_MINUTES = 60

# -------------------------
# Data quality
# -------------------------
MIN_MASTER_MATCH_RATE = 0.70