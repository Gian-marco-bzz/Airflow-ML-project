"""
DAG específico para realizar el merge y transformaciones de tráfico.
Usa las rutas ya definidas en config.py
Merge actualizado: Tráfico.Servicio ↔ Maestra.NODE NAME
"""

from __future__ import annotations
import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException
from airflow.models import Variable

# Importamos SOLO las variables que YA EXISTEN en tu config.py
from pipeline.config import (
    MAPPING_FOLDER,
    PROCESSED_FOLDER,
    RETRIES,
    RETRY_DELAY_MINUTES,
    MASTER_PARQUET_CURRENT,        # ← Ya existe: Maestra_Unificada_TFM_V2.parquet
    CRITICAL_CHANNELS_PARQUET,     # ← Ya existe: Canales_HC.parquet
)

# Importamos las funciones de transformations.py
from pipeline.transformations import (
    pipeline_append_and_merge,
    make_ml_ready
)

log = logging.getLogger(__name__)

# =============================================================================
# RUTAS
# =============================================================================

STAGING_RAW_PARQUET = os.path.join(PROCESSED_FOLDER, "_staging_raw_traffic.parquet")
MAESTRA_V2_PATH = MASTER_PARQUET_CURRENT
CRITICAL_PATH = CRITICAL_CHANNELS_PARQUET

PARQUET_OUTPUT = os.path.join(PROCESSED_FOLDER, "traffic_enriched.parquet")
ML_READY_PARQUET_OUTPUT = os.path.join(PROCESSED_FOLDER, "traffic_ml_ready.parquet")

COL_TIMESTAMP = ["START TIME", "END TIME"]

default_args = {
    'owner': 'airflow',
    'retries': RETRIES,
    'retry_delay': timedelta(minutes=RETRY_DELAY_MINUTES),
}

with DAG(
    dag_id='traffic_transform_dag',
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=['traffic', 'transform', 'merge'],
    description='Realiza el merge y transformaciones de tráfico usando la maestra V2',
) as dag:

    @task
    def verify_input_files():
        log.info("=" * 60)
        log.info("VERIFICANDO ARCHIVOS DE ENTRADA")
        log.info("=" * 60)
        missing = []
        for label, path in [
            ("Staging", STAGING_RAW_PARQUET),
            ("Maestra V2", MAESTRA_V2_PATH),
            ("Críticos", CRITICAL_PATH)
        ]:
            if os.path.exists(path):
                size = os.path.getsize(path) / (1024 * 1024)
                log.info(f"✅ {label}: {path} ({size:.2f} MB)")
            else:
                missing.append(f"{label}: {path}")
        if missing:
            error_msg = f"❌ FALTAN ARCHIVOS:\n" + "\n".join(missing)
            log.error(error_msg)
            raise FileNotFoundError(error_msg)
        return True

    @task
    def explore_data():
        log.info("=" * 60)
        log.info("EXPLORANDO DATOS")
        log.info("=" * 60)
        df_traffic = pd.read_parquet(STAGING_RAW_PARQUET).head(1000)
        df_maestra = pd.read_parquet(MAESTRA_V2_PATH).head(100)
        df_criticos = pd.read_parquet(CRITICAL_PATH).head(100)
        log.info(f"Tráfico: {len(df_traffic)} filas, Maestra: {len(df_maestra)} filas, Críticos: {len(df_criticos)} filas")
        return True

    @task
    def execute_merge_and_transform():
        log.info("=" * 70)
        log.info("🔄 INICIANDO MERGE Y TRANSFORMACIONES")
        log.info("=" * 70)

        df_traffic = pd.read_parquet(STAGING_RAW_PARQUET)
        df_maestra = pd.read_parquet(MAESTRA_V2_PATH)
        df_criticos = pd.read_parquet(CRITICAL_PATH)

        log.info(f"Tráfico: {len(df_traffic)} filas")
        log.info(f"Maestra: {len(df_maestra)} filas")
        log.info(f"Críticos: {len(df_criticos)} filas")

        # -------------------------
        # MERGE: nueva lógica
        # -------------------------
        log.info("🔄 Ejecutando pipeline_append_and_merge con nueva lógica de merge")
        df_enriched = pipeline_append_and_merge(
            df_traffic,
            df_maestra,
            df_criticos,
            key_traffic_col="Servicio",       # Tráfico.Servicio
            key_maestra_col="NODE NAME"       # Maestra.NODE NAME
        )
        log.info(f"✅ Dataset enriched: {len(df_enriched)} filas, {len(df_enriched.columns)} columnas")

        # -------------------------
        # Generar ML Ready
        # -------------------------
        log.info("🤖 Generando ML Ready")
        df_ml = make_ml_ready(
            df_enriched,
            timestamp_cols=[c.upper() for c in COL_TIMESTAMP],
            drop_cols=[]
        )
        log.info(f"✅ ML Ready: {len(df_ml)} filas, {len(df_ml.columns)} columnas")

        # -------------------------
        # Guardar resultados
        # -------------------------
        os.makedirs(PROCESSED_FOLDER, exist_ok=True)
        df_enriched.to_parquet(PARQUET_OUTPUT, index=False, compression='snappy')
        df_ml.to_parquet(ML_READY_PARQUET_OUTPUT, index=False, compression='snappy')

        log.info(f"📦 Enriched guardado → {PARQUET_OUTPUT}")
        log.info(f"📦 ML Ready guardado → {ML_READY_PARQUET_OUTPUT}")

        return {
            'enriched_rows': len(df_enriched),
            'enriched_cols': len(df_enriched.columns),
            'ml_rows': len(df_ml),
            'ml_cols': len(df_ml.columns),
        }

    @task
    def validate_results(merge_result: dict):
        log.info("=" * 60)
        log.info("🔍 VALIDANDO RESULTADOS")
        log.info("=" * 60)

        for path in [PARQUET_OUTPUT, ML_READY_PARQUET_OUTPUT]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"No se encuentra {path}")

        log.info("✅ Todos los archivos existen")
        log.info(f"Enriched rows: {merge_result['enriched_rows']}, ML Ready rows: {merge_result['ml_rows']}")
        return True

    # Asignar flujo
    verify = verify_input_files()
    explore = explore_data()
    merge = execute_merge_and_transform()
    validate = validate_results(merge)

    verify >> explore >> merge >> validate