from __future__ import annotations
import os
import shutil
from datetime import datetime, timedelta
from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException
from airflow.sensors.filesystem import FileSensor

# Importamos rutas y nombres desde el archivo central de configuración
from pipeline.config import (
    MENSUAL_FOLDER, MAPPING_FOLDER,
    MASTER_FILE_BASE, MASTER_FILE_CURRENT
)

# --- CONFIGURACIÓN DE RUTAS ---
# Rutas de entrada (Fuentes en Excel)
MAESTRA_BASE_EXCEL = os.path.join(MAPPING_FOLDER, MASTER_FILE_BASE)
NOMBRES_RESCATADOS = os.path.join(MAPPING_FOLDER, 'Nombres_Rescatados.xlsx')

# Ruta de salida final consolidada (Forzamos .parquet para alto rendimiento)
nombre_v2_limpio = os.path.splitext(MASTER_FILE_CURRENT)[0]
CURRENT_MASTER_V2_PARQUET = os.path.join(MAPPING_FOLDER, f"{nombre_v2_limpio}.parquet")

default_args = {
    'owner': 'airflow',
    'retries': 2, 
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='tfm_master_monthly_pipeline',
    start_date=datetime(2026, 1, 1),
    schedule='@monthly',
    catchup=False,
    default_args=default_args,
    tags=['master', 'monthly', 'tfm'],
) as dag:

    # 1. Sensor: Espera a que la carpeta de datos del mes exista o tenga contenido
    wait_for_inputs = FileSensor(
        task_id='wait_for_monthly_inputs',
        fs_conn_id='fs_default',
        filepath='data/raw/mensual',
        mode='reschedule',
        poke_interval=60,
        timeout=60 * 60 * 24 # 1 día
    )

    @task
    def run_tag() -> str:
        """Genera el sufijo de tiempo para los archivos del mes (ej: 202602)"""
        return datetime.utcnow().strftime('%Y%m')

    @task
    def ensure_files():
        """Verifica que el archivo maestro inicial (Excel) esté en la carpeta mapping"""
        if not os.path.exists(MAESTRA_BASE_EXCEL):
            raise AirflowSkipException(f"No existe el Excel base requerido en: {MAESTRA_BASE_EXCEL}")
        print(f"✅ Excel base detectado: {MAESTRA_BASE_EXCEL}")
        return True

    @task
    def build_mapping_task(tag: str) -> str:
        """Llama a la lógica que escanea los Excel mensuales para crear el mapping de IDs"""
        from pipeline.mapping_builder import build_identifier_master
        
        # Definimos nombre base; la función interna añadirá el .parquet
        out_path_base = os.path.join(MAPPING_FOLDER, f'Tabla_Maestra_Identificadores_{tag}')
        
        actual_path = build_identifier_master(MENSUAL_FOLDER, out_path_base)
        
        if not actual_path or not os.path.exists(actual_path):
            raise ValueError(f"Fallo al crear el mapping en: {actual_path}")
            
        return actual_path

    @task
    def enrich_master_monthly_task(tag: str) -> str:
        """Cruce de la Maestra Base con los Nombres Rescatados para generar la Maestra V2 del mes"""
        from pipeline.enrichment import enrich_master
        
        # Ruta base de salida (la función enrich_master forzará la extensión .parquet)
        out_path_base = os.path.join(MAPPING_FOLDER, f"Maestra_Unificada_TFM_V2_{tag}")
        
        actual_parquet_path = enrich_master(MAESTRA_BASE_EXCEL, NOMBRES_RESCATADOS, out_path_base)
        
        print(f"✨ Master enriquecida creada: {actual_parquet_path}")
        return actual_parquet_path

    @task
    def promote_to_current(enriched_path: str) -> str:
        """Promociona el Parquet del mes como el archivo oficial 'Maestra_Unificada_TFM_V2.parquet'"""
        os.makedirs(os.path.dirname(CURRENT_MASTER_V2_PARQUET), exist_ok=True)
        
        # Copia el archivo generado (que ya es parquet) al nombre estándar para el DAG de tráfico
        shutil.copy2(enriched_path, CURRENT_MASTER_V2_PARQUET)
        
        print(f"✅ Maestra V2 oficial actualizada en: {CURRENT_MASTER_V2_PARQUET}")
        return CURRENT_MASTER_V2_PARQUET

    # --- DEFINICIÓN DEL FLUJO (DEPENDENCIAS) ---
    
    # Inicialización
    tag_value = run_tag()
    check_files = ensure_files()

    # Orden de ejecución
    wait_for_inputs >> check_files
    
    # Ejecución de procesos de transformación
    mapping_file = build_mapping_task(tag_value)
    enriched_file = enrich_master_monthly_task(tag_value)
    
    # Promoción final
    current_master = promote_to_current(enriched_file)

    # Relaciones lógicas
    check_files >> mapping_file
    check_files >> enriched_file >> current_master