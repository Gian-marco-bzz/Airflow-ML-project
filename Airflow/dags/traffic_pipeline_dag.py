from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException
from airflow.models import Variable
from airflow.sensors.filesystem import FileSensor

from pipeline.config import (
    TRAFFIC_FOLDER,
    POKE_INTERVAL_SECONDS, SENSOR_TIMEOUT_SECONDS,
    RETRIES, RETRY_DELAY_MINUTES, TASK_EXECUTION_TIMEOUT_MINUTES,
)

from pipeline.traffic_etl import (
    list_input_files, compute_new_files,
    stage_raw_traffic_parquet, transform_merge_and_write,
    update_processed_state,
)

STATE_VARIABLE = "tfm_traffic_processed_files"

default_args = {"retries": RETRIES, "retry_delay": timedelta(minutes=RETRY_DELAY_MINUTES)}

with DAG(
    dag_id="traffic_incremental_pipeline",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
    tags=["traffic", "incremental", "tfm"],
    params={"reset_state": False},
) as dag:

    wait_for_traffic_dir = FileSensor(
        task_id="wait_for_traffic_dir",
        fs_conn_id="fs_default",
        filepath="data/raw/traffic",
        poke_interval=POKE_INTERVAL_SECONDS,
        timeout=SENSOR_TIMEOUT_SECONDS,
        mode="reschedule",
    )

    wait_for_master_parquet = FileSensor(
        task_id="wait_for_master_parquet",
        fs_conn_id="fs_default",
        filepath="data/mapping/Maestra_Unificada_TFM_V2.parquet",
        poke_interval=POKE_INTERVAL_SECONDS,
        timeout=SENSOR_TIMEOUT_SECONDS,
        mode="reschedule",
    )

    wait_for_critical_parquet = FileSensor(
        task_id="wait_for_critical_parquet",
        fs_conn_id="fs_default",
        filepath="data/mapping/Canales_HC.parquet",
        poke_interval=POKE_INTERVAL_SECONDS,
        timeout=SENSOR_TIMEOUT_SECONDS,
        mode="reschedule",
    )

    @task
    def maybe_reset_state(**context):
        reset = context["params"].get("reset_state", False)
        if reset:
            try:
                Variable.delete(STATE_VARIABLE)
                print("Estado incremental eliminado correctamente.")
            except KeyError:
                print("No existía estado previo.")
        else:
            print("No se solicitó reset.")

    @task(execution_timeout=timedelta(minutes=TASK_EXECUTION_TIMEOUT_MINUTES))
    def scan_for_new_files() -> list[str]:
        all_files = list_input_files(TRAFFIC_FOLDER)
        print(">>> TRAFFIC_FOLDER:", TRAFFIC_FOLDER)
        print(">>> Archivos encontrados:", all_files)

        if not all_files:
            raise AirflowSkipException("No hay archivos .xlsx/.xls/.csv en raw/traffic")

        new_files = compute_new_files(all_files)
        print(">>> Archivos nuevos:", new_files)

        if not new_files:
            raise AirflowSkipException("No hay archivos nuevos para procesar")

        return new_files

    @task(execution_timeout=timedelta(minutes=TASK_EXECUTION_TIMEOUT_MINUTES))
    def stage_raw(files: list[str]) -> str:
        return stage_raw_traffic_parquet(files)

    @task(execution_timeout=timedelta(minutes=TASK_EXECUTION_TIMEOUT_MINUTES))
    def transform_and_write(staged_path: str) -> dict:
        return transform_merge_and_write(staged_path)

    @task
    def persist_state(files: list[str]) -> int:
        return update_processed_state(files)

    reset_task = maybe_reset_state()
    files = scan_for_new_files()
    staged = stage_raw(files)
    outputs = transform_and_write(staged)
    persisted = persist_state(files)

    wait_for_traffic_dir >> reset_task
    wait_for_master_parquet >> reset_task
    wait_for_critical_parquet >> reset_task

    reset_task >> files >> staged >> outputs >> persisted