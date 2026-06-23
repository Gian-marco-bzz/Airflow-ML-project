## DAGs
- `dags/master_pipeline_dag.py`: actualiza la maestra mensualmente y publica `Maestra_Unificada_TFM_V2.xlsx`.
- `dags/traffic_pipeline_dag.py`: procesa tráfico incrementalmente y genera:
  - `traffic_enriched.*`
  - `traffic_ml_ready.*` (listo para ML)

## Datos
- `data/raw/traffic/`: CSVs entrantes
- `data/raw/mensual/`: excels mensuales para construir/actualizar la maestra
- `data/mapping/`: maestra, críticos, rescates
- `data/processed/`: outputs parquet/xlsx
- `data/state/`: estado incremental