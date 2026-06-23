# Traffic Network Utilization - ML Pipeline

Proyecto de Machine Learning para predecir riesgo de sobre-utilizacion de interfaces de red en horizonte futuro.

Este repositorio esta disenado para:
- Entrenar y comparar modelos de riesgo (Logistic Regression, Random Forest, XGBoost).
- Generar artefactos de negocio y tecnicos (metricas, figuras, modelos, predicciones).
- Ejecutar QA de datos y artefactos para asegurar robustez.

## 1. Que problema resuelve

En operaciones de red, una interfaz con alta utilizacion sostenida puede provocar degradacion o caidas.
Este proyecto predice si una entidad (por ejemplo `NODE NAME`) tendra un evento de sobre-utilizacion en los proximos `H` dias.

Objetivo principal:
- Detectar riesgo con suficiente anticipacion para acciones preventivas (capacity planning, upgrades, mitigaciones).

## 2. Como esta definido el target

Definicion de evento actual:
- `event_now = 1` si `max(Rx_util, Tx_util) >= risk_threshold`

Definicion de target futuro:
- `target = 1` si ocurre al menos un evento en la ventana `(t, t+H]` para la misma entidad.
- `target = 0` en caso contrario.

Esto evita leakage porque el evento en `t` no se usa como target futuro.

Codigo clave:
- `src/data/create_target.py`

## 3. Estructura del proyecto

Estructura principal actual:

```text
ml_tfm/
├── config/
│   └── config.yaml
├── data/
├── outputs/
│   ├── eda/
│   └── ml/
├── scripts/
│   ├── train_pipeline.py
│   ├── eda_pipeline.py
│   ├── qa_pipeline.py
│   └── predict_batch.py
├── src/
│   ├── data/
│   ├── inference/
│   ├── modeling/
│   ├── qa/
│   ├── reporting/
│   └── utils/
├── requirements.txt
├── run_pipeline.py
└── README.md
```

## 4. Requisitos e instalacion

### 4.1. Crear entorno e instalar dependencias

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 4.2. Configurar proyecto

Archivo principal de configuracion:
- `config/config.yaml`
Campos importantes de `modeling`:
- `risk_threshold`: umbral de evento para construir target.
- `horizon_days`: horizonte futuro para etiquetado.
- `classification_threshold`: umbral de decision en prediccion.
- `split_strategy`: estrategia de split para tuning.
- `cost_fp`: coste relativo de un falso positivo (alerta innecesaria).
- `cost_fn`: coste relativo de un falso negativo (evento no detectado).
- `n_iter`, `n_jobs`: costo computacional del tuning.

## 5. Flujos de ejecucion

### 5.1. Runner unificado (recomendado)

```powershell
python run_pipeline.py --mode ml
python run_pipeline.py --mode eda
python run_pipeline.py --mode qa --qa-mode full --strict-qa
```

### 5.2. Entrenamiento ML directo

```powershell
python scripts/train_pipeline.py --config config/config.yaml
```

Opciones utiles:
- `--ml-only`: omite visualizaciones y artefactos adicionales.

### 5.3. EDA sin reentrenar

```powershell
python scripts/eda_pipeline.py --config config/config.yaml
```

### 5.4. QA

```powershell
python scripts/qa_pipeline.py --mode artifacts --strict
python scripts/qa_pipeline.py --mode nan
python scripts/qa_pipeline.py --mode full --strict
```

### 5.5. Inferencia batch

```powershell
python scripts/predict_batch.py --data data/raw/new_traffic.parquet --model outputs/ml/models/xgboost.pkl --output outputs/ml/predictions.csv
```

### 5.6. Regenerar solo figuras ML (sin reentrenar)

Si ya existen artefactos en `outputs/ml/metrics/` y `outputs/ml/models/`, puedes regenerar figuras comparativas y tablas de prediccion sin lanzar el entrenamiento completo:

```powershell
python scripts/regenerate_ml_figures.py
```

Este flujo actualiza principalmente:
- `model_metrics_comparison.png`
- `model_decision_frontier.png`
- `model_confusion_matrices.png`
- `operational_cost_comparison.png`
- `test_predictions.csv`
- `predictions.csv`

## 6. Que artefactos genera la pipeline ML

### 6.1. Modelos

En `outputs/ml/models/`:
- `preprocessor.pkl`
- `<best_model>.pkl`

### 6.2. Predicciones

En `outputs/ml/` y `outputs/ml/metrics/`:
- `test_predictions.csv`
- `predictions.csv` (version enriquecida para negocio, con columnas descriptivas + `risk_probability` y `risk_prediction`)

### 6.3. Metricas y comparativas

En `outputs/ml/metrics/`:
- `ml_metrics.csv`: metricas del modelo ganador en test.
- `classification_report.txt`: reporte de clasificacion.
- `model_results_by_model.csv`: resumen por modelo en tuning (CV + holdout).
- `test_metrics_by_model.csv`: comparativa de metricas en test para todos los modelos candidatos.
- `model_comparison.csv`: comparativa final resumida del modelo ganador.
- `threshold_metrics.csv`: recall/precision/F1 por umbral.
- `calibration_table.csv`: tabla de calibracion por bins.
- `operational_cost_by_model.csv`: comparativa de coste operativo estimado por modelo.
- `training_summary.json`: metadata de entrenamiento (modelo ganador, params, threshold).

### 6.4. Figuras

En `outputs/ml/figures/`:
- `pr_curve.png`
- `roc_curve.png`
- `confusion_matrix.png`
- `threshold_tradeoff.png`
- `calibration_curve.png`
- `model_metrics_comparison.png`
- `model_decision_frontier.png`
- `model_pr_overlay.png`
- `model_confusion_matrices.png`
- `operational_cost_comparison.png`

## 7. Como interpretar resultados (guia simple)

### 7.1. Para elegir modelo

Usa primero:
- `outputs/ml/metrics/test_metrics_by_model.csv`
- `outputs/ml/figures/model_metrics_comparison.png`
- `outputs/ml/figures/model_decision_frontier.png`
- `outputs/ml/figures/model_pr_overlay.png`
- `outputs/ml/figures/model_confusion_matrices.png`

Criterio recomendado:
- Si negocio prioriza no perder eventos criticos: maximizar `Recall` sujeto a precision minima.
- Si negocio prioriza reducir falsas alertas: maximizar `Precision` sujeto a recall minima.
- Si quieres equilibrio general: usar `F1` + `PR_AUC`.

Lectura rapida de los graficos de comparativa:
- `model_metrics_comparison.png`: vista compacta de Recall/Precision/F1/PR-AUC por modelo.
- `model_decision_frontier.png`: posiciona cada modelo en Precision vs Recall para visualizar trade-off.
- `model_pr_overlay.png`: compara las curvas PR completas (calidad de ranking por probabilidad).
- `model_confusion_matrices.png`: muestra TP/TN/FP/FN de cada modelo al mismo threshold.

### 7.2. Para elegir umbral

Usa:
- `outputs/ml/metrics/threshold_metrics.csv`
- `outputs/ml/figures/threshold_tradeoff.png`

Busca el threshold que cumple tu politica operativa.
Ejemplo:
- Politica A: `Recall >= 0.97` y precision maxima posible.
- Politica B: `Precision >= 0.90` y recall maxima posible.

### 7.3. Para validar calidad probabilistica

Usa:
- `outputs/ml/figures/calibration_curve.png`
- `outputs/ml/metrics/calibration_table.csv`

Si esta bien calibrado, una prediccion de 0.8 deberia corresponder aproximadamente a 80% de positivos observados en ese rango.

### 7.4. Para justificar decision economica/operativa

Usa:
- `outputs/ml/metrics/operational_cost_by_model.csv`

Este archivo calcula coste estimado por modelo usando:
- `cost_fp` (penalizacion por falso positivo)
- `cost_fn` (penalizacion por falso negativo)

Formula:
- `coste_total = FP * cost_fp + FN * cost_fn`

Recomendacion:
- Ajustar `cost_fn` y `cost_fp` en config segun impacto real de negocio.
- Repetir evaluacion si cambian las prioridades operativas.

## 8. Calidad y robustez

Este proyecto incorpora controles de calidad en varios niveles:
- Limpieza temprana de timestamps invalidos y NaNs.
- Validacion de integridad antes de entrenar.
- QA de artefactos y QA de NaN/Inf.
- Persistencia reproducible de configuracion y parametros.

Modulos de referencia:
- `src/utils/validation.py`
- `src/qa/artifacts.py`
- `src/qa/nan_audit.py`

## 9. Problemas frecuentes

### 9.1. La ejecucion tarda demasiado

Acciones:
- Bajar `n_iter` en `config/config.yaml`.
- Reducir `n_jobs` si hay contention de recursos.
- Ejecutar con `--ml-only` para omitir parte grafica/reporting.

### 9.2. Falla por memoria

Acciones:
- Reducir paralelismo (`n_jobs`).
- Revisar cardinalidad y columnas antes de entrenar.

### 9.3. Diferencias entre CV y test

Acciones:
- Revisar estrategia de split (`split_strategy`).
- Analizar posible drift temporal con artefactos EDA.

## 10. Flujo recomendado para equipos

1. Ajustar `config/config.yaml` segun objetivo operativo.
2. Ejecutar entrenamiento ML.
3. Revisar `test_metrics_by_model.csv` y figuras comparativas.
4. Definir threshold final con `threshold_metrics.csv`.
5. Correr QA final (`qa_pipeline.py --mode full --strict`).
6. Publicar modelo/preprocesador y versionar `training_summary.json`.

## 11. Notas finales

- El repositorio esta orientado a trazabilidad: cada decision de modelo y umbral debe apoyarse en CSV/figuras generadas.
- La seleccion de mejor modelo no debe basarse solo en una metrica aislada.
- En operaciones reales, recomienda monitorear metricas por periodo (semanal/mensual) para detectar degradacion o drift.
