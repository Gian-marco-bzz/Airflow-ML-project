"""Validaciones QA para integridad de artefactos."""

from pathlib import Path

import joblib
import pandas as pd

from src.utils.io import get_logger, load_config

logger = get_logger(__name__)


def _check_file_exists_non_empty(path: Path, label: str) -> bool:
    if not path.exists():
        logger.error(f"[FAIL] Missing {label}: {path}")
        return False
    size = path.stat().st_size
    if size <= 0:
        logger.error(f"[FAIL] Empty {label}: {path}")
        return False
    logger.info(f"[OK] {label}: {path} ({size} bytes)")
    return True


def run_artifact_checks(config_path: str = "config/config.yaml", strict: bool = False) -> int:
    """Ejecuta validaciones de integridad sobre salidas ML."""
    try:
        cfg = load_config(config_path)

        model_dir = Path(cfg["paths"]["model_dir"])
        output_dir = Path(cfg["paths"]["output_dir"]) / "ml"
        preprocessor_path = model_dir / "preprocessor.pkl"
        model_path = model_dir / "xgboost.pkl"
        predictions_path = output_dir / "test_predictions.csv"

        checks = [
            _check_file_exists_non_empty(preprocessor_path, "preprocessor"),
            _check_file_exists_non_empty(model_path, "modelo entrenado"),
            _check_file_exists_non_empty(predictions_path, "predicciones de prueba"),
        ]

        if not all(checks):
            return 1

        preprocessor = joblib.load(preprocessor_path)
        model = joblib.load(model_path)
        logger.info("[OK] Pickles cargados correctamente")

        predictions = pd.read_csv(predictions_path)
        if {"actual", "probability", "prediction"}.issubset(predictions.columns):
            prob_col = "probability"
            pred_col = "prediction"
        elif {"actual", "predicted_probability", "predicted_risk"}.issubset(predictions.columns):
            prob_col = "predicted_probability"
            pred_col = "predicted_risk"
        else:
            logger.error(
                "[FAIL] El CSV de predicciones no coincide con los esquemas esperados: "
                "{'actual','probability','prediction'} or "
                "{'actual','predicted_probability','predicted_risk'}"
            )
            return 1

        if predictions.empty:
            logger.error("[FAIL] El CSV de predicciones esta vacio")
            return 1

        if predictions[prob_col].isna().any():
            logger.error(f"[FAIL] Se encontraron NaNs en la columna '{prob_col}'")
            return 1

        if not predictions[prob_col].between(0, 1).all():
            logger.error(f"[FAIL] Valores de '{prob_col}' fuera de [0, 1]")
            return 1

        unique_pred = set(predictions[pred_col].dropna().unique().tolist())
        if not unique_pred.issubset({0, 1}):
            logger.error(f"[FAIL] Etiquetas invalidas en '{pred_col}': {sorted(unique_pred)}")
            return 1

        logger.info(f"[OK] Filas de prediccion: {len(predictions):,}")
        logger.info(f"[OK] Tasa positiva: {predictions[pred_col].mean():.4f}")

        if strict:
            if not hasattr(preprocessor, "transform"):
                logger.error("[FAIL] El objeto preprocessor no expone transform()")
                return 1
            if not hasattr(model, "predict_proba"):
                logger.error("[FAIL] El objeto modelo no expone predict_proba()")
                return 1
            logger.info("[OK] Validaciones estrictas superadas")

        logger.info("[PASS] QA de artefactos completado correctamente")
        return 0

    except Exception:
        logger.exception("[FAIL] QA de artefactos fallo por excepcion")
        return 1
