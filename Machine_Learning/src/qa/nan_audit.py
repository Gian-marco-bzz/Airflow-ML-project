"""Validaciones QA de seguridad NaN/Inf en el flujo de preprocesamiento."""

import numpy as np
import pandas as pd

from src.utils.io import load_config, get_logger
from src.data.load_data import load_snapshot, filter_invalid_timestamps, filter_rows_with_any_nan
from src.data.create_target import create_binary_target
from src.utils.preprocessing import Preprocessor

logger = get_logger(__name__)


def run_nan_audit(config_path: str = "config/config.yaml") -> int:
    """Ejecuta auditoria NaN/Inf sobre datos y preprocesamiento."""
    logger.info("=" * 80)
    logger.info("AUDITORIA NaN/Inf - PIPELINE")
    logger.info("=" * 80)

    cfg = load_config(config_path)

    input_path = cfg["paths"]["input_parquet"]
    ts_col = cfg["schema"]["timestamp_col"]
    rx_col = cfg["schema"]["rx_util_col"]
    tx_col = cfg["schema"]["tx_util_col"]
    entity_col = cfg["schema"].get("entity_col", "NODE NAME")
    id_like_cols = cfg["schema"].get("id_like_cols", [])
    threshold = cfg["modeling"]["risk_threshold"]
    horizon = cfg["modeling"]["horizon_days"]
    min_freq = cfg["modeling"]["min_category_freq"]
    drop_leakage = cfg["modeling"].get("drop_cols_leakage", [])

    errors = []

    try:
        df = load_snapshot(input_path, validate=True)
        rows_raw, cols_raw = df.shape
        nans_raw_total = int(df.isna().sum().sum())

        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df_clean, n_dropped_ts = filter_invalid_timestamps(df, ts_col)
        rows_after_ts = len(df_clean)

        df_clean, _ = filter_rows_with_any_nan(df_clean)

        df_target = create_binary_target(
            df_clean,
            rx_util_col=rx_col,
            tx_util_col=tx_col,
            threshold=threshold,
            target_col="overutilization_risk",
            horizon_days=horizon,
            date_col=ts_col,
            entity_col=entity_col,
        )

        drop_cols = [rx_col, tx_col, entity_col, "Porcentaje de Utilizacion"] + id_like_cols + drop_leakage
        df_fe = df_target.drop(columns=drop_cols, errors="ignore")

        y = df_fe["overutilization_risk"].astype(int)
        X = df_fe.drop(columns=["overutilization_risk"], errors="ignore")

        dt_cols = X.select_dtypes(
            include=["datetime", "datetimetz", "datetime64[ns]", "datetime64[ns, UTC]"]
        ).columns.tolist()
        X = X.drop(columns=dt_cols, errors="ignore")

        if X.isna().sum().sum() > 0:
            errors.append("NaNs en X antes del preprocesamiento")

        split_idx = int(0.8 * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        prep = Preprocessor(min_category_freq=min_freq)
        prep.fit(X_train, y_train)
        X_train_clean = prep.transform(X_train)
        X_test_clean = prep.transform(X_test)

        if X_train_clean.isna().sum().sum() > 0:
            errors.append("NaNs en X_train despues de transform")
        if X_test_clean.isna().sum().sum() > 0:
            errors.append("NaNs en X_test despues de transform")

        X_all_clean = pd.concat([X_train_clean, X_test_clean], ignore_index=True)
        arr = X_all_clean.to_numpy(dtype=float, copy=False)
        inf_count = int(np.isinf(arr).sum())
        if inf_count > 0:
            errors.append("Infs encontrados despues de transform")

        logger.info(f"Crudo: {rows_raw:,} filas x {cols_raw} columnas")
        logger.info(f"Tras timestamp: {rows_after_ts:,} filas (descartadas {n_dropped_ts:,})")
        logger.info(f"NaNs en crudo: {nans_raw_total:,}")
        logger.info(f"NaNs tras preprocesamiento: {int(X_all_clean.isna().sum().sum())}")
        logger.info(f"Infs tras preprocesamiento: {inf_count}")

        if errors:
            for err in errors:
                logger.error(err)
            return 1

        logger.info("[PASS] Auditoria NaN/Inf completada correctamente")
        return 0

    except Exception:
        logger.exception("[FAIL] Auditoria NaN/Inf fallo por excepcion")
        return 1
