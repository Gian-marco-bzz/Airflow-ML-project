"""
Módulo de creación del target.

Responsable de:
- Definir evento de riesgo: max(Rx_util, Tx_util) >= threshold
- Construir label futuro: ¿evento en (t, t+H] para misma entidad?
- Evitar data leakage: evento actual no entra en label futuro
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _coerce_datetime(series: pd.Series) -> tuple:
    """
    Convierte serie a datetime con validación de NaNs.

    Args:
        series: Columna temporal (string o datetime)

    Returns:
        (series_datetime: pd.Series, n_nans: int)

    Example:
        >>> ts_col, n_invalid = _coerce_datetime(df['FECHA_ARCHIVO'])
        >>> if n_invalid > 0:
        ...     logger.warning(f"Found {n_invalid} invalid timestamps")
    """
    ts = pd.to_datetime(series, errors="coerce")
    n_nans = ts.isna().sum()

    if n_nans > 0:
        logger.warning(
            f"_coerce_datetime: Found {n_nans:,} invalid timestamps "
            f"({100*n_nans/len(series):.2f}%)"
        )

    return ts, n_nans


def _future_window_label_by_entity(
    event_now: pd.Series,
    event_dt: pd.Series,
    entity_series: pd.Series,
    horizon_days: int,
) -> pd.Series:
    """
    Construye y(t+H)=1 si el evento ocurre en los próximos H días
    para la MISMA entidad (NODE NAME).

    - No usa el evento actual en t para el label futuro: busca en (t, t+H].
    - Agrupa por la columna de entidad.
    """
    out = pd.Series(0, index=event_now.index, dtype="int64")

    if horizon_days <= 0:
        return event_now.astype(int)

    work = pd.DataFrame(
        {
            "_event_now": event_now.astype(int),
            "_event_dt": pd.to_datetime(event_dt, errors="coerce"),
            "_entity": entity_series.astype(str),
        },
        index=event_now.index,
    )

    work = work[work["_event_dt"].notna()].copy()
    if work.empty:
        return out

    for _, grp in work.groupby("_entity", sort=False):
        grp = grp.sort_values("_event_dt")

        dt_values = grp["_event_dt"].to_numpy(dtype="datetime64[ns]")
        idx_values = grp.index.to_numpy()

        positive_dates = grp.loc[
            grp["_event_now"] == 1, "_event_dt"
        ].to_numpy(dtype="datetime64[ns]")

        if len(positive_dates) == 0:
            continue

        positive_dates = np.sort(positive_dates)

        for row_idx, current_dt in zip(idx_values, dt_values):
            right_limit = current_dt + np.timedelta64(horizon_days, "D")
            pos = np.searchsorted(positive_dates, current_dt, side="right")

            if pos < len(positive_dates) and positive_dates[pos] <= right_limit:
                out.loc[row_idx] = 1

    return out.astype(int)


def create_binary_target(
    df: pd.DataFrame,
    rx_util_col: str,
    tx_util_col: str,
    threshold: float,
    target_col: str = "overutilization_risk",
    *,
    horizon_days: int = 0,
    date_col: Optional[str] = None,
    entity_col: Optional[str] = None,
    current_event_col: str = "overutilization_event_now",
) -> pd.DataFrame:
    """
    Crea target binario con validación de NaNs.

    **Lógica:**

    - horizon_days <= 0 (legacy):
      target = 1 si max(Rx_util, Tx_util) >= threshold en tiempo t

    - horizon_days > 0 (futuro):
      1. Define evento actual: max(Rx, Tx) >= threshold
      2. Busca si evento ocurre en (t, t+H] para MISMA entidad
      3. Evita leakage: evento actual NO entra en label futuro

    **Validaciones:**
    - threshold debe estar en (0, 100]
    - horizon_days debe ser > 0 si dato_col se pasa
    - date_col y entity_col deben existir si horizon_days > 0
    - Detecta y reporta NaNs en columnas clave

    Args:
        df: DataFrame crudo con datos de interfaces
        rx_util_col: Nombre col "Received Percent Utilization"
        tx_util_col: Nombre col "Transmit Percent Utilization"
        threshold: Umbral % para considerar sobre-utilización (ej: 80.0)
        target_col: Nombre del target output (default: "overutilization_risk")
        horizon_days: Horizonte futuro dias (default: 0 → legacy, usar evento actual)
        date_col: Columna temporal (requerida si horizon_days > 0)
        entity_col: Columna de agrupación (requerida si horizon_days > 0)
        current_event_col: Nombre del evento actual (default: "overutilization_event_now")

    Returns:
        df con 2 columnas nuevas:
        - {current_event_col}: evento en t (0/1)
        - {target_col}: evento en futuro o actual (0/1)

    Raises:
        ValueError: Si threshold fuera de rango
        ValueError: Si horizon_days > 0 pero falta date_col/entity_col
        ValueError: Si columnas requeridas no existen en df

    Example:
        >>> df_target = create_binary_target(
        ...     df,
        ...     rx_util_col="Received Percent Utilization",
        ...     tx_util_col="Transmit Percent Utilization",
        ...     threshold=80.0,
        ...     horizon_days=30,
        ...     date_col="FECHA_ARCHIVO",
        ...     entity_col="NODE NAME"
        ... )
        >>> print(df_target[[target_col]].value_counts())
    """
    logger.info(
        f"create_binary_target: threshold={threshold}, horizon={horizon_days} days, "
        f"input shape={df.shape}"
    )

    # Validaciones
    if threshold <= 0 or threshold > 100:
        raise ValueError(
            f"threshold={threshold} debe estar en (0, 100], "
            f"típicamente 70-95 para redes"
        )

    if horizon_days < 0:
        raise ValueError(
            f"horizon_days={horizon_days} debe ser >= 0 "
            f"(0=legacy, >0=predicción futura)"
        )

    # Validar columnas requeridas
    required_cols = [rx_util_col, tx_util_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas faltantes en df: {missing}. "
            f"Disponibles: {list(df.columns)}"
        )

    out = df.copy()

    # Evento actual: sobre-utilización en t
    logger.info(f"Computing event (max(Rx, Tx) >= {threshold}%)...")

    out[current_event_col] = np.where(
        (out[rx_util_col] >= threshold) | (out[tx_util_col] >= threshold),
        1,
        0,
    ).astype(int)

    event_count = out[current_event_col].sum()
    event_pct = 100.0 * event_count / len(out) if len(out) > 0 else 0

    logger.info(
        f"Current event: {event_count:,} cases ({event_pct:.2f}% positive class)"
    )

    # Modo legacy: target = evento actual
    if horizon_days <= 0:
        logger.info("Legacy mode: target = current event (no future forecasting)")
        out[target_col] = out[current_event_col].astype(int)
        return out

    # Modo futuro: construir label con horizonte
    logger.info(f"Future mode: building label for {horizon_days} day horizon...")

    if date_col is None or date_col not in out.columns:
        raise ValueError(
            f"horizon_days > 0 requiere date_col, got={date_col}. "
            f"Columnas disponibles: {list(out.columns)}"
        )

    if entity_col is None or entity_col not in out.columns:
        raise ValueError(
            f"horizon_days > 0 requiere entity_col, got={entity_col}. "
            f"Columnas disponibles: {list(out.columns)}"
        )

    # Coercionar datetime y validar NaNs
    event_dt, n_invalid_ts = _coerce_datetime(out[date_col])

    if n_invalid_ts > 0:
        logger.warning(
            f"Found {n_invalid_ts:,} invalid timestamps in '{date_col}'. "
            f"Estas filas tendrán target=0"
        )

    # Construir etiqueta futura (agrupa por entidad)
    out[target_col] = _future_window_label_by_entity(
        event_now=out[current_event_col],
        event_dt=event_dt,
        entity_series=out[entity_col],
        horizon_days=horizon_days,
    )

    future_event_count = out[target_col].sum()
    future_event_pct = 100.0 * future_event_count / len(out) if len(out) > 0 else 0

    logger.info(
        f"Future label ({horizon_days} days): {future_event_count:,} cases "
        f"({future_event_pct:.2f}% positive class)"
    )

    # Post-creation validations
    if out[[target_col]].isna().any().any():
        n_nans = out[target_col].isna().sum()
        logger.warning(
            f"Found {n_nans:,} NaNs in {target_col} after creation. "
            f"Setting to 0 (conservative)"
        )
        out[target_col] = out[target_col].fillna(0).astype(int)

    logger.info(
        f"create_binary_target completed: output shape={out.shape}, "
        f"new cols=[{current_event_col}, {target_col}]"
    )

    return out