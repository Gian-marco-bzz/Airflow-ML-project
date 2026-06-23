"""
Módulo de carga de datos.

Responsable de:
- Cargar parquet del S3 / local
- Validación básica de integridad
- Reportar estadísticas de carga (NaNs, shapes, tipos)
"""

import logging
from typing import Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_snapshot(parquet_path: str, validate: bool = True) -> pd.DataFrame:
    """
    Carga dataset parquet con validaciones opcionales.

    Args:
        parquet_path: Ruta al archivo traffic_ml_ready.parquet
        validate: Si True, reporta NaNs y shape

    Returns:
        DataFrame cargado (sin modificaciones)

    Raises:
        FileNotFoundError: Si archivo no existe
        Exception: Si parquet es inválido

    Example:
        >>> df = load_snapshot("data/raw/traffic_ml_ready.parquet")
        >>> print(df.shape)  # (646214, 35)
    """
    try:
        df = pd.read_parquet(parquet_path)

        if validate:
            n_rows, n_cols = df.shape
            n_nans = int(df.isna().sum().sum())
            logger.info(f"✓ Loaded {parquet_path}: {n_rows:,} rows × {n_cols} cols")

            if n_nans > 0:
                logger.warning(f"  → Found {n_nans:,} NaNs ({100*n_nans/n_rows/n_cols:.2f}% of data)")
                # Reportar por columna (top 5)
                col_nans = df.isna().sum().sort_values(ascending=False).head(5)
                for col, count in col_nans.items():
                    if count > 0:
                        logger.warning(f"     • {col}: {count:,} NaNs")
            else:
                logger.info("  → No NaNs detected")

        return df

    except FileNotFoundError:
        logger.error(f"File not found: {parquet_path}")
        raise
    except Exception as e:
        logger.error(f"Failed to load parquet: {e}")
        raise


def filter_invalid_timestamps(
    df: pd.DataFrame,
    timestamp_col: str
) -> Tuple[pd.DataFrame, int]:
    """
    Filtra filas con timestamp inválido (NaN después de coerce).

    Args:
        df: DataFrame con columna temporal
        timestamp_col: Nombre de la columna timestamp

    Returns:
        (df_cleaned: DataFrame sin rows inválidas, n_dropped: int)

    Example:
        >>> df['FECHA_ARCHIVO'] = pd.to_datetime(df['FECHA_ARCHIVO'], errors='coerce')
        >>> df_clean, n_dropped = filter_invalid_timestamps(df, 'FECHA_ARCHIVO')
        >>> print(f"Dropped {n_dropped} rows with invalid timestamps")
    """
    rows_before = len(df)

    # Elimina filas donde el timestamp es NaN (después de coerce).
    invalid_ts = df[timestamp_col].isna().sum()

    if invalid_ts > 0:
        logger.warning(f"Found {invalid_ts:,} rows with invalid timestamp in '{timestamp_col}'")
        df = df.loc[df[timestamp_col].notna()].copy()

    rows_after = len(df)
    n_dropped = rows_before - rows_after

    if n_dropped > 0:
        logger.info(f"Dropped {n_dropped:,} rows ({100*n_dropped/rows_before:.2f}% data loss)")

    return df, n_dropped


def filter_rows_with_any_nan(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Remove rows containing at least one NaN/NaT value.

    Args:
        df: Input DataFrame

    Returns:
        (df_cleaned, n_dropped)
    """
    rows_before = len(df)
    if rows_before == 0:
        return df, 0

    mask_complete = df.notna().all(axis=1)
    df_clean = df.loc[mask_complete].copy()
    n_dropped = rows_before - len(df_clean)

    if n_dropped > 0:
        logger.warning(
            f"Dropped {n_dropped:,} rows with at least one NaN/NaT "
            f"({100*n_dropped/rows_before:.2f}% data loss)"
        )

    return df_clean, n_dropped