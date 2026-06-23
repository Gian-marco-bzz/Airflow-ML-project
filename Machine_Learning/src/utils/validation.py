"""
Módulo de validación de integridad de datos.

Responsable de:
- Detectar NaNs, Infs, y valores inválidos
- Reportar estadísticas de limpieza de datos
- Validar pre-requisitos antes de entrenar modelos
"""

import logging
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def check_missing_values(df: pd.DataFrame, name: str = "DataFrame") -> Dict[str, int]:
    """
    Detecta y reporta NaNs por columna.
    
    Args:
        df: DataFrame a validar
        name: Identificador del DataFrame (para logs)
    
    Returns:
        Dict {columna: count_nans}
    
    Example:
        >>> nans = check_missing_values(X_train, name="X_train")
        >>> if nans:
        ...     logger.warning(f"Found NaNs: {nans}")
    """
    nans = df.isna().sum()
    nans = nans[nans > 0].to_dict()
    
    if nans:
        logger.warning(f"[{name}] Found NaNs: {nans}")
    else:
        logger.info(f"[{name}] ✓ No NaNs detected ({df.shape[0]:,} rows)")
    
    return nans


def check_infinite_values(X: np.ndarray, name: str = "Array") -> int:
    """
    Detecta valores infinitos (inf, -inf).
    
    Args:
        X: Array numérico a validar
        name: Identificador del array (para logs)
    
    Returns:
        Cuenta de infinitos encontrados
    
    Example:
        >>> X_array = np.array([[1, 2], [np.inf, 4]])
        >>> count = check_infinite_values(X_array, "X_test")
        >>> if count > 0:
        ...     logger.error(f"Found {count} infinite values")
    """
    count = int(np.isinf(X).sum())
    
    if count > 0:
        logger.error(f"[{name}] Found {count} infinite values (NaN->Inf conversion?)")
    else:
        logger.info(f"[{name}] ✓ No infinite values detected")
    
    return count


def validate_X_for_training(X: pd.DataFrame, y: pd.Series, name: str = "training set") -> Tuple[bool, list]:
    """
    Validación integral antes de fit/transform.
    
    Chequea:
    - No hay NaNs en X
    - No hay Infs en X
    - X e y tienen mismo largo
    - X no está vacío
    
    Args:
        X: Features (DataFrame)
        y: Target (Series)
        name: Identificador del dataset (para logs)
    
    Returns:
        (is_valid: bool, errors: list[str])
    
    Example:
        >>> valid, errors = validate_X_for_training(X_train, y_train)
        >>> if not valid:
        ...     for err in errors:
        ...         logger.error(err)
        ...     raise ValueError("Training set validation failed")
    """
    errors = []
    
    # Chequeo 1: vacío
    if X.empty:
        errors.append(f"[{name}] X is empty (shape={X.shape})")
    
    # Chequeo 2: NaNs
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        col_nans = X.isna().sum().sort_values(ascending=False).head(5)
        errors.append(f"[{name}] Found {nan_count} NaNs. Top: {col_nans.to_dict()}")
    
    # Chequeo 3: Infs
    X_numeric = X.select_dtypes(include=[np.number])
    if not X_numeric.empty:
        inf_count = int(np.isinf(X_numeric.to_numpy()).sum())
        if inf_count > 0:
            errors.append(f"[{name}] Found {inf_count} infinite values")
    
    # Chequeo 4: Largo X vs y
    if len(X) != len(y):
        errors.append(f"[{name}] len(X)={len(X)} != len(y)={len(y)}")
    
    # Chequeo 5: Largo y > 0
    if len(y) == 0:
        errors.append(f"[{name}] y is empty")
    
    is_valid = len(errors) == 0
    
    if is_valid:
        logger.info(f"[{name}] ✓ Validation passed ({X.shape})")
    else:
        logger.error(f"[{name}] Validation FAILED:")
        for err in errors:
            logger.error(f"  - {err}")
    
    return is_valid, errors


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_cols: list,
    name: str = "DataFrame"
) -> Tuple[bool, list]:
    """
    Valida que DataFrame tenga columnas requeridas.
    
    Args:
        df: DataFrame a validar
        required_cols: Lista de columnas que deben existir
        name: Identificador (para logs)
    
    Returns:
        (is_valid: bool, errors: list[str])
    """
    errors = []
    missing = [c for c in required_cols if c not in df.columns]
    
    if missing:
        errors.append(f"[{name}] Missing columns: {missing}")
    
    if not errors:
        logger.info(f"[{name}] ✓ Schema validation passed ({len(df.columns)} cols)")
    else:
        logger.error(f"[{name}] Schema validation FAILED: {errors}")
    
    return len(errors) == 0, errors


def report_cleaning_stats(
    rows_before: int,
    rows_after: int,
    operation: str = "Cleaning"
) -> None:
    """
    Reporta estadísticas de limpieza (cuántas filas se droperon).
    
    Args:
        rows_before: Filas antes
        rows_after: Filas después
        operation: Nombre de la operación (para logs)
    
    Example:
        >>> rows_b = df.shape[0]
        >>> df = df.dropna()
        >>> rows_a = df.shape[0]
        >>> report_cleaning_stats(rows_b, rows_a, "Drop NaN rows")
        [INFO] Drop NaN rows: 1895 rows removed (0.29% data loss)
    """
    dropped = rows_before - rows_after
    pct = 100.0 * dropped / rows_before if rows_before > 0 else 0
    
    if dropped > 0:
        logger.info(f"{operation}: {dropped:,} rows removed ({pct:.2f}% data loss)")
    else:
        logger.info(f"{operation}: No rows removed")
