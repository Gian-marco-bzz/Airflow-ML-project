from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ============================================================
# HELPERS DE PARSEO ROBUSTO (por valor, no por nombre de columna)
# ============================================================

_NUMBER_RE = re.compile(r"[-+]?\d[\d.,]*")
_THROUGHPUT_RE = re.compile(r"([-+]?\d[\d.,]*)\s*(gbps|mbps|kbps|bps)\b", re.IGNORECASE)
_BYTES_RE = re.compile(r"([-+]?\d[\d.,]*)\s*(tb|gb|mb|kb|b)\b", re.IGNORECASE)


def _to_float(num_str: str) -> float:
    """
    Convierte strings numéricos soportando:
    - separador miles con coma: '1,234.56'
    - decimal con coma (sin punto): '14,51'
    - espacios
    """
    s = str(num_str).strip().replace(" ", "")
    if s == "":
        return 0.0

    # Si hay coma y punto, asumimos coma como separador de miles.
    if "," in s and "." in s:
        s = s.replace(",", "")
    # Si hay coma y NO hay punto, asumimos coma como separador decimal.
    elif "," in s and "." not in s:
        s = s.replace(",", ".")

    try:
        return float(s)
    except Exception:
        return 0.0


def _parse_first_number(valor: Any) -> float:
    """
    Extrae el primer número que aparezca en el valor y lo convierte a float.
    Si no encuentra, devuelve 0.0.
    """
    if pd.isna(valor):
        return 0.0
    s = str(valor).strip()
    if s == "":
        return 0.0

    m = _NUMBER_RE.search(s)
    return _to_float(m.group(0)) if m else 0.0


# ============================================================
# NORMALIZADORES (según tus reglas)
# ============================================================

def normalize_percent(valor: Any) -> float:
    """
    Convierte '34 %' o '34%' a 0.34
    Ej: '14.51 %' -> 0.1451
    """
    if pd.isna(valor):
        return 0.0
    s = str(valor).strip().replace(" ", "").replace("%", "")
    if s == "":
        return 0.0
    return _to_float(s) / 100.0


def normalize_throughput_to_mbps(valor: Any) -> float:
    """
    Convierte throughput a Mbps soportando: bps, Kbps, Mbps, Gbps.
    - bps  -> / 1_000_000
    - Kbps -> / 1_000
    - Mbps -> x 1
    - Gbps -> x 1_000
    """
    if pd.isna(valor):
        return 0.0

    # Si ya viene numérico, asumimos que ya está en Mbps.
    if isinstance(valor, (int, float, np.number)) and not isinstance(valor, bool):
        return float(valor)

    s = str(valor).strip()
    if s == "":
        return 0.0

    m = _THROUGHPUT_RE.search(s)
    if not m:
        # Si no detecta unidad, intenta parsear el número y asumir Mbps.
        return _parse_first_number(s)

    num = _to_float(m.group(1))
    unit = m.group(2).lower()

    if unit == "gbps":
        return num * 1000.0
    if unit == "mbps":
        return num
    if unit == "kbps":
        return num / 1000.0
    # bps
    return num / 1_000_000.0


def normalize_bytes_to_mb(valor: Any) -> float:
    """
    Convierte volumen a MB soportando: B, KB, MB, GB, TB.
    - B  -> / (1024*1024)
    - KB -> / 1024
    - MB -> x 1
    - GB -> x 1024
    - TB -> x 1024*1024
    """
    if pd.isna(valor):
        return 0.0

    # Si ya viene numérico, asumimos que ya está en MB.
    if isinstance(valor, (int, float, np.number)) and not isinstance(valor, bool):
        return float(valor)

    s = str(valor).strip()
    if s == "":
        return 0.0

    m = _BYTES_RE.search(s)
    if not m:
        # Si no detecta unidad, intenta parsear el número y asumir MB.
        return _parse_first_number(s)

    num = _to_float(m.group(1))
    unit = m.group(2).lower()

    if unit == "tb":
        return num * 1024.0 * 1024.0
    if unit == "gb":
        return num * 1024.0
    if unit == "mb":
        return num
    if unit == "kb":
        return num / 1024.0
    # b
    return num / (1024.0 * 1024.0)


# ============================================================
# TRANSFORMACIÓN PRINCIPAL (USADA POR EL DAG)
# ============================================================

# Columnas objetivo (exactas) según tu especificación
PERCENT_COLS = {
    "Porcentaje de Utilizacion",
    "Received Percent Utilization",
    "Transmit Percent Utilization",
}

THROUGHPUT_COLS = {
    "Picos Recibidos bps",
    "Picos Transmitidos bps",
    "Velocidad",
    "Average Received bps",
    "Average Receive bps",   
    "Average Transmit bps",
}

BYTES_COLS = {
    "Total Bytes Recibidos",
    "Total Bytes Enviados",
}


def pipeline_append_and_merge(
    df_traffic: pd.DataFrame,
    df_maestra: pd.DataFrame,
    df_criticos: pd.DataFrame | None,
    key_traffic_col: str = "Servicio",
    key_maestra_col: str = "NODE NAME",
) -> pd.DataFrame:
    """
    Merge df_traffic con df_maestra usando key_traffic_col y key_maestra_col.
    Luego enriquece con df_criticos.
    Finalmente aplica transformaciones SOLO a las columnas especificadas:
    - Porcentajes -> [0..1]
    - Throughput -> Mbps
    - Bytes -> MB
    """

    # ---------- Validar existencia columnas ----------
    if key_traffic_col not in df_traffic.columns:
        log.error(f"Columnas disponibles en df_traffic: {df_traffic.columns.tolist()}")
        raise ValueError(f"Columna {key_traffic_col} no encontrada en df_traffic")

    if key_maestra_col not in df_maestra.columns:
        log.error(f"Columnas disponibles en df_maestra: {df_maestra.columns.tolist()}")
        raise ValueError(f"Columna {key_maestra_col} no encontrada en df_maestra")

    # ---------- Merge principal ----------
    df_merged = pd.merge(
        df_traffic,
        df_maestra,
        left_on=key_traffic_col,
        right_on=key_maestra_col,
        how="left",
    )

    # ---------- Enriquecer con df_criticos ----------
    if df_criticos is not None and not df_criticos.empty:
        if key_traffic_col in df_criticos.columns:
            df_merged = pd.merge(
                df_merged,
                df_criticos,
                on=key_traffic_col,
                how="left",
            )

    # ---------- Transformaciones POST-MERGE (solo columnas objetivo) ----------
    # Porcentajes
    for col in PERCENT_COLS:
        if col in df_merged.columns:
            log.info(f"Normalizando porcentaje en columna {col}")
            df_merged[col] = df_merged[col].apply(normalize_percent).astype(float)

    # Throughput -> Mbps
    for col in THROUGHPUT_COLS:
        if col in df_merged.columns:
            log.info(f"Normalizando throughput a Mbps en columna {col}")
            df_merged[col] = df_merged[col].apply(normalize_throughput_to_mbps).astype(float)

    # Bytes -> MB
    for col in BYTES_COLS:
        if col in df_merged.columns:
            log.info(f"Normalizando volumen a MB en columna {col}")
            df_merged[col] = df_merged[col].apply(normalize_bytes_to_mb).astype(float)

    return df_merged


def make_ml_ready(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """
    Prepara el DataFrame para ML:
    - Convierte strings tipo 'nan', 'None', etc. a NaN reales.
    - Mantiene columnas datetime como datetime.
    - Convierte columnas object a texto/categoría según cardinalidad.
    - Rellena NaNs numéricos con mediana.
    """
    out = df.copy()

    # --- 1. Convertir strings que representan NaN a NaN reales ---
    out = out.replace(
        ["nan", "NaN", "None", "NULL", "null", ""],
        np.nan
    )

    if "CANAL_CRITICO" in out.columns:
        out["CANAL_CRITICO"] = (out["CANAL_CRITICO"] == "SÍ").astype(int)

    # --- 2. Intentar convertir columnas object a datetime ---
    for c in out.columns:
        if out[c].dtype == "object":
            try:
                converted = pd.to_datetime(out[c], errors="raise")
                out[c] = converted
            except:
                pass

    # --- 3. Convertir texto a categoría si cardinalidad baja ---
    for c in out.select_dtypes(include=["object"]).columns:
        out[c] = out[c].astype(str)
        if out[c].nunique() < 5000:
            out[c] = out[c].astype("category")

    # --- 4. Rellenar NaNs en numéricos ---
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].fillna(out[num_cols].median())

    return out