import pandas as pd


def add_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Crea variables temporales a partir del timestamp.

    Agrega:
    - hour: hora del día
    - day_of_week: día de la semana (0=lunes, 6=domingo)
    - month: mes del año
    """
    out = df.copy()

    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    out["hour"] = out[timestamp_col].dt.hour
    out["day_of_week"] = out[timestamp_col].dt.dayofweek
    out["month"] = out[timestamp_col].dt.month

    # Elimina el timestamp crudo para evitar memorizar fechas exactas.
    out = out.drop(columns=[timestamp_col], errors="ignore")

    return out