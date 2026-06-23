from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def build_models(scale_pos_weight: float, random_state: int = 42) -> dict:
    """
    Crea estimadores BASE (sin "hardcodear" hiperparámetros que luego tuneas).

    Best practices:
    - Define aquí solo configuración estable / estructural:
      * random_state
      * métricas/objetivo
      * manejo de desbalance (class_weight / scale_pos_weight)
      * límites razonables (max_iter en LR)
    - NO fijar aquí parámetros que vayas a buscar en RandomizedSearchCV,
      para evitar duplicidad e inconsistencias.
    - Evitar oversubscription: NO pongas n_jobs=-1 aquí si CV ya paraleliza.
    """
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=random_state,
            # solver se tunea (o se fija en el espacio de búsqueda)
        ),
        "random_forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,  # importante: evita duplicar paralelismo con CV
        ),
        "xgboost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=1,  # importante: evita duplicar paralelismo con CV
        ),
    }
    return models