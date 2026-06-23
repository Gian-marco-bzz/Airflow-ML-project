import time
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------
# Espacio de busqueda para tuning
# ---------------------------------------------------------------------
def get_param_distributions() -> dict:
    """
    Espacio de busqueda compacto y conservador para datasets grandes.
    Objetivo:
    - Mantener recall alto
    - Subir precision
    - Reducir overfitting
    - Controlar tiempos de entrenamiento
    """
    return {
        "logistic_regression": {
            "model__C": [0.05, 0.1, 0.2, 0.5, 1.0],
        },
        "random_forest": {
            "model__n_estimators": [120, 180, 250],
            "model__max_depth": [6, 8, 10],
            "model__min_samples_leaf": [20, 50, 100],
            "model__min_samples_split": [20, 50, 100],
            "model__max_features": ["sqrt", 0.5],
            "model__max_samples": [0.6, 0.8],
        },
        "xgboost": {
            "model__n_estimators": [200, 350, 500],
            "model__max_depth": [3, 4],
            "model__learning_rate": [0.03, 0.05, 0.08],
            "model__subsample": [0.7, 0.85],
            "model__colsample_bytree": [0.7, 0.85],
            "model__reg_lambda": [2.0, 5.0, 10.0],
            "model__reg_alpha": [0.0, 0.5, 1.0],
            "model__min_child_weight": [5, 10],
            "model__gamma": [0.0, 1.0],
        },
    }


# ---------------------------------------------------------------------
# Helper privado de metricas
# ---------------------------------------------------------------------
def _compute_metrics(y_true, y_pred, y_score=None) -> Dict[str, float]:
    out = {
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_score is not None:
        out["pr_auc"] = average_precision_score(y_true, y_score)
        try:
            out["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            pass

    return out


def _sort_temporal_data(X, y, date_series: pd.Series):
    dt = pd.to_datetime(date_series, errors="coerce")
    order = np.argsort(dt.fillna(pd.Timestamp.max).to_numpy())
    X_sorted = X.iloc[order].copy() if hasattr(X, "iloc") else X[order]
    y_sorted = y.iloc[order].copy() if hasattr(y, "iloc") else y[order]
    dt_sorted = dt.iloc[order].copy() if hasattr(dt, "iloc") else dt[order]
    return X_sorted, y_sorted, dt_sorted


# ---------------------------------------------------------------------
# Tuning de modelos con holdout interno
# ---------------------------------------------------------------------
def tune_models_randomized(
    X_train,
    y_train,
    preprocessor,
    models: dict,
    cv_folds: int,
    n_iter: int = 10,
    n_jobs: int = -1,
    random_state: int = 42,
    verbose: int = 1,
    pre_dispatch: str = "2*n_jobs",
    split_strategy: str = "stratified",
    date_series: Optional[pd.Series] = None,
    classification_threshold: float = 0.5,
):
    """
    Tuning de modelos manteniendo entrenamiento agil.

    split_strategy:
      - stratified: para datasets estaticos
      - temporal: ordena por fecha y usa TimeSeriesSplit + holdout final temporal

    Devuelve:
            best_name, best_estimator, best_score, best_params, best_threshold,
            model_results (lista de métricas por modelo candidato),
            estimators_by_model (estimadores ajustados en train completo)
    """
    cv_local_folds = min(cv_folds, 3)
    param_distributions = get_param_distributions()

    if split_strategy.lower() == "temporal":
        if date_series is None:
            raise ValueError(
                "split_strategy='temporal' requiere date_series alineada con X_train / y_train."
            )

        aligned_dates = pd.Series(date_series).loc[X_train.index]
        X_work, y_work, _ = _sort_temporal_data(X_train, y_train, aligned_dates)

        # Holdout temporal: ultimo 15% del tiempo.
        split_point = int(len(X_work) * 0.85)
        X_fit = X_work.iloc[:split_point].copy()
        y_fit = y_work.iloc[:split_point].copy()
        X_val = X_work.iloc[split_point:].copy()
        y_val = y_work.iloc[split_point:].copy()

        cv = TimeSeriesSplit(n_splits=max(2, cv_local_folds))
    else:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.15,
            random_state=random_state,
        )
        fit_idx, val_idx = next(splitter.split(X_train, y_train))

        X_fit = X_train.iloc[fit_idx] if hasattr(X_train, "iloc") else X_train[fit_idx]
        y_fit = y_train.iloc[fit_idx] if hasattr(y_train, "iloc") else y_train[fit_idx]

        X_val = X_train.iloc[val_idx] if hasattr(X_train, "iloc") else X_train[val_idx]
        y_val = y_train.iloc[val_idx] if hasattr(y_train, "iloc") else y_train[val_idx]

        cv = StratifiedKFold(
            n_splits=cv_local_folds,
            shuffle=True,
            random_state=random_state,
        )

    scoring = {
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "pr_auc": "average_precision",
    }

    best_name, best_estimator, best_score, best_params = None, None, -np.inf, None
    best_threshold = float(classification_threshold)
    model_results = []
    candidate_estimators = {}
    t_global = time.perf_counter()

    print(
        f"\n[TUNING] Start | models={list(models.keys())} | split={split_strategy} | cv={cv_local_folds} | n_iter={n_iter} | n_jobs={n_jobs}"
    )
    print(
        f"[TUNING] Holdout interno | fit={len(X_fit):,} rows | val={len(X_val):,} rows\n"
    )

    for name, model in models.items():
        if name not in param_distributions:
            raise ValueError(f"No param_distributions defined for model '{name}'")

        model = clone(model)

        try:
            if "random_state" in model.get_params(deep=False):
                model.set_params(random_state=random_state)
        except Exception:
            pass

        n_iter_local = n_iter

        if name == "logistic_regression":
            n_iter_local = min(n_iter, len(param_distributions[name]["model__C"]))
            try:
                if "solver" in model.get_params(deep=False):
                    model.set_params(solver="liblinear")
                if "max_iter" in model.get_params(deep=False):
                    model.set_params(max_iter=300)
            except Exception:
                pass

        elif name in {"random_forest", "xgboost"}:
            try:
                params = model.get_params(deep=False)
                if "n_jobs" in params:
                    model.set_params(n_jobs=1)
                if name == "xgboost":
                    if "tree_method" in params:
                        model.set_params(tree_method="hist")
                    if "verbosity" in params:
                        model.set_params(verbosity=0)
            except Exception:
                pass

        pipe = Pipeline(
            steps=[
                ("prep", preprocessor),
                ("model", model),
            ]
        )

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_distributions[name],
            n_iter=n_iter_local,
            scoring=scoring,
            refit="pr_auc",
            cv=cv,
            n_jobs=n_jobs,
            verbose=verbose,
            random_state=random_state,
            return_train_score=False,
            pre_dispatch=pre_dispatch,
            error_score="raise",
        )

        print(f"[TUNING] {name}: fitting RandomizedSearchCV...")
        t0 = time.perf_counter()

        with warnings.catch_warnings():
            warnings.simplefilter("default")
            search.fit(X_fit, y_fit)

        elapsed = time.perf_counter() - t0

        best_pipe = search.best_estimator_
        best_cv_score = float(search.best_score_)
        best_model_params = dict(search.best_params_)
        candidate_estimators[name] = clone(best_pipe)

        if not hasattr(best_pipe, "predict_proba"):
            raise ValueError(f"El modelo '{name}' no soporta predict_proba.")

        y_score_fit = best_pipe.predict_proba(X_fit)[:, 1]
        y_pred_fit = (y_score_fit >= best_threshold).astype(int)
        m_fit = _compute_metrics(y_fit, y_pred_fit, y_score_fit)

        y_score_val = best_pipe.predict_proba(X_val)[:, 1]
        y_pred_val = (y_score_val >= best_threshold).astype(int)
        m_val = _compute_metrics(y_val, y_pred_val, y_score_val)

        gap_f1 = m_fit["f1"] - m_val["f1"]

        print(f"[TUNING] {name}: done | best CV PR-AUC={best_cv_score:.4f} | time={elapsed:,.1f}s")
        print(f" best params={best_model_params}")
        print(
            f" Holdout Train: recall={m_fit['recall']:.4f}, precision={m_fit['precision']:.4f}, f1={m_fit['f1']:.4f}, pr_auc={m_fit.get('pr_auc', np.nan):.4f}"
        )
        print(
            f" Holdout Valid: recall={m_val['recall']:.4f}, precision={m_val['precision']:.4f}, f1={m_val['f1']:.4f}, pr_auc={m_val.get('pr_auc', np.nan):.4f}"
        )
        print(f" GAP: df1={gap_f1:+.4f}\n")

        model_results.append(
            {
                "model": name,
                "cv_pr_auc": best_cv_score,
                "holdout_train_recall": float(m_fit["recall"]),
                "holdout_train_precision": float(m_fit["precision"]),
                "holdout_train_f1": float(m_fit["f1"]),
                "holdout_train_pr_auc": float(m_fit.get("pr_auc", np.nan)),
                "holdout_valid_recall": float(m_val["recall"]),
                "holdout_valid_precision": float(m_val["precision"]),
                "holdout_valid_f1": float(m_val["f1"]),
                "holdout_valid_pr_auc": float(m_val.get("pr_auc", np.nan)),
                "f1_gap_train_minus_valid": float(gap_f1),
                "best_params": str(best_model_params),
                "classification_threshold": float(best_threshold),
            }
        )

        selection_tuple = (
            float(m_val.get("pr_auc", -np.inf)),
            float(m_val["f1"]),
            float(m_val["recall"]),
            float(m_val["precision"]),
            -abs(float(gap_f1)),
        )

        current_best_tuple = None if best_params is None else best_params.get("_selection_tuple")
        if current_best_tuple is None or selection_tuple > current_best_tuple:
            best_name = name
            best_estimator = clone(best_pipe)
            best_score = best_cv_score
            best_params = dict(best_model_params)
            best_params["classification_threshold"] = best_threshold
            best_params["_selection_tuple"] = selection_tuple

    if best_estimator is None:
        raise RuntimeError("No se pudo seleccionar un modelo ganador.")

    best_params.pop("_selection_tuple", None)

    estimators_by_model = {}
    print("[TUNING] Refitting best estimator for each model on full train...")
    for model_name, estimator in candidate_estimators.items():
        t_refit_model = time.perf_counter()
        estimator.fit(X_train, y_train)
        estimators_by_model[model_name] = estimator
        print(
            f"[TUNING] Refit {model_name} done in {time.perf_counter() - t_refit_model:,.1f}s"
        )

    if best_name not in estimators_by_model:
        raise RuntimeError(f"No se encontró el estimador ganador '{best_name}' tras refit.")

    best_estimator = estimators_by_model[best_name]
    print(
        f"[TUNING] Best model: {best_name} | best_cv_pr_auc={best_score:.4f} | threshold={best_threshold:.4f}."
    )
    print(f"[TUNING] Finished in {time.perf_counter() - t_global:,.1f}s\n")

    return (
        best_name,
        best_estimator,
        best_score,
        best_params,
        best_threshold,
        model_results,
        estimators_by_model,
    )