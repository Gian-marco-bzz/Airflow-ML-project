"""
Pipeline principal de entrenamiento para el modelo de utilizacion de red.

Flujo:
1. Cargar y validar datos (auditoria NaN)
2. Crear target binario (evitar leakage)
3. Ingenieria de variables y preprocesamiento
4. Division train-test temporal
5. Ajuste de hiperparametros (RandomizedSearchCV)
6. Evaluacion del modelo en test
7. Generacion de artefactos de salida (CSV/PNG)

Uso:
    python scripts/train_pipeline.py

Codigos de salida:
    0 = Exito
    1 = Error
"""

import sys
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    auc,
    confusion_matrix,
    classification_report,
)

# Permite imports al ejecutar desde la raiz del repositorio
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_config, get_logger, ensure_dir
from src.utils.validation import (
    validate_X_for_training,
    report_cleaning_stats,
)
from src.data.load_data import (
    load_snapshot,
    filter_invalid_timestamps,
    filter_rows_with_any_nan,
)
from src.data.create_target import create_binary_target
from src.data.feature_engineering import add_time_features
from src.utils.preprocessing import Preprocessor
from src.modeling.models import build_models
from src.modeling.train import tune_models_randomized
from src.modeling.evaluate import compute_metrics, proba_to_class
from src.reporting.feature_importance import compute_feature_importance, plot_feature_importance

logger = get_logger(__name__)


def _to_serializable(value):
    """Convierte tipos numpy/pandas a tipos serializables para JSON."""
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value



def generate_model_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str,
    threshold: float,
) -> dict:
    """
    Genera y guarda curvas diagnosticas (PR, ROC, matriz de confusion).

    Args:
        y_true: Etiquetas reales (0/1)
        y_prob: Probabilidades predichas
        output_dir: Directorio donde guardar PNG

    Returns:
        Diccionario con rutas de imagenes generadas
    """
    ensure_dir(output_dir)
    paths = {}

    # --- Curva Precision-Recall ---
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2)
    plt.title("Curva Precision-Recall (Test)", fontsize=14)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    pr_path = Path(output_dir) / "pr_curve.png"
    plt.savefig(pr_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths["pr_curve"] = str(pr_path)
    logger.info(f"Curva PR guardada en {pr_path}")

    # --- Curva ROC ---
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    plt.title("Curva ROC (Test)", fontsize=14)
    plt.xlabel("Tasa de falsos positivos", fontsize=12)
    plt.ylabel("Tasa de verdaderos positivos", fontsize=12)
    plt.legend(loc="best", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = Path(output_dir) / "roc_curve.png"
    plt.savefig(roc_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths["roc_curve"] = str(roc_path)
    logger.info(f"Curva ROC guardada en {roc_path}")

    # --- Matriz de confusion ---
    y_pred = proba_to_class(y_prob, threshold)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, cmap="Blues", aspect="auto")
    plt.title(f"Matriz de confusion (threshold={threshold})", fontsize=14)
    plt.xlabel("Predicho", fontsize=12)
    plt.ylabel("Real", fontsize=12)
    plt.xticks([0, 1], ["Sin riesgo", "Riesgo"])
    plt.yticks([0, 1], ["Sin riesgo", "Riesgo"])
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14,
                fontweight="bold",
            )
    plt.colorbar()
    plt.tight_layout()
    cm_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths["confusion_matrix"] = str(cm_path)
    logger.info(f"Matriz de confusion guardada en {cm_path}")

    return paths


def generate_threshold_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str,
    metrics_output_dir: str,
) -> Path:
    """Genera curva precision/recall/F1 por umbral y la guarda como PNG."""
    ensure_dir(output_dir)

    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for thr in thresholds:
        metrics = compute_metrics(y_true, y_prob, threshold=float(thr))
        rows.append({"threshold": float(thr), **metrics})

    df_thr = pd.DataFrame(rows)

    plt.figure(figsize=(9, 6))
    plt.plot(df_thr["threshold"], df_thr["Recall"], label="Recall", linewidth=2)
    plt.plot(df_thr["threshold"], df_thr["Precision"], label="Precision", linewidth=2)
    plt.plot(df_thr["threshold"], df_thr["F1"], label="F1", linewidth=2)
    plt.title("Trade-off de metricas por umbral", fontsize=14)
    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.ylim(0.0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = Path(output_dir) / "threshold_tradeoff.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    ensure_dir(metrics_output_dir)
    df_thr.to_csv(Path(metrics_output_dir) / "threshold_metrics.csv", index=False)
    return out_path


def generate_calibration_plot(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_dir: str,
    metrics_output_dir: str,
    n_bins: int = 10,
) -> Path:
    """Genera plot de calibracion (reliability diagram)."""
    ensure_dir(output_dir)
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

    plt.figure(figsize=(7, 6))
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=2, label="Modelo")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfecta")
    plt.title("Curva de calibracion", fontsize=14)
    plt.xlabel("Probabilidad media predicha", fontsize=12)
    plt.ylabel("Fraccion positiva observada", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = Path(output_dir) / "calibration_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    ensure_dir(metrics_output_dir)
    pd.DataFrame(
        {
            "mean_predicted_probability": mean_pred,
            "observed_positive_rate": frac_pos,
        }
    ).to_csv(Path(metrics_output_dir) / "calibration_table.csv", index=False)
    return out_path


def generate_model_comparison_plots(
    test_metrics_df: pd.DataFrame,
    output_dir: str,
    best_model_name: str,
) -> dict:
    """Genera graficos comparativos entre modelos candidatos."""
    ensure_dir(output_dir)
    paths = {}

    df = test_metrics_df.copy()
    if df.empty:
        return paths

    df = df.sort_values(by="PR_AUC", ascending=False).reset_index(drop=True)
    model_names = df["model"].tolist()
    metrics = ["Recall", "Precision", "F1", "PR_AUC"]

    x = np.arange(len(model_names))
    width = 0.2

    plt.figure(figsize=(12, 7.2))
    for idx, metric_name in enumerate(metrics):
        plt.bar(x + (idx - 1.5) * width, df[metric_name], width=width, label=metric_name)

    plt.xticks(x, model_names, rotation=15, ha="right")
    plt.ylim(0.0, 1.02)
    plt.title("Comparativa de metricas en test por modelo", fontsize=14)
    plt.ylabel("Score", fontsize=12)
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.14), ncol=4, frameon=False)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.93], pad=1.2)

    bars_path = Path(output_dir) / "model_metrics_comparison.png"
    plt.savefig(bars_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths["model_metrics_comparison"] = str(bars_path)

    plt.figure(figsize=(9.5, 7.5))
    marker_cycle = ["o", "s", "^", "D", "P", "X"]
    for idx, (_, row) in enumerate(df.iterrows()):
        color = "#d62728" if row["model"] == best_model_name else "#1f77b4"
        label = (
            f"{row['model']} | Recall={row['Recall']:.3f} | "
            f"Precision={row['Precision']:.3f} | PR-AUC={row['PR_AUC']:.3f}"
        )
        plt.scatter(
            row["Recall"],
            row["Precision"],
            s=180,
            c=color,
            alpha=0.78,
            edgecolors="black",
            linewidths=0.8,
            marker=marker_cycle[idx % len(marker_cycle)],
            label=label,
        )

    # Curvas de iso-F1 para facilitar la lectura del trade-off precision/recall.
    x_vals = np.linspace(0.01, 0.999, 500)
    for f1_target in [0.90, 0.92, 0.94]:
        y_vals = (f1_target * x_vals) / (2 * x_vals - f1_target)
        valid = (y_vals > 0) & (y_vals <= 1)
        if valid.any():
            plt.plot(
                x_vals[valid],
                y_vals[valid],
                linestyle="--",
                linewidth=1,
                color="gray",
                alpha=0.55,
            )
            xpos = x_vals[valid][-1]
            ypos = y_vals[valid][-1]
            plt.text(
                xpos,
                ypos,
                f"F1={f1_target:.2f}",
                fontsize=9,
                color="gray",
                ha="right",
                va="bottom",
            )

    plt.title("Frontera Precision-Recall por modelo", fontsize=14)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    min_recall = max(float(df["Recall"].min()) - 0.02, 0.0)
    max_recall = min(float(df["Recall"].max()) + 0.01, 1.0)
    min_precision = max(float(df["Precision"].min()) - 0.03, 0.0)
    max_precision = min(float(df["Precision"].max()) + 0.02, 1.0)
    plt.xlim(min_recall, max_recall)
    plt.ylim(min_precision, max_precision)
    plt.grid(True, alpha=0.25)
    plt.legend(loc="lower right", frameon=True, framealpha=0.92, fontsize=9)
    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.92], pad=1.2)

    frontier_path = Path(output_dir) / "model_decision_frontier.png"
    plt.savefig(frontier_path, dpi=150, bbox_inches="tight")
    plt.close()
    paths["model_decision_frontier"] = str(frontier_path)

    return paths


def generate_pr_overlay_plot(
    y_true: np.ndarray,
    probs_by_model: dict,
    output_dir: str,
) -> Path:
    """Genera curva Precision-Recall superpuesta para todos los modelos."""
    ensure_dir(output_dir)

    plt.figure(figsize=(10, 7))
    for model_name, y_prob in probs_by_model.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, linewidth=2, label=f"{model_name} (PR-AUC={pr_auc:.4f})")

    plt.title("Curvas Precision-Recall por modelo", fontsize=14)
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False)
    plt.tight_layout(pad=1.2)

    out_path = Path(output_dir) / "model_pr_overlay.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


def generate_confusion_matrices_comparison(
    y_true: np.ndarray,
    probs_by_model: dict,
    threshold: float,
    output_dir: str,
) -> Path:
    """Genera matrices de confusion comparativas (subplots) para los modelos."""
    ensure_dir(output_dir)
    model_names = list(probs_by_model.keys())
    n_models = len(model_names)

    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(6.4 * n_models, 5.6),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.flatten()

    for idx, model_name in enumerate(model_names):
        y_prob = probs_by_model[model_name]
        y_pred = proba_to_class(y_prob, threshold)
        cm = confusion_matrix(y_true, y_pred)

        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_title(f"{model_name}\nthreshold={threshold}", fontsize=11)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Sin riesgo", "Riesgo"])
        ax.set_yticklabels(["Sin riesgo", "Riesgo"])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12,
                    fontweight="bold",
                )

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    out_path = Path(output_dir) / "model_confusion_matrices.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_confusion_matrices_from_counts(
    confusion_counts_df: pd.DataFrame,
    output_dir: str,
) -> Path:
    """Genera matrices comparativas a partir de TP/TN/FP/FN ya calculados."""
    ensure_dir(output_dir)
    df = confusion_counts_df.copy()
    if df.empty:
        raise ValueError("confusion_counts_df esta vacio")

    if not {"model", "tp", "tn", "fp", "fn"}.issubset(df.columns):
        raise ValueError("confusion_counts_df debe contener: model,tp,tn,fp,fn")

    model_names = df["model"].tolist()
    n_models = len(model_names)
    fig, axes = plt.subplots(
        1,
        n_models,
        figsize=(6.4 * n_models, 5.6),
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes.flatten()

    for idx, (_, row) in enumerate(df.iterrows()):
        cm = np.array(
            [
                [int(row["tn"]), int(row["fp"])],
                [int(row["fn"]), int(row["tp"])],
            ]
        )
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues", aspect="auto")
        ax.set_title(str(row["model"]), fontsize=11)
        ax.set_xlabel("Predicho")
        ax.set_ylabel("Real")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Sin riesgo", "Riesgo"])
        ax.set_yticklabels(["Sin riesgo", "Riesgo"])

        for i in range(2):
            for j in range(2):
                ax.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12,
                    fontweight="bold",
                )

    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    out_path = Path(output_dir) / "model_confusion_matrices.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def generate_operational_cost_table(
    y_true: np.ndarray,
    probs_by_model: dict,
    threshold: float,
    cost_fp: float,
    cost_fn: float,
    output_path: Path,
) -> Path:
    """Genera tabla de coste operativo estimado por modelo."""
    rows = []
    for model_name, y_prob in probs_by_model.items():
        y_pred = proba_to_class(y_prob, threshold)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        total_cost = fp * cost_fp + fn * cost_fn
        avg_cost_per_sample = total_cost / max(len(y_true), 1)
        rows.append(
            {
                "model": model_name,
                "threshold": float(threshold),
                "tp": int(tp),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "cost_fp": float(cost_fp),
                "cost_fn": float(cost_fn),
                "estimated_total_cost": float(total_cost),
                "estimated_cost_per_sample": float(avg_cost_per_sample),
            }
        )

    df_cost = pd.DataFrame(rows).sort_values(by="estimated_total_cost", ascending=True)
    df_cost.to_csv(output_path, index=False)
    return output_path


def main(
    config_path: str = "config/config.yaml",
    include_eda_report: bool = True,
) -> int:
    """
    Ejecuta el pipeline completo de entrenamiento.

    Args:
        config_path: Ruta al config.yaml

    Returns:
        Codigo de salida (0=exito, 1=error)
    """
    logger.info("=" * 80)
    logger.info("PIPELINE DE ENTRENAMIENTO - MODELO DE UTILIZACION DE RED")
    logger.info("=" * 80)

    try:
        # ====================================================================
        # PASO 1: Cargar configuracion
        # ====================================================================
        logger.info("\n[PASO 1] Cargando configuracion...")
        cfg = load_config(config_path)

        # Rutas
        input_parquet = cfg["paths"]["input_parquet"]
        output_root = Path(cfg["paths"]["output_dir"])
        ml_out_dir = output_root / "ml"
        metrics_dir = ml_out_dir / "metrics"
        model_dir = Path(cfg["paths"]["model_dir"])
        ml_fig_dir = ml_out_dir / "figures"

        ensure_dir(str(output_root))
        ensure_dir(str(ml_out_dir))
        ensure_dir(str(metrics_dir))
        ensure_dir(str(model_dir))
        ensure_dir(str(ml_fig_dir))

        # Esquema
        rx_col = cfg["schema"]["rx_util_col"]
        tx_col = cfg["schema"]["tx_util_col"]
        ts_col = cfg["schema"]["timestamp_col"]
        entity_col = cfg["schema"].get("entity_col", "NODE NAME")
        id_like_cols = cfg["schema"].get("id_like_cols", [])

        # Modelado
        test_size = cfg["modeling"]["test_size"]
        random_state = cfg["modeling"]["random_state"]
        cv_folds = cfg["modeling"]["cv_folds"]
        min_category_freq = cfg["modeling"]["min_category_freq"]
        threshold = cfg["modeling"]["risk_threshold"]
        horizon_days = cfg["modeling"]["horizon_days"]
        classification_threshold = cfg["modeling"]["classification_threshold"]
        cost_fp = float(cfg["modeling"].get("cost_fp", 1.0))
        cost_fn = float(cfg["modeling"].get("cost_fn", 5.0))
        drop_leakage = cfg["modeling"].get("drop_cols_leakage", [])

        logger.info("✓ Configuracion cargada")

        # ====================================================================
        # PASO 2: Cargar y validar datos
        # ====================================================================
        logger.info("\n[PASO 2] Cargando parquet...")
        df = load_snapshot(input_parquet, validate=True)
        rows_raw = len(df)

        # Convertir timestamps y filtrar filas invalidas/faltantes
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df_clean, _ = filter_invalid_timestamps(df, ts_col)
        report_cleaning_stats(rows_raw, len(df_clean), "Filtrado de timestamp")

        rows_after_ts = len(df_clean)
        df_clean, _ = filter_rows_with_any_nan(df_clean)
        report_cleaning_stats(rows_after_ts, len(df_clean), "Filtrado global de NaN")

        # ====================================================================
        # PASO 3: Crear target
        # ====================================================================
        logger.info("\n[PASO 3] Creando target binario...")
        df_target = create_binary_target(
            df_clean,
            rx_util_col=rx_col,
            tx_util_col=tx_col,
            threshold=threshold,
            target_col="overutilization_risk",
            horizon_days=horizon_days,
            date_col=ts_col,
            entity_col=entity_col,
        )
        df_target_for_export = df_target.copy()

        # ====================================================================
        # PASO 4: Ingenieria de variables
        # ====================================================================
        logger.info("\n[PASO 4] Ingenieria de variables...")

        # Agregar variables temporales
        df_target = add_time_features(df_target, ts_col)

        # Eliminar columnas propensas a leakage
        drop_cols = [rx_col, tx_col, entity_col, "Porcentaje de Utilizacion"] \
                    + id_like_cols + drop_leakage
        df_fe = df_target.drop(columns=drop_cols, errors="ignore")

        y = df_fe["overutilization_risk"].astype(int)
        X = df_fe.drop(columns=["overutilization_risk"], errors="ignore")

        # Eliminar columnas datetime
        dt_cols = X.select_dtypes(
            include=["datetime", "datetimetz", "datetime64[ns]", "datetime64[ns, UTC]"]
        ).columns.tolist()
        X = X.drop(columns=dt_cols, errors="ignore")

        logger.info(f"Variables: {X.shape[1]} en total ({len(X)} muestras)")

        # ====================================================================
        # PASO 5: Division train-test (temporal)
        # ====================================================================
        logger.info("\n[PASO 5] Division temporal train-test...")
        split_idx = int((1 - test_size) * len(X))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Train: {X_train.shape[0]:,} muestras")
        logger.info(f"Test:  {X_test.shape[0]:,} muestras")

        # Validar antes del preprocesamiento
        valid, errors = validate_X_for_training(X_train, y_train, name="train_set")
        if not valid:
            raise ValueError("\n".join(errors))

        # ====================================================================
        # PASO 6: Preprocesamiento
        # ====================================================================
        logger.info("\n[PASO 6] Preprocesamiento (fit en train, transform en test)...")
        prep = Preprocessor(min_category_freq=min_category_freq)
        prep.fit(X_train, y_train)
        X_train_clean = prep.transform(X_train)
        X_test_clean = prep.transform(X_test)

        logger.info(f"Train transformado: {X_train_clean.shape}")
        logger.info(f"Test transformado: {X_test_clean.shape}")

        # Guardar preprocesador
        prep_path = model_dir / "preprocessor.pkl"
        prep.save(str(prep_path))
        logger.info(f"✓ Preprocesador guardado en {prep_path}")

        # ====================================================================
        # PASO 7: Construir y ajustar modelos
        # ====================================================================
        logger.info("\n[PASO 7] Entrenamiento y ajuste de modelos...")

        # Peso de clase para desbalance
        pos_count = (y_train == 1).sum()
        neg_count = (y_train == 0).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        logger.info(f"Ratio de desbalance de clase: {scale_pos_weight:.2f}")

        # Construir modelos
        models = build_models(scale_pos_weight=scale_pos_weight, random_state=random_state)

        # Ajustar con RandomizedSearchCV
        tuned_models = tune_models_randomized(
            X_train=X_train_clean,
            y_train=y_train,
            preprocessor=prep,
            models=models,
            cv_folds=cv_folds,
            random_state=random_state,
            n_iter=cfg["modeling"]["n_iter"],
            n_jobs=cfg["modeling"].get("n_jobs", -1),
            classification_threshold=classification_threshold,
        )

        # ====================================================================
        # PASO 8: Evaluar el mejor modelo en test
        # ====================================================================
        logger.info("\n[PASO 8] Evaluando mejor modelo en test...")

        (
            best_model_name,
            best_model,
            best_cv_score,
            best_params,
            best_threshold,
            model_results,
            estimators_by_model,
        ) = tuned_models
        logger.info(f"Mejor modelo: {best_model_name} (CV PR-AUC={best_cv_score:.4f})")

        y_prob_test = best_model.predict_proba(X_test_clean)[:, 1]
        test_metrics = compute_metrics(
            y_test.values,
            y_prob_test,
            threshold=classification_threshold,
        )

        per_model_test_rows = []
        probs_by_model = {}
        for model_name, estimator in estimators_by_model.items():
            y_prob_model = estimator.predict_proba(X_test_clean)[:, 1]
            probs_by_model[model_name] = y_prob_model
            metrics_model = compute_metrics(
                y_test.values,
                y_prob_model,
                threshold=classification_threshold,
            )
            per_model_test_rows.append(
                {
                    "model": model_name,
                    **metrics_model,
                    "classification_threshold": float(classification_threshold),
                }
            )

        logger.info(f"\nMetricas en test (threshold={classification_threshold}):")
        logger.info(f"  Recall: {test_metrics['Recall']:.4f} (objetivo: > 0.75)")
        logger.info(f"  Precision: {test_metrics['Precision']:.4f}")
        logger.info(f"  F1: {test_metrics['F1']:.4f}")
        logger.info(f"  PR-AUC: {test_metrics['PR_AUC']:.4f}")

        metrics_rows = [
            {
                "model": best_model_name,
                **test_metrics,
                "cv_pr_auc": float(best_cv_score),
                "classification_threshold": float(classification_threshold),
            }
        ]
        metrics_comparison = pd.DataFrame(metrics_rows)
        model_comparison_path = metrics_dir / "model_comparison.csv"
        metrics_comparison.to_csv(model_comparison_path, index=False)

        all_models_test_df = pd.DataFrame(per_model_test_rows)
        all_models_test_df = all_models_test_df.sort_values(
            by=["PR_AUC", "F1"],
            ascending=False,
        )
        all_models_test_path = metrics_dir / "test_metrics_by_model.csv"
        all_models_test_df.to_csv(all_models_test_path, index=False)

        model_results_path = metrics_dir / "model_results_by_model.csv"
        pd.DataFrame(model_results).sort_values(
            by=["cv_pr_auc", "holdout_valid_f1"],
            ascending=False,
        ).to_csv(model_results_path, index=False)

        train_summary = {
            "best_model_name": best_model_name,
            "best_cv_pr_auc": float(best_cv_score),
            "classification_threshold": float(classification_threshold),
            "selected_threshold_from_tuning": float(best_threshold),
            "cost_assumptions": {
                "cost_fp": cost_fp,
                "cost_fn": cost_fn,
            },
            "best_params": _to_serializable(best_params),
        }
        summary_path = ml_out_dir / "training_summary.json"
        summary_path.write_text(
            json.dumps(train_summary, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        (metrics_dir / "training_summary.json").write_text(
            json.dumps(train_summary, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

        # Guardar mejor modelo
        model_path = model_dir / f"{best_model_name}.pkl"
        joblib.dump(best_model, model_path)
        logger.info(f"✓ Modelo guardado en {model_path}")

        if include_eda_report:
            # ====================================================================
            # PASO 9: Generar visualizaciones ML
            # ====================================================================
            logger.info("\n[PASO 9] Generando visualizaciones...")

            # Curvas diagnosticas
            generate_model_curves(
                y_test.values,
                y_prob_test,
                str(ml_fig_dir),
                classification_threshold,
            )

            # Importancia de variables
            feature_names = list(X_test_clean.columns)
            importance_series = compute_feature_importance(best_model, feature_names)
            plot_feature_importance(importance_series, str(ml_fig_dir), top_n=15)

            metrics_path = metrics_dir / "ml_metrics.csv"
            pd.DataFrame([test_metrics]).to_csv(metrics_path, index=False)

            cls_report_path = metrics_dir / "classification_report.txt"
            cls_report_path.write_text(
                classification_report(
                    y_test,
                    proba_to_class(y_prob_test, classification_threshold),
                    zero_division=0,
                ),
                encoding="utf-8",
            )

            threshold_plot_path = generate_threshold_curve(
                y_true=y_test.values,
                y_prob=y_prob_test,
                output_dir=str(ml_fig_dir),
                metrics_output_dir=str(metrics_dir),
            )
            calibration_plot_path = generate_calibration_plot(
                y_true=y_test.values,
                y_prob=y_prob_test,
                output_dir=str(ml_fig_dir),
                metrics_output_dir=str(metrics_dir),
            )
            model_compare_paths = generate_model_comparison_plots(
                test_metrics_df=all_models_test_df,
                output_dir=str(ml_fig_dir),
                best_model_name=best_model_name,
            )
            pr_overlay_path = generate_pr_overlay_plot(
                y_true=y_test.values,
                probs_by_model=probs_by_model,
                output_dir=str(ml_fig_dir),
            )
            confusion_compare_path = generate_confusion_matrices_comparison(
                y_true=y_test.values,
                probs_by_model=probs_by_model,
                threshold=classification_threshold,
                output_dir=str(ml_fig_dir),
            )
            cost_table_path = generate_operational_cost_table(
                y_true=y_test.values,
                probs_by_model=probs_by_model,
                threshold=classification_threshold,
                cost_fp=cost_fp,
                cost_fn=cost_fn,
                output_path=metrics_dir / "operational_cost_by_model.csv",
            )
            logger.info(f"Metricas CSV guardadas en {metrics_path}")
            logger.info(f"Reporte de clasificacion guardado en {cls_report_path}")
            logger.info(f"Curva de umbral guardada en {threshold_plot_path}")
            logger.info(f"Curva de calibracion guardada en {calibration_plot_path}")
            logger.info(f"Curvas PR por modelo guardadas en {pr_overlay_path}")
            logger.info(f"Matrices de confusion comparativas guardadas en {confusion_compare_path}")
            logger.info(f"Tabla de coste operativo guardada en {cost_table_path}")
            if model_compare_paths:
                logger.info(
                    f"Comparativa de modelos guardada en {model_compare_paths.get('model_metrics_comparison')}"
                )
                logger.info(
                    f"Frontera de decision guardada en {model_compare_paths.get('model_decision_frontier')}"
                )
        else:
            logger.info("\n[PASO 9-10] Se omiten visualizaciones/metricas (--ml-only)")

        logger.info(f"Comparativa de modelos guardada en {model_comparison_path}")
        logger.info(f"Resultados por modelo guardados en {model_results_path}")
        logger.info(f"Metricas de test por modelo guardadas en {all_models_test_path}")
        logger.info(f"Resumen de entrenamiento guardado en {summary_path}")

        # ====================================================================
        # PASO 11: Guardar predicciones de test
        # ====================================================================
        logger.info("\n[PASO 11] Guardando predicciones de test...")

        y_prob_all = best_model.predict_proba(X_test_clean)[:, 1]
        y_pred_all = proba_to_class(y_prob_all, classification_threshold)

        predictions_df = pd.DataFrame({
            "actual": y_test.values,
            "predicted_probability": y_prob_all,
            "predicted_risk": y_pred_all,
        })

        pred_path = ml_out_dir / "test_predictions.csv"
        predictions_df.to_csv(pred_path, index=False)
        logger.info(f"✓ Predicciones guardadas en {pred_path}")

        pred_metrics_path = metrics_dir / "test_predictions.csv"
        predictions_df.to_csv(pred_metrics_path, index=False)

        # Tabla de predicciones extendida para consumo de negocio.
        predictions_export = df_target_for_export.loc[X_test.index].copy()
        predictions_export["risk_probability"] = y_prob_all
        predictions_export["risk_prediction"] = y_pred_all

        preferred_columns = [
            "Servicio",
            "Nombre de Interface",
            "Total Bytes Recibidos",
            "Total Bytes Enviados",
            "Picos Recibidos bps",
            "Picos Transmitidos bps",
            "Velocidad",
            "Timestamp",
            "Average Receive bps",
            "Average Transmit bps",
            "SOURCE_FILE",
            "FECHA_ARCHIVO",
            "DS",
            "TIPO DE SEDE",
            "DEPARTAMENTO",
            "CIUDAD",
            "SEDE",
            "BW",
            "SIGLAPAPA",
            "UNIDAD",
            "DEPENDENCIA",
            "MEDIO",
            "TIPO DE SERVICIO",
            "PROVEEDOR",
            "IP LOOPBACK",
            "SAP",
            "PE-MPLS",
            "FECHA REVISIÓN",
            "overutilization_event_now",
            "overutilization_risk",
            "risk_probability",
            "risk_prediction",
        ]
        existing_preferred = [c for c in preferred_columns if c in predictions_export.columns]
        remaining_columns = [c for c in predictions_export.columns if c not in existing_preferred]
        predictions_export = predictions_export[existing_preferred + remaining_columns]

        predictions_export_path = ml_out_dir / "predictions.csv"
        predictions_export.to_csv(predictions_export_path, index=False)
        predictions_export.to_csv(metrics_dir / "predictions.csv", index=False)
        logger.info(f"✓ Tabla de predicciones extendida guardada en {predictions_export_path}")

        # ====================================================================
        # FIN
        # ====================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE DE ENTRENAMIENTO COMPLETADO CORRECTAMENTE")
        logger.info("=" * 80)
        logger.info(f"\nArchivos de salida:")
        logger.info(f"  Modelo:       {model_path}")
        logger.info(f"  Preprocesador:{prep_path}")
        logger.info(f"  Predicciones: {pred_path}")

        return 0

    except Exception as exc:
        logger.error(f"\nFALLO EN PIPELINE: {exc}", exc_info=True)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrenar pipeline de riesgo de trafico")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta al archivo de configuracion YAML",
    )
    parser.add_argument(
        "--ml-only",
        action="store_true",
        help="Omitir visualizaciones y artefactos adicionales",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    exit_code = main(
        config_path=args.config,
        include_eda_report=not args.ml_only,
    )
    sys.exit(exit_code)
