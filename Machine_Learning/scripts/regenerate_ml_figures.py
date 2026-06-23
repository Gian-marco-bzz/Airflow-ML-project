"""Regenera figuras ML desde artefactos existentes (sin reentrenar)."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import train_pipeline as tp


def generate_operational_cost_bar(cost_df: pd.DataFrame, output_dir: Path) -> Path:
    """Genera barras de coste total estimado por modelo."""
    output_dir.mkdir(parents=True, exist_ok=True)

    df = cost_df.sort_values(by="estimated_total_cost", ascending=True).copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(df["model"], df["estimated_total_cost"], color=["#2ca02c", "#1f77b4", "#d62728"])

    for bar, val in zip(bars, df["estimated_total_cost"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.set_title("Coste operativo estimado por modelo", fontsize=14)
    ax.set_xlabel("Modelo", fontsize=12)
    ax.set_ylabel("Coste total estimado", fontsize=12)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()

    out_path = output_dir / "operational_cost_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def regenerate_predictions_without_retraining(
    repo_root: Path,
    cfg: dict,
    best_model_name: str,
) -> dict:
    """Reconstruye predicciones de test usando artefactos guardados (sin fit)."""
    input_parquet = cfg["paths"]["input_parquet"]
    output_root = repo_root / cfg["paths"]["output_dir"]
    ml_out_dir = output_root / "ml"
    metrics_dir = ml_out_dir / "metrics"
    model_dir = repo_root / cfg["paths"]["model_dir"]

    rx_col = cfg["schema"]["rx_util_col"]
    tx_col = cfg["schema"]["tx_util_col"]
    ts_col = cfg["schema"]["timestamp_col"]
    entity_col = cfg["schema"].get("entity_col", "NODE NAME")
    id_like_cols = cfg["schema"].get("id_like_cols", [])

    test_size = cfg["modeling"]["test_size"]
    threshold = cfg["modeling"]["risk_threshold"]
    horizon_days = cfg["modeling"]["horizon_days"]
    classification_threshold = cfg["modeling"]["classification_threshold"]
    drop_leakage = cfg["modeling"].get("drop_cols_leakage", [])

    df = tp.load_snapshot(input_parquet, validate=True)
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
    df_clean, _ = tp.filter_invalid_timestamps(df, ts_col)
    df_clean, _ = tp.filter_rows_with_any_nan(df_clean)

    df_target = tp.create_binary_target(
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
    df_target = tp.add_time_features(df_target, ts_col)

    drop_cols = [rx_col, tx_col, entity_col, "Porcentaje de Utilizacion"] + id_like_cols + drop_leakage
    df_fe = df_target.drop(columns=drop_cols, errors="ignore")

    y = df_fe["overutilization_risk"].astype(int)
    X = df_fe.drop(columns=["overutilization_risk"], errors="ignore")

    dt_cols = X.select_dtypes(
        include=["datetime", "datetimetz", "datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()
    X = X.drop(columns=dt_cols, errors="ignore")

    split_idx = int((1 - test_size) * len(X))
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    prep_path = model_dir / "preprocessor.pkl"
    model_path = model_dir / f"{best_model_name}.pkl"
    if not prep_path.exists():
        raise FileNotFoundError(f"No existe preprocesador: {prep_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"No existe modelo ganador: {model_path}")

    prep = tp.Preprocessor.load(str(prep_path))
    model = joblib.load(model_path)

    X_test_clean = prep.transform(X_test)
    y_prob_all = model.predict_proba(X_test_clean)[:, 1]
    y_pred_all = tp.proba_to_class(y_prob_all, classification_threshold)

    predictions_df = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted_probability": y_prob_all,
            "predicted_risk": y_pred_all,
        }
    )

    test_pred_path = ml_out_dir / "test_predictions.csv"
    test_pred_metrics_path = metrics_dir / "test_predictions.csv"
    predictions_df.to_csv(test_pred_path, index=False)
    predictions_df.to_csv(test_pred_metrics_path, index=False)

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

    export_path = ml_out_dir / "predictions.csv"
    export_metrics_path = metrics_dir / "predictions.csv"
    predictions_export.to_csv(export_path, index=False)
    predictions_export.to_csv(export_metrics_path, index=False)

    return {
        "test_predictions": str(test_pred_path),
        "test_predictions_metrics": str(test_pred_metrics_path),
        "predictions": str(export_path),
        "predictions_metrics": str(export_metrics_path),
    }


def main() -> int:
    repo_root = Path(__file__).parent.parent
    cfg = tp.load_config(str(repo_root / "config" / "config.yaml"))
    ml_dir = repo_root / "outputs" / "ml"
    metrics_dir = ml_dir / "metrics"
    figures_dir = ml_dir / "figures"

    test_metrics_path = metrics_dir / "test_metrics_by_model.csv"
    cost_path = metrics_dir / "operational_cost_by_model.csv"
    summary_path = ml_dir / "training_summary.json"

    if not test_metrics_path.exists():
        raise FileNotFoundError(f"No existe {test_metrics_path}")
    if not cost_path.exists():
        raise FileNotFoundError(f"No existe {cost_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"No existe {summary_path}")

    test_metrics_df = pd.read_csv(test_metrics_path)
    cost_df = pd.read_csv(cost_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    best_model_name = summary.get("best_model_name", "")
    if not best_model_name:
        raise ValueError("No se encontro best_model_name en training_summary.json")

    paths = tp.generate_model_comparison_plots(
        test_metrics_df=test_metrics_df,
        output_dir=str(figures_dir),
        best_model_name=best_model_name,
    )

    cm_path = tp.generate_confusion_matrices_from_counts(
        confusion_counts_df=cost_df[["model", "tp", "tn", "fp", "fn"]],
        output_dir=str(figures_dir),
    )

    cost_bar_path = generate_operational_cost_bar(cost_df, figures_dir)
    prediction_paths = regenerate_predictions_without_retraining(repo_root, cfg, best_model_name)

    print("Figuras regeneradas sin reentrenar:")
    for key, value in paths.items():
        print(f"- {key}: {value}")
    print(f"- confusion: {cm_path}")
    print(f"- cost_bar: {cost_bar_path}")
    for key, value in prediction_paths.items():
        print(f"- {key}: {value}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
