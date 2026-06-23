"""
Pipeline de artefactos EDA usando artefactos ya entrenados.

Este script evita reentrenar y regenera salidas de analisis
(PNG/CSV/TXT) desde el ultimo snapshot y el modelo/preprocesador guardados.

Uso:
    python scripts/eda_pipeline.py
    python scripts/eda_pipeline.py --config config/config.yaml
"""

import sys
import argparse
from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import load_config, get_logger, ensure_dir
from src.utils.validation import report_cleaning_stats
from src.data.load_data import load_snapshot, filter_invalid_timestamps, filter_rows_with_any_nan
from src.data.create_target import create_binary_target
from src.data.feature_engineering import add_time_features
from src.modeling.evaluate import (
    compute_metrics,
    proba_to_class,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
)
from src.reporting.eda import make_eda_plots, eda_summary_text
from src.reporting.feature_importance import compute_feature_importance, plot_feature_importance

logger = get_logger(__name__)


def _resolve_model_path(model_dir: Path) -> Path:
    preferred = model_dir / "xgboost.pkl"
    if preferred.exists():
        return preferred

    candidates = [
        p for p in model_dir.glob("*.pkl")
        if p.name != "preprocessor.pkl"
    ]
    if not candidates:
        return Path("")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def main(config_path: str = "config/config.yaml") -> int:
    logger.info("=" * 80)
    logger.info("PIPELINE EDA (SIN REENTRENAR)")
    logger.info("=" * 80)

    try:
        cfg = load_config(config_path)

        input_parquet = cfg["paths"]["input_parquet"]
        model_dir = Path(cfg["paths"]["model_dir"])
        output_root = Path(cfg["paths"]["output_dir"])
        output_dir = output_root / "eda"
        tmp_dir = output_dir / "figures"

        ensure_dir(str(output_dir))
        ensure_dir(str(tmp_dir))

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

        logger.info("[1] Cargando datos...")
        df = load_snapshot(input_parquet, validate=True)
        rows_raw = len(df)

        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df_clean, _ = filter_invalid_timestamps(df, ts_col)
        report_cleaning_stats(rows_raw, len(df_clean), "Filtrado de timestamp")

        rows_after_ts = len(df_clean)
        df_clean, _ = filter_rows_with_any_nan(df_clean)
        report_cleaning_stats(rows_after_ts, len(df_clean), "Filtrado global de NaN")

        logger.info("[2] Reconstruyendo target/variables...")
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
        df_target = add_time_features(df_target, ts_col)

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

        logger.info(f"[3] Cargando artefactos desde {model_dir}...")
        preprocessor_path = model_dir / "preprocessor.pkl"
        model_path = _resolve_model_path(model_dir)
        has_model_artifacts = preprocessor_path.exists() and bool(str(model_path)) and model_path.exists()

        if has_model_artifacts:
            preprocessor = joblib.load(preprocessor_path)
            model = joblib.load(model_path)
            logger.info(f"Usando modelo: {model_path.name}")

            logger.info("[4] Scoring sobre particion test...")
            X_test_clean = preprocessor.transform(X_test)
            y_prob_test = model.predict_proba(X_test_clean)[:, 1]

            test_metrics = compute_metrics(
                y_test.values,
                y_prob_test,
                threshold=classification_threshold,
            )
        else:
            logger.warning("Modelo/preprocesador no encontrados. Ejecutando modo solo EDA.")
            test_metrics = {}

        logger.info("[5] Generando artefactos EDA...")
        eda_df = X_test.copy()
        eda_df["overutilization_risk"] = y_test.values
        eda_images = make_eda_plots(eda_df, "overutilization_risk", str(tmp_dir))
        eda_summary = eda_summary_text(eda_df, "overutilization_risk")

        # Diagnosticos globales EDA antes de decisiones de ML.
        profile_rows = []
        for col in eda_df.columns:
            s = eda_df[col]
            profile_rows.append({
                "column": col,
                "dtype": str(s.dtype),
                "n_rows": int(len(s)),
                "n_missing": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean() * 100.0),
                "n_unique": int(s.nunique(dropna=True)),
                "sample": str(s.dropna().iloc[0]) if s.dropna().shape[0] > 0 else "",
            })
        profile_df = pd.DataFrame(profile_rows)
        profile_df.to_csv(output_dir / "eda_column_profile.csv", index=False)

        constant_df = profile_df[profile_df["n_unique"] <= 1].copy()
        constant_df.to_csv(output_dir / "eda_constant_columns.csv", index=False)

        high_card_df = profile_df[(profile_df["dtype"].str.contains("object|category", case=False, na=False)) & (profile_df["n_unique"] > 100)].copy()
        high_card_df.to_csv(output_dir / "eda_high_cardinality_columns.csv", index=False)

        # Candidatos de leakage: nombres sospechosos y alta correlacion con target.
        suspicious_tokens = ["target", "risk", "util", "porcentaje", "received", "transmit"]
        leakage_name_rows = []
        for col in df_clean.columns:
            low = col.lower()
            if any(token in low for token in suspicious_tokens):
                leakage_name_rows.append({"column": col, "reason": "name_pattern"})

        leakage_corr_rows = []
        numeric_cols = [c for c in df_fe.select_dtypes(include=["number"]).columns if c != "overutilization_risk"]
        for col in numeric_cols:
            if df_fe[col].nunique(dropna=True) <= 1:
                continue
            corr = float(df_fe[[col, "overutilization_risk"]].corr(numeric_only=True).iloc[0, 1])
            if abs(corr) >= 0.9:
                leakage_corr_rows.append({"column": col, "reason": "high_target_corr", "abs_corr": abs(corr)})

        leakage_df = pd.concat([
            pd.DataFrame(leakage_name_rows),
            pd.DataFrame(leakage_corr_rows),
        ], ignore_index=True) if (leakage_name_rows or leakage_corr_rows) else pd.DataFrame(columns=["column", "reason", "abs_corr"])
        leakage_df.drop_duplicates(subset=["column", "reason"], inplace=True)
        leakage_df.to_csv(output_dir / "eda_leakage_candidates.csv", index=False)

        if ts_col in df_clean.columns:
            ts_stats = pd.DataFrame({
                "min_timestamp": [df_clean[ts_col].min()],
                "max_timestamp": [df_clean[ts_col].max()],
                "n_days": [(df_clean[ts_col].max() - df_clean[ts_col].min()).days if len(df_clean) > 0 else 0],
                "n_rows": [len(df_clean)],
            })
            ts_stats.to_csv(output_dir / "eda_temporal_coverage.csv", index=False)

            by_month = df_clean.assign(_month=df_clean[ts_col].dt.to_period("M").astype(str)).groupby("_month").size().reset_index(name="n_rows")
            by_month.to_csv(output_dir / "eda_rows_by_month.csv", index=False)

            plt.figure(figsize=(12, 5))
            plt.plot(by_month["_month"], by_month["n_rows"], marker="o", linewidth=1.5)
            plt.title("Volumen de registros por mes")
            plt.xlabel("Mes")
            plt.ylabel("Numero de filas")
            plt.xticks(rotation=45, ha="right")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(tmp_dir / "rows_by_month.png", dpi=150, bbox_inches="tight")
            plt.close()

            if "overutilization_risk" in df_target.columns:
                risk_tmp = pd.DataFrame({
                    "_month": df_clean[ts_col].dt.to_period("M").astype(str),
                    "overutilization_risk": df_target["overutilization_risk"].values,
                })
                risk_by_month = (
                    risk_tmp.groupby("_month")["overutilization_risk"]
                    .mean()
                    .reset_index(name="risk_rate")
                )
                risk_by_month.to_csv(output_dir / "eda_risk_rate_by_month.csv", index=False)

                plt.figure(figsize=(12, 5))
                plt.plot(risk_by_month["_month"], risk_by_month["risk_rate"], marker="o", linewidth=1.5)
                plt.title("Tasa de riesgo por mes")
                plt.xlabel("Mes")
                plt.ylabel("Tasa de riesgo")
                plt.xticks(rotation=45, ha="right")
                plt.grid(alpha=0.3)
                plt.tight_layout()
                plt.savefig(tmp_dir / "risk_rate_by_month.png", dpi=150, bbox_inches="tight")
                plt.close()

            # Drift score (primer mes vs ultimo mes) para variables numericas.
            num_for_drift = [c for c in df_fe.select_dtypes(include=["number"]).columns if c != "overutilization_risk"]
            month_df = df_clean.assign(_month=df_clean[ts_col].dt.to_period("M").astype(str))
            months = sorted(month_df["_month"].unique().tolist())
            if len(months) >= 2 and num_for_drift:
                first_m = months[0]
                last_m = months[-1]
                first_idx = month_df["_month"] == first_m
                last_idx = month_df["_month"] == last_m
                drift_rows = []
                for col in num_for_drift:
                    source_df = df_clean if col in df_clean.columns else df_target
                    if col not in source_df.columns:
                        continue
                    s_all = source_df[col].dropna()
                    if len(s_all) < 30 or s_all.nunique() <= 1:
                        continue
                    first_vals = source_df.loc[first_idx, col].dropna()
                    last_vals = source_df.loc[last_idx, col].dropna()
                    if len(first_vals) < 10 or len(last_vals) < 10:
                        continue
                    denom = float(s_all.std()) if float(s_all.std()) > 0 else 1.0
                    score = float(abs(last_vals.mean() - first_vals.mean()) / denom)
                    drift_rows.append({
                        "column": col,
                        "first_month": first_m,
                        "last_month": last_m,
                        "first_mean": float(first_vals.mean()),
                        "last_mean": float(last_vals.mean()),
                        "drift_score": score,
                    })
                if drift_rows:
                    drift_df = pd.DataFrame(drift_rows).sort_values("drift_score", ascending=False)
                    drift_df.to_csv(output_dir / "eda_feature_drift_score.csv", index=False)
                else:
                    pd.DataFrame(columns=[
                        "column",
                        "first_month",
                        "last_month",
                        "first_mean",
                        "last_mean",
                        "drift_score",
                    ]).to_csv(output_dir / "eda_feature_drift_score.csv", index=False)
            else:
                pd.DataFrame(columns=[
                    "column",
                    "first_month",
                    "last_month",
                    "first_mean",
                    "last_mean",
                    "drift_score",
                ]).to_csv(output_dir / "eda_feature_drift_score.csv", index=False)

        num_cols = eda_df.select_dtypes(include=["number"]).columns.tolist()
        outlier_rows = []
        for col in num_cols:
            if col == "overutilization_risk":
                continue
            s = eda_df[col].dropna()
            if len(s) < 20 or s.nunique() <= 1:
                continue
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            outlier_pct = float(((s < lower) | (s > upper)).mean() * 100.0)
            outlier_rows.append({"column": col, "outlier_pct_iqr": outlier_pct})
        pd.DataFrame(outlier_rows).sort_values("outlier_pct_iqr", ascending=False).to_csv(
            output_dir / "eda_outlier_report.csv", index=False
        )

        pr_curve_path = ""
        roc_curve_path = ""
        confusion_matrix_path = ""
        feat_imp_path = ""

        if has_model_artifacts:
            pr_curve_path = plot_pr_curve(
                y_test.values,
                y_prob_test,
                str(tmp_dir / "pr_curve.png"),
            )
            roc_curve_path = plot_roc_curve(
                y_test.values,
                y_prob_test,
                str(tmp_dir / "roc_curve.png"),
            )
            confusion_matrix_path = plot_confusion_matrix(
                y_test.values,
                y_prob_test,
                classification_threshold,
                str(tmp_dir / "confusion_matrix.png"),
            )

            importance_series = compute_feature_importance(model, list(X_test_clean.columns))
            feat_imp_path = plot_feature_importance(importance_series, str(tmp_dir), top_n=15)

            pd.DataFrame([test_metrics]).to_csv(output_dir / "eda_metrics.csv", index=False)
            (output_dir / "eda_classification_report.txt").write_text(
                classification_report(
                    y_test,
                    proba_to_class(y_prob_test, classification_threshold),
                    zero_division=0,
                ),
                encoding="utf-8",
            )

        pd.DataFrame([eda_summary]).to_csv(output_dir / "eda_summary.csv", index=False)

        manifest = {
            "eda_images": eda_images,
            "pr_curve": pr_curve_path,
            "roc_curve": roc_curve_path,
            "confusion_matrix": confusion_matrix_path,
            "feature_importance": feat_imp_path,
        }
        pd.Series(manifest).to_json(output_dir / "eda_artifacts_manifest.json", indent=2)

        logger.info("=" * 80)
        logger.info("PIPELINE EDA COMPLETADO")
        logger.info("=" * 80)
        logger.info(f"Directorio de artefactos: {output_dir}")
        return 0

    except Exception as exc:
        logger.error(f"Fallo en pipeline EDA: {exc}", exc_info=True)
        return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecutar pipeline EDA sin reentrenamiento")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta al archivo de configuracion YAML",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sys.exit(main(config_path=args.config))
