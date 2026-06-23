import math
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def _safe_close() -> None:
    plt.close("all")


def _save_current_figure(path: str, dpi: int = 150) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()
    return path


def make_eda_plots(df: pd.DataFrame, target_col: str, tmp_dir: str) -> dict:
    """Generate readable EDA figures and skip low-value/blank charts."""
    os.makedirs(tmp_dir, exist_ok=True)
    paths = {}
    _safe_close()

    max_rows_plot = 20000
    df_plot = df.sample(n=min(len(df), max_rows_plot), random_state=42) if len(df) > max_rows_plot else df.copy()

    num_cols_all = df.select_dtypes(include=["number"]).columns.tolist()
    num_cols_all = [c for c in num_cols_all if c != target_col]
    cat_cols_all = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_candidates = []
    for col in num_cols_all:
        s = df_plot[col].dropna()
        if len(s) < 100 or s.nunique() <= 1:
            continue
        numeric_candidates.append((col, float(s.std())))
    numeric_candidates.sort(key=lambda x: x[1], reverse=True)
    plot_cols = [col for col, _ in numeric_candidates[:6]]

    cat_candidates = []
    for col in cat_cols_all:
        nunique = int(df_plot[col].astype(str).nunique(dropna=False))
        if 2 <= nunique <= 40:
            cat_candidates.append((col, nunique))
    plot_cat_cols = [col for col, _ in sorted(cat_candidates, key=lambda x: x[1])[:6]]

    if target_col in df.columns:
        class_counts = df[target_col].value_counts(dropna=False).sort_index()
        if not class_counts.empty:
            plt.figure(figsize=(8, 5))
            class_counts.plot(kind="bar", color="#4F81BD")
            plt.title("Balance de clases (0 = no riesgo, 1 = riesgo)")
            plt.xlabel("Clase")
            plt.ylabel("Numero de filas")
            paths["class_balance"] = _save_current_figure(os.path.join(tmp_dir, "class_balance.png"))

    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(20)
    if not miss.empty:
        plt.figure(figsize=(11, 7))
        sns.barplot(x=miss.values, y=miss.index, orient="h", color="#5B9BD5")
        plt.title("Top 20 columnas con mas valores faltantes (%)")
        plt.xlabel("Proporcion de missing")
        plt.ylabel("Columna")
        paths["missing_top20"] = _save_current_figure(os.path.join(tmp_dir, "missing_top20.png"))

    if df_plot.shape[1] <= 50 and df_plot.shape[0] <= 2000 and df_plot.isna().any().any():
        plt.figure(figsize=(13, 7))
        sns.heatmap(df_plot.isna(), cbar=False)
        plt.title("Mapa de valores faltantes")
        paths["missing_heatmap"] = _save_current_figure(os.path.join(tmp_dir, "missing_heatmap.png"))

    if plot_cols:
        n = len(plot_cols)
        cols = 3
        rows = math.ceil(n / cols)
        _, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, col in zip(axes, plot_cols):
            s = df_plot[col].dropna()
            sns.histplot(s, bins=30, kde=True, ax=ax, color="#5B9BD5")
            ax.set_title(f"Distribucion: {col}")
            ax.set_xlabel(col)
            ax.set_ylabel("Frecuencia")
        for ax in axes[len(plot_cols):]:
            ax.axis("off")
        paths["numeric_distributions"] = _save_current_figure(os.path.join(tmp_dir, "numeric_distributions.png"))

    if plot_cols:
        n = len(plot_cols)
        cols = 3
        rows = math.ceil(n / cols)
        _, axes = plt.subplots(rows, cols, figsize=(18, 4.5 * rows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, col in zip(axes, plot_cols):
            sns.boxplot(x=df_plot[col], ax=ax, color="#A5A5A5")
            ax.set_title(f"Boxplot: {col}")
            ax.set_xlabel(col)
        for ax in axes[len(plot_cols):]:
            ax.axis("off")
        paths["numeric_boxplots"] = _save_current_figure(os.path.join(tmp_dir, "numeric_boxplots.png"))

    corr_cols = []
    for col in num_cols_all[:25]:
        s = df[col].dropna()
        if len(s) >= 100 and s.nunique() > 1:
            corr_cols.append(col)
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(numeric_only=True)
        corr = corr.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if corr.shape[0] >= 2:
            plt.figure(figsize=(12, 9))
            sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
            plt.title("Matriz de correlacion (variables numericas)")
            paths["correlation_matrix"] = _save_current_figure(os.path.join(tmp_dir, "correlation_matrix.png"))

    if target_col in df.columns and pd.api.types.is_numeric_dtype(df[target_col]):
        corr_base = [c for c in num_cols_all if c in df.columns and df[c].nunique(dropna=True) > 1]
        if corr_base:
            corr_target = (
                df[corr_base + [target_col]]
                .corr(numeric_only=True)[target_col]
                .drop(target_col)
                .sort_values(key=lambda s: s.abs(), ascending=False)
                .head(15)
            )
            corr_target = corr_target.replace([np.inf, -np.inf], np.nan).dropna()
            if not corr_target.empty:
                plt.figure(figsize=(11, 8))
                sns.barplot(
                    x=corr_target.values,
                    y=corr_target.index,
                    orient="h",
                    hue=corr_target.index,
                    palette="Blues_r",
                    legend=False,
                )
                plt.title("Top correlaciones con el target")
                plt.xlabel("Correlacion")
                plt.ylabel("Variable")
                paths["target_correlations"] = _save_current_figure(os.path.join(tmp_dir, "target_correlations.png"))

    if plot_cat_cols:
        n = len(plot_cat_cols)
        cols = 3
        rows = math.ceil(n / cols)
        _, axes = plt.subplots(rows, cols, figsize=(18, 4.8 * rows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, col in zip(axes, plot_cat_cols):
            top_vals = df_plot[col].astype(str).value_counts(dropna=False).head(10)
            sns.barplot(x=top_vals.values, y=top_vals.index, orient="h", ax=ax, color="#70AD47")
            ax.set_title(f"Top categorias: {col}")
            ax.set_xlabel("Frecuencia")
            ax.set_ylabel(col)
        for ax in axes[len(plot_cat_cols):]:
            ax.axis("off")
        paths["categorical_distributions"] = _save_current_figure(os.path.join(tmp_dir, "categorical_distributions.png"))

    if target_col in df_plot.columns and plot_cat_cols:
        rel_cols = plot_cat_cols[:4]
        n = len(rel_cols)
        cols = 2
        rows = math.ceil(n / cols)
        _, axes = plt.subplots(rows, cols, figsize=(16, 4.8 * rows))
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
        for ax, col in zip(axes, rel_cols):
            top_labels = df_plot[col].astype(str).value_counts(dropna=False).head(10).index
            subset = df_plot[df_plot[col].astype(str).isin(top_labels)]
            temp = pd.crosstab(subset[col].astype(str), subset[target_col], normalize="index")
            if not temp.empty:
                temp.plot(kind="bar", stacked=True, ax=ax, colormap="Blues")
                ax.set_title(f"{col} vs {target_col}")
                ax.set_ylabel("Proporcion")
                ax.set_xlabel(col)
                ax.legend(title=target_col, fontsize=8)
            else:
                ax.axis("off")
        for ax in axes[len(rel_cols):]:
            ax.axis("off")
        paths["categorical_vs_target"] = _save_current_figure(os.path.join(tmp_dir, "categorical_vs_target.png"))

    return paths


def eda_summary_text(df: pd.DataFrame, target_col: str) -> dict:
    """Return a compact EDA summary dictionary."""
    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "positive_rate": float(df[target_col].mean()) if target_col in df.columns else None,
        "missing_overall_pct": float(df.isna().mean().mean()),
        "n_numeric_cols": int(df.select_dtypes(include=["number"]).shape[1]),
        "n_categorical_cols": int(df.select_dtypes(include=["object", "category", "bool"]).shape[1]),
        "total_missing_cells": int(df.isna().sum().sum()),
        "top_missing_columns": df.isna().mean().sort_values(ascending=False).head(10).to_dict(),
    }

    if target_col in df.columns:
        summary["target_distribution"] = df[target_col].value_counts(normalize=True).to_dict()

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        summary["numeric_summary"] = df[num_cols].describe().to_dict()

    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if cat_cols:
        summary["categorical_cardinality"] = {col: int(df[col].nunique(dropna=False)) for col in cat_cols}
        summary["top_categories"] = {
            col: df[col].astype(str).value_counts(dropna=False).head(5).to_dict()
            for col in cat_cols[:10]
        }

    return summary
