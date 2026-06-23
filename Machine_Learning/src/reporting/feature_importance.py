import os
import pandas as pd
import matplotlib.pyplot as plt


def compute_feature_importance(model, feature_names) -> pd.Series:

    """
    Extrae 'importancia' de features si el modelo la soporta.
    - RandomForest / XGBoost -> feature_importances_
    - LogisticRegression -> coef_
    """

    if hasattr(model, "feature_importances_"):

        imp = model.feature_importances_
        s = pd.Series(imp, index=feature_names).sort_values(ascending=False)
        return s

    elif hasattr(model, "coef_"):

        coef = model.coef_

        if coef.ndim > 1:
            coef = coef[0]

        s = pd.Series(abs(coef), index=feature_names).sort_values(ascending=False)
        return s

    else:
        return pd.Series(dtype=float)


def plot_feature_importance(importance: pd.Series, tmp_dir: str, top_n: int = 15) -> str:

    """
    Crea un gráfico de barras con las top N features y devuelve la ruta.
    """

    os.makedirs(tmp_dir, exist_ok=True)

    if importance.empty:
        return ""

    top = importance.head(top_n).sort_values()

    plt.figure(figsize=(12, 8))
    top.plot(kind="barh")

    plt.title(f"Top {top_n} - Importancia de variables", fontsize=16)
    plt.xlabel("Importancia", fontsize=12)
    plt.ylabel("Feature", fontsize=12)

    plt.grid(axis="x", linestyle="--", alpha=0.6)

    out_path = os.path.join(tmp_dir, "feature_importance.png")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return out_path