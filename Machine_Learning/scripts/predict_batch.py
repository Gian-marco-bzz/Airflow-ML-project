"""
Script de inferencia por lotes para modelos entrenados.

Carga un modelo y preprocesador guardados, y genera predicciones
sobre datos nuevos de trafico.

Uso:
    python scripts/predict_batch.py \
        --data "ruta/a/nuevo_trafico.parquet" \
        --model "outputs/ml/models/xgboost.pkl" \
        --output "outputs/ml/predictions.csv"
"""

import sys
import argparse
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import get_logger, ensure_dir
from src.inference.predict import predict_with_threshold

logger = get_logger(__name__)


def main() -> int:
    """Ejecuta la prediccion por lotes."""

    # ========================================================================
    # Parseo de argumentos
    # ========================================================================
    parser = argparse.ArgumentParser(
        description="Inferencia por lotes sobre datos nuevos de trafico",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python scripts/predict_batch.py \\
    --data data/raw/new_traffic.parquet \\
    --model outputs/ml/models/xgboost.pkl \\
    --output predictions.csv

  python scripts/predict_batch.py \\
    --data new_training_data.csv \\
    --model outputs/ml/models/random_forest.pkl \\
    --use-preprocessor outputs/ml/models/preprocessor.pkl \\
    --output scores.csv
        """,
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Ruta de datos de entrada (parquet o CSV)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Ruta al modelo entrenado (*.pkl)",
    )
    parser.add_argument(
        "--use-preprocessor",
        default="outputs/ml/models/preprocessor.pkl",
        help="Ruta al preprocesador (default: outputs/ml/models/preprocessor.pkl)",
    )
    parser.add_argument(
        "--output",
        default="outputs/ml/predictions.csv",
        help="Ruta de salida (default: outputs/ml/predictions.csv)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Umbral de clasificacion (default: 0.70)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10000,
        help="Procesar datos en lotes (default: 10000)",
    )

    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        logger.error("El umbral debe estar en [0, 1]")
        return 1
    if args.batch_size <= 0:
        logger.error("El tamano de lote debe ser > 0")
        return 1

    # ========================================================================
    # Preparacion
    # ========================================================================
    logger.info("=" * 80)
    logger.info("PREDICCION POR LOTES")
    logger.info("=" * 80)

    try:
        # Cargar datos
        logger.info(f"\n[1] Cargando datos desde {args.data}...")
        if args.data.endswith(".parquet"):
            X = pd.read_parquet(args.data)
        else:
            X = pd.read_csv(args.data)

        logger.info(f"✓ Cargadas {len(X):,} filas, {X.shape[1]} variables")

        # ====================================================================
        # Cargar modelo y preprocesador
        # ====================================================================
        logger.info(f"\n[2] Cargando modelo desde {args.model}...")
        model = joblib.load(args.model)
        logger.info(f"✓ Modelo cargado (tipo: {type(model).__name__})")

        logger.info(f"\n[3] Cargando preprocesador desde {args.use_preprocessor}...")
        try:
            preprocessor = joblib.load(args.use_preprocessor)
            logger.info("✓ Preprocesador cargado")
            has_preprocessor = True
        except FileNotFoundError:
            logger.warning("Preprocesador no encontrado, se usaran variables en crudo")
            has_preprocessor = False

        # ====================================================================
        # Prediccion
        # ====================================================================
        logger.info(f"\n[4] Generando predicciones (threshold={args.threshold})...")

        if has_preprocessor:
            X_clean = preprocessor.transform(X)
            logger.info(f"✓ Preprocesador aplicado: {len(X_clean):,} filas")
        else:
            X_clean = X

        # Prediccion en lotes
        predictions_prob = []
        predictions_class = []

        for batch_start in range(0, len(X_clean), args.batch_size):
            batch_end = min(batch_start + args.batch_size, len(X_clean))
            X_batch = X_clean.iloc[batch_start:batch_end]

            prob, pred = predict_with_threshold(
                model,
                X_batch,
                classification_threshold=args.threshold,
            )

            predictions_prob.extend(prob)
            predictions_class.extend(pred)

            logger.info(
                f"  Procesadas {batch_end:,}/{len(X_clean):,} muestras "
                f"({100*batch_end/len(X_clean):.1f}%)"
            )

        # ====================================================================
        # Guardar resultados
        # ====================================================================
        logger.info(f"\n[5] Guardando resultados en {args.output}...")

        results = pd.DataFrame({
            "risk_probability": predictions_prob,
            "risk_prediction": predictions_class,
        })

        output_path = Path(args.output)
        ensure_dir(str(output_path.parent))
        results.to_csv(output_path, index=False)

        logger.info(f"✓ Guardado en {output_path}")
        logger.info(
            f"\nResumen:"
            f"\n  Total de predicciones: {len(results):,}"
            f"\n  Predicciones positivas: {results['risk_prediction'].sum():,} "
            f"({100*results['risk_prediction'].mean():.2f}%)"
            f"\n  Probabilidad media: {results['risk_probability'].mean():.4f}"
        )

        logger.info("\n" + "=" * 80)
        logger.info("PREDICCION POR LOTES COMPLETADA")
        logger.info("=" * 80)

        return 0

    except Exception as exc:
        logger.error(f"\nFALLO EN PREDICCION: {exc}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
