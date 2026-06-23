"""Despachador unificado de ejecución para ML, EDA y QA."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scripts.train_pipeline import main as run_ml
from scripts.eda_pipeline import main as run_eda
from scripts.qa_pipeline import run_qa


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecutar pipelines del proyecto")
    parser.add_argument(
        "--mode",
        choices=["ml", "eda", "qa", "full"],
        default="ml",
        help="Modo de pipeline a ejecutar",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta al archivo de configuracion",
    )
    parser.add_argument(
        "--ml-only",
        action="store_true",
        help="Para mode=ml: omite artefactos visuales adicionales",
    )
    parser.add_argument(
        "--qa-mode",
        choices=["artifacts", "nan", "full"],
        default="full",
        help="Para mode=qa: selecciona submodo de QA",
    )
    parser.add_argument(
        "--strict-qa",
        action="store_true",
        help="Para artefactos QA: habilita validaciones estrictas",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.mode == "ml":
        return run_ml(config_path=args.config, include_eda_report=not args.ml_only)

    if args.mode == "eda":
        return run_eda(config_path=args.config)

    if args.mode == "qa":
        return run_qa(mode=args.qa_mode, config_path=args.config, strict=args.strict_qa)

    # full = ml -> eda -> qa(artifacts)
    rc_ml = run_ml(config_path=args.config, include_eda_report=not args.ml_only)
    rc_eda = run_eda(config_path=args.config)
    rc_qa = run_qa(mode="artifacts", config_path=args.config, strict=args.strict_qa)
    return 0 if (rc_ml == 0 and rc_eda == 0 and rc_qa == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
