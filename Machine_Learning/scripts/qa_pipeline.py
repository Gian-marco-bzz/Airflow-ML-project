"""
Pipeline unificado de QA para integridad de artefactos y auditoria NaN.

Uso:
    python scripts/qa_pipeline.py --mode artifacts
    python scripts/qa_pipeline.py --mode nan
    python scripts/qa_pipeline.py --mode full
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import get_logger
from src.qa.artifacts import run_artifact_checks
from src.qa.nan_audit import run_nan_audit

logger = get_logger(__name__)


def run_qa(mode: str, config_path: str, strict: bool) -> int:
    """Ejecuta validaciones QA segun el modo seleccionado."""
    if mode == "artifacts":
        logger.info("Ejecutando validaciones de integridad de artefactos...")
        return run_artifact_checks(config_path=config_path, strict=strict)

    if mode == "nan":
        logger.info("Ejecutando auditoria NaN del pipeline...")
        return run_nan_audit(config_path=config_path)

    if mode == "full":
        logger.info("Ejecutando QA completo (artefactos + auditoria NaN)...")
        rc_artifacts = run_artifact_checks(config_path=config_path, strict=strict)
        rc_nan = run_nan_audit(config_path=config_path)
        return 0 if (rc_artifacts == 0 and rc_nan == 0) else 1

    logger.error(f"Modo QA desconocido: {mode}")
    return 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline unificado de QA")
    parser.add_argument(
        "--mode",
        choices=["artifacts", "nan", "full"],
        default="full",
        help="Modo QA a ejecutar",
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Ruta al archivo de configuracion",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Habilita validaciones estrictas de artefactos",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    return run_qa(mode=args.mode, config_path=args.config, strict=args.strict)


if __name__ == "__main__":
    raise SystemExit(main())
