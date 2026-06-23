"""State management for incremental processing.
Stores processed files in BOTH:
- Airflow Variable (metadata DB)
- Disk JSON file (/opt/airflow/data/state/...)
"""
from __future__ import annotations
import json
import os
from typing import Dict, Set


def _safe_read_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def _safe_write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def load_disk_state(state_file: str) -> Set[str]:
    data = _safe_read_json(state_file)
    files = data.get("processed_files", [])
    return set(files) if isinstance(files, list) else set()


def save_disk_state(state_file: str, processed_files: Set[str]) -> None:
    _safe_write_json(state_file, {"processed_files": sorted(processed_files)})


def load_airflow_state(variable_name: str) -> Set[str]:
    """Load processed file list from Airflow Variable."""
    try:
        from airflow.models import Variable
        raw = Variable.get(variable_name, default_var="[]")
        data = json.loads(raw)
        return set(data) if isinstance(data, list) else set()
    except Exception:
        return set()


def save_airflow_state(variable_name: str, processed_files: Set[str]) -> None:
    try:
        from airflow.models import Variable
        Variable.set(variable_name, json.dumps(sorted(processed_files)))
    except Exception:
        pass