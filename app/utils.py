import json
import re
import unicodedata
from pathlib import Path

import joblib


def clean_column_name(name: str) -> str:
    """
    Normalizuje názov stĺpca do čistého formátu vhodného pre kľúče a zobrazenie.
    """
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


def build_drug_clean_maps(drug_groups: dict, cleaner=clean_column_name, max_len: int = 30):
    """
    Vytvorí mapy z "čistých" názvov liekov na ich skrátené a úplné názvy.
    """
    clean_to_abbr = {}
    clean_to_desc = {}

    for _, items in (drug_groups or {}).items():
        for raw in items:
            clean_name = cleaner(raw)

            short_name = str(raw).strip()
            if len(short_name) > max_len:
                short_name = short_name[: max_len - 1].rstrip() + "…"

            clean_to_abbr[clean_name] = short_name
            clean_to_desc[clean_name] = str(raw)

    return clean_to_abbr, clean_to_desc


def load_model_bundle(project_root: Path, model_cfg: dict) -> dict:
    """
    Načíta všetky súbory potrebné pre zvolený model.
    """
    model = joblib.load(project_root / model_cfg["model"])

    with open(project_root / model_cfg["threshold"], "r", encoding="utf-8") as file:
        threshold = float(json.load(file)["threshold"])

    with open(project_root / model_cfg["features"], "r", encoding="utf-8") as file:
        expected = list(json.load(file)["expected"])

    bundle = {
        "model": model,
        "threshold": threshold,
        "expected": expected,
    }

    for key in ("colmap", "preprocess", "score_cfg"):
        path = model_cfg.get(key)
        if path:
            with open(project_root / path, "r", encoding="utf-8") as file:
                bundle[key] = json.load(file)

    imputer_path = model_cfg.get("imputer")
    if imputer_path:
        full_path = project_root / imputer_path
        if full_path.exists():
            bundle["imputer"] = joblib.load(full_path)

    lime_df_ref_path = model_cfg.get("lime_df_ref")
    if lime_df_ref_path:
        full_path = project_root / lime_df_ref_path
        if full_path.exists():
            try:
                bundle["lime_df_ref"] = joblib.load(full_path)
            except Exception:
                bundle["lime_df_ref"] = None

    return bundle
