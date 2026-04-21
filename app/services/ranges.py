from pathlib import Path
import pandas as pd

from ..utils import clean_column_name


def compute_global_ranges(project_root: Path, available_models: dict) -> dict:
    '''Pre každý numerický vstupný parameter spočíta globálny rozsah (min, max).'''
    global_ranges = {}

    for _, cfg in available_models.items():
        df = pd.read_csv(project_root / cfg["dataset"], sep=";")
        tgt = cfg["target"]

        for col in df.columns:
            if col == tgt:
                continue
            if df[col].dtype == object:
                continue

            clean = clean_column_name(col)
            min_val = float(df[col].min())
            max_val = float(df[col].max())

            if clean not in global_ranges:
                global_ranges[clean] = [min_val, max_val]
            else:
                global_ranges[clean][0] = min(global_ranges[clean][0], min_val)
                global_ranges[clean][1] = max(global_ranges[clean][1], max_val)

    return {k: (round(v[0], 2), round(v[1], 2)) for k, v in global_ranges.items()}
