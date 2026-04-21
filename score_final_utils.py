# Tento modul implementuje podporu pre:
#  - štandardizáciu názvov stĺpcov 
#  - výber laboratórnych parametrov z tréningových dát 
#  - návrh jednoduchého odvodeného atribútu "Skore_final",
#    ktorý sumarizuje rizikové signály z vybraných laboratórnych premenných
#    a binárnych premenných (komorbidity, lieky)
#  - generovanie reportov pre DP
#  - pomocné funkcie pre zostavenie finálneho zoznamu featur a uloženie
#    očakávaného poradia stĺpcov pre nasadenie do aplikácie

import json
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact
from sklearn.metrics import roc_curve



# Helpers: čistenie názvov stĺpcov
def clean_column_name(name: str) -> str:
    """Upraví názov stĺpca do tvaru bez diakritiky a špeciálnych znakov."""
    name = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("_")
    return name


def _clean_list(cols):
    return [clean_column_name(c) for c in cols]


# BH-FDR korekcia (Benjamini–Hochberg)
def bh_fdr(pvals: pd.Series) -> pd.Series:
    p = pvals.values.astype(float)
    n = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, n + 1, dtype=float)

    q_sorted = p_sorted * n / ranks
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0.0, 1.0)

    q = np.empty_like(q_sorted)
    q[order] = q_sorted
    return pd.Series(q, index=pvals.index)


# 3) Výber lab parametrov (MWU + point-biserial korelácia)
def select_labs_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lab_cols: list,
    alpha: float = 0.05,
    min_abs_r: float = 0.05,
    min_non_nan: int = 30,
    min_pos: int = 5,
):
    y = pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int)

    rows_mwu = []
    rows_corr = []

    for col in lab_cols:
        if col not in X_train.columns:
            continue

        s = pd.to_numeric(X_train[col], errors="coerce")
        mask = s.notna()
        if mask.sum() < min_non_nan:
            continue

        yy = y[mask]
        ss = s[mask]

        if yy.sum() < min_pos or yy.nunique() < 2:
            continue

        x_pos = ss[yy == 1].values
        x_neg = ss[yy == 0].values

        # Mann–Whitney U test
        try:
            p_mwu = stats.mannwhitneyu(x_pos, x_neg, alternative="two-sided").pvalue
        except Exception:
            p_mwu = np.nan

        mean_pos = float(np.mean(x_pos)) if len(x_pos) else np.nan
        mean_neg = float(np.mean(x_neg)) if len(x_neg) else np.nan
        med_pos = float(np.median(x_pos)) if len(x_pos) else np.nan
        med_neg = float(np.median(x_neg)) if len(x_neg) else np.nan

        rows_mwu.append({
            "Name": col,
            "p_value": float(p_mwu) if np.isfinite(p_mwu) else np.nan,
            "mean_pos": mean_pos,
            "mean_neg": mean_neg,
            "median_pos": med_pos,
            "median_neg": med_neg,
            "delta_mean": mean_pos - mean_neg if np.isfinite(mean_pos) and np.isfinite(mean_neg) else np.nan,
            "delta_median": med_pos - med_neg if np.isfinite(med_pos) and np.isfinite(med_neg) else np.nan,
        })

        # Point-biserial korelácia
        try:
            r, p_corr = stats.pointbiserialr(yy.values, ss.values)
            rows_corr.append({"Name": col, "Correlation": float(r), "corr_p_value": float(p_corr)})
        except Exception:
            pass

    diff_table = pd.DataFrame(rows_mwu)
    corr_table = pd.DataFrame(rows_corr)

    if diff_table.empty or corr_table.empty:
        return [], diff_table, corr_table

    diff_table["q_value"] = bh_fdr(diff_table["p_value"].fillna(1.0))
    corr_table["corr_q_value"] = bh_fdr(corr_table["corr_p_value"].fillna(1.0))

    diff_table["Significant_FDR"] = np.where(diff_table["q_value"] < alpha, "Yes", "No")
    corr_table["Significant_FDR"] = np.where(
        (corr_table["corr_q_value"] < alpha) & (corr_table["Correlation"].abs() >= min_abs_r),
        "Yes", "No"
    )

    diff_sig = diff_table[diff_table["q_value"] < alpha].copy()
    corr_sig = corr_table[
        (corr_table["corr_q_value"] < alpha) & (corr_table["Correlation"].abs() >= min_abs_r)
    ].copy()

    selected = sorted(set(diff_sig["Name"]).union(set(corr_sig["Name"])))
    return selected, diff_table.sort_values("q_value"), corr_table.sort_values("corr_q_value")



# ROC thresholdy pre lab premenné (pravidlá do Skore_final)
def compute_lab_rules_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    lab_vars: list,
    min_non_nan: int = 30,
    min_unique: int = 5,
    min_pos: int = 5,
):
    rules = []
    y = pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int)

    for col in lab_vars:
        if col not in X_train.columns:
            continue

        x_full = pd.to_numeric(X_train[col], errors="coerce")
        mask = x_full.notna()
        x = x_full[mask]
        yy = y[mask]

        if len(x) < min_non_nan:
            continue
        if x.nunique() < min_unique:
            continue
        if yy.sum() < min_pos or yy.nunique() < 2:
            continue

        fpr, tpr, thr = roc_curve(yy, x)
        youden = tpr - fpr

        finite_mask = np.isfinite(thr)
        thr2 = thr[finite_mask] if finite_mask.any() else thr
        youden2 = youden[finite_mask] if finite_mask.any() else youden

        idx = int(np.argmax(youden2))
        best_thr = float(thr2[idx])

        med_pos = float(pd.to_numeric(X_train.loc[y == 1, col], errors="coerce").median())
        med_neg = float(pd.to_numeric(X_train.loc[y == 0, col], errors="coerce").median())
        direction = "higher_risk" if med_pos > med_neg else "lower_risk"

        rules.append({"col": col, "thr": best_thr, "direction": direction})

    return rules



# FIT score config – návrh pravidiel pre Skore_final + výber binárnych rizikových stĺpcov
def fit_score_config(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    core_labs: list,
    base_vars: list,
    lab_cols_all: list | None = None,
    do_lab_selection: bool = True,
    alpha: float = 0.05,
    min_abs_r: float = 0.05,
    comorbid_cols: list | None = None,
    drug_cols: list | None = None,
    exclude_from_score: set | None = None,
    min_non_nan: int = 30,
    min_unique: int = 5,
    min_pos: int = 5,
):
    exclude_from_score = exclude_from_score or set()

    core_labs_c = _clean_list(core_labs)
    base_vars_c = _clean_list(base_vars)
    exclude_c = set(_clean_list(list(exclude_from_score)))

    selected_labs = []
    if lab_cols_all is not None and do_lab_selection:
        lab_cols_all_c = _clean_list(lab_cols_all)
        lab_cols_present = [c for c in lab_cols_all_c if c in X_fit.columns]
        selected_labs, _, _ = select_labs_train_only(
            X_fit, y_fit,
            lab_cols=lab_cols_present,
            alpha=alpha,
            min_abs_r=min_abs_r,
            min_non_nan=min_non_nan,
            min_pos=min_pos
        )

    lab_set = sorted(set(core_labs_c).union(set(selected_labs)))
    lab_vars_final = [c for c in (base_vars_c + lab_set) if c in X_fit.columns]

    lab_rules_all = compute_lab_rules_train_only(
        X_fit, y_fit, lab_vars_final,
        min_non_nan=min_non_nan, min_unique=min_unique, min_pos=min_pos
    )

    lab_rules = [r for r in lab_rules_all if r["col"] not in exclude_c]

    bin_cols = []
    if comorbid_cols:
        bin_cols += _clean_list(comorbid_cols)
    if drug_cols:
        bin_cols += _clean_list(drug_cols)
    bin_cols = [c for c in bin_cols if c in X_fit.columns]

    binary_risk_cols, binary_audit = compute_binary_risk_cols_train_only(
        X_train=X_fit,
        y_train=y_fit,
        binary_cols=bin_cols,
        alpha=alpha,
        min_present=10,
        min_pos_present=3,
        min_or=1.2,
    )

    return {
        "lab_rules": lab_rules,
        "binary_risk_cols": binary_risk_cols,
        "meta": {
            "base_vars": base_vars_c,
            "core_labs": core_labs_c,
            "selected_labs": selected_labs,
            "lab_vars_final": lab_vars_final,
            "excluded_from_score": sorted(list(exclude_c)),
            "binary_total_present": len(bin_cols),
            "binary_risk_selected": len(binary_risk_cols),
        },
    }



# Výpočet Skore_final

def apply_score_config(X: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    X2 = X.copy()
    score = np.zeros(len(X2), dtype=int)

    for r in cfg["lab_rules"]:
        col = r["col"]
        thr = float(r["thr"])
        direction = r["direction"]

        if col not in X2.columns:
            continue

        x = pd.to_numeric(X2[col], errors="coerce")
        if direction == "higher_risk":
            score += (x > thr).fillna(False).astype(int).to_numpy()
        else:
            score += (x < thr).fillna(False).astype(int).to_numpy()

    bin_cols = [c for c in cfg.get("binary_risk_cols", []) if c in X2.columns]
    if bin_cols:
        tmp = X2[bin_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        tmp = (tmp > 0).astype(int)
        score += tmp.sum(axis=1).to_numpy()

    X2["Skore_final"] = score.astype(int)
    return X2



# Štatistiky do DP
def _to_txt_table(df: pd.DataFrame, cols: list[str], top_k: int = 30) -> str:
    if df is None or df.empty:
        return "(empty)\n"
    cols = [c for c in cols if c in df.columns]
    return df[cols].head(top_k).to_string(index=False) + "\n"


def compute_and_save_train_lab_stats_for_dp(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    lab_cols_all: list[str],
    out_dir: str | Path,
    tag: str,
    alpha: float = 0.05,
    min_abs_r: float = 0.05,
    min_non_nan: int = 30,
    min_pos: int = 5,
    top_k: int = 30,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lab_cols_all_c = _clean_list(lab_cols_all)
    lab_present = [c for c in lab_cols_all_c if c in X_fit.columns]

    selected, diff_tbl, corr_tbl = select_labs_train_only(
        X_train=X_fit,
        y_train=y_fit,
        lab_cols=lab_present,
        alpha=alpha,
        min_abs_r=min_abs_r,
        min_non_nan=min_non_nan,
        min_pos=min_pos,
    )

    diff_csv = out_dir / f"train_mwu_all_{tag}.csv"
    corr_csv = out_dir / f"train_corr_all_{tag}.csv"
    sel_txt  = out_dir / f"train_lab_stats_report_{tag}.txt"

    if diff_tbl is not None and not diff_tbl.empty:
        diff_tbl.to_csv(diff_csv, index=False, encoding="utf-8")
    if corr_tbl is not None and not corr_tbl.empty:
        corr_tbl.to_csv(corr_csv, index=False, encoding="utf-8")

    with open(sel_txt, "w", encoding="utf-8") as f:
        f.write(f"TRAIN LAB STATS (train-only) | {tag}\n")
        f.write("=" * 90 + "\n\n")
        f.write(f"alpha={alpha} | min_abs_r={min_abs_r} | min_non_nan={min_non_nan} | min_pos={min_pos}\n")
        f.write(f"labs_present={len(lab_present)} | selected_union={len(selected)}\n\n")

        f.write("SELECTED_LABS (union):\n")
        f.write(", ".join(selected) + "\n\n")

        f.write(f"TOP {top_k} MWU (sorted by q_value):\n")
        f.write(_to_txt_table(diff_tbl, cols=[
            "Name", "p_value", "q_value",
            "mean_pos", "mean_neg", "median_pos", "median_neg",
            "delta_mean", "delta_median",
            "Significant_FDR",
        ], top_k=top_k))
        f.write("\n")

        f.write(f"TOP {top_k} CORR (sorted by corr_q_value):\n")
        f.write(_to_txt_table(corr_tbl, cols=[
            "Name", "Correlation", "corr_p_value", "corr_q_value",
            "Significant_FDR",
        ], top_k=top_k))
        f.write("\n")

    print("\n" + "=" * 90)
    print(f"TRAIN LAB STATS (DP) | {tag}")
    print("=" * 90)
    print(f"labs_present={len(lab_present)} | selected={len(selected)}")
    print(f"Saved: {sel_txt}")
    if diff_tbl is not None and not diff_tbl.empty:
        print(f"Saved: {diff_csv}")
    if corr_tbl is not None and not corr_tbl.empty:
        print(f"Saved: {corr_csv}")

    return {"selected": selected}



# Strength ranking + výber TOP labov (MWU + corr)
def pick_top_labs_by_stats(
    diff_table: pd.DataFrame,
    corr_table: pd.DataFrame,
    top_n: int = 15,
    alpha: float = 0.05,
    min_abs_r: float = 0.05,
) -> tuple[list[str], pd.DataFrame]:
    # MWU časť
    mwu = diff_table.copy()
    if "q_value" not in mwu.columns:
        raise ValueError("diff_table musí obsahovať 'q_value'")

    mwu = mwu[["Name", "q_value"]].dropna()
    mwu["q_value"] = mwu["q_value"].astype(float).clip(1e-300, 1.0)
    mwu = mwu[mwu["q_value"] < alpha].copy()
    mwu["mwu_strength"] = -np.log10(mwu["q_value"])

    # Point-biserial korelácia 
    corr = corr_table.copy()
    if "corr_q_value" not in corr.columns:
        raise ValueError("corr_table musí obsahovať 'corr_q_value'")

    corr = corr[["Name", "Correlation", "corr_q_value"]].dropna()
    corr["corr_q_value"] = corr["corr_q_value"].astype(float).clip(1e-300, 1.0)
    corr["Correlation"] = corr["Correlation"].astype(float)
    corr = corr[(corr["corr_q_value"] < alpha) & (corr["Correlation"].abs() >= min_abs_r)].copy()
    corr["corr_strength"] = corr["Correlation"].abs() * (-np.log10(corr["corr_q_value"]))

    # MWU + corr do jedného rankingu
    ranked = pd.merge(
        mwu[["Name", "mwu_strength"]],
        corr[["Name", "corr_strength"]],
        on="Name",
        how="outer",
    ).fillna(0.0)

    ranked["strength"] = ranked["mwu_strength"] + ranked["corr_strength"]
    ranked = ranked.sort_values("strength", ascending=False).reset_index(drop=True)

    top_labs = ranked["Name"].head(top_n).tolist()
    return top_labs, ranked



# Zostavenie finálneho zoznamu labov
def compose_final_labs_stats_plus_must(
    top_labs: list[str],
    ranked: pd.DataFrame,
    must_labs: list[str],
    target_total: int = 15,
) -> list[str]:
    top = list(dict.fromkeys(top_labs))
    must = list(dict.fromkeys(must_labs))

    final = top.copy()
    for m in must:
        if m not in final:
            final.append(m)

    if len(final) < target_total:
        for name in ranked["Name"].tolist():
            if name not in final:
                final.append(name)
            if len(final) >= target_total:
                break

    if len(final) > target_total:
        strength_map = dict(zip(ranked["Name"].tolist(), ranked["strength"].tolist())) if (
            ranked is not None and not ranked.empty and "Name" in ranked.columns and "strength" in ranked.columns
        ) else {}

        must_set = set(must)
        removable = [x for x in final if x not in must_set]
        removable_sorted = sorted(removable, key=lambda x: float(strength_map.get(x, -1e18)))

        while len(final) > target_total and removable_sorted:
            drop = removable_sorted.pop(0)
            if drop in final:
                final.remove(drop)

        final = final[:target_total]

    return final



# Overlap report
def overlap_report(core_clean: list[str], top_clean: list[str]) -> dict:
    core = set(core_clean)
    top = set(top_clean)
    inter = sorted(list(core & top))
    return {
        "core_n": len(core_clean),
        "top_n": len(top_clean),
        "intersection_n": len(inter),
        "intersection": inter,
        "core_missing_in_top": sorted(list(core - top)),
        "top_not_in_core": sorted(list(top - core)),
        "coverage_core_in_top_pct": round(100.0 * len(inter) / max(len(core), 1), 1),
    }



# Features helpers + expected JSON
def build_feature_list_for_model(
    final_labs_clean: list[str],
    base_vars: list[str],
    comorbid_cols: list[str],
    drug_cols: list[str],
    include_score: bool,
) -> list[str]:
    base_c = _clean_list(base_vars)
    bin_c = _clean_list(comorbid_cols) + _clean_list(drug_cols)

    feats = []
    feats += base_c
    feats += list(dict.fromkeys(final_labs_clean))
    feats += list(dict.fromkeys(bin_c))

    if include_score:
        feats += [clean_column_name("Skore_final")]

    return list(dict.fromkeys(feats))


def slice_X_to_features(X: pd.DataFrame, features: list[str], tag: str = "") -> pd.DataFrame:
    present = [c for c in features if c in X.columns]
    missing = [c for c in features if c not in X.columns]
    if missing:
        print(f"WARN missing features{(' | ' + tag) if tag else ''} (first 20):", missing[:20])
    return X[present].copy()


def save_features_expected(out_dir: str | Path, base: str, expected: list[str]) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{base}_features.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"expected": list(expected)}, f, ensure_ascii=False, indent=2)
    return p



# Binárne rizikové stĺpce
def compute_binary_risk_cols_train_only(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    binary_cols: list[str],
    alpha: float = 0.05,
    min_present: int = 10,
    min_pos_present: int = 3,
    min_or: float = 1.2,
):
    y = pd.to_numeric(y_train, errors="coerce").fillna(0).astype(int)

    rows = []
    for col in binary_cols:
        if col not in X_train.columns:
            continue

        x = pd.to_numeric(X_train[col], errors="coerce").fillna(0)
        x = (x > 0).astype(int)

        present = int(x.sum())
        pos_present = int(((x == 1) & (y == 1)).sum())

        if present < min_present:
            continue
        if pos_present < min_pos_present:
            continue
        if y.nunique() < 2:
            continue

        a = int(((x == 1) & (y == 1)).sum())
        b = int(((x == 1) & (y == 0)).sum())
        c = int(((x == 0) & (y == 1)).sum())
        d = int(((x == 0) & (y == 0)).sum())

        try:
            _, p = fisher_exact([[a, b], [c, d]], alternative="two-sided")
        except Exception:
            p = 1.0

        or_hat = ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))

        rows.append({
            "Name": col,
            "a_pos_present": a,
            "b_neg_present": b,
            "c_pos_absent": c,
            "d_neg_absent": d,
            "present": present,
            "pos_present": pos_present,
            "OR": float(or_hat),
            "p_value": float(p),
        })

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return [], tbl

    tbl["q_value"] = bh_fdr(tbl["p_value"])
    tbl = tbl.sort_values(["q_value", "OR"], ascending=[True, False]).reset_index(drop=True)

    risk_tbl = tbl[(tbl["q_value"] < alpha) & (tbl["OR"] > float(min_or))].copy()
    risk_cols = risk_tbl["Name"].tolist()

    return risk_cols, tbl