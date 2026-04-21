import base64
from io import BytesIO
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import shap
from lime.lime_tabular import LimeTabularExplainer

from ..utils import clean_column_name


def _fig_to_b64() -> str:
    '''Uloží aktuálny matplotlib graf do PNG v pamäti a vráti ho ako base64 string.'''
    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    out = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    plt.close()
    return out


def _pretty_feature_label(
    feat: str,
    reference_info: Dict[str, Dict[str, Any]],
    include_unit: bool = False,
) -> str:
    '''Získa "čistý" názov featury, vrátane jednotky ak je k dispozícii.'''
    info = reference_info.get(feat, {}) or {}
    abbr = info.get("abbr", feat)
    unit = info.get("unit", "")

    if include_unit and unit:
        return f"{abbr} ({unit})"
    return abbr


def _to_dataframe_like(original_X: pd.DataFrame, arr: Any) -> pd.DataFrame:
    '''Zabezpečí, že pole `arr` bude DataFrame s rovnakým indexom a vhodnými názvami stĺpcov.'''
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if arr.shape[1] == len(original_X.columns):
        cols = list(original_X.columns)
    else:
        cols = [f"x{i}" for i in range(arr.shape[1])]

    return pd.DataFrame(arr, columns=cols, index=original_X.index)



def _safe_clip(x: float, low: float, high: float) -> float:
    '''Orezáva hodnotu `x` do rozsahu [low, high], pričom nevyvolá chybu pri nečíselných hodnotách.'''
    try:
        return float(np.clip(float(x), low, high))
    except Exception:
        return 0.0


def _strength_label_from_abs_contrib(contrib_abs: float) -> str:
    '''Na základe absolútnej hodnoty príspevku určí slovný popis jeho "síly".'''
    if contrib_abs >= 2.0:
        return "veľmi silný"
    elif contrib_abs >= 1.0:
        return "silný"
    elif contrib_abs >= 0.5:
        return "stredný"
    elif contrib_abs >= 0.2:
        return "mierny"
    return "slabý"


def _fmt_odds_multiplier(multiplier: float) -> str:
    '''Formátuje násobiteľ pre odds (exp(coef)) do čitateľného tvaru.'''
    if not np.isfinite(multiplier):
        return "neurčené"
    return f"×{multiplier:.2f}"


def explain_shap_tree_clinical(
    model: Any,
    X: pd.DataFrame,
    expected: List[str],
    reference_info: Dict[str, dict],
    top_n: int = 10,
    exclude_features: Optional[set] = None,
    df_ref: Optional[pd.DataFrame] = None,
    target_name: Optional[str] = None,
    bkg_max_rows: int = 80,
    nsamples: int = 300,
    random_state: int = 42,
) -> Tuple[Optional[str], List[dict]]:
    """
    SHAP waterfall pre finálny kalibrovaný model.
    Vysvetľuje priamo model.predict_proba(...) pre pozitívnu triedu (CDI+).
    Graf je zobrazený v percentuálnych bodoch.
    """

    shap_waterfall_png = None
    factors: List[dict] = []

    try:
        import re
        from matplotlib.ticker import FuncFormatter

        exclude_features = set(exclude_features or set())

    
        # Príprava vstupu pacienta
        X_use = X.copy()

        for col in expected:
            if col not in X_use.columns:
                X_use[col] = 0.0

        X_use = X_use[expected].copy()

   
        # Príprava background datasetu
        if df_ref is not None:
            X_bkg = df_ref.copy()

            if target_name and target_name in X_bkg.columns:
                X_bkg = X_bkg.drop(columns=[target_name], errors="ignore")

            for col in expected:
                if col not in X_bkg.columns:
                    X_bkg[col] = 0.0

            X_bkg = X_bkg[expected].copy()
        else:
            # fallback, ak nie je referenčný dataset
            X_bkg = X_use.copy()

        if len(X_bkg) > bkg_max_rows:
            X_bkg = X_bkg.sample(n=bkg_max_rows, random_state=random_state)

       
        # Zistenie indexu pozitívnej triedy
        classes = getattr(model, "classes_", None)

        if classes is None and hasattr(model, "named_steps"):
            for step in reversed(list(model.named_steps.values())):
                if hasattr(step, "classes_"):
                    classes = step.classes_
                    break

        pos_idx = 1
        if classes is not None:
            cl = list(classes)
            if 1 in cl:
                pos_idx = cl.index(1)
            elif True in cl:
                pos_idx = cl.index(True)
            else:
                pos_idx = len(cl) - 1

  
        # Predikcia finálnej kalibrovanej pravdepodobnosti
        def pred_calibrated(arr: np.ndarray) -> np.ndarray:
            df_arr = pd.DataFrame(arr, columns=expected)
            proba = model.predict_proba(df_arr)
            proba = np.asarray(proba, dtype=float)

            if proba.ndim == 1:
                return proba

            if proba.shape[1] == 1:
                return proba[:, 0]

            return proba[:, pos_idx]

  
        # Kernel SHAP nad finálnym modelom
        np.random.seed(random_state)

        explainer = shap.KernelExplainer(
            pred_calibrated,
            X_bkg.values
        )

        sv_raw = explainer.shap_values(
            X_use.iloc[[0]].values,
            nsamples=nsamples
        )

        ssv = np.asarray(sv_raw, dtype=float)
        if ssv.ndim == 2:
            ssv = ssv[0]
        ssv = np.asarray(ssv, dtype=float).ravel()

        ev = float(np.asarray(explainer.expected_value).ravel()[0])

        # Finálna kalibrovaná predikcia pre kontrolu
        fx = float(pred_calibrated(X_use.iloc[[0]].values)[0])

      
        # Podiel vplyvu (%)
        abs_sum = float(np.sum(np.abs(ssv))) if np.isfinite(ssv).all() else 0.0
        if abs_sum <= 0:
            abs_sum = 1e-9

        x0 = X_use.iloc[0].to_dict()

        def fmt_val(v: Any) -> str:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "—"
            try:
                fv = float(v)
                if abs(fv) >= 100:
                    return f"{fv:.0f}"
                if abs(fv) >= 10:
                    return f"{fv:.1f}"
                return f"{fv:.2f}"
            except Exception:
                return str(v)

        for i, feat_clean in enumerate(expected):
            if feat_clean in exclude_features:
                continue

            info = reference_info.get(feat_clean, {}) or {}
            abbr = info.get("abbr", feat_clean)
            unit = info.get("unit", "")
            desc = info.get("desc", "")
            pretty_name = _pretty_feature_label(feat_clean, reference_info, include_unit=True)

            val = fmt_val(x0.get(feat_clean, X_use.iloc[0, i]))
            shap_i = float(ssv[i])
            impact_pct = float(abs(shap_i) / abs_sum * 100.0)

            factors.append({
                "feature": feat_clean,
                "name": pretty_name,
                "abbr": abbr,
                "desc": desc,
                "value": val,
                "unit": unit,
                "shap": shap_i,                 
                "shap_pp": shap_i * 100.0,      
                "impact_pct": round(impact_pct, 1),
                "direction": "up" if shap_i > 0 else ("down" if shap_i < 0 else "neutral"),
            })

        #  Waterfall graf
        #    - všetky SHAP hodnoty
        #    - zobrazenie len top_n cez max_display
        #    - v percentuálnych bodoch

        display_idx = [
            i for i, feat in enumerate(expected)
            if feat not in exclude_features
        ]

        ssv_display = ssv[display_idx]
        feat_display = [expected[i] for i in display_idx]

        feat_names_display = [
            _pretty_feature_label(fc, reference_info, include_unit=False)
            for fc in feat_display
        ]

        # Prepočet na percentuálne body
        ssv_display_pp = ssv_display * 100.0
        ev_pp = ev * 100.0
        fx_pp = fx * 100.0

        exp = shap.Explanation(
            values=ssv_display_pp,
            base_values=ev_pp,
            feature_names=feat_names_display,
        )

        plt.figure(figsize=(10, 7))
        shap.plots.waterfall(exp, max_display=top_n, show=False)

        ax = plt.gca()
        ax.set_title(
            "Vplyv na kalibrovanú pravdepodobnosť CDI",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Percentuálne body", fontsize=12)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}"))

        # odstránenie rušivých textov v stĺpcoch
        for t in ax.texts:
            txt = t.get_text().strip()
            if re.fullmatch(r"[+-]?0(\.0+)?", txt):
                t.set_text("")

        shap_waterfall_png = _fig_to_b64()

        # Zoradenie faktorov do tabuliek
        factors = sorted(factors, key=lambda d: abs(d["shap"]), reverse=True)

    except Exception as e:
        print("SHAP ERROR:", e)
        traceback.print_exc()

    return shap_waterfall_png, factors


def explain_odds_and_local_factors(
    model: Any,
    expected: List[str],
    vals: List[float],
    reference_info: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    
    '''Vysvetlenie pomocou OR (odds ratios) pre logistickú regresiu a extrakcia lokálnych faktorov pre konkrétneho pacienta.'''
    
    odds_png = None
    local_factors: List[Dict[str, Any]] = []

    try:
        logreg_model = model
        scale_step = None

        if hasattr(logreg_model, "base_estimator_") and logreg_model.base_estimator_ is not None:
            logreg_model = logreg_model.base_estimator_
        elif hasattr(logreg_model, "calibrated_classifiers_") and logreg_model.calibrated_classifiers_:
            try:
                logreg_model = logreg_model.calibrated_classifiers_[0].estimator
            except Exception:
                pass

        if hasattr(logreg_model, "named_steps"):
            if "scale" in logreg_model.named_steps:
                scale_step = logreg_model.named_steps["scale"]

            for step in reversed(list(logreg_model.named_steps.values())):
                if hasattr(step, "coef_"):
                    logreg_model = step
                    break

        coefs = np.asarray(logreg_model.coef_).ravel().astype(float)
        if len(coefs) != len(expected):
            raise ValueError("Nesedia rozmery coef vs features")

        vals_arr = np.asarray(vals, dtype=float).reshape(1, -1)
        if scale_step is not None:
            vals_scaled = scale_step.transform(vals_arr)[0]
        else:
            vals_scaled = vals_arr[0]

        pretty_expected = [
            _pretty_feature_label(f, reference_info, include_unit=True)
            for f in expected
        ]

        coefs_for_odds = np.clip(coefs, -6, 6)
        odds = np.exp(coefs_for_odds)

        idx = np.argsort(np.abs(coefs))[-10:]
        labels = np.array(pretty_expected)[idx]
        values_pct = (odds[idx] - 1.0) * 100.0

        # Farby: červená = zvyšuje riziko, zelená = znižuje riziko
        colors = []
        for v in values_pct:
            if v >= 0:
                colors.append("#d62728")
            else:
                colors.append("#2ca02c")

        fig, ax = plt.subplots(figsize=(9, 6))
        ax.barh(labels, values_pct, color=colors, height=0.6)
        ax.axvline(0, linestyle="--", color="#444444", linewidth=1.2)

        ax.set_xlabel("← Znižuje riziko CDI          Zvyšuje riziko CDI →", fontsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.tick_params(axis="x", labelsize=10)

        ax.set_title(
            "Najvýznamnejšie faktory – globálny prehľad (logistická regresia)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#d62728", label="Zvyšuje riziko CDI"),
            Patch(facecolor="#2ca02c", label="Znižuje riziko CDI"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

        plt.tight_layout()
        odds_png = _fig_to_b64()

        # Lokálne príspevky konkrétneho pacienta
        local_contrib = coefs * vals_scaled
        local_idx = np.argsort(np.abs(local_contrib))[-5:]

        for i in local_idx:
            contrib = float(local_contrib[i])
            contrib_clipped = _safe_clip(contrib, -6, 6)
            odds_multiplier = float(np.exp(contrib_clipped))

            if contrib > 0:
                effect = "zvyšuje modelové skóre"
            elif contrib < 0:
                effect = "znižuje modelové skóre"
            else:
                effect = "bez výrazného efektu"

            strength = _strength_label_from_abs_contrib(abs(contrib))

            local_factors.append({
                "feature": expected[i],
                "name": _pretty_feature_label(expected[i], reference_info, include_unit=True),
                "value": round(float(vals_arr[0][i]), 3),
                "scaled_value": round(float(vals_scaled[i]), 3),
                "impact": round(float(contrib), 3),
                "impact_display": _fmt_odds_multiplier(odds_multiplier),
                "effect": effect,
                "effect_display": f"{strength} – {effect}",
                "raw_contrib": round(contrib, 4),
                "odds_multiplier": round(odds_multiplier, 3),
            })

        local_factors = sorted(local_factors, key=lambda x: abs(x["raw_contrib"]), reverse=True)

    except Exception as e:
        print("ODDS/LOCAL FACTORS ERROR:", e)
        traceback.print_exc()

    return odds_png, local_factors


def explain_lime_clinical(
    model: Any,
    df_ref: pd.DataFrame,
    target_name: str,
    expected: List[str],
    X: pd.DataFrame,
    reference_info: Dict[str, Dict[str, Any]],
    num_features: int = 10,
    bkg_max_rows: int = 2000,
    random_state: int = 42,
    debug: bool = False,
    exclude_features: Optional[set] = None,
) -> Optional[str]:
    lime_png: Optional[str] = None
    ''' Vysvetlenie pomocou LIME pre konkrétneho pacienta. Graf zobrazuje faktory, ktoré zvyšujú alebo znižují riziko CDI pre daného pacienta.'''
    try:
        from matplotlib.ticker import FuncFormatter

        exclude_features = set(exclude_features or set())
        X_bkg = df_ref.copy()
        X_bkg.columns = [clean_column_name(c) for c in X_bkg.columns]

        target_clean = clean_column_name(target_name)
        if target_clean in X_bkg.columns:
            X_bkg = X_bkg.drop(columns=[target_clean])

        for col in expected:
            if col not in X_bkg.columns:
                X_bkg[col] = 0

        X_bkg = X_bkg[expected]

        if len(X_bkg) > bkg_max_rows:
            X_bkg = X_bkg.sample(n=bkg_max_rows, random_state=random_state)

        pretty_names = [
            _pretty_feature_label(f, reference_info, include_unit=False)
            for f in expected
        ]

        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            for step in reversed(list(model.named_steps.values())):
                if hasattr(step, "classes_"):
                    classes = step.classes_
                    break

        pos_index = 1
        neg_index = 0
        if classes is not None:
            cl = list(classes)
            if 1 in cl:
                pos_index = cl.index(1)
            elif True in cl:
                pos_index = cl.index(True)
            else:
                pos_index = len(cl) - 1

            if 0 in cl:
                neg_index = cl.index(0)
            elif False in cl:
                neg_index = cl.index(False)
            else:
                neg_index = 0

        explainer = LimeTabularExplainer(
            training_data=X_bkg.values,
            feature_names=pretty_names,
            class_names=["CDI−", "CDI+"],
            discretize_continuous=True,
            random_state=random_state,
            verbose=False,
            mode="classification",
        )

        def lime_pred(z: np.ndarray) -> np.ndarray:
            df_z = pd.DataFrame(z, columns=expected)
            proba = model.predict_proba(df_z)
            proba = np.asarray(proba, dtype=float)

            if proba.ndim == 1:
                proba = np.vstack([1 - proba, proba]).T
            if proba.shape[1] == 1:
                proba = np.hstack([1 - proba, proba])

            if proba.shape[1] >= 2:
                proba = proba[:, [neg_index, pos_index]]

            return proba

        if debug:
            test = lime_pred(X.iloc[0][expected].values.reshape(1, -1))
            print("[LIME DEBUG] proba:", test, "classes_:", classes)

        row = X.iloc[0][expected].astype(float).values

        exp = explainer.explain_instance(
            data_row=row,
            predict_fn=lime_pred,
            num_features=num_features,
            labels=[1],
            num_samples=10000,
        )

        available_labels = sorted(list(getattr(exp, "local_exp", {}).keys()))
        if debug:
            print("[LIME DEBUG] available_labels:", available_labels)

        label_to_plot = 1 if 1 in available_labels else (available_labels[0] if available_labels else 1)

        weights = exp.as_list(label=label_to_plot)
        filtered = []

        clean_to_pretty = {
            f: _pretty_feature_label(f, reference_info, include_unit=False)
            for f in expected
        }
        excluded_pretty = {clean_to_pretty[f] for f in exclude_features if f in clean_to_pretty}

        for rule, w in weights:
            if any(p in rule for p in excluded_pretty):
                continue
            filtered.append((rule, w))

        filtered = sorted(filtered, key=lambda t: abs(t[1]), reverse=True)[:num_features]
        filtered = list(reversed(filtered))

        labels = [r for r, _ in filtered]
        vals_w = np.array([float(w) for _, w in filtered], dtype=float)

        # Prepočet na percentuálne body
        vals_w_pp = vals_w * 100.0

        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.barh(range(len(labels)), vals_w_pp)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=10)
        ax.tick_params(axis="x", labelsize=10)

        for patch, w in zip(ax.patches, vals_w_pp):
            if w >= 0:
                patch.set_facecolor("#d62728")
            else:
                patch.set_facecolor("#2ca02c")

        ax.set_title(
            "Faktory ovplyvňujúce predikciu pre tohto pacienta (LIME)",
            fontsize=13,
            fontweight="bold",
            pad=12,
        )

        ax.set_xlabel("← Znižuje riziko CDI          Zvyšuje riziko CDI →", fontsize=11)
        ax.axvline(0, linestyle="--", linewidth=1.2, color="#444444")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}"))

        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="#d62728", label="Zvyšuje riziko CDI"),
            Patch(facecolor="#2ca02c", label="Znižuje riziko CDI"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9)

        plt.tight_layout()
        lime_png = _fig_to_b64()

        if debug:
            true_proba = float(model.predict_proba(X.iloc[[0]][expected])[0][pos_index])
            print(f"[LIME DEBUG] true_proba={true_proba:.6f}")
            print(f"[LIME DEBUG] local_pred={getattr(exp, 'local_pred', None)}")
            print(f"[LIME DEBUG] predict_proba={getattr(exp, 'predict_proba', None)}")

    except Exception as e:
        print("LIME ERROR:", e)
        traceback.print_exc()

    return lime_png