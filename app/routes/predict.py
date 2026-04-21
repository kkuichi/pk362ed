import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, abort, render_template, request

from ..constants import (
    AVAILABLE_MODELS,
    DEMOGRAFIA,
    DRUG_GROUPS,
    KOMORBIDITIES,
    LAB_GROUPS,
    MODEL_LABELS,
    REFERENCE_INFO,
)
from ..services.explainability import (
    explain_lime_clinical,
    explain_odds_and_local_factors,
    explain_shap_tree_clinical,
)
from ..services.ranges import compute_global_ranges
from ..utils import build_drug_clean_maps, clean_column_name, load_model_bundle


# Blueprint pre časť aplikácie, ktorá rieši predikciu
predict_bp = Blueprint("predict", __name__)

# Cesta ku koreňu projektu
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_global_ranges() -> dict:
    """
    Načíta minimá a maximá pre numerické vstupy.
    Tieto rozsahy sa potom používajú vo formulári napríklad na kontrolu hodnôt.
    """
    try:
        return compute_global_ranges(PROJECT_ROOT, AVAILABLE_MODELS)
    except Exception:
        return {}


# Rozsahy vstupov dostupné pre formulár
GLOBAL_RANGES = _load_global_ranges()


def _load_score_function():
    """
    Zo súboru score_final_utils.py načíta funkciu apply_score_config.
    Táto funkcia sa používa vtedy, keď model pracuje aj s odvodeným atribútom
    Skore_final.
    """
    score_utils_path = PROJECT_ROOT / "score_final_utils.py"
    spec = importlib.util.spec_from_file_location("score_final_utils", score_utils_path)
    if spec is None or spec.loader is None:
        raise ImportError("Nepodarilo sa načítať score_final_utils.py")

    score_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(score_utils)
    return score_utils.apply_score_config


# Pomocná funkcia na výpočet Skore_final
apply_score_config = _load_score_function()


# Zozbiera všetky numerické vstupy z formulára.
NUMERIC_INPUT_FIELDS = {clean_column_name(x) for x in DEMOGRAFIA}
for group_items in LAB_GROUPS.values():
    for feature_name in group_items:
        NUMERIC_INPUT_FIELDS.add(clean_column_name(feature_name))


def _parse_float(value, missing_value=np.nan) -> float:
    """
    Prevedie hodnotu z formulára na číslo typu float.
    Ak je pole prázdne alebo má neplatnú hodnotu, vráti sa missing_value.
    Predvolene je to NaN, aby sa takáto hodnota dala neskôr dopočítať imputérom.
    """
    if value is None or str(value).strip() == "":
        return missing_value

    try:
        return float(str(value).replace(",", "."))
    except Exception:
        return missing_value


@predict_bp.get("/index")
def index():
    """
    Zobrazí vstupný formulár pre predikciu CDI.
    """
    # Pripraví dvojice (pôvodný názov, clean názov) pre lieky
    drug_pairs = {
        group: [(raw_drug, clean_column_name(raw_drug)) for raw_drug in items]
        for group, items in DRUG_GROUPS.items()
    }

    # To isté pre demografiu a komorbidity
    demografia = [(raw, clean_column_name(raw)) for raw in DEMOGRAFIA]
    komorbidity = [(raw, clean_column_name(raw)) for raw in KOMORBIDITIES]

    # Pre laboratórne hodnoty pripraví zoskupenie do kategórií
    lab_groups_render = {}
    for group_name, group_items in LAB_GROUPS.items():
        pairs = []
        for feature_name in sorted(group_items):
            clean_name = clean_column_name(feature_name)
            info = REFERENCE_INFO.get(clean_name, {})
            display_name = info.get("abbr") or feature_name.replace("_", " ")
            pairs.append((display_name, clean_name))
        lab_groups_render[group_name] = pairs

    return render_template(
        "pages/predict_form.html",
        models=list(AVAILABLE_MODELS.keys()),
        drug_groups=DRUG_GROUPS,
        drug_pairs=drug_pairs,
        demografia=demografia,
        komorbidity=komorbidity,
        lab_groups=lab_groups_render,
        ranges=GLOBAL_RANGES,
        model_labels=MODEL_LABELS,
        reference_info=REFERENCE_INFO,
    )


@predict_bp.post("/predict")
def predict():
    """
    Spracuje údaje z formulára, pripraví vstup pre model,
    vykoná predikciu a odošle výsledok na výsledkovú stránku.
    """
    form = request.form

    # Overenie, či bol vybraný platný model
    model_key = form.get("model")
    if not model_key or model_key not in AVAILABLE_MODELS:
        abort(400, description="Neplatný model.")

    cfg = AVAILABLE_MODELS[model_key]
    model_type = cfg.get("type", "")

    # Načítanie uloženého balíka s modelom a ďalšími potrebnými súbormi
    bundle = load_model_bundle(PROJECT_ROOT, cfg)
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    expected_all = list(bundle["expected"])

    preprocess = bundle.get("preprocess", {}) or {}
    imputer = bundle.get("imputer")
    score_cfg = bundle.get("score_cfg")
    df_ref = bundle.get("lime_df_ref")

    # Vyčistený názov pre odvodený atribút Skore_final
    score_key = clean_column_name("Skore_final")

    # Pomocné mapy pre názvy liekov, aby boli lepšie čitateľné vo vysvetliteľnosti
    drug_clean_to_abbr, drug_clean_to_desc = build_drug_clean_maps(
        DRUG_GROUPS, clean_column_name, max_len=30
    )

    # Vytvorenie lokálnej kópie referenčných informácií
    reference_info_local = dict(REFERENCE_INFO)
    for feat in expected_all:
        if feat in reference_info_local:
            continue

        if feat in drug_clean_to_abbr:
            reference_info_local[feat] = {
                "abbr": drug_clean_to_abbr[feat],
                "desc": drug_clean_to_desc[feat],
                "ref": None,
                "unit": "",
            }

    # Príprava vstupov bez Skore_final
    base_features = [feat for feat in expected_all if feat != score_key]

    raw_values = []
    for feat in base_features:
        raw_value = form.get(feat)

        if feat in NUMERIC_INPUT_FIELDS:
            # Pri numerických vstupoch necháva prázdne pole ako NaN.
            # Vďaka tomu sa chýbajúce hodnoty dajú prípadne dopočítať imputáciou.
            raw_values.append(_parse_float(raw_value, missing_value=np.nan))
        else:
            # Checkboxy fungujú binárne:
            # nezaškrtnuté = 0, zaškrtnuté = 1
            raw_values.append(1.0 if raw_value not in (None, "", "0") else 0.0)

    # Z pripravených hodnôt vytvorí  DataFrame
    X = pd.DataFrame([raw_values], columns=base_features)

    # Ak má model uložený imputér, pokúsi sa dopočítať chýbajúce hodnoty
    if imputer is not None:
        # Zistí, na akých stĺpcoch bol imputér pôvodne trénovaný
        imputer_features = list(getattr(imputer, "feature_names_in_", []))

        if imputer_features:
            # Vytvorí dočasný DataFrame presne s takou štruktúrou,
            # akú imputér očakáva
            X_impute = pd.DataFrame(np.nan, index=X.index, columns=imputer_features)

            common_cols = [col for col in imputer_features if col in X.columns]
            X_impute.loc[:, common_cols] = X[common_cols]

            # Spustí imputáciu
            X_imputed = pd.DataFrame(
                imputer.transform(X_impute),
                columns=imputer_features,
                index=X.index,
            )

            # Späť si vezme len tie stĺpce, ktoré reálne používa aktuálny model
            cols_back = [col for col in X.columns if col in X_imputed.columns]
            X.loc[:, cols_back] = X_imputed[cols_back]

    # Po imputácii vypočíta Skore_final
    if score_cfg is not None and score_key in expected_all:
        scored_frame = apply_score_config(X.copy(), score_cfg)
        X[score_key] = float(scored_frame.loc[0, score_key])

    for feat in expected_all:
        if feat not in X.columns:
            X[feat] = 0.0

    # Zachová presné poradie stĺpcov podľa toho, čo model očakáva
    X = X[expected_all]

    # Ak bundle obsahuje stĺpce, ktoré sa majú pred predikciou vyhodiť,
    # odstráni ich
    drop_cols = preprocess.get("drop_cols", []) or []
    if drop_cols:
        X = X.drop(columns=[col for col in drop_cols if col in X.columns], errors="ignore")

    # Finálny zoznam vstupov pre model a vysvetliteľnosť
    expected_final = list(X.columns)
    values_final = X.iloc[0][expected_final].astype(float).tolist()

    # Samotná predikcia pravdepodobnosti CDI
    probability = float(model.predict_proba(X)[0][1])
    risk_class = int(probability >= threshold)

    # Výpočet relatívneho rizika voči rozhodovaciemu prahu modelu
    threshold_safe = threshold if threshold and threshold > 0 else 1e-9
    relative_risk = probability / threshold_safe
    margin_pp = (probability - threshold) * 100.0

    # Slovné zhrnutie úrovne rizika
    if relative_risk < 0.7:
        relative_label = "Nízke (<0.7× prahu)"
    elif relative_risk < 1.0:
        relative_label = "Hraničné (0.7–1× prahu)"
    elif relative_risk < 2.0:
        relative_label = "Stredné (1–2× nad prahom)"
    elif relative_risk < 3.0:
        relative_label = "Vysoké (2–3× nad prahom)"
    else:
        relative_label = "Veľmi vysoké (>3× nad prahom)"

    # Pozícia ukazovateľa v grafickom zobrazení relatívneho rizika (0–100 %)
    rel = max(relative_risk, 0.0)
    bar_pos = min(math.log(rel + 1.0) / math.log(4.0) * 100.0, 100.0)

    wave_feature = clean_column_name("Vlna")
    exclude_features = {wave_feature} if wave_feature in expected_final else set()

    # Premenné pre vysvetliteľnosť
    shap_waterfall_png = None
    odds_png = None
    lime_png = None
    local_factors = []
    shap_factors = []

    # Stromové modely sú vysvetľované pomocou SHAP
    if model_type in ("gb", "rf", "xgb"):
        shap_waterfall_png, shap_factors = explain_shap_tree_clinical(
            model=model,
            X=X,
            expected=expected_final,
            reference_info=reference_info_local,
            top_n=10,
            exclude_features=exclude_features,
            df_ref=df_ref,
            target_name=cfg.get("target"),
            bkg_max_rows=80,
            nsamples=300,
        )

    # Logistická regresia je vysvetľovaná cez odds ratio a lokálne faktory
    if model_type == "logreg":
        odds_png, local_factors = explain_odds_and_local_factors(
            model=model,
            expected=expected_final,
            vals=values_final,
            reference_info=reference_info_local,
        )

    # LIME vysvetlenie
    if df_ref is not None:
        lime_png = explain_lime_clinical(
            model=model,
            df_ref=df_ref,
            target_name=cfg["target"],
            expected=expected_final,
            X=X,
            reference_info=reference_info_local,
            num_features=10,
            exclude_features=exclude_features,
            debug=False,
        )

    model_label = MODEL_LABELS.get(model_key, model_key)

    # Vykreslenie stránky
    return render_template(
        "pages/predict_result.html",
        model_name=model_label,
        model_type=model_type,
        probability=round(probability, 4),
        risk_class=risk_class,
        shap_waterfall_png=shap_waterfall_png,
        odds_png=odds_png,
        lime_png=lime_png,
        local_factors=local_factors,
        shap_factors=shap_factors,
        relative_risk=round(relative_risk, 2),
        relative_label=relative_label,
        bar_pos=round(bar_pos, 1),
        threshold=round(threshold, 4),
        margin_pp=round(margin_pp, 1),
    )