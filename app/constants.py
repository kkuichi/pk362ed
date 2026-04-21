
from .utils import clean_column_name

# Modely a ich konfigurácia
AVAILABLE_MODELS = {

    "20_MICE_XGBoost_score1_ADASYN": {
        "model":      "models_2/20_MICE_XGBoost_score1_ADASYN.joblib",
        "threshold":  "models_2/20_MICE_XGBoost_score1_ADASYN_threshold.json",
        "features":   "models_2/20_MICE_XGBoost_score1_ADASYN_features.json",
        "colmap":     "models_2/20_MICE_XGBoost_score1_ADASYN_colmap.json",
        "preprocess": "models_2/20_MICE_XGBoost_score1_ADASYN_preprocess.json",
        "imputer":    "models_2/20_MICE_XGBoost_score1_ADASYN_imputer.joblib",
        "score_cfg":  "models_2/20_MICE_XGBoost_score1_ADASYN_score_cfg.json",
        "lime_df_ref":"models_2/20_MICE_XGBoost_score1_ADASYN_lime_df_ref.joblib",
        "dataset":    "scoring_dataset_20_MICE_FINAL.csv",
        "target":     "A04_7",
        "type":       "xgb",
        "score":      1,
    },

    "20_KNN_LogReg_score1_None": {
        "model":      "models_2/20_KNN_LogReg_score1_None.joblib",
        "threshold":  "models_2/20_KNN_LogReg_score1_None_threshold.json",
        "features":   "models_2/20_KNN_LogReg_score1_None_features.json",
        "colmap":     "models_2/20_KNN_LogReg_score1_None_colmap.json",
        "preprocess": "models_2/20_KNN_LogReg_score1_None_preprocess.json",
        "imputer":    "models_2/20_KNN_LogReg_score1_None_imputer.joblib",
        "score_cfg":  "models_2/20_KNN_LogReg_score1_None_score_cfg.json",
        "lime_df_ref":"models_2/20_KNN_LogReg_score1_None_lime_df_ref.joblib",
        "dataset":    "scoring_dataset_20_KNN_FINAL.csv",
        "target":     "A04_7",
        "type":       "logreg",
        "score":      1,
    },

    "20_MICE_RandomForest_score0_None": {
        "model":      "models_2/20_MICE_RandomForest_score0_None.joblib",
        "threshold":  "models_2/20_MICE_RandomForest_score0_None_threshold.json",
        "features":   "models_2/20_MICE_RandomForest_score0_None_features.json",
        "colmap":     "models_2/20_MICE_RandomForest_score0_None_colmap.json",
        "preprocess": "models_2/20_MICE_RandomForest_score0_None_preprocess.json",
        "imputer":    "models_2/20_MICE_RandomForest_score0_None_imputer.joblib",
        "lime_df_ref":"models_2/20_MICE_RandomForest_score0_None_lime_df_ref.joblib",
        "dataset":    "scoring_dataset_20_MICE_FINAL.csv",
        "target":     "A04_7",
        "type":       "rf",
        "score":      0,
    },

}

MODEL_LABELS = {
    "20_MICE_XGBoost_score1_ADASYN": "XGBoost (odporúčaný)",
    "20_KNN_LogReg_score1_None": "Logistická regresia",
    "20_MICE_RandomForest_score0_None": "Random Forest",
}


# Demografia
DEMOGRAFIA = ["Vek", "Dĺžka hospitalizácie"]


# Laboratórne hodnoty a ich zoskupenie do kategórií
LAB_GROUPS = {
    "Zápalová a imunitná odpoveď": {
        "WBC_max",
        "Neu_abs_max",
        "Ly_abs_min",
        "NE_LY_NLR_max",
        "NE_LY_NLR_last",
        "S_CRP_max",
        "S_CRP_last",
        "S_IL6_max",
        "S_IL6_last",
        "S_IL6_min",
    },
    "Funkcia obličiek a elektrolytová rovnováha": {
        "S_K_min",
        "S_Na_min",
        "S_Urea_max",
        "S_Kreat_min",
    },
    "Pečeňové a svalové poškodenie": {
        "S_AST_min",
    },
    "Koagulácia": {
        "D_dimer_HS_max",
    },
    "Hematologické parametre": {
        "HGB_min",
        "PDW_max",
    },
    "Vitamín D": {
        "S_VITD_first",
        "S_VITD_max",
    },
}


# Komorbidity
KOMORBIDITIES = [
    "Hypertenzia",
    "Diabetes mellitus",
    "Kardiovaskulárne ochorenia",
    "Chronické respiračné ochorenia",
    "Renálne ochorenia",
    "Pečeňové ochorenia",
    "Onkologické ochorenia",
    "Imunosupresia",
]


# Referenčné hodnoty pre laboratórne parametre (pre zobrazenie v insights)
REFERENCE_INFO = {

    # DEMOGRAFIA 
    clean_column_name("Vek"): {
        "abbr": "Vek",
        "desc": "Vek pacienta.",
        "ref": (0, 100),
        "unit": "roky",
    },
    clean_column_name("Dĺžka hospitalizácie"): {
        "abbr": "Dni hospit.",
        "desc": "Dĺžka hospitalizácie (Length of stay).",
        "ref": (0, 100),
        "unit": "dni",
    },

    # ZÁPAL A IMUNITNÁ ODPOVEĎ
    clean_column_name("WBC_max"): {
        "abbr": "WBC max",
        "desc": "Leukocyty – maximálna hodnota.",
        "ref": (4.0, 10.0),
        "unit": "×10⁹/l",
    },
    clean_column_name("Neu_abs_max"): {
        "abbr": "NEU max",
        "desc": "Neutrofily – maximálna hodnota.",
        "ref": (2.0, 7.0),
        "unit": "×10⁹/l",
    },
    clean_column_name("Ly_abs_min"): {
        "abbr": "LY min",
        "desc": "Lymfocyty – minimálna hodnota.",
        "ref": (1.5, 4.0),
        "unit": "×10⁹/l",
    },
    clean_column_name("NE_LY_NLR_max"): {
        "abbr": "NLR max",
        "desc": "Neutrofil/lymfocyt pomer – maximálna hodnota.",
        "ref": (0, 50.0),
        "unit": "",
    },
    clean_column_name("NE_LY_NLR_last"): {
        "abbr": "NLR last",
        "desc": "Neutrofil/lymfocyt pomer – posledná hodnota.",
        "ref": (0, 50.0),
        "unit": "",
    },
    clean_column_name("S_CRP_max"): {
        "abbr": "CRP max",
        "desc": "C-reaktívny proteín – maximálna hodnota.",
        "ref": (0.1, 2000.0),
        "unit": "mg/l",
    },
    clean_column_name("S_CRP_last"): {
        "abbr": "CRP last",
        "desc": "C-reaktívny proteín – posledná hodnota.",
        "ref": (0.1, 2000.0),
        "unit": "mg/l",
    },
    clean_column_name("S_IL6_max"): {
        "abbr": "IL-6 max",
        "desc": "Interleukín-6 – maximálna hodnota.",
        "ref": (1.5, 10000.0),
        "unit": "ng/l",
    },
    clean_column_name("S_IL6_last"): {
        "abbr": "IL-6 last",
        "desc": "Interleukín-6 – posledná hodnota.",
        "ref": (0.1, 10000.0),
        "unit": "ng/l",
    },
    clean_column_name("S_IL6_min"): {
        "abbr": "IL-6 min",
        "desc": "Interleukín-6 – minimálna hodnota.",
        "ref": (0.1, 7.0),
        "unit": "ng/l",
    },

    # FUNKCIA OBLIČIEK A ELEKTROLYTOVÁ ROVNOVÁHA
    clean_column_name("S_K_min"): {
        "abbr": "K min",
        "desc": "Draslík – minimálna hodnota.",
        "ref": (1, 10),
        "unit": "mmol/l",
    },
    clean_column_name("S_Na_min"): {
        "abbr": "Na min",
        "desc": "Sodík – minimálna hodnota.",
        "ref": (100, 146),
        "unit": "mmol/l",
    },
    clean_column_name("S_Urea_max"): {
        "abbr": "Urea max",
        "desc": "Močovina – maximálna hodnota.",
        "ref": (2.8, 50),
        "unit": "mmol/l",
    },
    clean_column_name("S_Kreat_min"): {
        "abbr": "Kreat min",
        "desc": "Kreatinín – minimálna hodnota.",
        "ref": (10, 104),
        "unit": "µmol/l",
    },

    # PEČEŇOVÉ A SVALOVÉ POŠKODENIE
    clean_column_name("S_AST_min"): {
        "abbr": "AST min",
        "desc": "AST – minimálna hodnota.",
        "ref": (0.1, 0.6),
        "unit": "µkat/l",
    },

    # KOAGULÁCIA
    clean_column_name("D_dimer_HS_max"): {
        "abbr": "D-dimér max",
        "desc": "D-dimér – maximálna hodnota.",
        "ref": (0.03, 10),
        "unit": "mg/l",
    },

    # HEMATOLOGICKÉ PARAMETRE
    clean_column_name("HGB_min"): {
        "abbr": "HGB min",
        "desc": "Hemoglobín – minimálna hodnota.",
        "ref": (5, 13),
        "unit": "g/dl",
    },
    clean_column_name("PDW_max"): {
        "abbr": "PDW max",
        "desc": "Šírka distribúcie trombocytov – maximálna hodnota.",
        "ref": (9.0, 17.0),
        "unit": "fL",
    },

    # VITAMÍN D
    clean_column_name("S_VITD_first"): {
        "abbr": "S VITD first",
        "desc": "Vitamín D – prvá hodnota.",
        "ref": (20, 1000),
        "unit": "µg/l",
    },
    clean_column_name("S_VITD_max"): {
        "abbr": "S VITD max",
        "desc": "Vitamín D – maximálna hodnota.",
        "ref": (20, 1000),
        "unit": "µg/l",
    },
    
}


# Skupiny liekov a ich zložky (pre zobrazenie v insights)
DRUG_GROUPS = {
    "Antivirotiká": [
        "MD652 | FABIFLU TABLETS", "5042D | VEKLURY", "9547D | PAXLOVID", "LAGEVRIO"
    ],
    "Imuno- (tlmenie reakcie)": [
        "02963 | PREDNISON 20 LÉČIVA", "00269 | PREDNISON 5 LÉČIVA", "84090 | DEXAMED 6",
        "1275C | DEXAMETAZÓN KRKA", "MD661 BIODEXONE-DEXAMETHASONE", "2410B HYDROCORTISONE",
        "RoActemra"
    ],
    "Imuno+ (stimulácia)": [
        "87299 | IMUNOR", "56930 IMMODIN",
    ],
    "Antibiotiká": [
        "9819A MOXIFLOXACIN", "58746 CIPROFLOXACINKABI 400"
    ],
    "Iné (PPI)": [
        "05044 OZZION", "4147C OMEMYL", "89662 NOLPAZA", "39397 PANTOPRAZOL",
        "62916 SMECTA", "30639 REASEC", "84370 LAGOSA", "93105 DEGAN "
    ],
    "Kašeľ": [
        "24859 PENTOXYPHILLINUM", "8893 ACC INJEKT",
        "24949 CODEIN ", "26846 OXANTIL"
    ],
    "Antikoagulanciá": [
        "FRAXIPARIN", "FRAGMIN"
    ]
}

# Názvy grafov v insights
PLOT_TITLES = {
    # demografia
    "demografia_pohlavie_barplot": "Výskyt CDI podľa pohlavia",
    "demografia_vek_boxplot": "Vek pacientov podľa CDI",
    "demografia_vek_vlny_violin": "Vek pacientov podľa CDI (vlny)",

    # overview grafy v ostatných sekciách
    "comorbidity_overview": "Prehľad komorbidít (CDI vs bez CDI)",
    "drug_groups_overview": "Prehľad skupín liekov (CDI vs bez CDI)",
}

INSIGHTS_CATEGORIES = {
    "Demografia": "demografia",
    "Laboratórne hodnoty": "labs",
    "Komorbidity": "comorb",
    "Skupiny liekov": "drug_groups",
    "Vlny": "waves",
}


