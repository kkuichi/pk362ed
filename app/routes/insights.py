import os
from flask import Blueprint, render_template
from ..constants import INSIGHTS_CATEGORIES, PLOT_TITLES

insights_bp = Blueprint("insights", __name__)


def _title_from_filename_key(key: str) -> str:
    """ Získanie názvu grafu z kľúča odobratím prefixov a nahradením podtržníkov medzerami."""
    if key in PLOT_TITLES:
        return PLOT_TITLES[key]

    if key.startswith("lab_"):
        x = key[len("lab_"):]
        x = x.replace("_", " ")
        return f"{x}"

    if key.startswith("group_"):
        x = key[len("group_"):]
        x = x.replace("_", " ")
        return f"{x}"

    if key.startswith("waves_"):
        x = key[len("waves_"):]
        x = x.replace("_", " ")
        return f"{x}"

    return key.replace("_", " ")


@insights_bp.get("/insights")
def insights():
    '''Zobrazí prehľad všetkých kategórií insights. Každá kategória odkazuje na stránku s grafmi v danej kategórii.'''
    return render_template("pages/insights_overview.html", folders=INSIGHTS_CATEGORIES)


@insights_bp.get("/insights/<category>")
def insights_category(category: str):
    '''Zobrazí všetky grafy v danej kategórii insights. Grafy sú načítané zo statických súborov (png) v priečinku `static/plots/{category}`.'''
    folder_path = os.path.join("static", "plots", category)

    folder_to_label = {v: k for k, v in INSIGHTS_CATEGORIES.items()}
    label = folder_to_label.get(category, category)

    imgs = []
    try:
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]
        files.sort(key=lambda s: s.lower())

        for f in files:
            key = os.path.splitext(f)[0]  # bez .png
            full_path = f"/static/plots/{category}/{f}"
            title = _title_from_filename_key(key)

            imgs.append({"file": full_path, "title": title})
    except Exception:
        imgs = []

    return render_template(
        "pages/insights_category.html",
        category=category,
        category_label=label,
        imgs=imgs,
    )
