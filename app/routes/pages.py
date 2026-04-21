from flask import Blueprint, render_template

pages_bp = Blueprint("pages", __name__)


@pages_bp.get("/")
def home():
    """Úvodná stránka aplikácie."""
    return render_template("pages/home.html")


@pages_bp.get("/project")
def project():
    """Stránka s opisom projektu a metodiky."""
    return render_template("pages/project.html")
