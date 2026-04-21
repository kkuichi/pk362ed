from pathlib import Path

from flask import Flask


def create_app() -> Flask:
    """
    Vytvorí a nakonfiguruje Flask aplikáciu.

    Šablóny a statické súbory sú uložené mimo balíka ``app``,
    preto sa najprv určí koreň projektu a následne sa nastavia
    cesty k priečinkom `templates` a `static`.
    """
    project_root = Path(__file__).resolve().parents[1]

    app = Flask(
        __name__,
        template_folder=str(project_root / "templates"),
        static_folder=str(project_root / "static"),
    )

    from app.routes.insights import insights_bp
    from app.routes.pages import pages_bp
    from app.routes.predict import predict_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(predict_bp)
    app.register_blueprint(insights_bp)

    return app
