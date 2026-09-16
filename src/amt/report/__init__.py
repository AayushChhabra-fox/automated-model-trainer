"""HTML rendering of the experimentation report.

One Jinja2 template, no chart library and no JavaScript, so the same bytes
render as an MLflow artifact, an email attachment and a page in the local web
surface. Jinja2 arrives with Flask, so this costs no new dependency.
"""

from __future__ import annotations

from pathlib import Path

from amt.schemas.conclusion import ExperimentationReport, ScientificConclusion

__all__ = ["TEMPLATE_DIR", "render_report"]

TEMPLATE_DIR = Path(__file__).parent
TEMPLATE_NAME = "experimentation_report.html.j2"


def render_report(
    report: ExperimentationReport,
    conclusion: ScientificConclusion | None = None,
) -> str:
    """Render ``report`` as a standalone HTML document."""
    from jinja2 import Environment, FileSystemLoader, select_autoescape

    environment = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    return environment.get_template(TEMPLATE_NAME).render(report=report, conclusion=conclusion)
