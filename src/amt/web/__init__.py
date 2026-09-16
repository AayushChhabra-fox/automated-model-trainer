"""Local web surface over the campaign graph (``amt web``).

Requires the ``web`` extra: ``uv sync --extra web``.
"""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Lazy re-export, so importing this package does not require Flask."""
    from amt.web.app import create_app as _create_app

    return _create_app(*args, **kwargs)
