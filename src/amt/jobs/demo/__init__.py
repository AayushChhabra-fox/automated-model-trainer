"""Synthetic data so the whole pipeline can run before real tables exist."""

from __future__ import annotations

import os
from typing import Any

__all__ = ["seed_demo_data"]


def seed_demo_data(
    catalog: str = "main",
    schema: str = "amt",
    users: int = 2000,
    items: int = 500,
    table: str = "interactions",
    spark: Any | None = None,
    output_dir: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """``amt demo seed`` entry point -- wrapper around :func:`synthetic.seed`.

    Writes to ``{catalog}.{schema}.{table}`` when a Spark session is available,
    and falls back to a local table under ``AMT_LOCAL_DATA_DIR`` otherwise.
    """
    from amt.jobs.demo.synthetic import seed
    from amt.jobs.entry import get_spark

    fq_table = table if "." in table else f"{catalog}.{schema}.{table}"
    return seed(
        table=fq_table,
        spark=spark if spark is not None else get_spark(),
        output_dir=output_dir or os.environ.get("AMT_LOCAL_DATA_DIR"),
        n_users=int(users),
        n_items=int(items),
        **overrides,
    )
