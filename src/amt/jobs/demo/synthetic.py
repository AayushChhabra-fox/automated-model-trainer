"""Synthetic interaction generator (``amt demo seed``).

Gives the pipeline a dataset with the two properties that actually decide
whether a sequential recommender beats a popularity baseline:

* **power-law popularity** -- so a popularity baseline is a genuinely strong
  reference point rather than a straw man;
* **sequential structure** -- users belong to latent taste clusters and walk
  a preferred genre, so there is real signal for SASRec to find that
  popularity cannot.

Without both, a demo campaign would reach a conclusion that tells you
nothing about the machinery under test.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["DEFAULTS", "generate", "write_local", "write_uc", "seed"]

DEFAULTS: dict[str, Any] = {
    "n_users": 2000,
    "n_items": 500,
    "n_genres": 8,
    "min_events": 5,
    "max_events": 60,
    "zipf_exponent": 1.1,
    "genre_affinity": 0.75,
    "start_ts": "2024-01-01",
    "days": 120,
    "seed": 7,
}


def generate(**overrides: Any):
    """Return a pandas DataFrame ``user_id, item_id, ts, item_genre``."""
    import pandas as pd

    cfg = {**DEFAULTS, **overrides}
    rng = np.random.default_rng(int(cfg["seed"]))

    n_items = int(cfg["n_items"])
    n_genres = int(cfg["n_genres"])

    # Power-law item popularity, plus a genre per item.
    ranks = np.arange(1, n_items + 1, dtype=np.float64)
    pop = ranks ** (-float(cfg["zipf_exponent"]))
    pop /= pop.sum()
    item_genre = rng.integers(0, n_genres, size=n_items)

    # Genre-conditional popularity: within a genre, keep the power law.
    genre_probs = []
    for g in range(n_genres):
        mask = item_genre == g
        weights = np.where(mask, pop, 0.0)
        total = weights.sum()
        genre_probs.append(weights / total if total > 0 else pop)

    affinity = float(cfg["genre_affinity"])
    start = pd.Timestamp(cfg["start_ts"])
    horizon = int(cfg["days"])
    rows_user, rows_item, rows_ts = [], [], []

    for user in range(int(cfg["n_users"])):
        fav = int(rng.integers(0, n_genres))
        n_events = int(rng.integers(int(cfg["min_events"]), int(cfg["max_events"]) + 1))
        # Sequential structure: a user drifts within their favourite genre and
        # only occasionally samples the global popularity distribution.
        in_genre = rng.random(n_events) < affinity
        picks = np.where(
            in_genre,
            rng.choice(n_items, size=n_events, p=genre_probs[fav]),
            rng.choice(n_items, size=n_events, p=pop),
        )
        # Monotonic timestamps, denser towards the end of the horizon.
        offsets = np.sort(rng.random(n_events)) * horizon
        rows_user.extend([f"u{user:06d}"] * n_events)
        rows_item.extend(f"i{int(p):05d}" for p in picks)
        rows_ts.extend(start + pd.to_timedelta(offsets, unit="D"))

    frame = pd.DataFrame(
        {"user_id": rows_user, "item_id": rows_item, "ts": rows_ts}
    )
    genre_of = {f"i{i:05d}": f"g{int(item_genre[i])}" for i in range(n_items)}
    frame["item_genre"] = frame["item_id"].map(genre_of)
    return frame.sort_values(["user_id", "ts"], kind="mergesort").reset_index(drop=True)


def write_local(frame, output_dir: str, table: str = "interactions") -> str:
    """Write the demo table next to where ``SplitGuard`` will look for it."""
    import os

    from amt.jobs.data.loaders import write_local_table
    from amt.jobs.materialize_splits import _parquet_available

    ext = ".parquet" if _parquet_available() else ".jsonl"
    os.makedirs(output_dir, exist_ok=True)
    return write_local_table(frame, os.path.join(output_dir, f"{table}{ext}"))


def write_uc(frame, spark: Any, table: str) -> str:
    """Write the demo table to Unity Catalog as Delta."""
    catalog_schema = ".".join(table.split(".")[:-1])
    if catalog_schema:
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog_schema}")
    spark.createDataFrame(frame).write.mode("overwrite").format("delta").saveAsTable(
        table
    )
    return table


def seed(
    table: str | None = None,
    spark: Any | None = None,
    output_dir: str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Generate and persist; returns a small descriptor for the caller."""
    frame = generate(**overrides)
    if spark is not None and table:
        location = write_uc(frame, spark, table)
        backend = "uc"
    else:
        if not output_dir:
            raise ValueError("local demo seeding requires output_dir")
        location = write_local(frame, output_dir, table=(table or "interactions").split(".")[-1])
        backend = "local"
    return {
        "backend": backend,
        "location": location,
        "rows": int(len(frame)),
        "n_users": int(frame["user_id"].nunique()),
        "n_items": int(frame["item_id"].nunique()),
    }
