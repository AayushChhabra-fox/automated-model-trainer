"""The few lines above the JSON at a checkpoint.

The reviewer's two questions were "what will this cost" and "what is it
actually fitting", and both were below the fold of the old preview.
"""

from __future__ import annotations

from amt.cli import (
    PREVIEW_CHARS,
    backlog_rows,
    checkpoint_document,
    shape_rows,
    summarize,
)

HC_FEATURE = {
    "checkpoint": "HC_FEATURE",
    "validation_rows": 44_495,
    "duplicate_shapes": {"hyp-1": ["hyp-2"], "hyp-2": ["hyp-1"]},
    "shapes": {
        "hyp-1": {
            "family": "lightgbm_reranker",
            "feature_fit_hash": "a" * 64,
            "candidate_generation": {"strategy": "popularity_topn", "n": 200},
            "fit_spec": {
                "objective": "lambdarank",
                "eval_metric": "ndcg",
                "num_boost_round": 200,
                "early_stopping_rounds": 20,
                "negative_samples_per_positive": 4,
                "seed": 42,
            },
            "feature_recipe": {"feature_columns": [{"name": "pop"}]},
        },
        "hyp-2": {
            "family": "lightgbm_reranker",
            "feature_fit_hash": "a" * 64,
            "candidate_generation": {"strategy": "popularity_topn", "n": 200},
            "fit_spec": {"num_boost_round": 200},
            "feature_recipe": {"feature_columns": []},
        },
    },
}

HC_BACKLOG = {
    "checkpoint": "HC_BACKLOG",
    "hypotheses": [
        {
            "hypothesis_id": "hyp-1",
            "title": "Impression-row booster",
            "family": "lightgbm_reranker",
            "metric_expected_to_improve": "ndcg@4",
            "expected_effect_size": "+0.01",
            "estimated_cost_minutes": 30,
            "status": "pending",
        },
        {
            "hypothesis_id": "hyp-2",
            "family": "sasrec",
            "title": "Sequence model",
            "metric_expected_to_improve": "ndcg@4",
            "expected_effect_size": "+0.02",
            "estimated_cost_minutes": 45,
            "status": "abandoned",
        },
    ],
}


def test_the_shape_summary_prices_the_candidate_set():
    """200 candidates against a 44k-row validation split is 8.9M rows of features."""
    first = shape_rows(HC_FEATURE)[0]

    assert first["family"] == "lightgbm_reranker"
    assert first["candidates"] == "popularity_topn n=200"
    assert first["est. rows"] == "8,899,000"


def test_the_shape_summary_names_the_six_fit_fields():
    assert shape_rows(HC_FEATURE)[0]["fit"].splitlines() == [
        "objective=lambdarank",
        "eval_metric=ndcg",
        "num_boost_round=200",
        "early_stopping_rounds=20",
        "negative_samples_per_positive=4",
        "seed=42",
    ]


def test_a_blank_fit_field_is_omitted_rather_than_shown_as_empty():
    """Blanks mean "the family default", which is not a value to read."""
    assert shape_rows(HC_FEATURE)[1]["fit"] == "num_boost_round=200"


def test_two_hypotheses_with_one_shape_are_flagged_as_one_experiment():
    rows = {row["id"]: row for row in shape_rows(HC_FEATURE)}

    assert "same shape as hyp-2" in rows["hyp-1"]["notes"]
    assert "same shape as hyp-1" in rows["hyp-2"]["notes"]
    assert "no feature columns" in rows["hyp-2"]["notes"]


def test_an_unpriced_split_says_so_instead_of_showing_zero():
    payload = {**HC_FEATURE, "validation_rows": 0}
    assert shape_rows(payload)[0]["est. rows"] == "unknown"


def test_the_backlog_summary_carries_family_cost_and_status():
    rows = backlog_rows(HC_BACKLOG)

    assert [row["id"] for row in rows] == ["hyp-1", "hyp-2"]
    assert rows[0]["family"] == "lightgbm_reranker"
    assert (rows[0]["cost"], rows[1]["cost"]) == ("30m", "45m")
    assert rows[1]["status"] == "abandoned"


def test_both_checklists_render_without_raising():
    assert summarize(HC_BACKLOG) is not None
    assert summarize(HC_FEATURE) is not None


def test_checkpoints_without_a_checklist_get_no_header():
    assert summarize({"checkpoint": "HC3"}) is None
    assert summarize({}) is None


def test_the_shapes_are_the_document_of_record_at_hc_feature():
    assert checkpoint_document(HC_FEATURE) == HC_FEATURE["shapes"]


def test_the_preview_is_large_enough_for_a_real_backlog():
    assert PREVIEW_CHARS >= 20_000
