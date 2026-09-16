"""Model families, the registry and the shared pyfunc serving wrapper.

Runs on CPU and skips the SASRec cases when torch is unavailable.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from amt.jobs.data import features
from amt.jobs.models import popularity, registry


def _torch():
    """Skip the SASRec cases when torch is not installed (CPU torch is enough)."""
    return pytest.importorskip("torch")


def _interactions(n_users: int = 30, n_events: int = 12) -> pd.DataFrame:
    rows = []
    for u in range(n_users):
        for e in range(n_events):
            rows.append(
                {
                    "user_id": f"u{u}",
                    # Item 0 is globally most frequent; the rest cycle.
                    "item_id": "i0" if e % 3 == 0 else f"i{(u + e) % 5 + 1}",
                    "ts": pd.Timestamp("2024-01-01") + pd.Timedelta(days=e),
                }
            )
    return pd.DataFrame(rows)


# -- features --------------------------------------------------------------


def test_item_index_reserves_zero_for_padding():
    index = features.build_item_index(_interactions())
    assert index.encode("i0") != 0
    assert index.encode("never-seen") == 0
    assert index.size == index.catalog_size + 1
    assert index.decode(index.encode("i0")) == "i0"


def test_item_index_order_is_deterministic_by_frequency():
    index = features.build_item_index(_interactions())
    # The most frequent item gets the first (lowest) model id.
    assert index.items[0] == "i0"
    again = features.build_item_index(_interactions())
    assert again.items == index.items


def test_as_item_list_accepts_json_and_native_forms():
    assert features.as_item_list('["a", "b"]') == ["a", "b"]
    assert features.as_item_list(["a"]) == ["a"]
    assert features.as_item_list(None) == []
    assert features.as_item_list("") == []


def test_pad_left_keeps_the_most_recent_items():
    assert features.pad_left([1, 2, 3], 5) == [0, 0, 1, 2, 3]
    assert features.pad_left([1, 2, 3, 4], 2) == [3, 4]


def test_build_eval_frame_keeps_raw_ids_and_sorts_users():
    df = pd.DataFrame(
        {
            "user_id": ["b", "a"],
            "history": [["i1"], ["i2"]],
            "targets": [["i3"], ["i4"]],
        }
    )
    frame = features.build_eval_frame(df)
    assert frame.user_ids == ["a", "b"]
    assert frame.histories == [["i2"], ["i1"]]
    assert frame.truths == [{"i4"}, {"i3"}]


# -- registry --------------------------------------------------------------


def test_registry_exposes_every_family():
    assert set(registry.FAMILIES) == {
        "popularity",
        "sasrec",
        "xgboost_classifier",
        "xgboost_reranker",
        "lightgbm_classifier",
        "lightgbm_reranker",
    }
    assert callable(registry.get_builder("popularity"))
    assert set(registry.FAMILIES) >= registry.RECIPE_FAMILIES


def test_recipe_families_agree_with_the_orchestrator():
    # Jobs code cannot import amt.schemas, so the two lists are kept in step here.
    from amt.schemas.feature_recipe import TABULAR_FAMILIES

    assert registry.RECIPE_FAMILIES == TABULAR_FAMILIES


def test_unknown_family_is_rejected():
    with pytest.raises(registry.UnknownFamily):
        registry.get_builder("transformer-xl-9000")


# -- popularity ------------------------------------------------------------


def test_popularity_ranks_by_train_frequency():
    model = popularity.fit(_interactions())
    assert model.items[0] == "i0"
    assert model.score("i0") > model.score("i1")


def test_popularity_never_recommends_a_seen_item():
    model = popularity.fit(_interactions())
    ranked = model.recommend([["i0", "i1"], []], k=3)
    assert "i0" not in ranked[0] and "i1" not in ranked[0]
    assert ranked[1][0] == "i0"
    assert len(ranked[0]) == 3


def test_popularity_ignores_validation_data():
    train = _interactions()
    # A validation frame that would flip the ranking if it leaked in.
    val = pd.DataFrame({"user_id": ["u0"], "history": [[]], "targets": [["i4"] * 500]})
    model = popularity.fit(train, val)
    assert model.items[0] == "i0"


def test_popularity_round_trips_through_disk(tmp_path):
    model = popularity.fit(_interactions())
    registry.save_model(model, str(tmp_path / "m"))
    loaded = registry.load_model(str(tmp_path / "m"))
    assert isinstance(loaded, popularity.PopularityModel)
    assert loaded.items == model.items
    assert loaded.recommend([[]], 3) == model.recommend([[]], 3)


# -- pyfunc wrapper --------------------------------------------------------


def _wrapped(model):
    wrapper = registry.RecommenderPyfunc()
    wrapper._model = model
    return wrapper


def test_pyfunc_predict_returns_top_k_lists():
    model = popularity.fit(_interactions())
    frame = pd.DataFrame({"history": [["i0"], []], "k": [2, 3]})
    out = _wrapped(model).predict(None, frame)
    assert list(out.columns) == ["items"]
    assert out["items"].tolist()[0] == model.recommend([["i0"]], 2)[0]
    assert len(out["items"].tolist()[1]) == 3


def test_pyfunc_predict_speaks_the_json_serving_form():
    model = popularity.fit(_interactions())
    frame = pd.DataFrame({"history": [json.dumps(["i0"])], "k": [2]})
    out = _wrapped(model).predict(None, frame)
    items = json.loads(out["items"].iloc[0])
    assert items == model.recommend([["i0"]], 2)[0]


def test_recommend_with_pyfunc_matches_the_model_directly():
    model = popularity.fit(_interactions())
    histories = [["i0"], ["i1", "i2"], []]
    assert registry.recommend_with_pyfunc(_wrapped(model), histories, 4) == (
        model.recommend(histories, 4)
    )


def test_pyfunc_signature_declares_the_history_column():
    signature = registry.pyfunc_signature()
    if signature is None:
        pytest.skip("mlflow not installed")
    assert [c.name for c in signature.inputs.inputs] == ["history", "k"]


# -- sasrec ----------------------------------------------------------------


def _sasrec_cfg(**overrides):
    cfg = {
        "model": {"family": "sasrec", "hidden_dim": 16, "num_layers": 1, "heads": 2,
                  "dropout": 0.0, "max_seq_len": 8},
        "training": {"learning_rate": 1e-2, "batch_size": 16, "epochs": 2,
                     "early_stopping_patience": 0, "seed": 3},
        "loss": {"type": "bce", "num_negatives": 4},
    }
    cfg.update(overrides)
    return cfg


def test_sasrec_config_flattening_does_not_need_torch():
    from amt.jobs.models import sasrec

    flat = sasrec.resolve_cfg(_sasrec_cfg())
    assert flat["hidden_dim"] == 16
    assert flat["loss"] == "bce"
    assert flat["num_negatives"] == 4
    # Unspecified knobs fall back to documented defaults.
    assert flat["primary_metric"] == sasrec.DEFAULTS["primary_metric"]


def test_sasrec_fits_and_recommends_unseen_items():
    _torch()
    from amt.jobs.models import sasrec

    train = _interactions(n_users=20, n_events=10)
    model = sasrec.fit(train, None, _sasrec_cfg())
    ranked = model.recommend([["i0", "i1"], []], k=3)
    assert len(ranked) == 2
    assert all(len(r) == 3 for r in ranked)
    assert "i0" not in ranked[0] and "i1" not in ranked[0]
    assert all(item in set(model.index.items) for r in ranked for item in r)


def test_sasrec_logs_a_curve_per_epoch():
    _torch()
    from amt.jobs.models import sasrec

    seen = []
    sasrec.fit(
        _interactions(n_users=12, n_events=8),
        None,
        _sasrec_cfg(),
        logger=lambda step, metrics: seen.append((step, dict(metrics))),
    )
    assert [s for s, _ in seen] == [1, 2]
    assert all("train/loss" in m for _, m in seen)


def test_sasrec_early_stops_on_the_validation_metric():
    _torch()
    from amt.jobs.models import sasrec

    cfg = _sasrec_cfg()
    cfg["training"]["epochs"] = 6
    cfg["training"]["early_stopping_patience"] = 0
    calls = {"n": 0}

    def _evaluator(_model):
        calls["n"] += 1
        # Best at the first epoch, then strictly worse: must stop immediately.
        return {"ndcg@10": 1.0 / calls["n"]}

    model = sasrec.fit(
        _interactions(n_users=12, n_events=8), None, cfg, evaluator=_evaluator
    )
    assert model.cfg["best_epoch"] == 1
    assert len(model.history) < 6


def test_sasrec_supports_full_softmax_cross_entropy():
    _torch()
    from amt.jobs.models import sasrec

    cfg = _sasrec_cfg()
    cfg["loss"] = {"type": "ce"}
    model = sasrec.fit(_interactions(n_users=12, n_events=8), None, cfg)
    assert model.history
    assert model.n_params() > 0


def test_sasrec_round_trips_through_disk_and_the_registry(tmp_path):
    _torch()
    from amt.jobs.models import sasrec

    model = sasrec.fit(_interactions(n_users=12, n_events=8), None, _sasrec_cfg())
    registry.save_model(model, str(tmp_path / "m"))
    loaded = registry.load_model(str(tmp_path / "m"))
    assert isinstance(loaded, sasrec.SASRecModel)
    assert loaded.recommend([["i0"]], 3) == model.recommend([["i0"]], 3)


def test_sasrec_pyfunc_wrapper_serves_the_same_ranking():
    _torch()
    from amt.jobs.models import sasrec

    model = sasrec.fit(_interactions(n_users=12, n_events=8), None, _sasrec_cfg())
    histories = [["i0", "i2"], []]
    assert registry.recommend_with_pyfunc(_wrapped(model), histories, 3) == (
        model.recommend(histories, 3)
    )


# -- tabular booster families ----------------------------------------------


def _booster(family: str):
    """Skip when the native booster library cannot be loaded (e.g. no libomp)."""
    module = "lightgbm" if family.startswith("lightgbm") else "xgboost"
    try:
        __import__(module)
    except Exception as exc:  # ImportError, or OSError from a missing libomp
        pytest.skip(f"{module} unavailable: {exc}")


def _tabular_interactions(n_users: int = 40, n_events: int = 10) -> pd.DataFrame:
    """Wider catalog than ``_interactions``: the boosters need sampleable negatives."""
    rows = []
    for u in range(n_users):
        for e in range(n_events):
            rows.append(
                {
                    "user_id": f"u{u}",
                    "item_id": "i0" if e % 4 == 0 else f"i{(u * 3 + e) % 30 + 1}",
                    "ts": pd.Timestamp("2024-01-01") + pd.Timedelta(days=e),
                    "item_genre": "news" if (u + e) % 2 else "sports",
                }
            )
    return pd.DataFrame(rows)


def _validation_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u0", "u1"],
            "history": [["i1", "i2"], ["i3"]],
            "targets": [["i0"], ["i4"]],
        }
    )


def _recipe(**overrides) -> dict:
    recipe = {
        "sources": [
            {"name": "interactions", "role": "interactions", "split": "train"},
            {
                "name": "attrs",
                "role": "item_attributes",
                "split": "train",
                "columns": ["item_genre"],
            },
        ],
        "feature_columns": [
            {"name": "pop", "kind": "count", "of": "item_id", "over": "train"},
            {"name": "genre", "kind": "cast", "source": "item_genre", "to": "code"},
            {"name": "affinity", "kind": "count", "of": "item_genre", "over": "history"},
            {"name": "hist_len", "kind": "column", "source": "history_length"},
            {"name": "affinity_rate", "kind": "ratio",
             "numerator": "affinity", "denominator": "hist_len"},
            {"name": "rank_bucket", "kind": "bucket",
             "source": "item_train_rank", "boundaries": [5.0, 15.0]},
            {"name": "item_hash", "kind": "hash", "source": "item_id", "buckets": 8},
            {"name": "bias", "kind": "literal", "value": 1.0},
        ],
        "label": {"strategy": "holdout_next_item"},
        "group_key": "user_id",
        "candidate_generation": {"strategy": "popularity_topn", "n": 8, "exclude_seen": True},
        "filters": [],
    }
    recipe.update(overrides)
    return recipe


def _tabular_cfg(family: str, **overrides) -> dict:
    cfg = {
        "model": {"family": family, "max_depth": 3},
        "training": {"learning_rate": 0.2},
        "fit_spec": {
            "num_boost_round": 8,
            "early_stopping_rounds": 4,
            "negative_samples_per_positive": 3,
            "booster_params": {"min_child_weight": 1},
        },
        "feature_recipe": _recipe(),
    }
    cfg.update(overrides)
    return cfg


BOOSTER_FAMILIES = [
    "xgboost_classifier",
    "xgboost_reranker",
    "lightgbm_classifier",
    "lightgbm_reranker",
]


def test_tabular_cfg_resolution_separates_booster_params_from_amt_knobs():
    from amt.jobs.models import tabular_common

    flat = tabular_common.resolve_cfg(
        {**_tabular_cfg("xgboost_classifier"), "config_id": "cfg-1", "rationale": "prose"}
    )
    assert flat["num_boost_round"] == 8
    assert flat["negative_samples_per_positive"] == 3
    # Campaign plumbing must never reach the booster.
    assert flat["booster_params"] == {
        "min_child_weight": 1,
        "max_depth": 3,
        "learning_rate": 0.2,
    }


def test_an_approved_recipe_freezes_the_amt_knobs_against_the_trainer():
    """The Trainer tunes booster params; the shape is the human's, not its.

    `objective` went through this route in `vod-nextitem-001`: a `training`
    section naming one the approved FitSpec did not have.
    """
    from amt.jobs.models import tabular_common

    drifted = _tabular_cfg("xgboost_reranker")
    drifted["training"] = {
        "learning_rate": 0.2,
        "objective": "rank:ndcg",
        "max_train_rows": 10,
        "negative_samples_per_positive": 99,
    }
    flat = tabular_common.resolve_cfg(drifted)

    assert flat["negative_samples_per_positive"] == 3  # the approved fit_spec's
    assert flat["max_train_rows"] == tabular_common.DEFAULTS["max_train_rows"]
    assert "objective" not in flat
    # Rejected, not smuggled in as a hyperparameter.
    assert flat["booster_params"] == {
        "min_child_weight": 1,
        "max_depth": 3,
        "learning_rate": 0.2,
    }

    without_recipe = {**drifted, "feature_recipe": None}
    assert tabular_common.resolve_cfg(without_recipe)["max_train_rows"] == 10


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("xgboost_reranker", ("xgboost",)),
        ("lightgbm_classifier", ("lightgbm",)),
        ("sasrec", ()),
        ("popularity", ()),
        ("", ()),
    ],
)
def test_each_family_declares_the_packages_a_job_cluster_must_install(family, expected):
    """A booster on a fresh job cluster brings its own backend (B7)."""
    assert registry.requires_pip(family) == expected


def test_tabular_recipe_is_required():
    from amt.jobs.models import tabular_common

    with pytest.raises(tabular_common.UnsupportedRecipe):
        tabular_common.fit_family(
            "xgboost_classifier", _tabular_interactions(), None, {"model": {"max_depth": 2}}
        )


def test_tabular_rejects_an_unsupported_transform():
    from amt.jobs.models import tabular_common

    recipe = _recipe(
        feature_columns=[{"name": "x", "kind": "sql", "source": "SELECT 1"}]
    )
    with pytest.raises(tabular_common.UnsupportedRecipe):
        tabular_common.fit_family(
            "xgboost_classifier",
            _tabular_interactions(),
            None,
            _tabular_cfg("xgboost_classifier", feature_recipe=recipe),
        )


def test_tabular_recipe_may_never_read_the_qa_split():
    from amt.jobs.models import tabular_common

    recipe = _recipe(
        sources=[{"name": "leak", "role": "interactions", "split": "qa"}]
    )
    with pytest.raises(tabular_common.UnsupportedRecipe):
        tabular_common.fit_family(
            "lightgbm_classifier",
            _tabular_interactions(),
            None,
            _tabular_cfg("lightgbm_classifier", feature_recipe=recipe),
        )


def test_feature_rows_are_built_the_same_way_for_fit_and_serving():
    """One builder, so a feature that needs more than a history cannot be trained on."""
    from amt.jobs.models import tabular_common

    recipe = _recipe()
    state = tabular_common.build_state(recipe, _tabular_interactions())
    frame, groups, items = tabular_common.build_rows(
        recipe, state, [["i1", "i2"], []], [["i0", "i3"], ["i0"]]
    )
    assert list(frame.columns) == [f["name"] for f in recipe["feature_columns"]]
    assert groups.tolist() == [0, 0, 1]
    assert items == ["i0", "i3", "i0"]
    # history_length is per history, popularity is per item.
    assert frame["hist_len"].tolist() == [2.0, 2.0, 0.0]
    assert frame["pop"].iloc[0] == frame["pop"].iloc[2]
    # A candidate whose genre never appears in the history has no affinity.
    assert frame["affinity"].iloc[2] == 0.0
    assert frame["bias"].tolist() == [1.0, 1.0, 1.0]


def test_popularity_candidates_exclude_seen_items():
    from amt.jobs.models import tabular_common

    recipe = _recipe()
    state = tabular_common.build_state(recipe, _tabular_interactions())
    candidates = tabular_common._candidates_for(recipe, state, ["i0", "i1"])
    assert len(candidates) == recipe["candidate_generation"]["n"]
    assert "i0" not in candidates and "i1" not in candidates


def test_closed_form_filters_drop_rows_without_sql():
    from amt.jobs.models import tabular_common

    recipe = _recipe(filters=[{"column": "hist_len", "op": "gt", "value": 0}])
    state = tabular_common.build_state(recipe, _tabular_interactions())
    frame, groups, _items = tabular_common.build_rows(
        recipe, state, [["i1"], []], [["i0"], ["i0"]]
    )
    kept, labels, groups = tabular_common._apply_filters(
        recipe, frame, np.asarray([1.0, 0.0]), groups
    )
    assert len(kept) == 1
    assert labels.tolist() == [1.0]


def _fit_tabular(family: str, **cfg_overrides):
    _booster(family)
    from amt.jobs.models import registry as reg

    return reg.get_builder(family)(
        _tabular_interactions(), _validation_frame(), _tabular_cfg(family, **cfg_overrides)
    )


@pytest.mark.parametrize("family", BOOSTER_FAMILIES)
def test_tabular_families_fit_and_recommend_unseen_items(family):
    model = _fit_tabular(family)
    ranked = model.recommend([["i1", "i2"], []], k=3)
    assert len(ranked) == 2
    assert all(len(r) == 3 for r in ranked)
    assert "i1" not in ranked[0] and "i2" not in ranked[0]
    assert model.feature_names == [f["name"] for f in _recipe()["feature_columns"]]


@pytest.mark.parametrize("family", BOOSTER_FAMILIES)
def test_tabular_families_round_trip_through_disk(tmp_path, family):
    model = _fit_tabular(family)
    registry.save_model(model, str(tmp_path / family))
    loaded = registry.load_model(str(tmp_path / family))
    assert loaded.family == family
    assert loaded.feature_names == model.feature_names
    assert loaded.recommend([["i1"]], 3) == model.recommend([["i1"]], 3)


def test_tabular_pyfunc_wrapper_serves_the_same_ranking():
    model = _fit_tabular("xgboost_classifier")
    histories = [["i1", "i2"], []]
    assert registry.recommend_with_pyfunc(_wrapped(model), histories, 3) == (
        model.recommend(histories, 3)
    )


def test_tabular_logs_a_curve_and_reports_the_harness_metric():
    seen: list[tuple[int, dict]] = []
    _booster("lightgbm_reranker")
    from amt.jobs.models import lightgbm_reranker

    model = lightgbm_reranker.fit(
        _tabular_interactions(),
        _validation_frame(),
        _tabular_cfg("lightgbm_reranker"),
        logger=lambda step, metrics: seen.append((step, dict(metrics))),
        evaluator=lambda _model: {"ndcg@10": 0.37},
    )
    assert seen
    # Convergence is judged on the harness metric, not the booster's own loss.
    assert model.history[-1]["val/ndcg@10"] == pytest.approx(0.37)


def test_tabular_rankers_group_candidates_by_request():
    model = _fit_tabular("xgboost_reranker")
    assert model.recipe["group_key"] == "user_id"
    assert model.cfg["n_train_rows"] > 0


class _StubBackend:
    """Stands in for xgboost/lightgbm so the recipe path is tested without them."""

    name = "stub"
    artifact = "booster.json"

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def fit(self, train, validation, params, rounds, early_stopping_rounds, ranker):
        self.calls.append(
            {
                "params": dict(params),
                "ranker": ranker,
                "rounds": rounds,
                "early_stopping_rounds": early_stopping_rounds,
                "n_rows": len(train["X"]),
                "positives": float(np.sum(train["y"])),
                "group_sizes": list(train["group_sizes"]),
                "n_val_rows": None if validation is None else len(validation["X"]),
            }
        )
        metric = params.get("eval_metric") or params.get("metric")
        curve = {"train": {metric: [0.5, 0.4]}, "validation": {metric: [0.6, 0.55]}}
        return {"weights": [1.0] * train["X"].shape[1]}, curve, 2

    def predict(self, booster, frame):
        return frame.to_numpy(dtype=float) @ np.asarray(booster["weights"])

    def save(self, booster, path):
        import os

        with open(os.path.join(path, self.artifact), "w", encoding="utf-8") as fh:
            json.dump(booster, fh)

    def load(self, path):
        import os

        with open(os.path.join(path, self.artifact), encoding="utf-8") as fh:
            return json.load(fh)


@pytest.fixture
def stub_backend(monkeypatch):
    from amt.jobs.models import tabular_common

    stub = _StubBackend()
    monkeypatch.setitem(tabular_common.BACKENDS, "xgboost", stub)
    monkeypatch.setitem(tabular_common.BACKENDS, "lightgbm", stub)
    return stub


@pytest.mark.parametrize("family", BOOSTER_FAMILIES)
def test_tabular_fit_builds_grouped_labelled_rows(stub_backend, family):
    from amt.jobs.models import tabular_common

    model = tabular_common.fit_family(
        family, _tabular_interactions(), _validation_frame(), _tabular_cfg(family)
    )
    call = stub_backend.calls[-1]
    assert call["ranker"] is family.endswith("_reranker")
    assert call["rounds"] == 8 and call["early_stopping_rounds"] == 4
    # One positive per user group, the rest sampled negatives.
    assert call["positives"] == len(call["group_sizes"])
    assert set(call["group_sizes"]) == {4}
    # Validation is used for early stopping only, never for feature state.
    assert call["n_val_rows"] == 8
    assert model.cfg["n_features"] == len(_recipe()["feature_columns"])
    assert len(model.recommend([["i1"], []], 3)[0]) == 3


def test_tabular_fit_passes_the_family_objective_and_tuned_params(stub_backend):
    from amt.jobs.models import tabular_common

    tabular_common.fit_family(
        "lightgbm_reranker",
        _tabular_interactions(),
        _validation_frame(),
        _tabular_cfg("lightgbm_reranker"),
    )
    params = stub_backend.calls[-1]["params"]
    assert params["objective"] == "lambdarank"
    assert params["learning_rate"] == 0.2 and params["max_depth"] == 3
    # Campaign plumbing stays out of the booster.
    assert "feature_recipe" not in params and "rationale" not in params


def test_tabular_negatives_are_never_the_positive(stub_backend):
    from amt.jobs.models import tabular_common

    recipe = _recipe(
        candidate_generation={"strategy": "popularity_topn", "n": 8, "exclude_seen": True}
    )
    state = tabular_common.build_state(recipe, _tabular_interactions())
    rng = np.random.default_rng(0)
    negatives = tabular_common._sample_negatives(state, ["i1"], ["i0"], 5, rng, True)
    assert "i0" not in negatives and "i1" not in negatives
    assert len(set(negatives)) == len(negatives) == 5
