"""Model families, the registry and the shared pyfunc serving wrapper.

Runs on CPU and skips the SASRec cases when torch is unavailable.
"""

from __future__ import annotations

import json

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


def test_registry_exposes_both_families():
    assert set(registry.FAMILIES) == {"popularity", "sasrec"}
    assert callable(registry.get_builder("popularity"))


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
