"""The experimentation report, and the gate that decides whether it means anything.

If the offline metric cannot be shown to track the product number for the
ranker already in production, the campaign stops at SPEC_REVIEW before any
candidate number is read. That ordering is the point.
"""

from __future__ import annotations

import pytest

from amt.jobs.eval.paired import ab_design, spearman, top_slot_conversions
from amt.report import render_report
from amt.schemas.conclusion import (
    GROUPED_RERANK_ASSUMPTIONS,
    AbDesign,
    CohortRow,
    ExperimentationReport,
    ProductUnits,
    ProxySanity,
)

# --- proxy correlation ---------------------------------------------------


def test_two_series_that_move_together_correlate():
    offline = [0.10, 0.12, 0.11, 0.15, 0.14, 0.18, 0.17]
    product = [2.0, 2.4, 2.2, 3.1, 2.9, 3.6, 3.4]

    assert spearman(offline, product) == pytest.approx(1.0)


def test_rank_correlation_ignores_the_difference_in_scale():
    """Our label is playback; theirs is a click. Only co-movement is evidence."""
    assert spearman([1, 2, 3, 4], [100, 200, 300, 400]) == pytest.approx(1.0)


def test_two_series_that_disagree_correlate_negatively():
    assert spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) == pytest.approx(-1.0)


def test_ties_do_not_invent_an_ordering():
    assert spearman([1, 1, 2, 2], [1, 1, 2, 2]) == pytest.approx(1.0)


def test_a_flat_series_has_no_correlation_to_report():
    """Undefined, not zero: a metric that never moved is not evidence of anything."""
    import math

    assert math.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4]))


def test_too_few_days_is_undefined_rather_than_confident():
    import math

    assert math.isnan(spearman([1, 2], [1, 2]))


def test_misaligned_series_are_a_programming_error():
    with pytest.raises(ValueError, match="align"):
        spearman([1, 2, 3], [1, 2])


# --- product units -------------------------------------------------------


def test_the_incumbents_number_is_its_own_observed_top_slot_rate():
    """Which is why it matches the dashboard: theta at slot 0 is 1."""
    # Four renders; the slot-0 item converted in two of them.
    ranked = [["a", "b"], ["c", "d"], ["e", "f"], ["g", "h"]]
    truth = [{"a"}, set(), {"e"}, set()]
    gains = [{"a": 1.0}, {}, {"e": 1.0}, {}]

    units = top_slot_conversions(ranked, truth, gains)

    assert units["per_1k"] == pytest.approx(500.0)
    assert units["n_renders"] == 4


def test_a_candidate_that_surfaces_a_buried_conversion_is_credited_by_its_propensity():
    """Slot 1 was examined a tenth as often, so recovering it counts ten times."""
    ranked = [["b", "a"]]
    truth = [{"b"}]
    gains = [{"b": 10.0}]

    assert top_slot_conversions(ranked, truth, gains)["per_1k"] == pytest.approx(10_000.0)


def test_a_top_pick_that_did_not_convert_earns_nothing():
    assert top_slot_conversions([["x", "y"]], [{"y"}], [{"y": 5.0}])["per_1k"] == 0.0


def test_the_bootstrap_is_memory_bounded_at_a_real_containers_scale(monkeypatch):
    """A single (2000, 500k) index array is 8 GB, which is how a QA job dies.

    The chunked draw is asserted to allocate no more than the chunk budget per
    call, and the seeded result must not depend on the chunking.
    """
    import numpy as np

    from amt.jobs.eval import paired

    sizes: list[int] = []

    class _Spy:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def integers(self, low, high=None, size=None, **kwargs):
            if isinstance(size, tuple):
                sizes.append(size[0] * size[1])
            return self._inner.integers(low, high, size=size, **kwargs)

    real_default_rng = np.random.default_rng
    monkeypatch.setattr(paired.np.random, "default_rng", lambda seed: _Spy(real_default_rng(seed)))
    values = np.linspace(0.0, 1.0, 200_000)

    means = paired._bootstrap_means(values, n_boot=2000, seed=7)

    assert means.size == 2000
    assert sizes, "the resample was drawn at all"
    assert max(sizes) <= paired._BOOTSTRAP_CHUNK_ROWS
    assert len(sizes) > 1, "200k x 2000 must not be drawn in one allocation"


def test_chunking_does_not_change_the_interval(monkeypatch):
    """Same seed, same numbers, whatever the chunk size."""
    import numpy as np

    from amt.jobs.eval import paired

    values = np.linspace(-1.0, 1.0, 5_000)
    monkeypatch.setattr(paired, "_BOOTSTRAP_CHUNK_ROWS", 10_000_000)
    whole = paired._bootstrap_means(values, n_boot=500, seed=11)
    monkeypatch.setattr(paired, "_BOOTSTRAP_CHUNK_ROWS", 50_000)
    chunked = paired._bootstrap_means(values, n_boot=500, seed=11)

    assert np.array_equal(whole, chunked)


def test_the_four_metrics_agree_whether_computed_together_or_apart():
    """`per_user_metrics` shares one hit matrix; it must change no number."""
    import numpy as np

    from amt.jobs.eval.harness import (
        hitrate_at_k,
        mrr,
        ndcg_at_k,
        per_user_metrics,
        recall_at_k,
    )

    ranked = [["a", "b", "c"], ["c", "a", "b"], ["x", "y", "z"], ["b", "c", "a"]]
    truth = [{"b", "c"}, {"c"}, set(), {"a"}]
    gains = [{"b": 8.0, "c": 1.0}, {"c": 3.0}, {}, {"a": 1.0}]

    shared = per_user_metrics(ranked, truth, ks=(1, 3), gains=gains)

    for k in (1, 3):
        assert np.array_equal(
            shared[f"recall@{k}"], recall_at_k(ranked, truth, k, gains), equal_nan=True
        )
        assert np.array_equal(
            shared[f"ndcg@{k}"], ndcg_at_k(ranked, truth, k, gains), equal_nan=True
        )
        assert np.array_equal(
            shared[f"hitrate@{k}"], hitrate_at_k(ranked, truth, k, gains), equal_nan=True
        )
        assert np.array_equal(shared[f"mrr@{k}"], mrr(ranked, truth, k, gains), equal_nan=True)


def test_the_product_unit_carries_a_confidence_interval():
    ranked = [["a"], ["b"], ["c"], ["d"], ["e"], ["f"]]
    truth = [{"a"}, set(), {"c"}, set(), {"e"}, set()]

    units = top_slot_conversions(ranked, truth)

    assert units["ci_low"] < units["per_1k"] < units["ci_high"]


def test_unweighted_units_are_just_the_hit_rate():
    assert top_slot_conversions([["a"], ["b"]], [{"a"}, {"b"}])["per_1k"] == 1000.0


# --- A/B sizing ----------------------------------------------------------


def test_a_bigger_effect_needs_less_traffic():
    small = ab_design(0.40, 0.41, renders_per_day=100_000)
    large = ab_design(0.40, 0.50, renders_per_day=100_000)

    assert large["required_renders"] < small["required_renders"]
    assert large["days_at_current_traffic"] < small["days_at_current_traffic"]


def test_the_sizing_reports_the_relative_effect_a_product_owner_asks_for():
    design = ab_design(0.40, 0.44, renders_per_day=1_000)

    assert design["absolute_effect"] == pytest.approx(0.04)
    assert design["relative_effect"] == pytest.approx(0.1)


def test_no_effect_is_said_plainly_rather_than_as_an_infinity():
    design = ab_design(0.4, 0.4, renders_per_day=1_000)

    assert design["required_renders"] == 0.0
    assert design["days_at_current_traffic"] == 0.0


def test_without_a_traffic_estimate_the_days_are_not_invented():
    design = ab_design(0.40, 0.45)

    assert design["required_renders"] > 0
    assert design["days_at_current_traffic"] == 0.0


# --- the rendered report -------------------------------------------------


def _report(**overrides) -> ExperimentationReport:
    fields = {
        "campaign_id": "fyl-ctv-rerank-2026q3",
        "primary_metric": "ndcg@4",
        "primary_delta": 0.0121,
        "primary_ci": (0.004, 0.020),
        "primary_significant": True,
        "candidate_metrics": {"ndcg@4": 0.412, "hitrate@1": 0.298},
        "baseline_metrics": {"ndcg@4": 0.400, "hitrate@1": 0.291},
        "secondary_deltas": {"hitrate@1": 0.007},
        "product_units": ProductUnits(
            candidate=41.2,
            candidate_ci=(38.0, 44.1),
            baseline=38.7,
            baseline_ci=(36.2, 41.0),
            n_renders=612_004,
        ),
        "cohort_table": [
            CohortRow(
                name="ROKU",
                metric="ndcg@4",
                candidate=0.42,
                baseline=0.40,
                delta=0.02,
                relative_delta=0.05,
                n=200_000,
            ),
            CohortRow(
                name="FIRETV",
                metric="ndcg@4",
                candidate=0.36,
                baseline=0.40,
                delta=-0.04,
                relative_delta=-0.10,
                n=90_000,
                regressed=True,
            ),
        ],
        "regressions": ["FIRETV"],
        "proxy_sanity": ProxySanity(
            passed=True,
            checked=True,
            rho=0.71,
            rho_external=0.71,
            rho_own_label=0.88,
            days=7,
            source="external",
        ),
        "propensities": {"ROKU|0": 1.0, "ROKU|1": 0.12},
        "ab_design": AbDesign(
            baseline_rate=0.0387,
            candidate_rate=0.0412,
            absolute_effect=0.0025,
            relative_effect=0.0646,
            required_renders=190_000.0,
            days_at_current_traffic=2.2,
        ),
        "operational": {"latency_ms_p95": 24.1},
        "assumptions": list(GROUPED_RERANK_ASSUMPTIONS),
        "evaluation_mode": "grouped_rerank",
        "relevance_weighting": "pbm",
        "contract_hash": "a" * 64,
    }
    fields.update(overrides)
    return ExperimentationReport(**fields)


def test_the_report_leads_with_the_product_unit_not_the_ranking_metric():
    html = render_report(_report())

    assert html.index("product units") < html.index("The offline effect")
    assert "+2.50" in html, "the delta in conversions per 1,000 renders"
    assert "top-slot conversions per 1,000 renders" in html


def test_the_proxy_verdict_comes_before_any_candidate_number():
    html = render_report(_report())
    assert html.index("Is the offline metric measuring the product?") < html.index("41.2")


def test_a_failed_proxy_check_says_the_numbers_are_not_evidence():
    html = render_report(
        _report(
            proxy_sanity=ProxySanity(
                passed=False,
                checked=True,
                rho=0.21,
                days=7,
                threshold=0.5,
                reason="rho=0.210 is below the 0.5 the contract pre-registered",
            )
        )
    )

    assert "Failed" in html
    assert "not</em> evidence" in html
    assert "0.210 is below" in html


def test_an_unchecked_proxy_says_the_assumption_is_untested():
    html = render_report(_report(proxy_sanity=ProxySanity(checked=False)))
    assert "untested" in html


def test_a_regressed_cohort_is_named_rather_than_averaged_away():
    html = render_report(_report())

    assert "FIRETV" in html and "regressed" in html
    assert "guardrail" in html


def test_the_propensities_are_printed_so_the_weighting_is_auditable():
    html = render_report(_report())

    assert "ROKU|1" in html
    # 1 / 0.12: the weight a slot-1 conversion carries.
    assert "8.3\u00d7" in html


def test_the_caveats_are_data_not_prose_an_agent_can_edit_out():
    html = render_report(_report())

    for assumption in GROUPED_RERANK_ASSUMPTIONS:
        assert assumption[:40] in html
    assert "not causal lift" in html


def test_a_next_item_report_carries_no_grouped_caveats_or_propensities():
    html = render_report(
        _report(
            evaluation_mode="next_item",
            relevance_weighting="none",
            assumptions=[],
            propensities={},
        )
    )

    assert "Position-bias correction" not in html
    assert "What this does not say" not in html


def test_an_empty_report_still_renders():
    """A SPEC_REVIEW campaign gets the same page as a passing one."""
    html = render_report(ExperimentationReport())
    assert "<html" in html and "</html>" in html


def test_the_validation_verdict_is_computed_not_narrated():
    from types import SimpleNamespace

    from amt.graph.nodes.conclusion import validation_verdict

    champions = [
        SimpleNamespace(hypothesis_id="H1", primary_metric_value=0.7502),
        SimpleNamespace(hypothesis_id="H4", primary_metric_value=0.7535),
    ]
    verdict = validation_verdict({"ndcg@4": 0.7911}, champions, "ndcg@4")

    assert verdict["any_beats_baseline"] is False
    h4 = verdict["champions"][1]
    assert h4["beats_baseline"] is False
    assert abs(h4["delta"] - (-0.0376)) < 1e-4
    assert abs(h4["relative_delta"] - (-0.0475)) < 1e-3
    assert validation_verdict({}, [], "ndcg@4") is None
