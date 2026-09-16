"""The proving campaign's EvaluationContract, as code.

`amt` normally derives this from the ProblemSpec the agent drafts at HC2/HC3.
This module exists because the *first* run of the grouped-rerank materialization
should be judged against a contract a human wrote, so a failure is attributable
to the runtime rather than to the agent. It is also the reference the operator
compares the agent's draft against at HC3.

Every number here was confirmed against the table by ``scripts/probe_fyl.py``
and ``scripts/probe_fyl_boundaries.py``; nothing is inferred from a column name.
"""

from __future__ import annotations

from amt.schemas.common import CohortDef, MetricDef, SplitRef
from amt.schemas.evaluation_contract import (
    AcceptanceCriteria,
    ColumnMap,
    DatasetDefinition,
    EvaluationContract,
    ProductMetricSource,
    SplitPolicy,
    TemporalRange,
)

CAMPAIGN = "fyl-ctv-rerank-2026q3"
SOURCE = "pnr_prod.silver.pr_video_impressions"

#: "Home Page For You" / "Latest For You". The dashboard hard-codes this id as
#: FYL_CONTAINER_ID, so it is the same rail the product number is reported on.
CONTAINER = "671107897793"

#: CTV only. Position mechanics differ by surface -- a D-pad and a swipe
#: carousel are not the same rail -- so mixing them would let surface mix
#: masquerade as ranking quality. Mobile and tvOS are follow-on campaigns.
CTV_CLIENTS = ("ROKU", "FIRETV", "TIZEN", "LG", "VIZIO", "ANDROIDTV")

#: A schema-example value the STM page documents; it is not a real account.
SCHEMA_EXAMPLE_ACCOUNT = "af44641e-a0ed-4d88-92d0-6fa68e0cbf1a"

#: Whole Sun-Sat weeks: 07-27 is a Monday, 08-30 and 09-06 are Sundays, and the
#: table's last impression is 09-12 (a Saturday). So validation and QA are one
#: complete week each and weekday mix is balanced across the splits. QA is the
#: most recent week -- a forward test on held-out time.
WINDOW_START = "2026-07-27"
TRAIN_END = "2026-08-30"
VALIDATION_END = "2026-09-06"
QA_END = "2026-09-13"

#: Four tiles are visible on CTV: slots 0-3 each carry ~13M impressions over
#: the window and slot 4 drops to 3.5M. Slot 4's conversion rate also *rises*
#: back to 3.2% from slot 3's 1.7%, which is the signature of a different
#: population (a user who scrolled) rather than a less-examined tile.
K = 4
PRIMARY_METRIC = f"ndcg@{K}"

#: Frozen in the contract so the population cannot drift between runs. 62.6M
#: impressions over the window is 9.7M train renders; 10 % keeps `build_rows`
#: in single-node range, and 20 % leaves QA 518k renders -- far more than
#: enough to bound a half-point CTR effect.
TRAIN_SAMPLE = "abs(hash(session_id)) % 10 = 0"
EVAL_SAMPLE = "abs(hash(session_id)) % 5 = 0"

FILTERS = [
    f"parent_id = '{CONTAINER}'",
    "parent_type = 'container'",
    "client_name IN (" + ", ".join(f"'{c}'" for c in CTV_CLIENTS) + ")",
    "item_index IS NOT NULL",
    f"account_id <> '{SCHEMA_EXAMPLE_ACCOUNT}'",
    f"event_received >= '{WINDOW_START}'",
    # One predicate for both sampling rates: a session is in the train sample or
    # the evaluation sample, and which window it lands in decides which applies.
    f"(({TRAIN_SAMPLE} AND event_received < '{TRAIN_END}') "
    f"OR ({EVAL_SAMPLE} AND event_received >= '{TRAIN_END}'))",
]


def _placeholder(name: str) -> SplitRef:
    """Materialization returns the real reference; this is what HC3 approves."""
    return SplitRef(name=name, table="", delta_version=0, row_count=0, membership_hash="")


def contract(spec_hash: str = "fyl-ctv-rerank-2026q3-spec") -> EvaluationContract:
    return EvaluationContract(
        spec_hash=spec_hash,
        evaluation_mode="grouped_rerank",
        relevance_weighting="pbm",
        propensity_floor=0.02,
        dataset=DatasetDefinition(
            source_table=SOURCE,
            source_tables=[SOURCE],
            columns=ColumnMap(user="account_id", item="entity_id", timestamp="event_received"),
            filters=FILTERS,
            min_interactions_per_user=1,
            # One render of one container for one session at one instant.
            group_key=["session_id", "page_id", "parent_id", "event_received"],
            position_column="item_index",
            label_column="conversion_flag",
            label_positive_values=[True],
            snapshot_id=f"{SOURCE}@{WINDOW_START}..{QA_END}",
            dataset_version="2026-09-16",
            temporal_range=TemporalRange(start=WINDOW_START, end=QA_END),
            population_eligibility=(
                "CTV renders of container 671107897793 with at least two distinct "
                "impressed titles, from the frozen session-hash sample."
            ),
            exclusions=[
                "mobile (D2C_MOB_LANDSCAPE_FORYOU) and tvOS (D2C_LR_LANDSCAPE_FORYOU): "
                "different position mechanics, so different propensities",
                "renders with fewer than two distinct candidates: nothing to re-rank",
                f"account_id {SCHEMA_EXAMPLE_ACCOUNT} (a schema example, not an account)",
            ],
            label_construction=(
                "conversion_flag: a content_click on the impressed entity in the same "
                "account/session/container/page within -2..+90s, followed by a "
                "media_content_start within 90s. Autoplay is excluded by construction."
            ),
        ),
        train_split=_placeholder("train"),
        validation_split=_placeholder("validation"),
        qa_split=_placeholder("qa"),
        split_policy=SplitPolicy(
            strategy="temporal",
            train_end=TRAIN_END,
            validation_end=VALIDATION_END,
            qa_end=QA_END,
            rationale=(
                "Whole Sun-Sat weeks for validation and QA so weekday mix is balanced, "
                "with QA the most recent complete week: a forward test on held-out time. "
                "Item statistics used as features come from the train window only."
            ),
        ),
        user_population="CTV sessions that saw the rail in the window, hash-sampled.",
        item_population="Titles the rail actually impressed. A re-ranker cannot surface more.",
        label_definition="conversion_flag = true on the impressed row.",
        candidate_generation_rules=(
            "logged_impressions: the candidate set is exactly what the render showed, "
            "deduplicated per title keeping the earlier slot."
        ),
        cohorts=[
            CohortDef(
                name="client_name",
                description="The six CTV clients. Slot-0 conversion ranges 33-45% across them.",
            ),
            CohortDef(
                name="history_length",
                description="Prior renders for the account inside the train window.",
            ),
        ],
        metrics=[
            MetricDef(
                name=PRIMARY_METRIC,
                definition=(
                    "Position-weighted NDCG over the impressed candidates. A conversion "
                    "at slot p carries gain 1/theta_p, theta estimated per client on train."
                ),
                k=K,
                implementation="harness:ndcg_at_k",
                is_primary=True,
                tolerance=1e-3,
            ),
            MetricDef(
                name="hitrate@1",
                definition=(
                    "Did the model put a converting title first? The slot-0 metric, and "
                    "the bridge to the dashboard's Container CTR."
                ),
                k=1,
                implementation="harness:hitrate_at_k",
            ),
            MetricDef(
                name=f"mrr@{K}",
                definition="Reciprocal rank of the first converting title.",
                k=K,
                implementation="harness:mrr",
            ),
            MetricDef(
                name="auc",
                definition="Does the score separate converters from non-converters at all?",
                implementation="harness:binary_metrics",
            ),
            MetricDef(
                name="logloss",
                definition="Calibration of the candidate scores.",
                direction="minimize",
                implementation="harness:binary_metrics",
            ),
        ],
        primary_metric=PRIMARY_METRIC,
        product_metric_source=ProductMetricSource(
            table="product_api_dev.cursor_agent.hp_dash_home_container_daily",
            date_column="event_date",
            value_column="container_ctr_pct",
            filter_sql=(
                f"container_id = '{CONTAINER}' AND cohort = 'all' AND page_id = '727846815041'"
            ),
            description=(
                "The dashboard's own Container CTR: slot-0 content_click rows over slot-0 "
                "content_impression rows, built from bronze dtc_visitid rather than from "
                "pr_video_impressions. A different source, so the correlation is a real "
                "cross-source test of 'our label tracks their metric'."
            ),
        ),
        acceptance_criteria=AcceptanceCriteria(
            primary_metric=PRIMARY_METRIC,
            min_relative_improvement=0.0,
            require_ci_excludes_zero=True,
            max_cohort_regression=0.02,
            min_proxy_correlation=0.5,
            min_proxy_days=7,
            notes=(
                "Pre-registered at HC3: the pbm ndcg@4 paired CI must exclude zero, "
                "pbm hitrate@1 must not regress, no client_name cohort may regress more "
                "than 2% relative, and the proxy correlation must reach 0.5 over at "
                "least 7 QA days."
            ),
        ),
    )
