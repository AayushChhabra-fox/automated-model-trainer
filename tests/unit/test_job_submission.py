"""What a submitted run asks for, and who gets billed for it.

Every AMT job is a one-time run, so the task built here is the only place the
family's libraries and the campaign's cost tags can be declared.
"""

from __future__ import annotations

from typing import Any

import pytest

from amt.databricks.jobs import build_task
from amt.executor.databricks import DatabricksExecutor
from amt.settings import Settings

JOB_CLUSTER = {"spark_version": "15.4.x-cpu-ml-scala2.12", "node_type_id": "n", "num_workers": 2}
TAGS = {"amt": "true", "amt_role": "train", "amt_campaign_id": "c-1"}


def _task(cluster: dict, **kwargs: Any):
    return build_task(
        task_key="amt_job",
        wheel_uri="/Volumes/cat/schema/amt_artifacts/wheels/abc/amt-0.1.0-py3-none-any.whl",
        parameters=["--role", "train"],
        cluster=cluster,
        **kwargs,
    )


def test_a_family_library_is_installed_alongside_the_wheel() -> None:
    task = _task(JOB_CLUSTER, extra_libraries=["xgboost"])

    installed = [lib.whl or lib.pypi.package for lib in task.libraries]

    assert installed[0].endswith(".whl")
    assert "xgboost" in installed


def test_cost_tags_land_on_the_job_cluster() -> None:
    task = _task(JOB_CLUSTER, custom_tags=TAGS)

    assert task.new_cluster.custom_tags == TAGS


def test_an_existing_cluster_drops_the_per_run_tags() -> None:
    """Why pinning one costs you attribution.

    A static `amt=true` cluster tag still reaches billing, so the loss is not
    "untagged" -- it is that `amt_role` and `amt_campaign_id` vary per run and
    have nowhere to live, leaving cost attributable to AMT but not to a
    campaign, role or hypothesis.
    """
    task = _task({"existing_cluster_id": "0914-091216-h6cd8ahc"}, custom_tags=TAGS)

    assert task.existing_cluster_id == "0914-091216-h6cd8ahc"
    assert task.new_cluster is None


def _executor(campaign_id: str = "vod-nextitem-001") -> DatabricksExecutor:
    return DatabricksExecutor(
        settings=Settings(_env_file=None, AMT_CATALOG="main", AMT_SCHEMA="amt"),
        client=object(),
        campaign_id=campaign_id,
        progress=None,
    )


@pytest.mark.parametrize(
    ("family", "expected"),
    [("lightgbm_reranker", ("lightgbm",)), ("xgboost_classifier", ("xgboost",)), ("sasrec", ())],
)
def test_the_executor_asks_for_what_the_family_needs(family, expected) -> None:
    assert _executor()._libraries(family) == expected


def test_the_family_comes_from_the_config_or_else_the_frozen_champion() -> None:
    """QA loads the champion's artifact, so it needs the backend too."""
    executor = _executor()

    assert executor._family({"family": "xgboost_reranker"}) == "xgboost_reranker"
    assert executor._family(None, {"family": "lightgbm_classifier"}) == "lightgbm_classifier"
    assert executor._family(None, None) == ""


def test_every_run_is_attributable_to_a_campaign_role_and_hypothesis() -> None:
    tags = _executor()._cost_tags("train", hypothesis_id="H2")

    assert tags == {
        "amt": "true",
        "amt_role": "train",
        "amt_campaign_id": "vod-nextitem-001",
        "amt_hypothesis_id": "H2",
    }


def test_a_run_outside_a_campaign_still_reads_as_amt() -> None:
    assert _executor(campaign_id="")._cost_tags("materialize") == {
        "amt": "true",
        "amt_role": "materialize",
    }
