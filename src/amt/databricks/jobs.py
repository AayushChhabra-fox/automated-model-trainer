"""One-time Databricks job runs (``jobs.submit``) for AMT compute.

AMT never creates persistent jobs: every unit of compute is a one-time run of
the ``amt-job`` wheel entry point, which keeps the workspace free of stale job
definitions and makes each run's parameters part of its own record.

The SDK is imported lazily so the orchestrator (and the unit tests) import
this module without ``databricks-sdk`` installed.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "SUCCESS_RESULT_STATES",
    "TERMINAL_LIFECYCLE_STATES",
    "JobRunError",
    "build_task",
    "get_run",
    "get_run_output",
    "is_terminal",
    "run_page_url",
    "run_state",
    "submit_run",
    "wait_for_terminal",
]

TERMINAL_LIFECYCLE_STATES = {"TERMINATED", "SKIPPED", "INTERNAL_ERROR"}
SUCCESS_RESULT_STATES = {"SUCCESS"}

DEFAULT_POLL_SECONDS = 15.0
DEFAULT_TIMEOUT_SECONDS = 3 * 60 * 60.0


class JobRunError(RuntimeError):
    """A submitted run reached a non-success terminal state (or timed out)."""


def _sdk():
    try:
        from databricks.sdk.service import compute, jobs
    except Exception as exc:  # pragma: no cover
        raise JobRunError(
            "databricks-sdk is required to submit Databricks jobs"
        ) from exc
    return compute, jobs


def build_task(
    *,
    task_key: str,
    wheel_uri: str,
    entry_point: str = "amt-job",
    package_name: str = "amt",
    parameters: Sequence[str] = (),
    cluster: Mapping[str, Any] | None = None,
    extra_libraries: Sequence[str] = (),
    environment_vars: Mapping[str, str] | None = None,
    custom_tags: Mapping[str, str] | None = None,
):
    """A single ``python_wheel_task`` on a new job cluster (or an existing one)."""
    compute, jobs = _sdk()
    cluster = dict(cluster or {})

    libraries = [compute.Library(whl=wheel_uri)]
    libraries.extend(compute.Library(pypi=compute.PythonPyPiLibrary(package=p))
                     for p in extra_libraries)

    kwargs: dict[str, Any] = {
        "task_key": task_key,
        "python_wheel_task": jobs.PythonWheelTask(
            package_name=package_name,
            entry_point=entry_point,
            parameters=list(parameters),
        ),
        "libraries": libraries,
    }

    if cluster.get("existing_cluster_id"):
        kwargs["existing_cluster_id"] = cluster["existing_cluster_id"]
    else:
        spec: dict[str, Any] = {
            "spark_version": cluster.get("spark_version", "15.4.x-cpu-ml-scala2.12"),
            "node_type_id": cluster.get("node_type_id", "Standard_DS3_v2"),
            "num_workers": int(cluster.get("num_workers", 0)),
        }
        if environment_vars:
            spec["spark_env_vars"] = dict(environment_vars)
        if custom_tags:
            # Tags are a property of the cluster, so they only reach
            # `system.billing.usage.custom_tags` on a job cluster. A run on an
            # existing all-purpose cluster is billed to that cluster and cannot
            # be attributed here at all.
            spec["custom_tags"] = {str(k): str(v) for k, v in custom_tags.items()}
        for optional in ("policy_id", "data_security_mode", "instance_pool_id"):
            if cluster.get(optional):
                spec[optional] = cluster[optional]
        kwargs["new_cluster"] = compute.ClusterSpec(**spec)

    return jobs.SubmitTask(**kwargs)


def submit_run(
    client: Any,
    *,
    run_name: str,
    wheel_uri: str,
    parameters: Sequence[str],
    cluster: Mapping[str, Any] | None = None,
    entry_point: str = "amt-job",
    package_name: str = "amt",
    task_key: str = "amt_job",
    timeout_seconds: int = 0,
    extra_libraries: Sequence[str] = (),
    environment_vars: Mapping[str, str] | None = None,
    custom_tags: Mapping[str, str] | None = None,
) -> int:
    """Submit a one-time run and return its ``run_id`` (does not wait)."""
    task = build_task(
        task_key=task_key,
        wheel_uri=wheel_uri,
        entry_point=entry_point,
        package_name=package_name,
        parameters=parameters,
        cluster=cluster,
        extra_libraries=extra_libraries,
        environment_vars=environment_vars,
        custom_tags=custom_tags,
    )
    waiter = client.jobs.submit(
        run_name=run_name, tasks=[task], timeout_seconds=timeout_seconds
    )
    response = getattr(waiter, "response", waiter)
    run_id = getattr(response, "run_id", None)
    if run_id is None:
        raise JobRunError(f"jobs.submit returned no run_id: {response!r}")
    return int(run_id)


def get_run(client: Any, run_id: int):
    return client.jobs.get_run(run_id=run_id)


def run_state(run: Any) -> tuple[str, str, str]:
    """``(life_cycle_state, result_state, state_message)`` as plain strings."""
    state = getattr(run, "state", None)
    if state is None:
        return ("", "", "")

    def _name(value: Any) -> str:
        if value is None:
            return ""
        return str(getattr(value, "value", value))

    return (
        _name(getattr(state, "life_cycle_state", None)),
        _name(getattr(state, "result_state", None)),
        str(getattr(state, "state_message", "") or ""),
    )


def is_terminal(run: Any) -> bool:
    life_cycle, _result, _msg = run_state(run)
    return life_cycle in TERMINAL_LIFECYCLE_STATES


def run_page_url(run: Any) -> str:
    return str(getattr(run, "run_page_url", "") or "")


def wait_for_terminal(
    client: Any,
    run_id: int,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    on_poll: Any = None,
):
    """Poll until the run is terminal; raise on failure or timeout."""
    deadline = time.monotonic() + float(timeout_seconds)
    run = get_run(client, run_id)
    while not is_terminal(run):
        if time.monotonic() > deadline:
            raise JobRunError(
                f"run {run_id} did not reach a terminal state within "
                f"{timeout_seconds}s ({run_page_url(run)})"
            )
        if on_poll is not None:
            on_poll(run)
        time.sleep(float(poll_seconds))
        run = get_run(client, run_id)

    life_cycle, result, message = run_state(run)
    if result not in SUCCESS_RESULT_STATES:
        raise JobRunError(
            f"run {run_id} finished {life_cycle}/{result}: {message} "
            f"({run_page_url(run)})"
        )
    return run


def get_run_output(client: Any, run: Any, max_chars: int = 4000) -> str:
    """Best-effort driver output for a failed run, for error reporting.

    ``error`` alone is often just the exception type (a wheel entry point that
    exits non-zero reports ``SystemExit: 2``), so the logs come along too.
    """
    tasks = getattr(run, "tasks", None) or []
    parts: list[str] = []
    for task in tasks:
        try:
            output = client.jobs.get_run_output(run_id=task.run_id)
        except Exception:  # pragma: no cover - diagnostics only
            continue
        for attr in ("error", "logs"):
            value = str(getattr(output, attr, None) or "").strip()
            if value:
                parts.append(value[-max_chars:])
    return "\n".join(parts)
