"""What a failed Databricks run reports back to the orchestrator."""

from __future__ import annotations

from typing import Any

from amt.databricks.jobs import get_run_output


class _Output:
    def __init__(self, error: str = "", logs: str = "") -> None:
        self.error = error
        self.logs = logs


class _Task:
    def __init__(self, run_id: int) -> None:
        self.run_id = run_id


class _Jobs:
    def __init__(self, outputs: dict[int, _Output]) -> None:
        self.outputs = outputs

    def get_run_output(self, run_id: int) -> _Output:
        return self.outputs[run_id]


class _Run:
    def __init__(self, task_run_ids: list[int]) -> None:
        self.tasks = [_Task(run_id) for run_id in task_run_ids]


class _Client:
    def __init__(self, outputs: dict[int, _Output]) -> None:
        self.jobs: Any = _Jobs(outputs)


def test_the_logs_come_along_because_the_error_is_only_the_exception_type() -> None:
    """A wheel entry point that exits non-zero reports nothing but SystemExit."""
    client = _Client(
        {7: _Output(error="SystemExit: 2", logs="amt-job: error: unrecognized arguments")}
    )

    detail = get_run_output(client, _Run([7]))

    assert "SystemExit: 2" in detail
    assert "unrecognized arguments" in detail


def test_a_run_that_said_nothing_reports_nothing() -> None:
    assert get_run_output(_Client({7: _Output()}), _Run([7])) == ""


def test_output_no_one_can_fetch_is_not_a_second_failure() -> None:
    assert get_run_output(_Client({}), _Run([7])) == ""
