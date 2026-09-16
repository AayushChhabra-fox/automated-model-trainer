"""Background execution of campaigns for the local web surface.

The graph is already the backend: ``get_state`` is the read model and
``Command(resume=...)`` is the write. What a browser needs that a terminal does
not is somewhere for the twenty minutes between a decision and the next
checkpoint to go, so each campaign gets a thread and a ring buffer of events
the page renders as a progress log.

Deliberately in-process and deliberately not durable. Losing this process
loses the progress log and nothing else: the campaign itself lives in the
checkpointer, exactly as it does for the CLI.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from amt import monitor

__all__ = ["CampaignRunner", "Event", "RunnerBusy", "Runners"]

#: How many progress lines a campaign page shows. A trainer loop over four
#: hypotheses produces a few dozen; a thousand is generous and bounded.
EVENT_LIMIT = 1000


class RunnerBusy(RuntimeError):
    """A campaign is already running. The caller answers 409 rather than racing.

    Two threads resuming one campaign would interleave writes to the same
    checkpoint thread, and LangGraph offers no way to make that safe. The lock
    is the whole concurrency model.
    """


@dataclass(frozen=True)
class Event:
    """One line of the progress log."""

    at: float
    kind: str
    message: str

    @property
    def clock(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.at))


@dataclass
class CampaignRunner:
    """One campaign's thread, lock and event buffer."""

    campaign_id: str
    events: deque[Event] = field(default_factory=lambda: deque(maxlen=EVENT_LIMIT))
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _thread: threading.Thread | None = None
    _error: str = ""

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def error(self) -> str:
        return self._error

    def log(self, kind: str, message: str) -> None:
        self.events.append(Event(time.time(), kind, message))

    def start(self, work: Callable[[Callable[[str, str], None]], None]) -> None:
        """Run ``work`` in a thread, passing it a logger. Refuses to overlap."""
        if not self._lock.acquire(blocking=False):
            raise RunnerBusy(f"campaign {self.campaign_id} is already running")

        def _run() -> None:
            try:
                self._error = ""
                work(self.log)
            except Exception as exc:  # surfaced on the page, not swallowed
                self._error = f"{type(exc).__name__}: {exc}"
                self.log("error", self._error)
            finally:
                self._lock.release()

        self._thread = threading.Thread(target=_run, name=f"amt-{self.campaign_id}", daemon=True)
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the current run. Used by tests; the browser polls instead."""
        if self._thread is not None:
            self._thread.join(timeout)


class Runners:
    """The process's runners, one per campaign, created on first use."""

    def __init__(self, build: Callable[[str], Any]) -> None:
        self._build = build
        self._runners: dict[str, CampaignRunner] = {}
        self._guard = threading.Lock()

    def for_campaign(self, campaign_id: str) -> CampaignRunner:
        with self._guard:
            runner = self._runners.get(campaign_id)
            if runner is None:
                runner = CampaignRunner(campaign_id)
                self._runners[campaign_id] = runner
            return runner

    def resume(self, campaign_id: str, payload: Any) -> CampaignRunner:
        """Answer a checkpoint (or continue from one) in the background."""
        runner = self.for_campaign(campaign_id)
        graph = self._build(campaign_id)

        def work(log: Callable[[str, str], None]) -> None:
            log("resume", _describe(payload))
            for update in graph.stream(
                payload, monitor.config_for(campaign_id), stream_mode="updates"
            ):
                for node, delta in (update or {}).items():
                    if not isinstance(delta, dict):
                        continue
                    stage = str(delta.get("stage") or "")
                    log("node", f"{node}{f'  [{stage}]' if stage and stage != node else ''}")
            pending = monitor.pending_interrupt(graph, campaign_id)
            if pending:
                log("checkpoint", f"awaiting {pending.get('checkpoint')}")
            else:
                state = monitor.state(graph, campaign_id)
                log("done", f"status={state.get('status')}")

        runner.start(work)
        return runner


def _describe(payload: Any) -> str:
    """A resume payload as one readable line for the log."""
    resume = getattr(payload, "resume", None)
    if resume is None:
        return "continuing from the checkpoint"
    if not isinstance(resume, dict):
        return str(resume)
    parts = ["approved" if resume.get("approved") else "rejected"]
    for key in ("feedback", "keep", "drop", "decision"):
        if resume.get(key):
            parts.append(f"{key}={resume[key]}")
    if resume.get("freeze"):
        parts.append("freeze")
    return " ".join(parts)
