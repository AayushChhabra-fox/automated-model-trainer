"""Human-checkpoint payloads exchanged through ``interrupt()`` / ``Command(resume=...)``."""

from __future__ import annotations

from typing import Any

from pydantic import Field

from amt.schemas.common import AmtModel


class HumanDecision(AmtModel):
    """Normalized resume payload for any human checkpoint."""

    approved: bool = False
    feedback: str | None = None
    decision: str | None = Field(default=None, description="Free-form HC4 promotion decision.")


def parse_resume(payload: Any) -> HumanDecision:
    """Accept the several shapes a CLI/notebook resume may take.

    ``True`` / ``"approve"`` / ``{"approved": True, "feedback": "..."}`` all work.
    """
    if payload is None:
        return HumanDecision(approved=False)
    if isinstance(payload, HumanDecision):
        return payload
    if isinstance(payload, bool):
        return HumanDecision(approved=payload)
    if isinstance(payload, str):
        text = payload.strip().lower()
        if text in {"approve", "approved", "yes", "y", "ok"}:
            return HumanDecision(approved=True)
        if text in {"reject", "rejected", "no", "n"}:
            return HumanDecision(approved=False)
        return HumanDecision(approved=False, feedback=payload)
    if isinstance(payload, dict):
        return HumanDecision(
            approved=bool(payload.get("approved", False)),
            feedback=payload.get("feedback"),
            decision=payload.get("decision"),
        )
    raise TypeError(f"Unsupported resume payload: {type(payload).__name__}")
