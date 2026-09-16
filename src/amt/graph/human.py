"""Human-checkpoint payloads exchanged through ``interrupt()`` / ``Command(resume=...)``."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from pydantic import Field

from amt.schemas.common import AmtModel
from amt.schemas.hypothesis import Hypothesis


class HumanDecision(AmtModel):
    """Normalized resume payload for any human checkpoint.

    ``keep`` and ``drop`` make a checkpoint a checklist rather than an
    all-or-nothing vote. Striking one item out of five used to mean rejecting
    the whole list and paying for the agent to rewrite it.
    """

    approved: bool = False
    feedback: str | None = None
    decision: str | None = Field(default=None, description="Free-form HC4 promotion decision.")
    keep: list[str] = Field(
        default_factory=list,
        description="Hypothesis ids to keep; anything else under review is abandoned.",
    )
    drop: list[str] = Field(default_factory=list, description="Hypothesis ids to abandon.")
    freeze: bool = Field(
        default=False,
        description="At HC_CHAMPION: stop exploring and take this champion to QA.",
    )


_ID_SEPARATORS = re.compile(r"[,\s;]+")
_VERB_PREFIX = re.compile(r"^(keep|drop|freeze)\b[:\s]*", re.IGNORECASE)


def _ids(value: Any) -> list[str]:
    """Accept a list, or the free text a CLI flag or a form field produces.

    ``"H1,H3"``, ``"H1, H3"``, ``"H1 H3"`` and ``"keep H1,H3"`` all yield
    ``["H1", "H3"]`` — an operator typing exactly what a placeholder shows must
    not be told the hypothesis does not exist.
    """
    if value is None:
        return []
    if isinstance(value, str):
        text = _VERB_PREFIX.sub("", value.strip())
        return [part for part in _ID_SEPARATORS.split(text) if part]
    return [str(part).strip() for part in value if str(part).strip()]


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
            keep=_ids(payload.get("keep")),
            drop=_ids(payload.get("drop")),
            freeze=bool(payload.get("freeze", False)),
        )
    raise TypeError(f"Unsupported resume payload: {type(payload).__name__}")


def gate_update(state: Mapping[str, Any], decision: HumanDecision) -> dict:
    """The fields every human checkpoint returns, including the bounce counter.

    ``gate_cycles`` is the count the previous campaign had no way to see: eight
    reject/approve rounds cost no compute, so nothing in `status` moved and the
    loop looked idle rather than expensive.
    """
    return {
        "human_feedback": decision.feedback,
        "awaiting": None,
        "status": "running",
        "gate_cycles": int(state.get("gate_cycles", 0) or 0) + (0 if decision.approved else 1),
    }


class UnknownHypothesis(ValueError):
    """A ``keep`` / ``drop`` list names an id that is not under review.

    Raised rather than ignored: a typo in ``--keep`` would otherwise abandon
    every hypothesis the operator meant to keep, silently and irreversibly.
    The node has not committed its state when this raises, so the checkpoint is
    still pending and the operator can answer it again.
    """


def apply_checklist(
    hypotheses: Sequence[Hypothesis],
    decision: HumanDecision,
    scope: Iterable[str] | None = None,
) -> list[Hypothesis]:
    """Abandon what the operator struck out, and reorder what they kept.

    ``drop`` names what goes; ``keep``, when given, names what stays and
    everything else *in scope* goes. ``scope`` is the set of ids the checkpoint
    actually put in front of the human -- HC_FEATURE reviews only the tabular
    shapes, and must not abandon a SASRec hypothesis nobody was asked about.
    The order of ``keep`` becomes the new ``priority``, so reordering the
    checklist reorders the campaign.
    """
    known = {h.hypothesis_id: h for h in hypotheses}
    reviewed = set(known) if scope is None else {i for i in scope if i in known}

    unknown = sorted({i for i in (*decision.keep, *decision.drop) if i not in known})
    if unknown:
        raise UnknownHypothesis(
            f"no hypothesis named {unknown} in this backlog; under review: {sorted(reviewed)}"
        )

    keep = [i for i in decision.keep if i in reviewed]
    dropped = {i for i in decision.drop if i in reviewed}
    if keep:
        dropped |= reviewed - set(keep)
    if not dropped and not keep:
        return list(hypotheses)

    order = {hypothesis_id: rank for rank, hypothesis_id in enumerate(keep, start=1)}
    updated = []
    for hypothesis in hypotheses:
        if hypothesis.hypothesis_id in dropped:
            updated.append(hypothesis.model_copy(update={"status": "abandoned"}))
        elif hypothesis.hypothesis_id in order:
            updated.append(
                hypothesis.model_copy(update={"priority": order[hypothesis.hypothesis_id]})
            )
        else:
            updated.append(hypothesis)
    return updated
