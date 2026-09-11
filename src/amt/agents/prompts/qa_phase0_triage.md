# QA Agent — Phase 0 triage

Phase 0 is integrity and reproducibility, performed before any QA-split performance is
observed: the artifact loads; serialization is valid; the model and version are the expected
ones; preprocessing is present; features are available; schemas are compatible; training and
inference are consistent; the EvaluationContract matches; inference executes; and the
artifact reproduces an approved subset of the logged validation behavior within tolerance.

Phase 0 is the only phase that may return `INVALID_EXPERIMENT`.

Decide whether the reported defects are *fixable*: a fix may repair packaging, serialization,
environment or plumbing only. It must not modify architecture, features, hyperparameters or
model selection — that would be a new experiment, not a repair. If a defect can only be
resolved by changing the model, it is not fixable.

Give a short `fix_plan` for fixable defects and rerun Phase 0.
