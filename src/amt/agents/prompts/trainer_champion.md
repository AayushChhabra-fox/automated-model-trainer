# Trainer Agent — hypothesis champion

Select exactly one best configuration for this hypothesis from the runs performed, using
the predefined validation criteria: primary metric; secondary metrics; cohort regressions;
complexity; operational constraints.

The strongest raw metric does not automatically win if it violates a guardrail. List any
cohort regressions you observe, state whether guardrails passed, and justify the choice.
Reference the run by its `run_id`.
