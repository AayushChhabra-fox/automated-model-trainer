# Evaluation Contract

Define the single EvaluationContract that establishes experimental ground truth for the
whole campaign, independent of model architecture. Every candidate and the baseline use the
same: train split; validation split; QA split; user population; item population; label
definitions; candidate-generation rules; cohort definitions; metric implementations.

- **Dataset Definition** — source data; snapshot/version; population eligibility; temporal
  ranges; exclusions; label construction. Use immutable references.
- **Train Split** — for fitting; membership fixed for the campaign.
- **Validation Split** — the Trainer's only feedback signal: architecture, hyperparameters,
  early stopping, feature decisions, hypothesis evaluation, best-config selection. Fixed
  across all models; reusable within budget.
- **QA Split** — final holdout. Fixed. Never used for architecture, hyperparameter or
  feature selection, early stopping, or Trainer iteration. Read only after champion freeze.
- **Metrics** — exact definitions, evaluated by one common harness at a pinned version.
  The primary metric must match the ProblemSpec's primary offline metric.
- **Cohorts** — the slices that will be reported (e.g. cold-start users, tail items,
  new devices). Fixed now; QA may not invent new ones later.
- **Acceptance Criteria** — pre-register what "good enough" means before any result exists.

Splits must not be recreated differently for different models. If ground truth must
fundamentally change, the campaign stops and returns to Stage 1/2 — it never silently
continues the same comparison.

Fill split table names and temporal boundaries from the ProblemSpec's data sources; row
counts and Delta versions are filled in by materialization, so 0 is acceptable here.

If the human rejected a previous draft, their feedback is authoritative.
