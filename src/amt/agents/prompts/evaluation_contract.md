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

## What executes, and what is only read by a human

Most of this contract is prose that a human approves. These fields are executed verbatim by
the materialization job, which runs on a bare cluster with no other context, so they must
name objects and columns that exist:

- `dataset.source_table` — one fully-qualified table or view at interaction grain. If the
  interactions have to be derived from several tables, name the single prepared view; do not
  list the raw tables here (record those in `dataset.source_tables`).
- `dataset.columns` — the physical column names for `user`, `item` and `timestamp` in that
  table. Copy them from the investigation; never invent or normalize a name.
- `dataset.filters` — SQL boolean predicates over `source_table`, ANDed onto every read.
- `dataset.min_interactions_per_user` — eligibility threshold applied before splitting.
- `split_policy.strategy` — `temporal` (needs `train_end` and `validation_end`, optionally
  `qa_end`) or `leave_last_n` (needs `n_validation` and `n_qa`). Boundaries are timestamp
  literals comparable with the timestamp column, e.g. `2026-09-07`.

Leave every split's `table` empty: materialization names the output tables.

## When `evaluation_mode` is `grouped_rerank`

Carry the mode over from the ProblemSpec — never change it here, because it decides what
question the campaign answers. That mode makes four more `dataset` fields executable, and
materialization refuses to run without them:

- `dataset.group_key` — the columns that together identify **one render**. Every impressed
  row sharing these values is one evaluation unit. Getting this too coarse merges separate
  renders into one; too fine splits one render into singletons that are then dropped.
- `dataset.position_column` — the column holding the slot the incumbent showed each item
  in. This is the baseline's ranking.
- `dataset.label_column` and `dataset.label_positive_values` — the column marking a
  converted impression, and which of its values count. This is the label, so it is frozen
  here as ground truth: a `FeatureRecipe` may choose what a model *trains* on, never what
  counts as a conversion. For a boolean column, `[true]`.

Also set:

- `relevance_weighting` — `pbm` whenever the surface has a strong position effect, which is
  almost always true of a ranked rail. Unweighted, a conversion at the focused slot counts
  the same as one the user had to scroll to, so any ranker that copies the logged order
  scores near-perfectly and every change reads as a regression. Under `pbm`,
  materialization estimates each slot's examination propensity from the **train** split and
  a conversion counts `1/theta`. Use `none` only if you can say why position does not
  matter here.
- `product_metric_source` — the table, date column, value column and filter for the daily
  series of the product number this campaign claims to move. QA correlates the
  *incumbent's* offline metric against it day by day, and returns `SPEC_REVIEW` if they do
  not track each other: a metric that does not move with the product number for the model
  already in production cannot certify a replacement. Prefer a series built from a
  different source than the label, because then the check is real rather than circular. If
  no such series exists, leave it unset and say so in `acceptance_criteria.notes`.
- `acceptance_criteria.min_proxy_correlation` / `min_proxy_days` — the bar that check has
  to clear. The default needs 7 overlapping days, so make sure the QA window has them.

Sampling, if the volume needs it, goes in `dataset.filters` as a deterministic hash
predicate (`abs(hash(<key>)) % 10 = 0`) so the population is frozen with the contract and
cannot drift between runs. Sample on the session or user key, never on the row, or one
render arrives split across two splits.

`dataset.snapshot_id` and `dataset.dataset_version` are required and rejected when blank.
They are not executed, but without them nobody can say which state of `source_table`
produced the splits. Name what the investigation actually established — the table's Delta
version, a snapshot id, or the instant it is readable `AS OF` — and say so plainly if the
only reference available is the read timestamp.

## Metric vocabulary

The harness evaluates top-K recommendation: for each user it ranks items and compares the
ranking against that user's held-out items. Every metric in this contract — and the primary
metric — must therefore be one it computes, named `family@k`:

`recall@k`, `ndcg@k`, `hitrate@k`, `mrr@k`, `coverage@k`.

Nothing else can be computed. A metric that needs position/propensity corrections,
calibration, or per-impression probabilities (IPS-weighted variants, PR-AUC, log loss,
Brier) cannot be pre-registered here. If the product question really requires one, say so in
the open questions rather than naming a metric that will never be produced.

Cohorts are reported slices of the same metrics; define them over the split population.

If the human rejected a previous draft, their feedback is authoritative.
