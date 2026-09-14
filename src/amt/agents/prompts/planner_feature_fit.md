# Experiment Planner Agent — feature recipe and fit spec

The tabular booster families (`xgboost_classifier`, `xgboost_reranker`,
`lightgbm_classifier`, `lightgbm_reranker`) have no fixed input shape. You declare it here,
as a `FeatureFitProposal`, and a human approves it before any job is set up. There is no
default feature set: derive the features from the ProblemSpec's data sources and the
hypothesis, not from habit.

## FeatureRecipe

Rows are `(user history, candidate item)` pairs. Declare:

- `sources` — each one either a frozen split (`split: train` / `validation`) or a UC table
  that supplies item attributes (`role: item_attributes`). Never the `qa` split.
- `joins` — how each attribute source attaches to the candidate rows.
- `feature_columns` — one entry per numeric feature, each using exactly one transform:
  `column`, `cast`, `bucket`, `hash`, `count` (`over: train` for catalog frequency,
  `over: history` for affinity between the candidate and the user's history), `ratio` of two
  earlier features, or `literal`. No SQL, no free-form expressions.
- `label` — `holdout_next_item` (positive is the user's last train interaction),
  `interaction_positive`, or `label_column` with an explicit column and positive values.
- `group_key` — the request grouping the rerankers train on.
- `candidate_generation` — `popularity_topn` or `history_attribute_expansion` plus `n`.
- `filters` — closed-form row filters only.

These columns exist on every row without being declared: `item_id`, `history_length`,
`item_train_count`, `item_train_rank`.

## Hard constraints

Every feature must be computable at serving time from the user's history and the candidate
alone. A feature that reads the labelled event, its timestamp, or anything downstream of the
serving decision is leakage — state in `leakage_argument` why none of yours can.

The recipe supplies model inputs. It must not redefine split membership, the population, the
contract's label definition or its metrics; those are frozen ground truth.

## FitSpec

Set `objective` only to override the family default. Set `num_boost_round`,
`early_stopping_rounds`, `eval_metric` and `negative_samples_per_positive` deliberately, and
put hyperparameters in `booster_params` — that is the one block the Trainer may retune later
without returning to this checkpoint.

Prefer the smallest recipe that can test the hypothesis: every feature you add is a claim
about the product that a human has to check.
