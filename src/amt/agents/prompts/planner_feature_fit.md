# Experiment Planner Agent — feature recipe and fit spec

The tabular booster families (`xgboost_classifier`, `xgboost_reranker`,
`lightgbm_classifier`, `lightgbm_reranker`) have no fixed input shape. You declare it here,
as a `FeatureFitProposal`, and a human approves it before any job is set up. There is no
default feature set: derive the features from the ProblemSpec's data sources and the
hypothesis, not from habit.

## FeatureRecipe

Rows are `(user history, candidate item)` pairs. Declare:

- `sources` — each one either a frozen split (`split: train` / `validation`) or a UC table
  that supplies item attributes (`role: item_attributes`). Never the `qa` split. An
  attribute source attaches at the item grain; there is no join specification because no
  other grain is supported.
- `feature_columns` — one entry per numeric feature, each using exactly one transform:
  `column`, `cast`, `bucket`, `hash`, `count` (`over: train` for catalog frequency,
  `over: history` for affinity between the candidate and the user's history), `ratio` of two
  earlier features, or `literal`. No SQL, no free-form expressions.
- `label` — `holdout_next_item` (positive is the user's last train interaction),
  `interaction_positive`, or `label_column` with an explicit column and positive values.
- `group_key` — the request grouping the rerankers train on.
- `candidate_generation` — the strategy plus `n`.
- `filters` — closed-form row filters only.

### The candidate-row budget

Feature building materialises one row per (evaluation unit × candidate). For the
generating strategies (`popularity_topn`, `history_attribute_expansion`)
`n × validation_split.row_count` is the size of the frame this job builds in memory;
`max_candidate_rows` in the context is the cap. Compute the product, state it in
`rationale`, and choose an `n` that stays under the cap — a plausible-looking `n = 200`
against a 44 000-row validation split is 8.8 million rows.

For `logged_impressions` the candidate set is closed at what the container showed:
rows = renders × tiles shown (3–5), and `n` is **not applied** — do not compute or
reason about `n × renders`, and do not pick `n` to fit a budget. Set `n` to the
contract's `k` and move on.

These columns exist on every row without being declared: `item_id`, `history_length`,
`item_train_count`, `item_train_rank`.

### What a `grouped_rerank` recipe can actually read

A render-grain split carries **only** `candidates`, `positions`, `labels`, `weights`,
`user_id`, `ts` and the cohort columns the contract materialized (typically
`client_name`). It carries **no per-candidate attributes**. Therefore:

- Every per-item feature (`content_type`, `series_id`, `duration_seconds`, `genres`, ...)
  must come from a `role: item_attributes` source — a UC table keyed by the contract's
  item column, such as `pnr_prod.silver.pr_item_feature`. Listing those names under a
  `split:` source does nothing; the job refuses a feature whose source resolves to no
  values rather than training on constants.
- Render-level features are the materialized cohort columns and nothing else. Read
  `client_name` with a `cast … to: code`; there is no other request context.
- Point-in-time item fields that change per render (`days_since_air_date`, `promo_value`)
  are **not available**: the split does not carry them and the snapshot's current value
  is banned. Leave recency out until materialization carries it.
- There is **no account history** in grouped mode. `count … over: history` counts within
  the render's own candidate list, not the account's past. A hypothesis that needs prior
  conversions per account is infeasible here; say so instead of describing semantics the
  runtime does not have.
- The contract's relevance weights **are** applied in training: as sample weights for the
  classifier families, as linear-gain integer grades for the reranker families. Do not
  re-describe the weighting in the recipe; it is the contract's.

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
