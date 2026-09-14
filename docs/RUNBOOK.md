# AMT Runbook

Operating guide for running an experiment campaign with the Automated Model Trainer. For
what the system is and why it is shaped this way, see `SPEC.md` and `PLAN.md`; for a visual
map, open `campaign-map.html`.

## 1. One-time setup

### 1.1 Databricks service principal

AMT never runs as a user. Create (or reuse) a service principal and grant it:

| Resource | Grant | Why |
|---|---|---|
| SQL warehouse (`DATABRICKS_WAREHOUSE_ID`) | `CAN USE` | Stage 1 read-only data probing |
| Source tables | `SELECT` | Probing + split materialization |
| `{catalog}.{schema}` | `USE SCHEMA`, `CREATE TABLE`, `MODIFY` | Writes `split_train/validation/qa` |
| `{catalog}.{schema}` volume `amt_artifacts` | `READ VOLUME`, `WRITE VOLUME` | Wheel + JSON payload uploads |
| Workspace folder `/Shared/amt` | `CAN MANAGE` | MLflow experiments |
| Cluster policy or existing cluster | create / attach | Job compute |
| UC model registry (optional) | `CREATE MODEL` on schema | `amt promote` on PASS |

Generate an OAuth secret for the principal (Workspace admin → Service principals → Secrets).

Optional hardening: a second principal for QA jobs with `SELECT` on `split_qa` only. Not wired
by default; see `PLAN.md` §9.

### 1.2 Local environment

```bash
git clone <repo> && cd automated-model-trainer
uv sync --all-extras
cp .env.example .env      # fill in values below
```

| Variable | Required | Notes |
|---|---|---|
| `DATABRICKS_HOST` | yes | `https://adb-…azuredatabricks.net` (no trailing slash) |
| `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` | yes | Service-principal OAuth M2M |
| `DATABRICKS_WAREHOUSE_ID` | yes | SQL warehouse for Stage 1 probing |
| `AMT_LLM_API_KEY` | yes | Agents run locally against whichever LLM provider you configure |
| `AMT_LLM_BASE_URL` | some | Required for `huggingface` / `openai-compat` models; omit for hosted OpenAI and Anthropic |
| `AMT_CATALOG` / `AMT_SCHEMA` | yes | Where splits are materialized |
| `AMT_VOLUME_PATH` | yes | `/Volumes/{catalog}/{schema}/amt_artifacts` |
| `AMT_EXISTING_CLUSTER_ID` | one of | Reuse a running cluster (fastest iteration) |
| `AMT_SPARK_VERSION` / `AMT_NODE_TYPE_ID` / `AMT_NUM_WORKERS` | one of | Or a per-run job cluster |
| `AMT_MLFLOW_EXPERIMENT_ROOT` | no | Default `/Shared/amt` |
| `AMT_*_MODEL` | no | Per-agent model as `provider:model_id` (see §1.2.1) |
| `AMT_CHECKPOINT_DB` | no | Default `~/.amt/checkpoints.sqlite` |

Smoke-check auth before anything else:

```bash
uv run python -c "from amt.databricks.client import get_client; print(get_client().current_user.me().user_name)"
```

#### 1.2.1 Choosing an LLM provider

AMT is provider-agnostic: the agents only need a chat model that can return structured
output. Name each agent's model as `provider:model_id` and the orchestrator builds the right
client. Nothing else in a campaign changes when you switch providers.

| Provider prefix | Reaches | `AMT_LLM_BASE_URL` |
|---|---|---|
| `openai` | Hosted OpenAI (or an OpenAI-shaped proxy if you set a base URL) | optional |
| `anthropic` | Hosted Anthropic | ignored |
| `huggingface` | A Hugging Face Inference Endpoint | required |
| `openai-compat` | Any other OpenAI-compatible server: vLLM, TGI, Databricks model serving | required |

```bash
AMT_LLM_API_KEY=<your provider key>
AMT_PLANNER_MODEL=openai:gpt-4.1
AMT_PROBLEM_SPEC_MODEL=anthropic:claude-sonnet-5      # mixing providers is fine

# Self-hosted or Hugging Face instead:
AMT_LLM_BASE_URL=https://<endpoint>/v1
AMT_QA_MODEL=huggingface:meta-llama/Llama-3.1-70B-Instruct
```

Unprefixed names still resolve (`claude-*` to Anthropic, `gpt-*` to OpenAI, `org/name` to
Hugging Face), but prefer the explicit form. Whichever provider you pick must support
structured output reliably — the agents fail the stage rather than accept a malformed
contract.

### 1.3 Dry run without Databricks or an LLM

```bash
uv run amt start "Improve homepage row ranking" -c dryrun --mock
uv run amt approve dryrun --mock     # HC2
uv run amt approve dryrun --mock     # HC3 → autonomous → stops at HC4
uv run amt report dryrun --mock
uv run amt promote dryrun --mock
```

If this completes, the orchestrator is healthy. Everything after this point is configuration
and data.

### 1.4 Optional: synthetic data

```bash
uv run amt demo seed --users 5000 --items 2000
```

Writes `{catalog}.{schema}.demo_interactions` (`user_id, item_id, ts, item_genre`). Use it for
a first real campaign end-to-end before pointing at production tables.

## 2. Running a campaign

A campaign is one frozen question. Its id is the LangGraph thread id, so pick something you
can find later in MLflow: `hp-rows-seq-2026q3`, not `test1`.

### 2.1 Start — HC1

```bash
uv run amt start "<problem>" -c <campaign> \
  --max-iterations 6 --max-hypotheses 2 --max-compute-minutes 240
```

Write the problem the way you would brief a new ML engineer. Include:

- the product surface and the decision the model makes there;
- the source tables you believe are relevant (fully qualified);
- the product metric you actually care about;
- the current baseline, if one exists;
- constraints (latency, cold-start, fairness, catalog rules).

The Problem Specification Agent probes the tables you name (schema, volume, freshness, label
availability, leakage) via read-only SQL, drafts a `ProblemSpec`, and the CLI stops at HC2.

The budget flags are hard stops enforced by routing, not suggestions to the agent.

### 2.2 HC2 — approve the ProblemSpec

The CLI prints the spec as JSON. Check, in this order:

1. Decision and target — is this the prediction the product needs?
2. Data sources — right tables, right grain, right join keys, right timestamp column?
3. Label construction — does it match how the product defines success, without leakage?
4. Offline metric — is the primary metric one that plausibly moves the product metric?
5. Baseline — is it the thing you'd actually have to beat?

```bash
uv run amt approve <campaign>
uv run amt reject <campaign> -m "Label should be completion >= 70%, not any play. Exclude autoplay events."
```

Reject as many times as needed; each rejection re-runs Stage 1 with your feedback. Once
approved, the spec is frozen for the campaign.

### 2.3 HC3 — approve the EvaluationContract

After HC2 the system drafts the contract and materializes the three splits as Delta tables,
then stops. Check:

1. Temporal boundaries — train / validation / QA windows don't overlap and the QA window is
   recent enough to matter.
2. Population eligibility and exclusions — bots, internal users, min-interaction thresholds.
3. Split references — `delta_version`, `row_count`, `membership_hash` are all present.
4. Metric definitions and `harness_version` — matches what you expect to report.
5. Acceptance criteria — the bar QA will hold the champion to. Set it now; it cannot move
   later.

```bash
uv run amt approve <campaign>
```

Approval stamps `contract_hash`. From here to HC4 the campaign is autonomous.

If you reject at HC3 the splits are re-materialized. If you later realize the contract is wrong
(after training has started), do not edit anything — start a new campaign. That is by design.

### 2.3b HC_FEATURE — approve the feature recipe (booster families only)

If the Planner picks one of the gradient-boosting families (`xgboost_classifier`,
`xgboost_reranker`, `lightgbm_classifier`, `lightgbm_reranker`) the campaign stops once more
before any job is submitted. Those families have no fixed input shape, so you approve the
`FeatureRecipe` (sources, joins, feature columns, label, candidate generation, filters) and the
`FitSpec` that the Trainer will work inside:

```bash
uv run amt status <campaign>     # awaiting HC_FEATURE
uv run amt approve <campaign>
uv run amt reject  <campaign> -m "feature X reads the labelled event; drop it"
```

Read it for leakage first: every feature must be computable at serving time from the user's
history plus the candidate item. Then check the label strategy and the candidate set — those
decide what the model is actually learning to rank.

Approval stamps `feature_fit_hash`, which is the hash of the recipe plus the structural part of
the `FitSpec` (`booster_params` excluded). Consequences:

- later iterations that only retune hyperparameters run without asking you again;
- any change to features, label, candidates or structural fit fields returns here for
  re-approval and is visible as a new `feature_fit_hash`;
- `execute_training` refuses a config whose `feature_fit_hash` is not the approved one;
- `popularity` and `sasrec` skip this checkpoint entirely.

### 2.4 Autonomous phase — what to watch

The CLI blocks while jobs run. In a second terminal:

```bash
uv run amt status <campaign>
```

Key fields: `stage`, `current_hypothesis_id`, `budget` (iterations used / max), length of
`experiment_history`, `hypothesis_champions`, `frozen_champion`.

In Databricks, open MLflow experiment `/Shared/amt/<campaign>`. Useful filters:

```
tags."amt.campaign_id" = '<campaign>' and tags."amt.role" = 'train'
tags."amt.role" = 'qa1'
```

Every run carries `amt.hypothesis_id`, `amt.contract_hash`, `amt.config_hash`,
`amt.harness_version`. Each run's `result.json` artifact is what the orchestrator saw.

Typical wall-clock: baseline 5–10 min, each training iteration 10–40 min depending on cluster
and model family, QA phases 5–15 min each.

### 2.5 HC4 — read the conclusion and decide

```bash
uv run amt report <campaign>
```

Prints the Scientific Conclusion and the QA decision. Read it as a paper, not a scoreboard:

| QA decision | What it means | Your move |
|---|---|---|
| `PASS` | Frozen champion beat baseline on the untouched QA split within predefined criteria | `amt promote` → registers UC model `{catalog}.{schema}.amt_<campaign>` with alias `candidate`; plan an online A/B or shadow test |
| `FAIL_HYPOTHESIS` | Won on validation, did not generalize | Record it. Do **not** rerun the same hypothesis with tweaks against this QA split. New hypothesis → new campaign |
| `SPEC_REVIEW` | QA surfaced evidence the labels / population / metrics are wrong | Fix the spec or contract in a **new** campaign; this one is closed |
| `INVALID_EXPERIMENT` (terminal) | Phase 0 failed three times on a technical defect | Fix the artifact/inference bug in code, then start a new campaign |

```bash
uv run amt promote <campaign>                          # PASS only
uv run amt promote <campaign> --model-name main.recsys.homepage_ranker
```

A negative result is a finished campaign. The MLflow final run holds the report either way.

## 3. Interruptions and recovery

| Situation | Action |
|---|---|
| Laptop closed / CLI killed mid-run | `uv run amt resume <campaign>`. State is checkpointed after every node; the in-flight Databricks job is re-polled or re-submitted. |
| Waiting on you but you forgot which checkpoint | `uv run amt status <campaign>` prints `awaiting HC*`. |
| Databricks job failed (cluster, quota, OOM) | Fix the cause; `amt resume`. The trainer sees `status=failed` in the result and adapts (smaller batch, fewer layers). Three consecutive failures count against the iteration budget. |
| Auth error (`401`, `invalid_client`) | Rotate the SP secret, update `.env`, `amt resume`. |
| Wheel upload fails | Check `AMT_VOLUME_PATH` exists and SP has `WRITE VOLUME`. |
| Phase 0 `INVALID_EXPERIMENT` once or twice | Normal — usually a serialization or schema-signature issue. The system fixes plumbing and retries automatically. Only three strikes is terminal. |
| You want to change the contract | You can't. `amt start` a new campaign. |
| Checkpoint DB corrupted | Delete `~/.amt/checkpoints.sqlite`; the campaign is lost locally but every run remains in MLflow. |

## 4. Interpreting MLflow

| Tag `amt.role` | Split(s) read | Look at |
|---|---|---|
| `baseline` | train, validation | `val/*` — your reference numbers |
| `train` | train, validation | `val/<primary>`, `train/*`, `val/cohort/*`, per-epoch curves; `config.yaml` |
| `qa0` | validation subset | `checks` in `result.json`; `reproduction_deltas` should be ~0 |
| `qa1` | qa | `paired_deltas.<metric>.{mean,ci_low,ci_high,frac_improved}` — the CI excluding 0 is the claim |
| `final` | — | `conclusion.md`, `problem_spec.json`, `evaluation_contract.json`, QA metrics |

Metric names: `recall@k`, `ndcg@k`, `hitrate@k`, `mrr@k`, `coverage@k` with k ∈ {5, 10, 20}
by default. Cohort keys look like `history_length=cold_start`, `recency=recency_0_7d`.

## 5. Things the system will refuse to do

These are enforced in code (routing, `SplitGuard`, hash assertions). If you hit one, the fix is
a new campaign, not a workaround.

- Run a training or baseline job with `qa` in its allowed splits.
- Train a booster family on a feature/fit shape you did not approve, or on one that changed
  after you approved it.
- Run QA before exactly one `FrozenChampion` exists.
- Re-enter the trainer after freeze.
- Continue after `contract_hash` changes.
- Exceed `--max-iterations` or `--max-compute-minutes`.
- Retry Phase 0 more than three times.

## 6. Cost and time guardrails

- Start with `--max-hypotheses 1 --max-iterations 4` and an existing small cluster; widen once
  the first campaign completes.
- Prefer `AMT_EXISTING_CLUSTER_ID` during development — job-cluster spin-up adds ~5 min per run.
- The Problem Specification Agent is capped at `AMT_MAX_PROBE_STEPS` (default 8) SQL probes.
- Per-agent models can be downgraded — or pointed at a cheaper provider — via `AMT_*_MODEL`
  for rehearsal campaigns.

## 7. Model families

| Family | Kind | Input shape |
|---|---|---|
| `popularity` | baseline, trailing frequency | fixed in code |
| `sasrec` | sequence model | fixed in code |
| `xgboost_classifier` | pointwise `binary:logistic` | declared per hypothesis by an approved `FeatureRecipe` |
| `xgboost_reranker` | pairwise `rank:pairwise` | same, grouped by `group_key` |
| `lightgbm_classifier` | pointwise `binary` | same |
| `lightgbm_reranker` | `lambdarank` | same, grouped by `group_key` |

The four booster families share one runtime (`src/amt/jobs/models/tabular_common.py`) that
executes the recipe: it builds `(history, candidate)` rows from the frozen splits plus the
item-attribute tables the recipe names, fits the booster, and at serving time generates
candidates and scores them through the same feature builder. They need `xgboost` / `lightgbm`
from the `jobs` extra (on macOS also `brew install libomp`); the unit tests skip when the
native library cannot load.

### Adding a model family

1. Implement `fit / recommend / pyfunc wrapper` in `src/amt/jobs/models/<family>.py`
   following `sasrec.py` — or, for a recipe-driven family, a thin module over
   `tabular_common.fit_family` following `lightgbm_reranker.py`.
2. Register it in `src/amt/jobs/models/registry.py` (and in `RECIPE_FAMILIES` plus
   `amt.schemas.feature_recipe.TABULAR_FAMILIES` if it takes a recipe).
3. Add it to the Planner's permitted changes in the campaign's ExperimentPlan (the agent reads
   the registry to know what exists).
4. Unit-test it under `tests/unit/test_models.py`; no Spark or Databricks needed.
5. Bump nothing in the harness — evaluation is model-agnostic by design.

## 8. Quick reference

```bash
amt start "<problem>" -c <id> [--max-iterations N] [--max-hypotheses N] [--max-compute-minutes M] [--mock]
amt approve <id>                 # HC2 / HC3 / HC_FEATURE
amt reject  <id> -m "<feedback>" # HC2 / HC3 / HC_FEATURE, re-runs the stage
amt status  <id>                 # compact state + which checkpoint is pending
amt resume  <id>                 # continue after crash / kill
amt report  <id>                 # Scientific Conclusion + MLflow refs (HC4)
amt promote <id> [--model-name <uc.name>]   # HC4 decision; registers on PASS
amt demo seed [--users N --items N]
```
