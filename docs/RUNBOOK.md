# AMT Runbook

Operating guide for running an experiment campaign with the Automated Model Trainer. For
what the system is and why it is shaped this way, see `SPEC.md` and `PLAN.md`; for a visual
map, open `campaign-map.html`.

There are two surfaces over the same graph. **`amt web` is the primary one**: a local Flask
app where you read each checkpoint's document in full, answer it, watch progress, and rewind
a gate you already approved. The CLI does everything the browser does and is what you use for
scripting, `--mock` runs and CI. Neither is a cache of the other — both read campaign state
through `amt.monitor` and write it through the same `Command(resume=...)`, so they cannot
disagree about a campaign.

For a worked example of a real campaign end to end, including the numbers to expect at each
gate, see `RUN_FYL_CAMPAIGN.md`.

## 1. One-time setup

### 1.1 Databricks identity

AMT normally runs as a service principal. Create (or reuse) one and grant it:

| Resource | Grant | Why |
|---|---|---|
| SQL warehouse (`DATABRICKS_WAREHOUSE_ID`) | `CAN USE` | Stage 1 read-only data probing |
| Source tables | `SELECT` | Probing + split materialization |
| `{catalog}.{schema}` | `USE SCHEMA`, `CREATE TABLE`, `MODIFY` | Writes `<campaign>_split_train/validation/qa` |
| `{catalog}.{schema}` volume `amt_artifacts` | `READ VOLUME`, `WRITE VOLUME` | Wheel + JSON payload uploads |
| Workspace folder `/Shared/amt` | `CAN MANAGE` | MLflow experiments |
| Cluster creation (or a policy allowing it) | create | Per-run job clusters, which is the default |
| UC model registry (optional) | `CREATE MODEL` on schema | `amt promote` on PASS |

Generate an OAuth secret for the principal (Workspace admin → Service principals → Secrets).

Running as yourself instead: set `DATABRICKS_TOKEN` to a PAT and it takes precedence over the
OAuth pair. Every job then runs under your own grants, which is the quickest way onto source
data a principal cannot reach yet. The PAT needs the `files` scope or Volume uploads fail with
`does not have required scopes: files`.

Compute defaults to a **per-run job cluster**, which is what makes cost attributable: tags are
a property of a job cluster, so each run carries `amt`, `amt_campaign_id`, `amt_role` and
`amt_hypothesis_id` into `system.billing.usage.custom_tags`. Job clusters are created in
single-user mode so they can install the AMT wheel from a Unity Catalog volume. Pinning an
existing all-purpose cluster with `AMT_EXISTING_CLUSTER_ID` still works and is faster to
iterate on, but `amt doctor` will warn and tell you exactly what you give up — see §6.

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
| `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` | one of | Service-principal OAuth M2M |
| `DATABRICKS_TOKEN` | one of | PAT (needs the `files` scope); wins over the OAuth pair |
| `DATABRICKS_WAREHOUSE_ID` | yes | SQL warehouse for Stage 1 probing |
| `AMT_LLM_API_KEY` | yes | Agents run locally against whichever LLM provider you configure |
| `AMT_LLM_BASE_URL` | some | Required for `huggingface` / `openai-compat` models; omit for hosted OpenAI and Anthropic |
| `AMT_CATALOG` / `AMT_SCHEMA` | yes | Where splits are materialized |
| `AMT_VOLUME_PATH` | yes | `/Volumes/{catalog}/{schema}/amt_artifacts` |
| `AMT_SPARK_VERSION` / `AMT_NODE_TYPE_ID` / `AMT_NUM_WORKERS` | yes | The per-run job cluster. Prefer an ML runtime so the booster libraries are already present |
| `AMT_EXISTING_CLUSTER_ID` | no | Pin an all-purpose cluster instead. Faster to iterate, but loses per-run cost attribution and bills at the All-Purpose rate |
| `AMT_MAX_CANDIDATE_ROWS` | no | Default 50M. The candidate-row budget `HC_FEATURE` prices a shape against |
| `AMT_MLFLOW_EXPERIMENT_ROOT` | no | Default `/Shared/amt`; one experiment per campaign underneath |
| `AMT_MLFLOW_EXPERIMENT_ID` | no | Pin every job to one existing experiment instead |
| `AMT_*_MODEL` | no | Per-agent model as `provider:model_id` (see §1.2.1) |
| `AMT_CHECKPOINT_DB` | no | Default `~/.amt/checkpoints.sqlite`; the campaign list reads it directly |
| `AMT_REPORTS_DIR` | no | Where `report.html` is written locally as well as to MLflow |

Then check everything a campaign needs *before* it spends a cluster minute:

```bash
set -a && . ./.env && set +a    # MLflow reads credentials from the process environment
uv run amt doctor --tables <the tables your campaign will read>
```

`doctor` verifies auth, the warehouse, a writable artifact volume, the job-cluster spec
against the node types this workspace actually offers, the MLflow experiment root, that every
source table is readable, and one real structured-output round-trip against your LLM. Only
failures exit non-zero; a `warn` is a finding you may accept, and it says what accepting it
costs. Almost every "job died twenty minutes in" has an answer here.

```bash
uv run amt config    # resolved settings, masked, with the source of each value
```

Read the **source** column. An `AMT_*` variable exported in your shell silently shadows
`.env`, and that column is the only thing that will tell you.

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
uv run amt run dryrun --mock     # drives every checkpoint inline
```

Or answer the gates one at a time, which is what a script would do:

```bash
uv run amt approve dryrun --mock     # HC2
uv run amt approve dryrun --mock     # HC3
uv run amt approve dryrun --mock     # HC_BACKLOG
uv run amt approve dryrun --mock     # HC_CHAMPION → QA → stops at HC4
uv run amt report  dryrun --mock
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

```bash
uv run amt web        # http://127.0.0.1:5000 — the campaign list, and `new campaign`
```

The campaign page is the monitor: the pending checkpoint at the top with its verbs, the
document of record in full underneath, then the hypothesis table, the experiment history and
a live progress log carrying each Databricks run URL. It refreshes every five seconds only
while a runner is live, so a page showing a checkpoint never reloads under your cursor. No
auth, by design — it binds to `127.0.0.1` and runs with your `.env`, exactly as the CLI does.

The rest of this section describes each checkpoint and gives the CLI equivalent of every
action, because the CLI is what you script.

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

AMT evaluates in one of **two modes**, and which one your question needs is the single most
important thing to get right in the problem statement. The contract's `evaluation_mode`
records the choice, and both modes run through the same harness, the same paired bootstrap
and the same freeze.

**`next_item`** — top-K recommendation over one `(user, item, timestamp)` interaction table.
A unit of evaluation is a user at a point in their history; candidates are generated; the
baseline is a model such as `popularity` or `sasrec`. Use it for "what should we show this
user next".

**`grouped_rerank`** — re-ranking what a surface actually showed. A unit is one *render*
of one container for one user at one time, identified by the contract's `group_key`.
Candidates are the items that render impressed (`logged_impressions`), relevance is the
conversion label on each tile, and **the baseline is `logged_order`: the incumbent ranking
itself**. `k` is the above-the-fold tile count. Use it for "does a new ranker beat the order
this rail is already shipping, on that rail's own metric, over a recent window" — which is
the question a container owner actually asks, and it is not next-item prediction.

Two things follow from choosing `grouped_rerank`. First, you are measuring **re-ordering, not
retrieval**: the candidate set is the incumbent's, so the campaign cannot tell you whether
better candidates exist. Second, position bias must be corrected or a ranker is rewarded for
copying the incumbent — set `relevance_weighting: pbm` and the materialisation estimates
per-client, per-slot examination propensities from the train split alone, turning each label
into a gain. Name in the problem statement the container, the page, the clients, the label
column and the tile count; those are human-owned facts and the agent will not infer them.

In both modes the harness computes `recall@k`, `ndcg@k`, `hitrate@k`, `mrr@k` and
`coverage@k`, plus `auc`, `pr_auc`, `logloss` and `ece` as **secondary metrics only** — the
primary metric has to be one that moves the surface. The contract maps the source's real
column names and carries its own SQL filters, so a raw event table usually needs no
preparation; name it directly. Build a view upstream only when the grain has to be derived.

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

1. Temporal boundaries — `split_policy.train_end` / `validation_end` / `qa_end` don't
   overlap and the QA window is recent enough to matter. These literals are what the job
   cuts on, so a boundary that is only described in prose is a bug, not a nuance.
2. The executable dataset block — `dataset.source_table` is the view you prepared,
   `dataset.columns` are its real column names, and `dataset.filters` /
   `min_interactions_per_user` are the eligibility you expect.
3. Population eligibility and exclusions — bots, internal users, min-interaction thresholds.
4. Split references — `delta_version`, `row_count`, `membership_hash` are all present.
5. `evaluation_mode`, and if it is `grouped_rerank`: `group_key`, `position_column`,
   `label_column`, `label_positive_values`, `relevance_weighting` and the propensity floor.
   A wrong `group_key` here silently produces splits that look plausible and mean nothing.
6. Metric definitions and `harness_version` — matches what you expect to report. The primary
   metric cannot be one of the pointwise diagnostics (`auc`, `pr_auc`, `logloss`, `ece`);
   the contract's own validation refuses that.
7. `product_metric_source` — the external daily series QA will correlate against. Pick one
   built from a *different* source than your label, or the proxy check proves nothing.
8. Acceptance criteria — the bar QA will hold the champion to, including
   `min_proxy_correlation` and the cohort-regression guardrail. Set it now; it cannot move
   later.

```bash
uv run amt approve <campaign>
```

Approval stamps `contract_hash`.

If you reject at HC3 the splits are re-materialized. If you later realize the contract is wrong
(after training has started), do not edit anything — start a new campaign. That is by design.

### 2.3a HC_BACKLOG — approve the hypothesis backlog

After HC3 the baseline runs and the Planner writes the campaign's whole backlog at once, then
stops. This is the only point at which you see every hypothesis before compute is spent on any
of them, and the backlog decides how the budget is spent:

In the browser this is a table with `keep` and `drop` boxes. From the CLI:

```bash
uv run amt show    <campaign>                     # the full backlog, not the panel preview
uv run amt approve <campaign> --keep H1,H3        # order sets run order; everything else is dropped
uv run amt approve <campaign> --drop H2           # or strike individually
uv run amt reject  <campaign> -m "all four ignore that titles turn over daily"
```

Prefer `drop` over `reject`. Dropping marks that hypothesis `abandoned` and keeps the rest
intact; rejecting costs a full round trip through the Planner to redraft the whole backlog.

Read each hypothesis against what this campaign can actually run:

1. Testable here — the experiment design only needs the frozen splits, the contract's metrics,
   and transforms the `FeatureRecipe` can express. A hypothesis about trailing-window demand or
   item-pair affinity has nothing to run on today; strike it rather than discovering that at
   HC_FEATURE.
2. Falsifiable — `falsification_criteria` names a result that would kill it, not a hedge.
3. Worth the compute — `estimated_cost_minutes` against `max_compute_minutes`, priority order
   matching what you'd want tried first if the budget runs out early.
4. Different from each other — variations of one idea waste hypotheses.

Hypotheses asking for something the runtime cannot express arrive already marked
`infeasible`, with the reason — the Planner is given the literal transform vocabulary,
candidate strategies and registry families as context, so this is caught before it costs a
job. Approval is once per campaign: the backlog is written once and
`planner_select_hypothesis` selects from it without ever generating more.

### 2.3b HC_FEATURE — approve every feature/fit shape, once

The gradient-boosting families (`xgboost_classifier`, `xgboost_reranker`,
`lightgbm_classifier`, `lightgbm_reranker`) have no fixed input shape, so each one needs a
`FeatureRecipe` (sources, feature columns, label, candidate generation, filters) and a
`FitSpec`. This gate shows you **every approved tabular hypothesis's shape at the same time**,
before any job is submitted, and it interrupts **once per campaign** rather than once per
hypothesis. If no surviving hypothesis is tabular, it does not interrupt at all.

```bash
uv run amt status  <campaign>                 # awaiting HC_FEATURE
uv run amt show    <campaign>                 # every shape, in full
uv run amt approve <campaign> --drop H3       # dropping a shape abandons its hypothesis
uv run amt reject  <campaign> -m "H1 reads the labelled event; build it from history only"
```

Read the table in this order:

1. **Leakage.** Every feature must be computable at serving time from what is known before
   the decision. This is the one thing nothing downstream can catch for you.
2. **Estimated candidate rows.** The row prices the shape: a plausible-looking `n` against a
   large validation split is tens of millions of rows of feature building, and nothing else
   in the payload says so. Under `logged_impressions` the candidate set is closed at what the
   surface showed, so `n` is never applied and the gate says so instead of pricing it.
3. **Duplicate shapes.** A shape identical to another hypothesis's is annotated as such.
   Two hypotheses with the same `feature_fit_hash` are one experiment run twice.
4. **The six `FitSpec` fields.** Objective, eval metric, rounds, early stopping, negative
   sampling, seed. Not `booster_params` — those are the Trainer's to change without asking.

Approval stamps `feature_fit_hash` over the recipe plus the structural part of the `FitSpec`.
What that buys you:

- later iterations that only retune hyperparameters run without asking again;
- `execute_training` refuses any config whose `feature_fit_hash` is not the approved one;
- a Trainer that asks for a structural change is **clamped, not bounced** — the approved
  shape is kept, only its `booster_params` are taken, and the rejected ask is recorded and
  shown in `amt status`. The old behaviour sent the campaign back through this gate on the
  agent's initiative, which is how a review turned into a loop;
- `popularity`, `sasrec` and `logged_order` skip this checkpoint entirely.

If an approved shape turns out to be unexecutable — a recipe reading columns the split does
not carry — do not reject something downstream. Rewind (§3).

### 2.4 The autonomous band, and HC_CHAMPION

Between HC_FEATURE and QA the Trainer loop runs unattended **within one hypothesis**:
propose a config, submit the job, interpret the result, iterate until it succeeds, plateaus,
falsifies itself, repeats a config it already ran (`no_op`), or runs out of budget. Then it
picks a champion for that hypothesis and **stops at `HC_CHAMPION`**.

This gate is per hypothesis, not per iteration, and it is on by default. What it shows: this
hypothesis's champion against the baseline and against every prior champion on the primary
metric and the contract's secondaries, cohort deltas, the budget left, and the Planner's
recommended next move. The delta column against the baseline is the rightmost one — on the
campaign page the table scrolls horizontally with the metric names pinned.

```bash
uv run amt approve <campaign>                    # continue to the next hypothesis
uv run amt approve <campaign> --drop H4,H5       # abandon hypotheses not worth the budget
```

In the browser there is also a **freeze** button: stop exploring, take the best champion so
far and go to QA. Use it when a hypothesis has clearly answered the question and the
remaining ones will not change the decision. `--no-gate-champions` disables the gate
entirely, which is for rehearsal campaigns, not for real ones.

While jobs run, watch the progress log on the campaign page — each Databricks run URL lands
there. From a second terminal:

```bash
uv run amt status <campaign>
```

Key fields: `stage`, `current_hypothesis_id`, `budget` (iterations used / max), length of
`experiment_history`, `hypothesis_champions`, `frozen_champion`, and two worth reading
closely: `gate_cycles`, flagged above two, which means you and the agent are disagreeing
rather than converging; and any clamped structural ask, which means the Trainer wanted a
shape you did not approve.

In Databricks, open MLflow experiment `/Shared/amt/<campaign>`. Useful filters:

```
tags."amt.campaign_id" = '<campaign>' and tags."amt.role" = 'train'
tags."amt.role" = 'qa1'
```

Every run carries `amt.hypothesis_id`, `amt.contract_hash`, `amt.config_hash`,
`amt.harness_version`. Each run's `result.json` artifact is what the orchestrator saw.

Typical wall-clock: baseline 5–10 min, each training iteration 10–40 min depending on cluster
and model family, QA phases 5–15 min each.

### 2.5 QA, the report, and HC4

QA runs in two phases on the frozen champion. Phase 0 is integrity: the artifact loads
through MLflow pyfunc, its schema matches, and it reproduces its own logged validation
numbers. Phase 1 is the holdout — the only read of the QA split in the whole campaign.

Phase 1 does one thing **before** it looks at the candidate. It computes the *incumbent's*
offline primary metric on daily slices of the QA split and correlates it (Spearman ρ) with
the external product series the contract named. If ρ is below `min_proxy_correlation`, or
there are fewer than seven overlapping days, the campaign is `SPEC_REVIEW` with that reason
and no candidate number is displayed. A metric that does not track the product number for the
ranker already in production cannot certify a new one.

Then read the report, which is what a product owner decides on:

```bash
uv run amt report <campaign>          # conclusion + MLflow references
```

The campaign page shows the headline rows inline and links the full HTML report, which is
also logged to the final MLflow run as `report.html`. Its order is deliberate: proxy sanity
first, then the effect in **product units** (top-slot conversions per 1,000 renders,
candidate against incumbent, with an interval), then the offline delta and its CI, then the
cohort table with regressions called out, then what an online test would cost — MDE at 80 %
power, required renders, days at this surface's measured traffic. Every number comes from
`Phase1Result`; the Planner writes the prose around them and never the numbers.

Read the conclusion as a paper, not a scoreboard:

| QA decision | What it means | Your move |
|---|---|---|
| `PASS` | Frozen champion beat baseline on the untouched QA split within predefined criteria | `amt promote` → registers UC model `{catalog}.{schema}.amt_<campaign>` with alias `candidate`; plan an online A/B or shadow test using the report's sizing |
| `FAIL_HYPOTHESIS` | Won on validation, did not generalize | Record it. Do **not** rerun the same hypothesis with tweaks against this QA split. New hypothesis → new campaign |
| `SPEC_REVIEW` | The labels / population / metrics are wrong, or the proxy-sanity gate failed | Fix the spec or contract in a **new** campaign; this one is closed |
| `INVALID_EXPERIMENT` (terminal) | Phase 0 failed the capped number of times on a technical defect | Fix the artifact/inference bug in code, then start a new campaign |

The `recommendation` is clamped to the measured numbers: a champion that lost to the
incumbent on validation reads `hypothesis_rejected` whatever the narrative says, and whatever
QA did or did not manage to run. The prose may explain a loss; it cannot promote one.

```bash
uv run amt promote <campaign>                          # PASS only
uv run amt promote <campaign> --model-name main.recsys.homepage_ranker
```

A negative result is a finished campaign. The MLflow final run holds the report either way.

## 3. Interruptions, rewinding, and recovery

An approved gate is final by design — the Trainer is clamped to the shape a human signed, and
nothing downstream can reopen it. When the approved document itself turns out to be
*unexecutable*, the honest remedy is to go back and approve a different one rather than let
the run proceed on a shape that trains on nothing. That is a rewind, and it is available on
the campaign page under **go back**, or:

```bash
uv run amt rewind <campaign> --to feature -m "the recipe reads columns the split does not carry"
uv run amt rewind <campaign> --to backlog -m "every hypothesis assumes item identity carries over"
uv run amt rewind <campaign> --to qa      -m "the pyfunc signature dropped the candidates"
```

| Target | Re-enters at | Costs you |
|---|---|---|
| `backlog` | after the baseline, into `planner_hypotheses` | The backlog is redrafted. Mid-flight hypotheses return to `pending` and their budget slot comes back; `abandoned` ones stay abandoned, because a human struck them deliberately. |
| `feature` | after the approved backlog, into `planner_feature_fit` | The shapes are redrafted; the backlog stands. |
| `qa` | after the freeze, into `qa_phase0` | Only QA's own results are cleared. **The model is untouched** — this is for an evaluation-plumbing fix, and nothing about the artifact changes. |

Nothing that ran is erased: `experiment_history` and every MLflow run stay. Rewinding into
training is refused once a champion is frozen — `--to qa`, or a new campaign, are the only
honest paths then. The message is not paperwork; it is the feedback the Planner redrafts from.

| Situation | Action |
|---|---|
| Laptop closed / CLI killed mid-run | `uv run amt resume <campaign>`, or **pick it up** on the campaign page. State is checkpointed after every node; the in-flight Databricks job is re-polled or re-submitted. |
| Waiting on you but you forgot which checkpoint | `uv run amt status <campaign>` prints `awaiting HC*`. |
| Databricks job failed (cluster, quota, OOM) | Fix the cause; `amt resume`. The trainer sees `status=failed` in the result and adapts (smaller batch, fewer layers). Three consecutive failures count against the iteration budget. |
| Auth error (`401`, `invalid_client`) | Rotate the SP secret, update `.env`, `amt resume`. |
| `more than one authorization method configured: oauth and pat` | Your `.env` holds both a PAT and the service-principal pair. AMT prefers the PAT and now pins that choice in `DATABRICKS_AUTH_TYPE` for libraries that build their own client; if you see this from a tool outside AMT, export `DATABRICKS_AUTH_TYPE=pat` or drop one identity. |
| Wheel upload fails | Check `AMT_VOLUME_PATH` exists and SP has `WRITE VOLUME`. |
| A job dies on an argument the code should know (`amt-job: error: unrecognized arguments`) | Only possible on a pinned all-purpose cluster, where wheel installs are cluster-scoped and accumulate: `uv run python scripts/clear_cluster_wheels.py $AMT_EXISTING_CLUSTER_ID`, then `amt resume`. Job clusters are per-run and cannot hit this. |
| An approved recipe cannot execute | `amt rewind <campaign> --to feature -m "<why>"`. Rejecting something downstream will not fix it. |
| The web page says a campaign is already running | A second resume while one is live is refused rather than raced (HTTP 409). Wait for the runner, or check the progress log for what it is doing. |
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
| `final` | — | `conclusion.md`, `report.html`, `problem_spec.json`, `evaluation_contract.json`, QA metrics |

Metric names: `recall@k`, `ndcg@k`, `hitrate@k`, `mrr@k`, `coverage@k`, with `k` taken from
the contract's own metric list (`{5, 10, 20}` only when the contract does not say otherwise —
a contract asking for `ndcg@4` is scored at 4). `auc`, `pr_auc`, `logloss` and `ece` appear as
secondaries. Cohort keys look like `history_length=cold_start`, `client_name=ROKU`.

Note the prefix. The harness emits bare metric names; the training job logs them under
`val/`, and both spellings reach campaign state. Anything comparing a contract's
`primary_metric` against a stored metrics dict goes through
`amt.schemas.run_result.metric_value`, which accepts either — a raw `.get` looks like it works
and silently returns nothing for half its inputs.

## 5. Things the system will refuse to do

These are enforced in code (routing, `SplitGuard`, hash assertions). If you hit one, the fix is
a new campaign, not a workaround.

- Run a training or baseline job with `qa` in its allowed splits.
- Train a booster family on a feature/fit shape you did not approve, or on one that changed
  after you approved it. A structural ask from the Trainer is clamped, not honoured.
- Generate new hypotheses after the backlog is approved, or select one that is `abandoned` or
  `infeasible`.
- Report a candidate's QA numbers when the proxy-sanity gate failed.
- Recommend promotion for a champion that lost to the incumbent on validation.
- Run QA before exactly one `FrozenChampion` exists.
- Re-enter the trainer after freeze, including by rewinding.
- Continue after `contract_hash` changes.
- Exceed `--max-iterations` or `--max-compute-minutes`.
- Retry Phase 0 past its cap.

## 6. Cost: attribution and guardrails

Every job cluster AMT submits carries four tags: `amt=true`, `amt_campaign_id`, `amt_role`
(`materialize` / `baseline` / `train` / `qa0` / `qa1` / `final`) and `amt_hypothesis_id`. They
land in `system.billing.usage.custom_tags`, so "what did this campaign cost" and "which
hypothesis burned the budget" are queries rather than estimates. See
`.cursor/rules/databricks-cost.md` for the query and the dashboard.

**What pinning a cluster costs.** Tags are a property of a job cluster. A pinned all-purpose
cluster (`AMT_EXISTING_CLUSTER_ID`) can carry a static `amt=true` cluster tag — that part
works, and its cost stays attributable to AMT — but `amt_campaign_id`, `amt_role` and
`amt_hypothesis_id` vary per run and have nowhere to live, so per-campaign and per-hypothesis
attribution is gone. It also bills at the All-Purpose rate: measured on this workspace,
$0.650/DBU against $0.200 for Jobs Compute — **3.25×**, not the "about twice" rule of thumb.
Set
against that: a warm cluster skips the ~4.5 minutes a job cluster spends starting, which over
a twelve-iteration campaign is about an hour. `amt doctor` reads the cluster and tells you
which of these apply, including two that are easy to miss — a runtime without `-ml-` in its
name pip-installs the booster libraries on every run and those installs accumulate
cluster-scoped, and a single-node cluster is fine for the boosters but slow for
materialisation, which pushes the whole source table through Spark.

Pin one while you are learning the gates. Leave it unset for a campaign whose cost you want
to attribute, and because materialisation wants the workers.

Other guardrails:

- Start with `--max-hypotheses 1 --max-iterations 4`; widen once the first campaign completes.
- The Problem Specification Agent is capped at `AMT_MAX_PROBE_STEPS` (default 8) SQL probes.
- `AMT_MAX_CANDIDATE_ROWS` (default 50M) is the budget `HC_FEATURE` prices each shape against.
- Per-agent models can be downgraded — or pointed at a cheaper provider — via `AMT_*_MODEL`
  for rehearsal campaigns.

## 7. Model families

| Family | Kind | Input shape |
|---|---|---|
| `popularity` | baseline, trailing frequency | fixed in code |
| `logged_order` | baseline for `grouped_rerank`: returns the candidates in the order the surface showed them, i.e. the incumbent | fixed in code |
| `sasrec` | sequence model | fixed in code |
| `xgboost_classifier` | pointwise `binary:logistic` | declared per hypothesis by an approved `FeatureRecipe` |
| `xgboost_reranker` | pairwise `rank:pairwise` | same, grouped by `group_key` |
| `lightgbm_classifier` | pointwise `binary` | same |
| `lightgbm_reranker` | `lambdarank` | same, grouped by `group_key` |

The four booster families share one runtime (`src/amt/jobs/models/tabular_common.py`) that
executes the recipe: it builds `(history, candidate)` rows from the frozen splits plus the
item-attribute tables the recipe names, fits the booster, and at serving time scores
candidates through the same feature builder. Under `logged_impressions` the candidates come
off the render row rather than being generated, so the recipe's `column` transform reaches
impression-row fields — position, item metadata, user context — directly, and most
first-campaign features need no new transform kind.

Each family declares its own dependency (`requires_pip` in the registry) and the executor
passes it as a per-run library install, which is what makes job clusters viable. Locally they
need `xgboost` / `lightgbm` from the `jobs` extra (on macOS also `brew install libomp`); the
unit tests skip when the native library cannot load.

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
amt web                          # the primary surface: list, campaign pages, reports
amt doctor [--tables a,b]        # everything a campaign needs, before it spends a minute
amt config                       # resolved settings, masked, with the source of each value

amt start "<problem>" -c <id> [--max-iterations N] [--max-hypotheses N] [--max-compute-minutes M] [--mock]
amt run     <id>                 # drive every checkpoint inline from one terminal
amt list                         # every campaign in the local checkpoint database
amt status  <id>                 # compact state + which checkpoint is pending
amt show    <id> [--key K]       # the checkpoint document in full, not the preview
amt approve <id> [--keep H1,H3] [--drop H2]   # HC2 / HC3 / HC_BACKLOG / HC_FEATURE / HC_CHAMPION / HC4
amt reject  <id> -m "<feedback>" # same checkpoints; re-runs the stage with the feedback
amt resume  <id>                 # continue after crash / kill
amt rewind  <id> --to backlog|feature|qa -m "<why>"   # reopen a gate you approved
amt report  <id>                 # Scientific Conclusion + MLflow refs (HC4)
amt promote <id> [--model-name <uc.name>]   # HC4 decision; registers on PASS
amt demo seed [--users N --items N]
```

Add `--mock` to any of these for an offline dry run with no Databricks and no LLM.
