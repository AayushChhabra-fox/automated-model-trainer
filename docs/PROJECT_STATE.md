# Project State — handoff

**Written:** 2026-09-15. **Revised:** 2026-09-17 after the gate redesign and the proving
campaign. **Tree:** `02e4e36`, committed and clean.

This document describes what is actually in the repository and what has actually been run.
Where something is documented but not implemented, it says so.

**Status in one paragraph.** The system now runs a campaign end to end against a real
Databricks workspace, through every stage: SQL probes into a ProblemSpec, a frozen
EvaluationContract, three materialised Delta splits, an incumbent baseline, a human-approved
hypothesis backlog, one batched review of every feature/fit shape, an autonomous Trainer loop
per hypothesis, a human review of each hypothesis champion, a single champion freeze, QA
Phase 0 integrity, QA Phase 1 behind a proxy-sanity gate, a scientific conclusion with an
experimentation report, and HC4. `lfy-rerank-001` did all of that on the Latest For You
container and returned an honest negative result (§2.4). The gradient-boosting families have
now run on a cluster; their local unit tests remain the 11 skipped tests on a machine without
`libomp`. What has still never run for real is **UC model registration**, because no campaign
has passed QA. Open defects are graded in [OPEN_ISSUES.md](OPEN_ISSUES.md).

---

## 1. What exists today

Status vocabulary:

- **runs** — executed against the real Databricks workspace at least once.
- **runs (mock only)** — exercised by tests and the mock executor; never run for real. After
  `lfy-rerank-001` very little is left in this category; where it remains, it says why.
- **partial** — the code path exists but does less than its name or the docs suggest.
- **unused** — defined, imported by nothing that runs; dead weight.
- **aspirational** — described in docs, not implemented.

### 1.1 Orchestrator core

| File | What it is | Status |
|---|---|---|
| `src/amt/cli.py` | Typer app: `start`, `approve`, `reject`, `resume`, `rewind`, `run`, `status`, `list`, `config`, `web`, `doctor`, `show`, `report`, `promote`, `demo seed`. | runs (`promote`'s UC registration never exercised for real) |
| `src/amt/settings.py` | pydantic-settings singleton; every knob is an `AMT_*`/`DATABRICKS_*` env var. | runs |
| `src/amt/llm.py` | `provider:model_id` → LangChain chat model; `anthropic`, `openai`, `huggingface`, `openai-compat`. | runs |
| `src/amt/testing.py` | `FakeLLM` + `CannedResponder` + `mock_stack()` behind `--mock`. | runs (tests) |
| `src/amt/monitor.py` | The read model: pure functions of `(graph, campaign_id)`. `state`, `pending_interrupt`, `list_campaigns`, `checkpoint_document`, `checkpoint_prose`, the per-gate row builders, `report_rows`, `rewind_targets`, `masked_settings`. No Rich, no Typer, no Flask. | runs |
| `src/amt/web/` | The primary surface: `app.py` routes, `runner.py` one locked thread per campaign, server-rendered templates. Binds 127.0.0.1, no auth, five-second meta refresh only while a runner is live. | runs |
| `src/amt/report/` | Jinja2 template plus `render_report` for the `ExperimentationReport`, logged to MLflow as `report.html`. | runs |
| `src/amt/_version.py` | `0.1.0` plus a `+{timestamp}.{tree_hash}` local segment when building a wheel. | runs |
| `src/amt/__init__.py` | Exports `__version__`. | runs |

### 1.2 Schemas (`src/amt/schemas/`) — the only cross-agent language

All inherit `AmtModel` (`extra="forbid"`, `validate_assignment=True`).

| File | Models | Status |
|---|---|---|
| `common.py` | `MetricDef`, `CohortDef`, `SplitRef`, `ExperimentBudget`, `stable_hash`, `is_harness_metric`. | runs |
| `problem_spec.py` | `ProblemSpec` + sections; validates that `primary_offline_metric` is harness-computable. | runs |
| `evaluation_contract.py` | `EvaluationContract`, `DatasetDefinition`, `ColumnMap`, `SplitPolicy`, `AcceptanceCriteria`, `SplitMaterialization`. | runs |
| `hypothesis.py` | `Hypothesis` (the SPEC's six questions as six required strings), `HypothesisBacklog`, `ExperimentPlan`. | runs |
| `experiment_config.py` | `ExperimentConfig` + `config_hash`. | runs |
| `feature_recipe.py` | `FeatureRecipe`, `FitSpec`, `FeatureFitProposal`, `feature_fit_hash`, closed transform vocabulary. | runs (orchestrator) / never executed on a cluster |
| `run_result.py` | `TrainingRunResult`, `RunSummary`. | runs |
| `champion.py` | `HypothesisChampion`, `FinalChampionSelection`, `FrozenChampion` (Pydantic-frozen). | runs |
| `qa.py` | `QADecision`, `Phase0Result`, `Phase1Result`, `PairedDelta`, `Phase0Triage`, `QAAssessment`, `ProxySanity`, `ProductUnits`, `AbDesign`, `CohortRow`, `ExperimentationReport`. | runs |
| `conclusion.py` | `ScientificConclusion`, `ProductInterpretation`, `ExperimentationReport`. | runs |

Fields defined but never read by any runtime, re-verified 2026-09-17: `CohortDef.expression`
and `.min_support`, `ExperimentConfig.custom_code_ref`, `DatasetDefinition.temporal_range`,
`MetricDef.implementation`. They are prose that agents read, not instructions the system
follows — cohorts are assigned from columns the materialisation carries, and metrics are
computed by the harness from the metric *name*.

Two entries left this list. `FeatureRecipe.joins` and `RecipeSource.join_keys` were **deleted**:
an attribute source attaches on the contract's item column, which was the only path
`train.py` ever took, so the field was a second way to say the same thing and a first way to
say something unexecutable. `DatasetDefinition.snapshot_id` and `.dataset_version` are still
not read by a job, but they are now required and rejected when blank, so a contract can no
longer claim reproducibility it does not have.

### 1.3 Agents (`src/amt/agents/`) — pure `inputs → Pydantic` functions

| File | Methods / prompt | Status |
|---|---|---|
| `base.py` | Prompt loading, context rendering, `.with_structured_output()`. | runs |
| `problem_spec_agent.py` | `investigate` (probe loop, ≤ `AMT_MAX_PROBE_STEPS`=8), `draft_spec`, `draft_contract`. | runs |
| `planner_agent.py` | `propose_hypotheses`, `build_plan`, `propose_feature_fit`, `select_final_champion`, `write_conclusion`. | all run |
| `trainer_agent.py` | `propose_config`, `interpret`, `select_champion`. | all run |
| `qa_agent.py` | `triage_phase0`, `decide`. | both run |
| `prompts/*.md` (13 files) | One per agent method. | live iff their method runs |

The probe loop is real: the agent picks `describe_table` / `profile` / `sample` / `done`, and
those are typed calls into `ReadOnlySQL` — the model never emits raw SQL. It stops on a repeated
`(action, target)` pair or at 8 steps.

### 1.4 Graph (`src/amt/graph/`)

20 nodes, 8 conditional routers, 5 human interrupts.

| Node | Side effect | Status |
|---|---|---|
| `problem_spec` → `hc_problem_spec` (**HC2**) | SQL warehouse | runs |
| `evaluation_contract` → `materialize_splits` → `hc_contract` (**HC3**) | Databricks job | runs |
| `run_baseline` | Databricks job | runs |
| `planner_hypotheses` → `hc_hypotheses` (**HC_BACKLOG**) | LLM | new; never exercised on a real campaign |
| `planner_select_hypothesis` → `planner_feature_fit` → `hc_feature_fit` (**HC_FEATURE**) | LLM | `planner_feature_fit` runs; approval never granted for real |
| `trainer_propose` → `execute_training` → `trainer_interpret` | Databricks job | runs |
| `select_hypothesis_champion` → `hc_champion` (**HC_CHAMPION**) → `planner_final_champion` → `freeze_champion` | — | runs |
| `qa_phase0` → `qa_phase1` (proxy sanity, then the candidate) → `qa_decision` | Databricks jobs | runs |
| `conclusion` → `log_final_mlflow` (conclusion + `report.html`) → `hc_conclusion` (**HC4**) | MLflow | runs |

`state.py` holds `CampaignState`. Fields that are written but never read:
`split_materialization`, `final_selection_rationale`, `promotion_decision`. Fields declared and
never written: `errors`, and the `awaiting_human` status value. `routing.py` is pure and well
tested; `route_after_qa_decision` exists there but is **not wired into the graph** — the edge
`qa_decision → conclusion` is unconditional.

### 1.5 Executors and Databricks plumbing

| File | What it is | Status |
|---|---|---|
| `executor/base.py` | `Executor` Protocol. | runs |
| `executor/databricks.py` | Submits one-time wheel jobs, polls, finds the MLflow run by `amt.job_run_id` tag, downloads `result.json`. | runs for materialize/baseline/train; QA paths never executed |
| `executor/local_mock.py` | Deterministic synthetic results for every role. | runs (tests, `--mock`) |
| `databricks/client.py` | `WorkspaceClient` factory; `pin_auth_env` sets `DATABRICKS_AUTH_TYPE` so MLflow's own client can't see two identities. | runs |
| `databricks/jobs.py` | `jobs.submit` builder, poll loop, run output. | runs |
| `databricks/sql.py` | Read-only guard: single statement, first keyword in `SELECT|WITH|DESCRIBE|SHOW`. | runs |
| `databricks/wheel.py` | Content-addressed build + upload to the UC Volume. | runs |
| `databricks/mlflow_client.py` | Experiment/run helpers, `amt.*` tag conventions, UC registration. | runs except registration (mock only) |
| `scripts/clear_cluster_wheels.py` | Ops script to uninstall stale `amt-*` wheels from a cluster. | runs (manual, untracked) |

### 1.6 Job runtime (`src/amt/jobs/`) — runs on the cluster, imports no LangGraph/LLM code

| File | What it is | Status |
|---|---|---|
| `entry.py` | `amt-job` console script; dispatches `materialize\|baseline\|train\|qa_phase0\|qa_phase1`, wraps each in an MLflow run, writes `result.json`. | runs (QA roles never invoked) |
| `materialize_splits.py` | Builds train (interaction grain) and validation/QA (user grain) tables; pins Delta version; hashes membership. | runs |
| `train.py` | Fits a family, evaluates validation via the harness, computes cohorts, logs everything. | runs |
| `qa.py` | Phase 0 (8 integrity checks) and Phase 1 (QA holdout, paired bootstrap, cohorts, latency/size). | never run for real |
| `data/loaders.py` | `SplitGuard` — the only supported read path. | runs |
| `data/features.py` | Item index, sequences, eval frames, pyfunc history coercion. `history_stats` is **unused**. | runs |
| `eval/harness.py` | `recall@k`, `ndcg@k`, `hitrate@k`, `mrr@k`, `coverage@k`; `HARNESS_VERSION = "1.0.0"`. | runs |
| `eval/cohorts.py` | Fixed `history_length` and `recency` buckets. | runs |
| `eval/paired.py` | Per-user deltas, seeded paired bootstrap CIs. | never run for real (Phase 1 only) |
| `models/registry.py` | Family table, save/load, shared `RecommenderPyfunc`. | runs |
| `models/popularity.py` | Train-frequency baseline excluding seen items. | runs |
| `models/sasrec.py` | PyTorch SASRec with harness-driven early stopping. | runs |
| `models/tabular_common.py` | Executes a `FeatureRecipe` + `FitSpec`: candidates, negatives, XGBoost/LightGBM. | never run on a cluster |
| `models/{xgboost,lightgbm}_{classifier,reranker}.py` | Four thin modules over `tabular_common.fit_family`. | never run on a cluster |
| `demo/synthetic.py` | Power-law synthetic interactions for `amt demo seed`. | runs (mock/demo only) |

### 1.7 Tests

`uv run pytest` → **244 passed, 11 skipped**. All 11 skips are in
`tests/unit/test_models.py:303`: xgboost (6) and lightgbm (5) cannot load `libomp.dylib` on this
machine. Install `brew install libomp` to un-skip them; nobody has.

Untested, and this is the honest risk list: most CLI commands (`doctor`'s cluster report and
the checklist verbs now have unit tests, the rest do not); `DatabricksExecutor._submit`, wheel
build and upload; the probe loop; Spark materialisation; and the executor's
`_to_phase0_result` / `_to_phase1_result` mappings, which are thinly covered and where the
mock uses different check names than the job, so the integration test cannot catch a mismatch
(OPEN_ISSUES B5). The integration test proves graph topology and the web surface proves the
routes; neither proves anything about Databricks. What covers Databricks is that
`lfy-rerank-001` ran.

### 1.8 Docs

| Doc | What it is for |
|---|---|
| `docs/SPEC.md` | Normative. The scientific constraints in it are hard rules, not guidance. |
| `docs/PLAN.md` | Architecture: repo layout, state, graph topology, executor contract, enforcement, and §10 on swapping an agent or the executor. |
| `docs/RUNBOOK.md` | Operator guide, gate by gate. `docs/runbook.html` is the same content, hand-written, not generated. |
| `docs/campaign-map.html` | The visual map: swimlanes, both evaluation modes, split access, hypothesis statuses, the agent-swap table. Hand-written; no generator. |
| `docs/RUN_FYL_CAMPAIGN.md` | The worked proving campaign, with the numbers to expect at each gate. Read this before running a real one. |
| `docs/OPEN_ISSUES.md` | Every known defect, graded A–C. The live list. |
| `docs/IMPROVEMENT_PLAN.md`, `docs/REVIEW_ITERATION.md` | Historical: what the redesign was responding to. |
| `docs/adr/0001-split-guard.md`, `0002-wheel-jobs.md` | The two decisions worth an ADR. |
| `CLAUDE.md` | The conventions an agent working in this repo must follow. |
| `.cursor/rules/databricks-cost.mdc` | The cost tags and the `system.billing` query that attributes a campaign. |

The prompt/schema tables in `PLAN.md` §10 and `campaign-map.html` are checked against
`src/amt/agents/` — all thirteen pairs verified on 2026-09-17.

---

## 2. Data: what we read and write

### 2.1 Source tables (Unity Catalog, read-only)

| Table | Role |
|---|---|
| `pnr_prod.silver.pr_video_interactions` | **The only table the pipeline reads.** Interaction grain. Columns used: `account_id` (user), `media_content_id` (item), `event_timestamp` (ts), plus filter columns `media_stream_type`, `entity_type_papi`, `media_is_autoplay`, `audio_only`, `total_minutes_viewed`. |
| `pnr_prod.silver.pr_item_feature` | Named in the ProblemSpec as the candidate/attribute source. **Not read by any executed code.** It would only be read by `tabular_common` once a booster recipe is approved. |
| `pnr_prod.silver.pr_user_feature_agg` | Named in the ProblemSpec, then explicitly excluded by the contract (current-state snapshot, no as-of date). Not read. |
| `pnr_prod.silver.pr_video_impressions` | Listed as investigated; the contract excludes every field of it as a model input. Not read. |

Access is via a **personal access token**, not the service principal — `pnr_prod` is not granted
to the SP.

### 2.2 Tables we write

Splits land in `product_api_dev.amt`, prefixed by campaign:

| Split | Table | Rows (campaign `vod-nextitem-001`) |
|---|---|---|
| train | `product_api_dev.amt.vod_nextitem_001_split_train` | 1,241,140 (interaction grain) |
| validation | `product_api_dev.amt.vod_nextitem_001_split_validation` | 44,495 (user grain) |
| qa | `product_api_dev.amt.vod_nextitem_001_split_qa` | 50,094 (user grain) |

All pinned at `delta_version = 0` with a `membership_hash` recorded in the contract; every read
re-verifies the row count. An earlier unprefixed set (`product_api_dev.amt.split_train`, etc.)
from before campaign prefixing still exists and is orphaned — safe to drop.

`materialize_splits` needs `SELECT` on the source, and `CREATE TABLE` / `DROP TABLE` /
`DESCRIBE HISTORY` on `product_api_dev.amt`. It creates the schema only if missing.

### 2.3 Volume and MLflow

- Artifacts: `/Volumes/product_api_dev/amt/amt_artifacts`, with `wheels/{tree_hash12}/amt-*.whl`
  and `campaigns/{campaign_id}/{job_run_id}/{contract,config,frozen}.json`.
- MLflow: experiment **`3680987379186329`** (`/Users/aayush.chhabra@fox.com/amt-exp`), pinned via
  `AMT_MLFLOW_EXPERIMENT_ID`. Per-campaign experiment creation under a root is implemented but
  currently bypassed by the pin.
- Every run carries `amt.campaign_id`, `amt.role`, `amt.job_run_id`, `amt.contract_hash`,
  `amt.config_hash`, `amt.hypothesis_id`, `amt.harness_version`, `amt.allowed_splits`.
- Orchestration state: SQLite at `~/.amt/checkpoints.sqlite`. Delete it and the campaign is gone
  locally; the MLflow runs survive.

### 2.4 The campaigns that have run

`amt list` shows five threads: `lfy-rerank-001`, `vod-nextitem-001`, `lfy-ctr-002`,
`lfy-ctr-001`, `dryrun`. Two matter.

**`lfy-rerank-001` — the proving campaign, finished, negative.** The first campaign to run the
whole graph on real data, and the one the gate redesign existed to make possible.

- Latest For You container `687599077569` on home page `727846815041`, CTV clients,
  `evaluation_mode: grouped_rerank`, `relevance_weighting: pbm`, primary `ndcg@4` with
  `hitrate@1` as the slot-0 bridge, baseline `logged_order` — the incumbent ordering.
- `contract_hash 59f76bda202a…`, `feature_fit_hash f3d688b18ea5…`, harness `1.1.0`,
  MLflow experiment `3680987379186329`, final run `5d2af74bc5f54fff8b66c3a2db0c88ef`.
- Backlog of four hypotheses: two abandoned by the operator, two converged. H1 (three LightGBM
  classifier configurations) championed at validation `ndcg@4` **0.75022**; H4 (a grouped
  LightGBM reranker) at **0.75349**. The incumbent scored **0.79109**. **Neither beat it** —
  H4 by −4.75 % relative.
- QA replayed the frozen H4 artifact: Phase 0 passed every integrity check including exact
  metric reproduction. Phase 1's proxy-sanity gate **failed** — seven-day Spearman ρ between
  the incumbent's daily offline proxy and the dashboard's Container CTR was **0.357** against a
  pre-registered floor of 0.5 — so the decision is `SPEC_REVIEW`. Independently, the candidate
  also lost outright on QA: `ndcg@4` −0.05244 (95 % CI [−0.05653, −0.04836], wholly negative)
  and `hitrate@1` −0.10211, −18.81 % relative.
- Cohorts: improved only on ANDROID (+1.44 %) and IOS (+0.50 %); regressed past the 2 % guardrail
  on all eleven CTV cohorts, from ROKU −3.80 % to COMCAST −14.08 %.
- Operationally cheap and not the bottleneck: 401.9 s to train, 1,479 parameters, QA inference
  2.66 ms p50 / 7.17 ms p95.
- `recommendation: hypothesis_rejected`, `negative_result: true`. The conclusion's
  recommendation was clamped to the measured verdict rather than left to the narrative.

Two operator signals from that run are worth keeping, because both are the gates working:
`gate_cycles = 4` (flagged high — the operator and the agent went round four times, mostly at
HC_FEATURE) and `trainer_asks_refused = 15` — the Trainer asked for a structural change fifteen
times and was clamped every time, which under the old design would have been fifteen bounces
back through a human gate.

What it establishes, and what it does not: the grouped re-ranking path works end to end on
production data, and the honest answer for this rail is that metadata-only re-ranking of the
incumbent's own impressions does not beat the incumbent. The failed proxy gate means the
offline metric is not yet trustworthy enough to certify *any* replacement for this rail, which
is a finding about the evaluation design and is the first thing a follow-on campaign must fix.

**`vod-nextitem-001` — abandoned mid-flight, kept for history.** A `next_item` campaign that
reached `HC_FEATURE` on hypothesis H2 before the redesign landed. Its `hypotheses_approved`
flag is unset because it predates `HC_BACKLOG`. It is not resumable against the current graph
and should not be used as a reference for how a campaign looks now.


---

## 3. Decisions made and locked

| Decision | Reason |
|---|---|
| LangGraph + SQLite checkpointer, human checkpoints via `interrupt()`. | Campaigns span hours and need to survive a closed laptop; `Command(resume=...)` gives a CLI-driven approval loop for free. |
| Compute is a **content-addressed wheel** submitted as one-time `python_wheel_task` runs. | ADR 0002. What ran is a git-identifiable source tree plus a recorded parameter list, not a generated notebook string. Unchanged tree ⇒ upload skipped. |
| **All split access goes through `SplitGuard`**; the executor grants splits per role. | ADR 0001. "Don't read the holdout" is unfalsifiable as a prompt and a violation produces a *better* number. Train/baseline get `train,validation`; QA Phase 0 gets `validation`; QA Phase 1 gets `qa` and nothing else. |
| Reads are pinned to the contract's Delta version and row count verified on every read. | Source tables keep receiving data; campaign ground truth must not move underneath a comparison. |
| MLflow is the only persistence layer for experiment detail; `CampaignState` stores IDs. | SPEC's minimal-persistence principle; keeps the resumable state small. |
| Results return via MLflow artifact (`result.json`) keyed by an `amt.job_run_id` tag, not stdout. | Driver log parsing is unreliable; the tag survives retries and the artifact doubles as the audit record. |
| The EvaluationContract carries **executable** fields (`dataset.source_table`, `dataset.columns`, `dataset.filters`, `split_policy` boundaries). | The first contracts were pure prose and materialization had nothing to run. Splits now derive from the frozen contract alone. |
| Contract and spec metrics are restricted to `{recall\|ndcg\|hitrate\|mrr\|coverage}@k`. | Agents kept proposing IPS-NDCG, PR-AUC and Brier, which the harness cannot compute; the campaign then died three stages later. Now it fails at draft time. |
| Split tables are prefixed with the campaign id. | Parallel campaigns were overwriting each other's `split_train`. |
| The hypothesis backlog is written **once**, after the baseline, and approved at HC_BACKLOG. | A human reading all five at once can strike the ones this harness cannot honestly test, before any compute is spent. The SPEC leaves this region autonomous; this gate is a deliberate deviation. |
| Booster families take a declarative `FeatureRecipe` + `FitSpec` approved at HC_FEATURE, with a closed transform vocabulary; `feature_fit_hash` re-gates structural changes. | The alternative was letting the Trainer emit feature code. A closed vocabulary is reviewable and cannot read the QA split. |
| Agents are provider-agnostic (`provider:model_id`, built only in `src/amt/llm.py`). | Swapping models is an env change; no provider SDK leaks into agent code. |
| PAT authentication, with `DATABRICKS_AUTH_TYPE` pinned. | `pnr_prod` is not granted to the service principal. Having both identities present made MLflow's internally-built client fail with "more than one authorization method configured". |
| Wheels carry a `+{timestamp}.{tree_hash}` local version. | A cluster silently kept serving an older same-version `amt` install; jobs failed on arguments the code clearly had. |
| `amt.jobs.*` imports no LangGraph/LangChain/LLM code. | The wheel must install on a cluster without the orchestrator's dependency tree. |

---

## 4. Decisions still open

Most of what was open on 2026-09-15 has been decided by the redesign or answered by
`lfy-rerank-001`. What remains:

**Is the offline proxy salvageable for this rail?** This is the live question, and it blocks
everything else. `lfy-rerank-001` measured ρ = 0.357 between the incumbent's daily offline
`ndcg@4` and the dashboard's Container CTR, against a pre-registered floor of 0.5. The two
numbers are built from different sources — we rank on playback-within-90s, the dashboard counts
clicks — and they also cover different populations. Options: join a `click_flag` from
`dtc_visitid_dim_plus` and rank on clicks, matching the dashboard's definition; align the
populations instead (the ≥10 s session filter and the window are the prime suspects); change
the aggregation grain the correlation is computed at; or accept that Container CTR is the wrong
product metric for this rail and pick one the offline label can track. Until one is chosen, no
campaign on this container can certify a model, because the gate will keep firing — correctly.

**Who enforces acceptance criteria?** Unchanged and still worth doing. The QA agent reads them
as prose and returns a boolean. Options: leave it to the LLM, or compute PASS/FAIL in code from
`Phase1Result` and let the agent write only the narrative. The second is what the SPEC's
"pre-registered" language implies, and `validation_verdict` is now the precedent for it.
See OPEN_ISSUES A4.

**Contract cohorts vs. job cohorts.** Still unresolved. The contract names cohorts; the jobs
compute their own fixed keys from columns the materialisation carries, so
`CohortDef.expression` is decorative and the cohort-regression guardrail cannot be evaluated
mechanically from the contract. `lfy-rerank-001` worked because `client_name` happened to match
a key the job computes. Options: constrain contract cohort names to the job's vocabulary and
validate them at HC3, teach `cohorts.py` to execute the expression, or drop cohort regression
from acceptance and make it advisory. OPEN_ISSUES A3.

**Job failure handling.** A Databricks failure still raises out of the graph rather than
reaching the Trainer as a failed `TrainingRunResult`. Options: catch it in `execute_training`
and let the Trainer adapt, or keep it fatal and resumable and stop claiming otherwise.
OPEN_ISSUES A1.

**Whether to add an `age_hours` transform.** The one vocabulary addition this rail would
plausibly use (`event_received − created_at`, roughly ten lines in `tabular_common`). Deferred
before `lfy-rerank-001` and still deferred; the proxy question outranks it.

**Second service principal for QA** (PLAN §9) — grants limited to `split_qa`, so the negative
constraints hold at the permission layer too. Not started; `SplitGuard` is the only enforcement.

---

## 5. Assumptions not verified against data

1. **The 2-minute / non-autoplay / non-audio filter equals "deliberate start."** It was chosen
   from column names and a schema probe, not validated against known engagement behaviour.
2. **`entity_type_papi` values.** The contract filters on `'sportevent'`; the product team said
   `sporting_event`. Nobody has counted rows per entity type to confirm which literal exists, or
   how much data the wrong one would silently discard.
3. **Population eligibility is computed over the whole window, including post-train activity.**
   The contract admits this. Its effect on the metric is unmeasured.
4. **`snapshot_id` and `dataset_version` are empty strings.** The contract's own
   `label_construction` text says a real immutable source reference "must be populated before
   approval/materialization". It was approved without one. Reproducibility rests on the Delta
   version of the derived splits, not on the source snapshot.
5. **Validation and QA are comparable populations.** 44,495 vs 50,094 users over 3 and 4 days;
   nobody has checked that the user mixes match.
6. **`ndcg@10` over the full catalog is a usable proxy for Latest-For-You CTR.** This is the
   central scientific assumption of the campaign and it is unvalidated. The deployed production
   model was never evaluated under this contract, so "beats baseline" means "beats popularity",
   not "beats what is in production".
7. **The cluster has `xgboost`, `lightgbm`, `pandas` and `pyarrow`.** The executor passes **no**
   `extra_libraries` and installs the wheel without its `jobs` extra, so every heavy dependency
   comes from the cluster image. `torch` is confirmed present (SASRec ran); the boosters are not.
8. **`tabular_common`'s defaults fit this data.** `max_train_rows = 500_000` against a
   1.24M-row train split, `max_validation_users = 2_000` against 44,495, 200 popularity
   candidates, 4 negatives per positive. All untested here.
9. **Candidate tables are servable.** `pr_item_feature` is assumed joinable at the item grain and
   available at serve time; never probed for that.

---

## 6. Tried and abandoned

| Tried | Why it was dropped |
|---|---|
| Framing the campaign as Latest-For-You container CTR, with impressions as the candidate set. | The harness computes ranking metrics over a fixed catalog; it has no notion of a container, a slot or an impression. Re-scoped to next-item ranking. |
| Metrics like IPS-NDCG@10, PR-AUC, Brier, calibration. | Not implementable by `eval/harness.py`. Now rejected by a schema validator instead of failing mid-campaign. |
| A prepared SQL view to normalise the interactions source. | Unnecessary once the contract carried a column map and filter list; the raw table is usable directly, and a view would have been an unversioned dependency. |
| Service-principal OAuth as the only auth path. | The SP has no grants on `pnr_prod`. PAT added and given precedence. |
| Letting the Trainer generate feature engineering code for the boosters. | Unreviewable and capable of reading anything. Replaced by the declarative recipe with a closed vocabulary. |
| Returning a non-zero exit code from `entry.main()` to signal job failure. | Databricks `python_wheel_task` reports SUCCESS anyway; the real error was invisible. The entry point now re-raises. |
| Creating an MLflow experiment per campaign under a root path. | The root `/Users/…/amt-exp` was itself an experiment, so creation failed with "path exists but is not a directory". Bypassed with a pinned experiment id; the per-campaign code is still there and still the default when no id is pinned. |
| Repo-wide `ruff format`. | The tree is not format-clean under the pinned ruff, so it rewrote 31 files and buried real edits. Reverted. Run the formatter on files you actually touch, or expect a noisy diff. |

---

## 7. SPEC coverage

| SPEC stage | State |
|---|---|
| 1 — Problem Specification Agent | Implemented and run, including the SQL probe loop. HC1 is the `--problem` argument to `amt start`, not an interrupt. |
| 2 — EvaluationContract | Implemented and run. Executable fields drive materialization. **Gaps:** immutable source references (`snapshot_id`, `dataset_version`) are optional and empty; cohort definitions are not executed. |
| 3 — Planner / hypotheses | Implemented and run, plus the non-SPEC HC_BACKLOG gate. Hypotheses carry all six required questions as schema fields. |
| 4 — Trainer / ExperimentConfig | Implemented and run. Config-driven for `popularity` and `sasrec`; recipe-driven for the boosters. Custom architecture code is deliberately not supported (ADR 0002). |
| 5 — Databricks execution | Implemented and run. Reads frozen train/validation, never QA, logs params/metrics/artifact/lineage. |
| 6 — Trainer optimization loop | Implemented and run; three iterations on H1. Convergence combines the agent's judgment with hard budget stops. |
| 7 — Hypothesis champion | Implemented and run for real. Reviewed by a human at `HC_CHAMPION`, once per hypothesis. |
| 8 — Champion freeze | Implemented (`FrozenChampion` is Pydantic-frozen) and executed for real. |
| 9 — QA Phase 0 / Phase 1 | Implemented in `jobs/qa.py` and executed for real: Phase 0's integrity checks, and Phase 1's proxy-sanity gate followed by paired bootstrap and cohorts. Phase 0's "fix and rerun" loop still records a fix plan and then resubmits the identical champion — no node applies the fix (OPEN_ISSUES B3). Phase 1 loads the model through MLflow pyfunc but scores it with the same harness as training, not an independent implementation as the SPEC prefers. |
| 10 — QA decision | Implemented; the decision is the LLM's. Acceptance criteria are prose in its prompt, not code. |
| 11 — Scientific conclusion + HC4 | Implemented and run for real, with the `ExperimentationReport` alongside the narrative. `recommendation` is clamped to the measured validation verdict. |
| 12 — MLflow output | Implemented and run for real, including `report.html`. UC registration on `amt promote` has still never run against a real workspace, because no campaign has passed QA yet. |
| Scientific constraints | The negative constraints (never train on validation/QA, one frozen contract, one candidate to QA, no re-entry after freeze) are enforced in routing and `SplitGuard`, and tested. Permission-level enforcement (second SP) is not built. |

---

## 8. Known bugs, TODOs, hardcoded values

### Bugs and wrong behaviour

**The live list is [OPEN_ISSUES.md](OPEN_ISSUES.md)**, graded A (correctness and cost) through
C (untidy or deferred on purpose). It is maintained per change; this section only records what
the gate redesign *closed*, so a reader does not go looking for a bug that no longer exists.

Fixed and verified since this document was first written:

| Was | Now |
|---|---|
| `AMT_PHASE0_MAX_ATTEMPTS` ignored; `routing.PHASE0_MAX_ATTEMPTS = 3` hardcoded | `qa_phase0` stamps `settings.phase0_max_attempts` into state and routing reads it from there, so the cap cannot change mid-campaign |
| `route_after_qa_decision` dead code | Deleted, with its tests |
| `Phase0Result` carried `preprocessing_present`, `features_available`, `train_inference_consistent`, which nothing populated | Deleted. The verdict comes from the job's own `checks` list |
| `DatasetDefinition.snapshot_id` / `dataset_version` optional and usually blank | Required, with a validator that rejects blanks |
| `FeatureRecipe.joins` / `RecipeSource.join_keys` — a second, partly unexecutable way to express an attribute join | Deleted; sources attach on the contract's item column, which is the only path `train.py` ever took |
| A Trainer structural ask bounced the campaign back through `HC_FEATURE` | Clamped in `_bind_feature_fit` and recorded in `structural_asks_clamped` |
| `HC_FEATURE` interrupted once per hypothesis | Batched: one interrupt per campaign over every tabular shape, skipped when none is tabular |
| The Planner could regenerate hypotheses mid-campaign | The backlog is written once at `HC_BACKLOG`; selection never generates |
| A hypothesis needing a transform that does not exist was discovered by a failing job | Marked `infeasible` at backlog time, with the reason, from `runtime_vocabulary()` |
| No human saw a hypothesis champion | `HC_CHAMPION`, once per hypothesis |
| Job clusters unusable, so runs needed a hand-prepared all-purpose cluster | Per-family `requires_pip` installs the dependency per run; job clusters run in single-user mode so they can read the wheel from a UC volume; cost tags attached |
| `amt doctor` failed on a pinned cluster with a claim that was untrue once tagged | Warns, reads the cluster, and names what is actually lost |
| The conclusion could recommend promoting a champion that lost | `recommendation` clamped to `validation_verdict`, which resolves the baseline through `metric_value` and so can actually see it |
| The checkpoint document was clipped at 20 000 characters in the browser | Served whole, with a plain-text route; `HC4`'s narrative renders as prose |
| Rewind existed only in the CLI | On the campaign page, with illegal targets explained rather than hidden |

### Hardcoded values worth knowing

| Value | Where |
|---|---|
| `phase0_max_attempts` default `3` | `settings.py` — no longer hardcoded in routing |
| `RECURSION_LIMIT = 150` (raised for the batched gate plus a full trainer loop), CLI budget defaults `6 / 2 / 240.0` | `monitor.py:56`, `cli.py` |
| `PREVIEW_CHARS = 20_000` — the terminal preview cap. The web surface deliberately ignores it | `monitor.py` |
| `AMT_MAX_CANDIDATE_ROWS` default `50_000_000` | `settings.py` |
| `EVENT_LIMIT = 1000` progress lines per campaign, in-process and not durable | `web/runner.py` |
| `DEFAULT_POLL_SECONDS = 15.0`, `DEFAULT_TIMEOUT_SECONDS = 10800.0` (3h per job) | `databricks/jobs.py:34-35` |
| SQL probe: `LIMIT 200` cap, `wait_timeout="50s"`, 32 profiled columns | `databricks/sql.py` |
| `HARNESS_VERSION = "1.1.0"` (bumped for weighted gains and the binary secondaries), `DEFAULT_KS = (5, 10, 20)` — used only when the contract's metrics do not name their own `k` | `jobs/eval/harness.py:38` (duplicated in `schemas/evaluation_contract.py:30`) |
| `propensity_floor` contract default `0.02`; the LFY campaign used `0.05` | `schemas/evaluation_contract.py` |
| Bootstrap: seed `20240917`, 2000 resamples, α=0.05 | `jobs/eval/paired.py` |
| Reproduction subset = 200 users; Phase 1 latency sampled on 64 users | `jobs/train.py:39`, `jobs/qa.py:351` |
| `max_train_rows = 500_000`, `max_validation_users = 2_000`, 200 candidates, 4 negatives, 200 boost rounds, early stop 20, seed 42 | `jobs/models/tabular_common.py:55-73`, `454-455` |
| SASRec defaults: hidden 64, 2 layers, 2 heads, dropout 0.2, seq 50, lr 1e-3, batch 256, 5 epochs, 8 negatives, patience 2 | `jobs/models/sasrec.py:25-41` |
| `SplitPlan` fallbacks: catalog `main`, schema `amt`, `user_id`/`item_id`/`ts`, `min_interactions_per_user = 2` | `jobs/materialize_splits.py:81-93` (overridden by the contract in practice) |
| Cohort buckets: `<1`, `1–4`, `5–19`, `20+`; `0–7d`, `8–30d`, `31d+` | `jobs/eval/cohorts.py:26-37` |
| Promoted model name `{catalog}.{schema}.amt_{campaign}`, alias `candidate` | `cli.py:310`, `databricks/mlflow_client.py:171-174` |

### TODOs a new owner should pick up first

1. Decide HC_FEATURE for H2 (§4) — everything else is blocked behind it.
2. Smoke-test `tabular_common` against a sample of the real split tables **before** spending
   cluster minutes, and confirm `xgboost`/`lightgbm` import on the target cluster.
3. Reconcile contract cohort names with job cohort keys, or the acceptance gate is decorative.
4. Write the first real QA run with eyes open: Phase 0's mapping and Phase 1's baseline URI
   (`runs:/{baseline_run_id}/model`) have never been exercised.
5. Add tests for `_to_phase0_result` / `_to_phase1_result` against payloads produced by
   `amt.jobs.qa`, and make the mock use the job's check names.
6. `brew install libomp` so the 11 skipped booster tests actually run.
7. Commit or discard the uncommitted 37-file working tree.
