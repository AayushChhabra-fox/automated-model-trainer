# Architecture Plan — Automated Model Trainer (AMT)

Normative spec: `docs/SPEC.md`. This document translates it into a concrete build.

## 1. Decisions

| Concern | Decision |
|---|---|
| Orchestrator | **LangGraph** (`StateGraph`) with `SqliteSaver` checkpointer for resumability. Human checkpoints use `interrupt()`; CLI resumes with `Command(resume=...)`. |
| Agent LLMs | Provider-agnostic: `settings` names each agent's model as `provider:model_id` and `amt.llm` builds the client — `ChatAnthropic` for `anthropic`, `ChatOpenAI` for `openai`, `huggingface` and `openai-compat` (any OpenAI-compatible endpoint: HF Inference Endpoints, vLLM, TGI, Databricks serving) via `AMT_LLM_BASE_URL`. All providers go through `.with_structured_output(PydanticModel)`, so agents never learn which vendor answered. Defaults: `anthropic:claude-opus-5` for Planner/Trainer/QA reasoning, `anthropic:claude-sonnet-5` for ProblemSpec data-probing chatter. One credential pair (`AMT_LLM_API_KEY`, `AMT_LLM_BASE_URL`) covers every provider. |
| Where agents run | Local CLI (`amt`). Agents never compute training themselves. |
| Where compute runs | **Databricks Jobs** (one-time `jobs.submit` runs, `python_wheel_task`). Wheel built from this repo and uploaded to a UC Volume. |
| Auth | Databricks SDK unified auth via env: `DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET` (OAuth M2M, service principal), or `DATABRICKS_TOKEN` (PAT) to run as a human under their own grants. No secrets in repo. |
| SQL access (Stage 1) | Databricks SQL Statement Execution API via SDK against `DATABRICKS_WAREHOUSE_ID`, read-only wrapper (rejects non-SELECT). |
| Persistence | MLflow (Databricks tracking) for all experiment detail; UC Delta tables only for materialized splits; SQLite checkpoint for orchestration state. |
| Reference models | `popularity` (baseline), `sasrec` (PyTorch sequence model), and four gradient-boosting families (`xgboost_classifier`, `xgboost_reranker`, `lightgbm_classifier`, `lightgbm_reranker`) whose inputs come from a human-approved `FeatureRecipe`. All config-driven. |
| Demo data | Synthetic interaction generator (`amt demo seed`) writing to a UC schema so the pipeline runs before pointing at real tables. |
| Package mgmt | `uv`, `pyproject.toml`, Python ≥3.10. |

## 2. Repo layout

```
automated-model-trainer/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── .env.example
├── docs/
│   ├── SPEC.md                 # product spec (normative)
│   ├── PLAN.md                 # this file
│   └── adr/                    # short ADRs for non-obvious choices
├── src/amt/
│   ├── __init__.py
│   ├── settings.py             # pydantic-settings; env-driven
│   ├── llm.py                  # provider:model_id -> chat model (anthropic / openai / hf / compat)
│   ├── cli.py                  # Typer app: init, start, approve, reject, resume, status, report, demo
│   ├── schemas/                # Pydantic v2 contracts (the only cross-agent language)
│   │   ├── problem_spec.py
│   │   ├── evaluation_contract.py
│   │   ├── hypothesis.py       # Hypothesis, ExperimentPlan
│   │   ├── experiment_config.py
│   │   ├── run_result.py       # TrainingRunResult
│   │   ├── champion.py         # HypothesisChampion, FrozenChampion
│   │   ├── qa.py               # Phase0Result, Phase1Result, QADecision
│   │   ├── conclusion.py       # ScientificConclusion
│   │   └── common.py           # MetricDef, Budget, SplitRef, ids, hashing helpers
│   ├── graph/
│   │   ├── state.py            # CampaignState (compact; references not payloads)
│   │   ├── build.py            # build_graph(executor, llm_factory) -> CompiledGraph
│   │   ├── routing.py          # conditional edge functions
│   │   └── nodes/
│   │       ├── problem_spec.py     # Stage 1 + HC2
│   │       ├── evaluation_contract.py  # Stage 2 + HC3 + split materialization
│   │       ├── planner.py          # Stage 3 (hypothesis select / plan), Stage 7b final champion
│   │       ├── trainer.py          # Stage 4/6: propose config → execute → interpret; convergence
│   │       ├── champion.py         # Stage 7a select, Stage 8 freeze
│   │       ├── qa.py               # Stage 9/10
│   │       └── conclusion.py       # Stage 11/12 + HC4
│   ├── agents/                 # LLM prompting layers, pure functions of (inputs) -> schema
│   │   ├── base.py             # structured-output helper, retry, prompt loading
│   │   ├── prompts/*.md        # system prompts per agent (derived from SPEC.md)
│   │   ├── problem_spec_agent.py
│   │   ├── planner_agent.py
│   │   ├── trainer_agent.py
│   │   └── qa_agent.py
│   ├── executor/
│   │   ├── base.py             # Executor Protocol
│   │   ├── databricks.py       # submit wheel jobs, poll, fetch result JSON from MLflow
│   │   └── local_mock.py       # deterministic fake for tests/dry-runs
│   ├── databricks/
│   │   ├── client.py           # WorkspaceClient factory (SP OAuth)
│   │   ├── sql.py              # read-only SQL wrapper + schema/profile helpers
│   │   ├── jobs.py             # submit_run / wait / logs
│   │   ├── wheel.py            # build + upload wheel to Volume
│   │   └── mlflow_client.py    # experiment/run helpers, tag conventions
│   └── jobs/                   # Code that runs ON Databricks (entrypoints in pyproject)
│       ├── entry.py            # dispatch: materialize | train | qa_phase0 | qa_phase1 | baseline
│       ├── materialize_splits.py
│       ├── train.py
│       ├── qa.py
│       ├── data/
│       │   ├── loaders.py      # SplitGuard: role-scoped access to split tables
│       │   └── features.py
│       ├── models/
│       │   ├── registry.py     # family -> builder
│       │   ├── popularity.py
│       │   ├── sasrec.py
│       │   ├── tabular_common.py   # executes an approved FeatureRecipe + FitSpec
│       │   ├── xgboost_classifier.py
│       │   ├── xgboost_reranker.py
│       │   ├── lightgbm_classifier.py
│       │   └── lightgbm_reranker.py
│       └── eval/
│           ├── harness.py      # metrics (recall@k, ndcg@k, mrr, hitrate@k, coverage), versioned
│           ├── cohorts.py
│           └── paired.py       # paired deltas + bootstrap CIs
└── tests/
    ├── unit/                   # schemas, harness math, routing, split guard
    └── integration/            # full graph with LocalMockExecutor + fake LLM
```

## 3. CampaignState (LangGraph state)

Keep it small; everything heavy lives in MLflow or UC.

```python
class CampaignState(TypedDict, total=False):
    campaign_id: str
    stage: Stage                       # enum mirroring SPEC workflow
    status: Literal["running","awaiting_human","done","aborted","spec_review"]
    problem_text: str
    problem_spec: ProblemSpec | None
    problem_spec_approved: bool
    evaluation_contract: EvaluationContract | None
    contract_approved: bool
    contract_hash: str                 # sha256 of frozen contract JSON
    baseline_run_id: str | None        # MLflow run for baseline on validation
    mlflow_experiment_id: str
    hypotheses: list[Hypothesis]       # prioritized backlog (ids + short text)
    current_hypothesis_id: str | None
    experiment_plan: ExperimentPlan | None
    approved_feature_recipe: FeatureRecipe | None   # booster families only
    approved_fit_spec: FitSpec | None
    feature_fit_hash: str              # recipe + structural FitSpec, booster_params excluded
    feature_fit_approved: bool
    experiment_history: list[RunSummary]   # compact: run_id, config_hash, primary metric, verdict
    budget: ExperimentBudget           # iterations_used/max, compute_used/max
    hypothesis_champions: list[HypothesisChampion]
    frozen_champion: FrozenChampion | None
    qa_phase0: Phase0Result | None
    qa_phase1: Phase1Result | None
    qa_decision: QADecision | None
    conclusion: ScientificConclusion | None
    human_feedback: str | None
```

## 4. Graph topology

```
START → problem_spec → [interrupt HC2] ─rejected→ problem_spec
                      └approved→ evaluation_contract → materialize_splits → [interrupt HC3]
                            ─rejected→ evaluation_contract
                            └approved→ run_baseline → planner_hypotheses
planner_hypotheses → [interrupt HC_BACKLOG] ─rejected→ planner_hypotheses
                    └approved─any tabular hypothesis→ planner_feature_fit
                             └none tabular→ planner_select_hypothesis
planner_feature_fit → [interrupt HC_FEATURE] ─rejected→ planner_feature_fit
   (all tabular hypotheses    └approved→ planner_select_hypothesis
    in one payload)
planner_select_hypothesis ─no hypotheses left / budget / frozen→ planner_final_champion
                          └→ trainer_propose → execute_training → trainer_interpret
trainer_interpret ─continue→ trainer_propose
                  └converged→ select_hypothesis_champion → hc_champion
hc_champion → [interrupt HC_CHAMPION] ─continue→ planner_select_hypothesis
                                      └freeze→ planner_final_champion
planner_final_champion → freeze_champion → qa_phase0
qa_phase0 ─INVALID_EXPERIMENT (fixable, attempts < settings.phase0_max_attempts)→ qa_phase0
          └pass→ qa_phase1 → qa_decision → conclusion → log_final_mlflow → [interrupt HC4] → END
qa_decision = SPEC_REVIEW → conclusion (status=spec_review) → [interrupt HC4] → END
```

Two edges that used to exist and deliberately do not: `trainer_propose → planner_feature_fit`
(the Trainer's structural ask is now clamped in `_bind_feature_fit`, not routed) and
`planner_select_hypothesis → planner_feature_fit` (every shape is approved before selection,
so a tabular family reaching selection without one is a bug and raises rather than routes).

`graph.update_state(..., as_node=...)` is the one write that re-enters an earlier node:
`amt.graph.human.rewind_update` returns the node to re-enter and the state to clear, and the
CLI's `amt rewind` and the web surface's `/c/<id>/rewind` both go through it. Rewind targets
are `backlog` → `run_baseline`, `feature` → `hc_hypotheses`, `qa` → `freeze_champion`; the
first two are refused once a champion is frozen.

Rules enforced in `routing.py`, not prompts: trainer loop cannot exceed `plan.max_iterations`; `execute_training` refuses if `contract_hash` differs from the one stamped at HC3; `qa_*` refuses if `frozen_champion is None` or more than one champion; `execute_training` refuses a booster family whose `feature_fit_hash` is not the one approved at HC_FEATURE.

HC_BACKLOG gates the hypothesis backlog: the Planner writes it once, after the baseline, and no hypothesis is planned until a human approves the list. A rejection revises the backlog with the feedback (hypotheses the feedback did not object to survive intact); approval happens once per campaign, not once per hypothesis. `planner_select_hypothesis` never generates hypotheses, so there is no path into training that skips the gate.

HC_FEATURE gates the declarative feature/fit shape of the booster families, **batched**: `planner_feature_fit` loops over every pending tabular hypothesis and the gate interrupts once, carrying `feature_fit_proposals: dict[str, FeatureFitProposal]` and priced candidate-row estimates for all of them. It is skipped entirely when no approved hypothesis is tabular, and `planner_feature_fit_node` raises on an empty tabular set rather than interrupting on an empty dict. Approval stores `approved_shapes: dict[str, ApprovedShape]`; `planner_select_hypothesis` copies the selected hypothesis's shape into `approved_feature_recipe` / `approved_fit_spec` / `feature_fit_hash`, so `assert_feature_fit_approved` and `execute_training` are unchanged. Hyperparameter-only iterations run unattended. A structural change requested by the Trainer is **clamped**: `_bind_feature_fit` keeps the approved structural payload, takes only `booster_params`, inherits approved `objective` / `eval_metric` when the Trainer sends blanks, and records the rejected ask in `structural_asks_clamped` for `describe()` to report. `popularity`, `sasrec` and `logged_order` skip the gate.

HC_CHAMPION gates each hypothesis's champion, between `select_hypothesis_champion` and the next selection. It carries the champion against the baseline and every prior champion on the primary and secondary metrics, cohort deltas, remaining budget, and the recommended next move. Verbs: `continue` (default), `freeze` (skip the remaining hypotheses, route to `planner_final_champion`), `drop` (abandon named hypotheses). `--no-gate-champions` disables it for rehearsal campaigns. This is what "autonomous within a hypothesis" means in the graph: the loop between `trainer_propose` and `trainer_interpret` is unattended, and its exit is a checkpoint.

## 5. Executor contract

```python
class Executor(Protocol):
    def materialize_splits(self, contract: EvaluationContract) -> SplitMaterialization: ...
    def run_baseline(self, spec, contract, split: Literal["validation","qa"]) -> TrainingRunResult: ...
    def run_training(self, config: ExperimentConfig, contract, plan) -> TrainingRunResult: ...
    def run_qa_phase0(self, frozen: FrozenChampion, contract) -> Phase0Result: ...
    def run_qa_phase1(self, frozen: FrozenChampion, baseline_ref, contract) -> Phase1Result: ...
```

`DatabricksExecutor`: builds wheel once per campaign (hash-cached) → uploads to `/Volumes/{catalog}/{schema}/amt_artifacts/` → `jobs.submit` with `python_wheel_task(package_name="amt", entry_point="amt-job", parameters=[...])` → polls to terminal state → reads the result JSON logged as MLflow artifact `result.json` on the run tagged with the job's `run_uuid`. Job cluster spec from settings (or `existing_cluster_id`).

Job parameters include `--role train|qa|baseline|materialize` and `--allowed-splits`. `SplitGuard` in `jobs/data/loaders.py` raises if code requests a split outside `allowed_splits`. Train jobs get `train,validation`; QA jobs get `qa` (+ `validation` only for Phase-0 reproduction subset). `materialize` additionally gets `--split-catalog/--split-schema` from settings: where the splits are written is a deployment fact, so it is passed per run rather than frozen into the contract.

A job runs on a cluster with no AMT settings and no access to the prose a human approved, so everything it executes has to be a field the contract carries: `dataset.source_table`, `dataset.columns`, `dataset.filters`, `dataset.min_interactions_per_user` and `split_policy` (strategy plus boundaries or per-user counts). A role that fails logs `result.json` and then *raises* — a wheel task's return value is discarded, so returning non-zero would report SUCCESS with a failed result.

## 6. Evaluation harness

Single implementation in `jobs/eval/harness.py`, `HARNESS_VERSION` recorded in the EvaluationContract and asserted by both train and QA jobs. Metrics: Recall@K, NDCG@K, HitRate@K, MRR, item coverage, plus `binary_metrics` (AUC, PR-AUC, log loss, ECE) as secondaries. Inputs: a ranked item list plus a ground-truth set **per unit**, where a unit is a user under `next_item` and a render under `grouped_rerank` — the harness does not know the difference, which is why the second mode needed no second harness. `evaluate()` takes an optional `gains` argument: under `relevance_weighting: pbm`, NDCG uses the per-item weight as its gain and HitRate / Recall / MRR count a weighted hit. `paired.py`: per-unit metric deltas, paired bootstrap 95% CI computed in chunks so a container-scale split does not materialise one giant resample array, fraction of units improved, rank correlation, top-slot conversion estimates and A/B sizing.

Metric names do not vary by mode: `ndcg@4` is `ndcg@4` weighted or not. The weighting is a contract field stamped into `contract_hash`, so baseline, candidates and QA cannot disagree about it. `k` comes from the contract's own `MetricDef` names rather than a default, so a contract asking for `ndcg@4` is scored at 4.

QA independence: QA job runs `qa.py` which loads the model via MLflow pyfunc (not the trainer's in-memory object) and computes metrics through the harness — same math, different code path for inference.

## 7. MLflow conventions

- One experiment per campaign: `/Shared/amt/{campaign_id}`.
- Tags on every run: `amt.campaign_id`, `amt.stage`, `amt.role` (baseline|train|qa0|qa1|final), `amt.hypothesis_id`, `amt.contract_hash`, `amt.spec_hash`, `amt.harness_version`, `amt.config_hash`, `amt.code_version` (git sha).
- Train runs log: params (flattened config), metrics (`train/*`, `val/*`, `val/cohort/*`), artifacts (`model/` pyfunc, `config.yaml`, `result.json`).
- Final run (Stage 12): re-logs frozen model reference, QA metrics, baseline deltas, `conclusion.md`, `report.html` (the rendered `ExperimentationReport`), `problem_spec.json`, `evaluation_contract.json`. Optionally registers to UC model registry `{catalog}.{schema}.{model_name}` with alias `candidate` when QA=PASS.
- Cluster tags on every submitted job: `amt=true`, `amt_campaign_id`, `amt_role`, `amt_hypothesis_id`, landing in `system.billing.usage.custom_tags`. These are a property of a *job* cluster and are inert on a pinned all-purpose cluster — see `.cursor/rules/databricks-cost.md`.

## 8. Human checkpoints and the two surfaces

Seven interrupts: HC2 (spec), HC3 (contract), HC_BACKLOG, HC_FEATURE, HC_CHAMPION (once per
hypothesis), HC4 (conclusion), with HC1 being the problem text itself. Each is an
`interrupt()` carrying a payload; each is answered with `Command(resume=payload)` parsed by
`amt.graph.human.parse_resume` into a `HumanDecision` with `approved`, `feedback`, `keep`,
`drop`, `freeze` and `decision`.

**Both surfaces read through `amt.monitor` and write through that same resume**, so they
cannot disagree about a campaign. `monitor` is pure functions of `(graph, campaign_id)` — no
Rich, no Typer, no Flask — holding `state`, `pending_interrupt`, `list_campaigns`,
`checkpoint_document`, the per-checkpoint row builders (`backlog_rows`, `shape_rows`,
`champion_rows`, `conclusion_rows`), `report_rows` and `rewind_targets`.

`src/amt/web/` is the primary surface: `app.py` routes, `runner.py` one thread per campaign
guarded by a per-campaign lock (a second resume while one is live returns 409), and
server-rendered templates with a five-second meta refresh only while a runner is live. It
binds to 127.0.0.1 with the operator's `.env` and adds no auth. `rewind_targets` decides
which rewinds to offer by *calling* `rewind_update` and catching its refusal, so the offered
buttons cannot drift from what the action permits.

```
amt web                            # the campaign list, campaign pages, reports
amt doctor --tables a,b            # auth, warehouse, volume, cluster spec, MLflow, LLM
amt config                         # resolved settings, masked, with each value's source

amt start "Improve homepage row ranking..." --campaign my-campaign
   → runs to HC2, prints ProblemSpec, exits with "awaiting approval"
amt run my-campaign                # or drive every checkpoint inline from one terminal
amt approve my-campaign [--keep H1,H3] [--drop H2]
amt reject my-campaign -m "use 30-day label window"   # feeds back, re-runs stage
amt show my-campaign [--key K]     # the checkpoint document in full
amt rewind my-campaign --to backlog|feature|qa -m "<why>"
amt status my-campaign
amt list
amt report my-campaign             # prints ScientificConclusion, MLflow links
amt promote my-campaign            # HC4 decision: registers model alias
```

Checkpointer DB at `~/.amt/checkpoints.sqlite`; thread_id = campaign_id. The saver uses
`JsonPlusSerializer(allowed_msgpack_modules=[...])` listing the `amt.schemas.*` modules, and
the connection sets `PRAGMA busy_timeout=5000` because the web process reads the same
database while a runner writes it.

## 9. Enforcement mechanisms (beyond prompts)

1. Immutable splits: materialized Delta tables; contract stores table name + Delta version; jobs read `VERSION AS OF`.
2. `SplitGuard` role scoping in job code (train jobs physically cannot load QA).
3. `contract_hash` stamped at HC3; every node and job asserts equality.
4. Graph routing prevents QA before freeze and prevents trainer re-entry after freeze.
5. `FrozenChampion` is a frozen Pydantic model (`model_config = ConfigDict(frozen=True)`); QA takes exactly one.
6. Harness version pinned in contract; jobs assert.
7. The backlog reaches the Trainer only through HC_BACKLOG: `planner_select_hypothesis` raises rather than inventing hypotheses when state carries none.
8. `feature_fit_hash` approved at HC_FEATURE; `execute_training` asserts equality, and the booster runtime accepts only the closed transform vocabulary (no SQL, no agent-authored Python) and never a `qa` source.
9. The Trainer cannot change an approved shape: `_bind_feature_fit` clamps rather than routes, and records the rejected ask.
10. A hypothesis outside the runtime vocabulary is marked `infeasible` by the graph and never reaches selection; one a human drops is `abandoned` and is not resurrected by a rewind.
11. Proxy sanity precedes candidate metrics in the QA Phase 1 job, and a failure routes to `SPEC_REVIEW` before any candidate number is read.
12. The conclusion's `recommendation` is clamped to `validation_verdict`: a champion that did not beat the incumbent reads `hypothesis_rejected` regardless of the narrative. The verdict resolves the baseline through `metric_value`, which tolerates the `val/` prefix that `train.py` logs under — a raw lookup across that gap silently reported every champion as losing.
13. Rewinding into training is refused once a champion is frozen (`rewind_update` raises); only `--to qa` remains, and it clears QA's own results without touching the artifact.
14. Optional (documented in README): second SP for QA jobs with grants only on `split_qa`.

## 10. Swapping an agent

Each agent method is a pure function from a context dict to exactly one Pydantic schema,
built by `AgentBase.agent(key, prompt_name, schema)`. The prompt is swappable, the returned
schema is the contract, and `graph/routing.py` plus the node assertions are what make a bad
swap fail loudly instead of quietly producing a worse campaign. Models are named
`provider:model_id` and built only in `src/amt/llm.py`; nothing else may import a provider SDK.

Prompts live in `src/amt/agents/prompts/`. The table below is verified against the source —
note `planner_final_champion.md` (not `planner_champion.md`) and `qa_phase0_triage.md` /
`qa_decision.md` (not `qa_triage.md` / `qa_decide.md`).

| Agent method | Prompt | Must return | Model env var |
|---|---|---|---|
| `ProblemSpecAgent.investigate` | `problem_spec_probe.md` | `ProbeRequest` | `AMT_PROBLEM_SPEC_MODEL` |
| `ProblemSpecAgent.draft_spec` | `problem_spec.md` | `ProblemSpec` | `AMT_PROBLEM_SPEC_MODEL` |
| `ProblemSpecAgent.draft_contract` | `evaluation_contract.md` | `EvaluationContractDraft` | `AMT_PROBLEM_SPEC_MODEL` |
| `PlannerAgent.propose_hypotheses` | `planner_hypotheses.md` | `HypothesisBacklog` | `AMT_PLANNER_MODEL` |
| `PlannerAgent.propose_feature_fit` | `planner_feature_fit.md` | `FeatureFitProposal` | `AMT_PLANNER_MODEL` |
| `PlannerAgent.build_plan` | `planner_plan.md` | `ExperimentPlan` | `AMT_PLANNER_MODEL` |
| `PlannerAgent.select_final_champion` | `planner_final_champion.md` | `FinalChampionSelection` | `AMT_PLANNER_MODEL` |
| `PlannerAgent.write_conclusion` | `planner_conclusion.md` | `ScientificConclusion` | `AMT_PLANNER_MODEL` |
| `TrainerAgent.propose_config` | `trainer_propose.md` | `ExperimentConfig` | `AMT_TRAINER_MODEL` |
| `TrainerAgent.interpret` | `trainer_interpret.md` | `TrainerInterpretation` | `AMT_TRAINER_MODEL` |
| `TrainerAgent.select_champion` | `trainer_champion.md` | `ChampionSelection` | `AMT_TRAINER_MODEL` |
| `QAAgent.triage_phase0` | `qa_phase0_triage.md` | `Phase0Triage` | `AMT_QA_MODEL` |
| `QAAgent.decide` | `qa_decision.md` | `QAAssessment` | `AMT_QA_MODEL` |

`Settings.model_for(agent)` resolves the four keys (`planner`, `trainer`, `qa`,
`problem_spec`) and falls back to the planner's model for anything unrecognised.

Three things a prompt swap cannot do, because code and not the prompt decides them:

- **It cannot widen the vocabulary.** `runtime_vocabulary()` is rendered into the Planner's
  context, and a hypothesis or recipe outside it is marked `infeasible` or rejected by
  `FeatureRecipe` validation. A prompt that invents a transform produces a caught error, not
  a new capability.
- **It cannot change an approved shape.** `_bind_feature_fit` keeps the structural payload a
  human signed and takes only `booster_params`, whatever the Trainer returns.
- **It cannot upgrade a losing result.** `conclusion_node` clamps `recommendation` to the
  measured `validation_verdict`.

Swapping the *executor* is the same shape of change: implement the five-method `Executor`
protocol (§5). `LocalMockExecutor` and `FakeLLM` in `tests/conftest.py` are the reference
implementations, and the integration test drives the whole graph through both.

## 11. Implementation phases (Opus)

- **Phase A — core**: pyproject, settings, schemas, state, databricks clients, mlflow helpers, executor protocol + LocalMockExecutor, unit tests for schemas/hashing.
- **Phase B — orchestration**: agents (prompts + structured output), nodes, routing, build_graph, CLI, integration test with fake LLM + mock executor covering: approve/reject loops, trainer convergence, freeze, QA pass/fail/spec_review paths.
- **Phase C — Databricks jobs**: materialize_splits, popularity + sasrec, harness, paired bootstrap, train/qa entrypoints, SplitGuard, DatabricksExecutor, wheel build/upload, synthetic demo data.
- **Phase D — verify**: ruff, pytest, graph compile, end-to-end dry run with mock executor; README quickstart.

Phases A/B and C are independent and run in parallel.
