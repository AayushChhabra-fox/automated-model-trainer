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
                            └approved→ run_baseline → planner_select_hypothesis
planner_select_hypothesis ─no hypotheses left / budget→ planner_final_champion
                          ─tabular family, shape not approved→ planner_feature_fit
                          └→ trainer_propose → execute_training → trainer_interpret
planner_feature_fit → [interrupt HC_FEATURE] ─rejected→ planner_feature_fit
                                             └approved→ trainer_propose
trainer_propose ─feature_fit_hash != approved→ planner_feature_fit
trainer_interpret ─continue→ trainer_propose
                  └converged→ select_hypothesis_champion → planner_select_hypothesis
planner_final_champion → freeze_champion → qa_phase0
qa_phase0 ─INVALID_EXPERIMENT (fixable, attempts<3)→ qa_phase0
          └pass→ qa_phase1 → qa_decision → conclusion → log_final_mlflow → [interrupt HC4] → END
qa_decision = SPEC_REVIEW → conclusion (status=spec_review) → [interrupt HC4] → END
```

Rules enforced in `routing.py`, not prompts: trainer loop cannot exceed `plan.max_iterations`; `execute_training` refuses if `contract_hash` differs from the one stamped at HC3; `qa_*` refuses if `frozen_champion is None` or more than one champion; `execute_training` refuses a booster family whose `feature_fit_hash` is not the one approved at HC_FEATURE.

HC_FEATURE gates the declarative feature/fit shape of the booster families: one approval per hypothesis, hyperparameter-only iterations run unattended, and any structural change to the `FeatureRecipe` or `FitSpec` re-enters the checkpoint. `popularity` and `sasrec` skip it.

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

Job parameters include `--role train|qa|baseline|materialize` and `--allowed-splits`. `SplitGuard` in `jobs/data/loaders.py` raises if code requests a split outside `allowed_splits`. Train jobs get `train,validation`; QA jobs get `qa` (+ `validation` only for Phase-0 reproduction subset).

## 6. Evaluation harness

Single implementation in `jobs/eval/harness.py`, `HARNESS_VERSION = "1.0.0"` recorded in the EvaluationContract and asserted by both train and QA jobs. Metrics: Recall@K, NDCG@K, HitRate@K, MRR, item coverage. Inputs: per-user ranked item list + ground-truth set; cohort labels optional. `paired.py`: per-user metric deltas, paired bootstrap 95% CI, fraction of users improved.

QA independence: QA job runs `qa.py` which loads the model via MLflow pyfunc (not the trainer's in-memory object) and computes metrics through the harness — same math, different code path for inference.

## 7. MLflow conventions

- One experiment per campaign: `/Shared/amt/{campaign_id}`.
- Tags on every run: `amt.campaign_id`, `amt.stage`, `amt.role` (baseline|train|qa0|qa1|final), `amt.hypothesis_id`, `amt.contract_hash`, `amt.spec_hash`, `amt.harness_version`, `amt.config_hash`, `amt.code_version` (git sha).
- Train runs log: params (flattened config), metrics (`train/*`, `val/*`, `val/cohort/*`), artifacts (`model/` pyfunc, `config.yaml`, `result.json`).
- Final run (Stage 12): re-logs frozen model reference, QA metrics, baseline deltas, `conclusion.md`, `problem_spec.json`, `evaluation_contract.json`. Optionally registers to UC model registry `{catalog}.{schema}.{model_name}` with alias `candidate` when QA=PASS.

## 8. Human checkpoints (CLI)

```
amt start "Improve homepage row ranking for Tubi..." --campaign my-campaign
   → runs to HC2, prints ProblemSpec, exits with "awaiting approval"
amt approve my-campaign            # resumes; runs to HC3
amt reject my-campaign -m "use 30-day label window"   # feeds back, re-runs stage
amt approve my-campaign            # resumes; autonomous until HC4
   → booster families stop once more at HC_FEATURE with the proposed
     FeatureRecipe + FitSpec; approve / reject the same way
amt status my-campaign
amt report my-campaign             # prints ScientificConclusion, MLflow links
amt promote my-campaign            # HC4 decision: registers model alias
```

Checkpointer DB at `~/.amt/checkpoints.sqlite`; thread_id = campaign_id.

## 9. Enforcement mechanisms (beyond prompts)

1. Immutable splits: materialized Delta tables; contract stores table name + Delta version; jobs read `VERSION AS OF`.
2. `SplitGuard` role scoping in job code (train jobs physically cannot load QA).
3. `contract_hash` stamped at HC3; every node and job asserts equality.
4. Graph routing prevents QA before freeze and prevents trainer re-entry after freeze.
5. `FrozenChampion` is a frozen Pydantic model (`model_config = ConfigDict(frozen=True)`); QA takes exactly one.
6. Harness version pinned in contract; jobs assert.
7. `feature_fit_hash` approved at HC_FEATURE; `execute_training` asserts equality, and the booster runtime accepts only the closed transform vocabulary (no SQL, no agent-authored Python) and never a `qa` source.
8. Optional (documented in README): second SP for QA jobs with grants only on `split_qa`.

## 10. Implementation phases (Opus)

- **Phase A — core**: pyproject, settings, schemas, state, databricks clients, mlflow helpers, executor protocol + LocalMockExecutor, unit tests for schemas/hashing.
- **Phase B — orchestration**: agents (prompts + structured output), nodes, routing, build_graph, CLI, integration test with fake LLM + mock executor covering: approve/reject loops, trainer convergence, freeze, QA pass/fail/spec_review paths.
- **Phase C — Databricks jobs**: materialize_splits, popularity + sasrec, harness, paired bootstrap, train/qa entrypoints, SplitGuard, DatabricksExecutor, wheel build/upload, synthetic demo data.
- **Phase D — verify**: ruff, pytest, graph compile, end-to-end dry run with mock executor; README quickstart.

Phases A/B and C are independent and run in parallel.
