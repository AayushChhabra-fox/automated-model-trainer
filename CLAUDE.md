# AMT — Automated Model Trainer & QA Agent

Autonomous ML experimentation system on Databricks: LangGraph orchestrator (local CLI) + Databricks Jobs for compute + MLflow for persistence.

## Read first
- `docs/SPEC.md` — normative product spec. Scientific constraints there are hard rules.
- `docs/PLAN.md` — architecture, repo layout, state, graph topology, executor contract, enforcement.

## Conventions
- Python ≥3.10, `uv`, `src/` layout, package `amt`. Pydantic v2 for all cross-agent contracts.
- Agents (`src/amt/agents/`) are pure functions: inputs → Pydantic schema via structured output. No side effects.
- Nodes (`src/amt/graph/nodes/`) own side effects (executor calls, MLflow) and return state deltas.
- Anything that executes on Databricks lives in `src/amt/jobs/` and must not import LangGraph/LLM code.
- Split access in jobs goes through `SplitGuard` only. Train/validation jobs never receive `qa` in `allowed_splits`.
- Every node and job asserts `contract_hash` matches the frozen EvaluationContract.
- The contract carries the fields materialization executes (`dataset.source_table`, `dataset.columns`, `dataset.filters`, `split_policy` boundaries); its metrics are limited to what the harness computes — `{recall|ndcg|hitrate|mrr|coverage}@k`.
- Never duplicate MLflow detail into `CampaignState`; store IDs/references.
- Secrets only via env (`DATABRICKS_HOST`, `DATABRICKS_CLIENT_ID`, `DATABRICKS_CLIENT_SECRET`, `AMT_LLM_API_KEY`).
- Agents are provider-agnostic: models are named `provider:model_id` and built in `src/amt/llm.py`. Never import a provider SDK outside that module.
- The hypothesis backlog is written once, after the baseline, and approved at `HC_BACKLOG` before any hypothesis is planned. `planner_select_hypothesis` selects; it never generates.
- The booster families (`xgboost_*`, `lightgbm_*`) take their inputs from a `FeatureRecipe` + `FitSpec` approved at `HC_FEATURE`. Only the closed transform vocabulary is executable, and `execute_training` refuses a config whose `feature_fit_hash` is not the approved one.

## Commands
```
uv sync --all-extras
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
uv run amt --help
```

## Testing
- Unit tests must not hit network. Use `LocalMockExecutor` and `FakeLLM` (in `tests/conftest.py`).
- Integration test drives the full graph through all HC interrupts and QA outcomes.
