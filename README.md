# AMT — Automated Model Trainer & QA Agent

Autonomous ML experimentation on Databricks, run as a closed-loop scientific process:
LangGraph orchestrator (local CLI) + Databricks Jobs for compute + MLflow for persistence.

See `docs/SPEC.md` (normative product spec) and `docs/PLAN.md` (architecture).

## Quickstart

```bash
uv sync --all-extras            # or: uv sync   (skips the heavy `jobs` extra)
cp .env.example .env            # fill in Databricks + Anthropic credentials
uv run amt --help
```

Dry-run the entire campaign with no network, no LLM and no Databricks:

```bash
uv run amt start "Improve homepage row ranking" --campaign demo --mock
uv run amt approve demo --mock          # HC2: ProblemSpec approved
uv run amt approve demo --mock          # HC3: EvaluationContract approved -> autonomous
uv run amt status demo
uv run amt report demo
uv run amt promote demo --mock          # HC4: promotion decision
```

Reject with feedback to re-run a stage:

```bash
uv run amt reject demo -m "use a 30-day label window" --mock
```

## Human checkpoints

| HC | Where | CLI |
|---|---|---|
| 1 | initial product problem | `amt start "<problem>"` |
| 2 | ProblemSpec approval | `amt approve` / `amt reject -m ...` |
| 3 | EvaluationContract approval | `amt approve` / `amt reject -m ...` |
| 4 | scientific conclusion & promotion | `amt report`, `amt promote` |

Everything between HC3 and HC4 runs autonomously inside the agreed experiment budget.

## Layout

```
src/amt/schemas/    Pydantic v2 contracts — the only cross-agent language
src/amt/agents/     LLM layers: pure functions (inputs) -> schema
src/amt/graph/      CampaignState, nodes (side effects), routing (hard rules)
src/amt/executor/   Executor Protocol, DatabricksExecutor, LocalMockExecutor
src/amt/databricks/ WorkspaceClient, read-only SQL, MLflow helpers
src/amt/jobs/       Code that runs ON Databricks (never imports LangGraph/LLM code)
```

## Enforcement (not prompts)

1. Splits are materialized once and referenced by Delta version + membership hash.
2. `SplitGuard` scopes split access per job role; train jobs cannot load `qa`.
3. `contract_hash` is stamped at HC3; every node and job asserts equality.
4. Routing forbids QA before champion freeze, and trainer re-entry after freeze.
5. `FrozenChampion` is an immutable Pydantic model; QA accepts exactly one.
6. `harness_version` is pinned in the EvaluationContract and asserted by all jobs.
7. Optional hardening: a second service principal for QA jobs, granted only on the QA split table.

## Development

```bash
uv run ruff check . && uv run ruff format --check .
uv run pytest -q
```

Unit tests never hit the network: they use `LocalMockExecutor` and `FakeLLM`.
