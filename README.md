# AMT — Automated Model Trainer & QA Agent

Autonomous ML experimentation on Databricks, run as a closed-loop scientific process:
LangGraph orchestrator (local CLI) + Databricks Jobs for compute + MLflow for persistence.

See `docs/SPEC.md` (normative product spec) and `docs/PLAN.md` (architecture).

## Quickstart

```bash
uv sync --all-extras            # or: uv sync   (skips the heavy `jobs` extra)
cp .env.example .env            # fill in Databricks credentials + an LLM provider key
uv run amt --help
```

Dry-run the entire campaign with no network, no LLM and no Databricks:

```bash
uv run amt start "Improve homepage row ranking" --campaign demo --mock
uv run amt approve demo --mock          # HC2: ProblemSpec approved
uv run amt approve demo --mock          # HC3: EvaluationContract approved -> baseline
uv run amt approve demo --mock          # HC_BACKLOG: hypotheses approved -> autonomous
uv run amt status demo
uv run amt report demo
uv run amt promote demo --mock          # HC4: promotion decision
```

Reject with feedback to re-run a stage:

```bash
uv run amt reject demo -m "use a 30-day label window" --mock
```

`amt run demo --mock` is the same thing as one interactive session.

## The browser surface

```bash
uv sync --extra web
uv run amt web            # http://127.0.0.1:8765
```

Same graph, same checkpointer, same decisions: the campaign page renders the pending
checkpoint with its verbs, a progress log of every node as it completes (including the
Databricks run URL), the hypothesis table and the experimentation report. The CLI stays
for scripting and `--mock`.

**No authentication, by design.** It runs on your machine as you, with your `.env`, and
binds to `127.0.0.1`. One operator, one process. Multi-user or hosted deployment is out
of scope — there is no tenancy, no authorization and no audit of who answered a
checkpoint, so do not put it behind a public address.

## Human checkpoints

| HC | Where | CLI |
|---|---|---|
| 1 | initial product problem | `amt start "<problem>"` |
| 2 | ProblemSpec approval | `amt approve` / `amt reject -m ...` |
| 3 | EvaluationContract approval | `amt approve` / `amt reject -m ...` |
| — | hypothesis backlog (`HC_BACKLOG`) | `amt approve` / `amt reject -m ...` |
| — | every booster's feature/fit shape, at once (`HC_FEATURE`) | `amt approve [--keep/--drop]` |
| — | each hypothesis champion (`HC_CHAMPION`) | `amt approve [--freeze]` |
| 4 | scientific conclusion & promotion | `amt report`, `amt promote` |

The four numbered checkpoints are the ones SPEC.md requires. The other three are additions:
`HC_BACKLOG` decides which hypotheses spend the budget, `HC_FEATURE` decides what every
booster model is allowed to see (all shapes in one review, not one gate per hypothesis), and
`HC_CHAMPION` shows each hypothesis's champion against the incumbent before the next
hypothesis runs. The Trainer loop *inside* a hypothesis stays autonomous, which is the point
of the band between HC3 and HC4.

At the three checklist gates `--keep`/`--drop` edit the list in place; rejecting costs a round
trip through the agent, so it is for "redraft this", not "not that one".

## Layout

```
src/amt/schemas/    Pydantic v2 contracts — the only cross-agent language
src/amt/llm.py      provider:model_id -> chat model; the only place a provider SDK is imported
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

## Status / known gaps

The repo was built in a sandbox without PyPI access, so `pytest` has only been run against a
stdlib shim for `src/amt/jobs` and statically verified elsewhere. First thing to do locally:
`uv sync --all-extras && uv run ruff check . --fix && uv run ruff format . && uv run pytest -q`.

Known follow-ups (tracked in `docs/PLAN.md` §10 Phase D):

- `config_hash` is computed two ways (`ExperimentConfig.config_hash` excludes prose/ids; the
  job hashes the whole uploaded dict). Unify on the schema definition in `jobs/entry.py`.
- Phase-0 reproduction tolerance defaults differ (`1e-6` in jobs vs `1e-3` in `Phase0Result`);
  pass the contract's tolerance through the executor.
- Jobs ignore `EvaluationContract.metrics` when choosing `ks` (falls back to 5/10/20).
- `Phase1Result.cohort_deltas` is not populated with per-cohort CIs yet.
- Real Databricks run has not been exercised end-to-end; start with `amt demo seed`.
