# ADR 0002 — Compute is a content-addressed wheel run as one-time jobs

**Status:** accepted · **Date:** 2026-09-11 · **Scope:** `src/amt/databricks/{wheel,jobs}.py`, `src/amt/executor/databricks.py`, `src/amt/jobs/entry.py`

## Context

The Trainer agent iterates: propose a config, run it, interpret, propose
again. Each iteration is a unit of Databricks compute. Three options were
available:

- **Notebook tasks** with generated source. Easy to submit, but the executed
  code is a string assembled at runtime — unversioned, untestable locally,
  and impossible to assert anything about. Reproducing an experiment means
  reproducing a notebook body.
- **A persistent Databricks job** that is updated per run. Leaves stale job
  definitions in the workspace and makes "what exactly ran" a function of
  workspace state at a point in time.
- **A wheel built from this repo, submitted as a one-time run.**

The SPEC requires every result to carry reproduction lineage and requires the
evaluation harness to be a single pinned implementation. That pushes hard
towards shipping the repo itself rather than generating code.

## Decision

Compute is `jobs.submit` (one-time runs) of a `python_wheel_task` with
`package_name="amt"`, `entry_point="amt-job"`, and a flat `--flag value`
parameter list.

1. **The wheel is content-addressed.** `wheel.source_tree_hash()` hashes
   `src/` and `pyproject.toml`; the artifact is uploaded to
   `{volume}/wheels/{name}-{hash12}.whl`. An unchanged tree resolves to an
   already-uploaded wheel and the build is skipped — otherwise every trainer
   iteration would pay a build-and-upload round trip. Docs and tests are
   excluded from the hash deliberately: they cannot change job behaviour.

2. **Large payloads go to the Volume, not the command line.** The
   EvaluationContract, ExperimentConfig and FrozenChampion are uploaded as
   JSON under `{volume}/campaigns/{campaign_id}/{job_run_id}/` and passed by
   path. `entry.load_json_arg` accepts either a path or an inline JSON
   string, so local dry runs stay convenient. The uploaded payload doubles as
   the audit record of exactly what the job was given.

3. **Results come back through MLflow, not the job API.** The executor
   generates a `job_run_id` UUID, passes it as `--job-run-id`, and the job
   stamps it as the `amt.job_run_id` tag. After the run reaches a terminal
   state the executor searches the campaign experiment for that tag and
   downloads the `result.json` artifact. This keeps MLflow as the single
   persistence layer (SPEC stage 12, "Minimal Persistence Principle") and
   avoids depending on driver stdout parsing.

4. **`result.json` is a plain dict with a fixed key set**, converted to the
   Pydantic schema on the orchestrator side. The job package must not import
   the orchestrator's dependency tree, and a plain-dict boundary lets the
   schemas evolve without breaking wheels already uploaded to a Volume.

5. **Job code is import-isolated.** `amt.jobs.*` imports no LangGraph,
   LangChain or LLM client, and imports MLflow, Spark and torch lazily. The
   wheel therefore installs on a cluster without the orchestrator's
   dependencies, and `import amt.jobs.entry` works in a bare test
   environment.

## Consequences

- What ran is a git-identifiable source tree plus a recorded parameter list,
  not a generated string. `amt.code_version` on the MLflow run completes the
  lineage.
- Trainer iterations after the first cost only submission + cluster startup.
- A source change invalidates the cache for every role at once, which is the
  correct behaviour when the harness is part of the hash: a harness change
  must not be mixed into an in-flight campaign's results.
- Cost: the trainer agent cannot emit arbitrary architecture code for a
  single run. The SPEC already prefers this ("For supported families, change
  configuration rather than generating a new training system"); a genuinely
  new architecture is a repo change, reviewed and versioned, which is the
  outcome we want.
- `existing_cluster_id` in settings bypasses job-cluster startup for
  development; production uses a fresh job cluster per run so no state leaks
  between experiments.
