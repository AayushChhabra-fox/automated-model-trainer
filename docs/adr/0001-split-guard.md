# ADR 0001 — Split access is enforced by code, not by prompt

**Status:** accepted · **Date:** 2026-09-11 · **Scope:** `src/amt/jobs/data/loaders.py`, `src/amt/jobs/{train,qa}.py`, `src/amt/executor/databricks.py`

## Context

The SPEC's hardest constraints are negative: never train on validation, never
train on QA, never tune against QA, never evaluate different models on
different populations. These are exactly the constraints that an LLM-driven
system is worst at keeping, because every one of them can be violated by a
single plausible-looking line of generated data-loading code — and the
violation is invisible in the output. A model that has seen the holdout
simply looks good.

Prompt instructions ("do not read the QA split") are unfalsifiable at
runtime: nothing checks them, and a failure produces a *better* number, so
nothing downstream looks suspicious either. The SPEC anticipates this and
asks for "enforcement via Databricks permissions, common evaluation code,
immutable split materialization, and orchestration controls over prompt-only
enforcement."

## Decision

All split access inside job code goes through one object, `SplitGuard`,
constructed from the `--allowed-splits` job parameter. There is no other
supported read path.

1. **The executor decides scope; the job enforces it.** `ROLE_SPLITS` in
   `executor/databricks.py` maps each role to its grant — train and baseline
   get `train,validation`; QA phase 0 gets `validation`; QA phase 1 gets `qa`
   and nothing else. The grant travels as a CLI parameter, so it is visible
   in the Databricks run record and in the `amt.allowed_splits` MLflow tag.

2. **Jobs assert their own scope before touching data.** `train.py` calls
   `require_subset_of(("train","validation"))` and `qa.phase1` calls
   `require_exactly(("qa",))`. A misconfigured submission fails at startup
   rather than producing a contaminated result. `require_exactly` in phase 1
   matters specifically because it makes "peek at QA, then go back and
   re-tune on validation" impossible inside a single job: the job cannot
   read validation at all.

3. **Reads are pinned to the contract's Delta version.** `read()` issues
   `SELECT * FROM {table} VERSION AS OF {delta_version}`, using the
   `SplitRef` recorded in the frozen EvaluationContract. Source tables keep
   receiving data; the campaign's ground truth does not move.

4. **Row counts are verified against the contract on every read.** If a
   split has been rewritten, the job raises `SplitContractMismatch` instead
   of silently evaluating on a different population. Combined with the
   `membership_hash` produced at materialization time, this makes population
   drift a loud failure.

5. **Split names are normalized through a fixed alias table.** `test`,
   `holdout` and `split_qa` all resolve to `qa`; an unrecognized name is a
   `SplitAccessViolation`, not a silently-empty read.

A pandas/parquet fallback exists for unit tests and local dry runs. It goes
through the identical guard and the identical row-count check — the fallback
is a different *storage* path, never a different *permission* path.

## Consequences

- The unit tests can assert the negative constraints directly
  (`test_split_guard.py::test_train_job_cannot_read_qa`), which no amount of
  prompt review can do.
- Adding a model family or a new job role cannot accidentally widen access:
  the grant lives in the executor's role table, not in model code.
- The guard is defence in depth, not a substitute for permissions. PLAN §9.7
  still recommends a second service principal whose grants cover only
  `split_qa`; `SplitGuard` is the layer that works before that is set up, and
  the layer that produces a comprehensible error when it is.
- Cost: job code cannot read a split opportunistically for debugging. That is
  the point.
