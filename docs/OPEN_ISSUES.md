# Open issues at the end of Phase 8

**Written:** 2026-09-16. **Tree:** `35dc686`, clean. 430 passed, 11 skipped.

Every item is something a reader could otherwise discover the expensive way.
Phases 1–7 are committed, Phase 8's runtime is proven on real data, Phase 9's
documentation is not written. Nothing below is a blocker for Phase 9; several
items are things Phase 9 must describe *honestly* rather than aspirationally.

Severity: **A** = will produce a wrong answer or lose money. **B** = will waste
an operator's time. **C** = untidy, or a deliberate deferral.

---

## A — correctness and cost

### A1. A Databricks job failure raises out of the graph

`ExecutorError` propagates and the CLI traces back. `RUNBOOK.md` claims the
Trainer "sees `status=failed` and adapts"; it does not. The campaign is
resumable, but the Trainer never learns the run failed and cannot route around
a flaky cluster.

Compounding it: **`amt resume` re-submits rather than re-polls.** No Databricks
run id is persisted, so an interrupted node starts a fresh job and you can pay
twice for the same work. During Phase 8 this cost two duplicate runs (~$1) and
about twenty minutes.

*Fix:* persist `databricks_run_id` in state at submit time; on resume, poll the
existing run before submitting. Then catch `ExecutorError` in
`execute_training` and hand the Trainer a failed `TrainingRunResult`.

### A2. A failed iteration still consumes budget

`record_iteration` is called with the run's wall clock whatever the outcome, so
cluster flakiness eats the compute budget. Defensible, but it means a campaign
can exhaust its budget having learned nothing.

### A3. Contract cohort names are still decorative

The contract names cohorts (`client_name`, `history_length`); the job computes
what it can. Phase 8 fixed this for `grouped_rerank` — `cohorts_for` reports
the dimensions the contract named *and* materialization carried onto the row,
and reports nothing otherwise, which is why the FYL baseline has real per-client
numbers. **`next_item` mode is unchanged:** it still computes fixed
`history_length` / `recency` buckets regardless of what the contract asked for,
so `max_cohort_regression` cannot be evaluated against the contract's own
cohorts there. `CohortDef.expression` remains unexecuted.

### A4. Acceptance criteria are enforced by an LLM, not by code

`AcceptanceCriteria` is prose in the QA agent's prompt; the PASS/FAIL is the
model's judgement. The SPEC's "pre-registered" language implies the opposite.
Phase 6 moved *one* criterion into code — the proxy-sanity gate returns
`SPEC_REVIEW` before the agent is consulted — which is the pattern the rest
should follow: compute PASS/FAIL from `Phase1Result` and let the agent write
only the narrative.

### A5. Jobs do not enforce `feature_fit_hash`

`CLAUDE.md` says `execute_training` refuses an unapproved hash. That check lives
only in the orchestrator (`assert_feature_fit_approved`); `train.py` logs the
hash and fits whatever it was given. A job submitted by hand bypasses the gate.

### A6. `catalog_size` can be 0 in QA phase 0

Silently corrupts reproduced `coverage@k`. Unchanged since it was first noted.

### A7. Striking every shape at HC_FEATURE ends in a traceback

Approving with all shapes dropped is a legitimate operator action ("none of
these is worth compute"). The campaign then has no champion and
`planner_final_champion` raises `ContractViolation: nothing to freeze`. Honest,
but a traceback where a terminal state belongs.

---

## B — operator time

### B1. `--mock` cannot dry-run a booster or a grouped campaign

`cli._build` calls `mock_stack()` with no arguments, so every `--mock` campaign
is `next_item` + `sasrec`. The two paths most worth rehearsing — the batched
HC_FEATURE gate and `grouped_rerank` — are unreachable from the CLI. The test
suite covers both; an operator cannot.

*Fix:* `--mock-family` and `--mock-mode` flags threaded into `mock_stack`.

### B2. The `Settings` default node type is from the wrong cloud

`node_type_id` defaults to `Standard_DS3_v2`, an Azure type, in an AWS
workspace. `amt doctor` now catches it against `clusters.list_node_types()`, so
it is no longer silent — but a fresh clone still defaults to something that
cannot launch.

### B3. QA phase 0's "fix and rerun" loop applies no fix

`qa_phase0` stores `fix_applied` from triage and routes back to itself with the
identical `FrozenChampion`. Three identical attempts, then `INVALID_EXPERIMENT`.
The retry cap is now configurable and honoured (Phase 1, B5); the retry itself
still does nothing.

### B4. `amt promote` advances even when QA did not pass

It warns, skips registration, and resumes HC4 anyway. Arguably right, but the
warning is easy to miss in a terminal.

### B5. The executor's QA mappings are thinly tested

`_to_phase0_result` maps a subset of the job's check names, and the mock uses
*different* names than the job, so the integration test cannot catch a
mismatch. Phase 8's `_keep`-on-nested-refs fix removed one class of failure
here (a job-side addition can no longer break the orchestrator's read), but the
name mismatch remains. Phase 1 removed the three never-populated `Phase0Result`
fields; `reproduction_metrics` is still never populated from job output.

### B6. Local materialization only rewrites `=` to `==`

Any other SQL filter syntax fails in the pandas path. The Spark path is fine.
The FYL contract's hash-sampling predicate is one such filter, so that contract
cannot be dry-run locally.

---

## C — untidy, or deferred on purpose

### C1. Repo-wide lint and format debt

`ruff check .` reports **52 findings**; `ruff format --check .` would rewrite
**27 files**. None of it is from Phases 1–8: every phase was scoped-formatted
per `.cursor/skills/scoped-formatting`, and each commit verified its own file
set against `HEAD`. Clearing it is a mechanical one-commit job that should not
be mixed into a feature change.

### C2. 11 tests skip without `libomp`

`brew install libomp` un-skips the six xgboost and five lightgbm cases. Nobody
has. The boosters have therefore **never been fitted on this machine**, and
never on a cluster either — the FYL campaign's first booster run will be their
first execution anywhere.

### C3. `days_since_air_date` is unusable for this container

100% NULL for `parent_id = 671107897793` (45% populated table-wide). The plan's
suggested hypothesis list names it as a feature. `duration_seconds`,
`content_type`, `genres_array`, `network_name` and `is_sportingevent` are all
100% populated. Planner guidance, not code.

### C4. State fields written but never read

`final_selection_rationale`, `promotion_decision`, `champion_reviewed`;
`errors` is declared and never written, and the `awaiting_human` status value is
never set. `CohortDef.min_support`, `ExperimentConfig.custom_code_ref` and
`MetricDef.implementation` are prose agents read, not instructions anything
follows. `FeatureRecipe.joins` and `RecipeSource.join_keys` were deleted in
Phase 3 for exactly this reason; these are the survivors.

### C5. `contract_hash` is computed two ways

`ExperimentConfig.config_hash` excludes prose and ids; the job hashes the whole
uploaded dict. Pre-existing, noted in the README, unresolved.

### C6. Phase-0 reproduction tolerance differs between layers

`1e-6` in the jobs, `1e-3` in `Phase0Result`. The contract's tolerance is not
threaded through the executor.

### C7. No second service principal for QA

PLAN §9.7 wants QA's grants limited to `split_qa` so the negative constraints
hold at the permission layer too. `SplitGuard` is still the only enforcement.

### C8. The web surface has no authentication

Deliberate and documented in the README: localhost, operator's credentials, one
process. There is no tenancy, no authorization and no record of *who* answered
a checkpoint. It must not go behind a public address. Listed here so it is a
decision rather than an oversight.

---

## Phase 8, specifically

### What is proven on real data

Materialization of 62.6M impressions into render-grain splits; per-client
position propensities estimated on train only; the `logged_order` incumbent
scored through the harness at the contract's own cutoffs; per-client cohorts;
job clusters with B9 cost tags; the wheel installing from a UC Volume.

```
train       967,119 renders   386,520 with a conversion
validation  498,543 renders   203,779
qa          517,920 renders   192,591

incumbent   ndcg@4 0.86430   hitrate@1 0.69641   mrr@4 0.81487
theta       slot 0 = 1.00, slot 1 = 0.11-0.18, slot 2 = 0.037-0.064,
            slot 3 = 0.026-0.046  (per client, train window only)
```

### What is not

- **The agent half has never run**: HC1 → HC4 with the LLM drafting the spec and
  contract, the planner proposing hypotheses, the Trainer looping. The prompts
  now describe `grouped_rerank` (`35dc686`), but no agent has been asked to
  produce such a contract. `docs/RUN_FYL_CAMPAIGN.md` is the operator guide.
- **No booster has ever been fitted**, here or on a cluster (C2).
- **QA phase 0 and phase 1 have never executed for real** in any mode. The
  proxy-sanity gate, the product-unit arithmetic and the A/B sizing are unit
  tested and have never seen a real `Phase1Result`.
- **The experimentation report has never been rendered from a real campaign.**
- **The propensities' effect on a fitted model is unmeasured.** We know the
  weighting changes the incumbent's own score (`hitrate@1` 0.696 weighted
  against ~0.78 unweighted). We do not know whether a booster can beat it.

### Cost to date

Eight job runs, ~80 minutes of i3.2xlarge ×2, roughly **$4**, all tagged
`{amt, amt_role, amt_campaign_id}`. Three of the eight were failures that found
defects. Nothing touched a production table; splits are in `product_api_dev.amt`.

### Still owned by the operator

1. Confirm the three cut dates. Verified to give whole Sun–Sat weeks and exactly
   7 dashboard-overlapping days, but the choice is a human's.
2. Confirm CTV-only for v1.
3. Decide whether the dashboard team should be told the campaign's label is
   stricter than their numerator (playback within 90s vs a click) **before** the
   report circulates. The proxy-sanity check measures the gap; it does not
   excuse not mentioning it.
