# Improvement plan — after `vod-nextitem-001`

Source material: `REVIEW_ITERATION.md` (operator log), the campaign's MLflow runs, and the tree at
`ff651b7`. Two problems dominate the log and the operator's own summary:

1. **The gates cost more than the experiments.** 8 reject/approve rounds at 4 `HC_FEATURE` gates,
   a deterministic bounce loop that consumed no budget and showed nothing in `status`, and one
   undocumented prose side-channel (`FeatureRecipe.notes`) that was the only reason the campaign
   terminated. 1 of 5 hypotheses was actually tested.
2. **The tool needs a second agent to drive it.** Twenty-minute silent waits, a preview that cuts
   off the fields that matter, 17 warning lines per command, no resolved-config readout, and no
   single-session way to answer a checkpoint.

Ordering principle for everything below: reuse first, refactor second, add last. Estimated line
deltas are net, and several items are negative.

---

## A. Gate design — one backlog, one shape review, no bounce loop

### A1. Checklist approval at `HC_BACKLOG` (the operator's request)

`HC_BACKLOG` already shows the whole backlog at once (`make_hc_hypotheses_node`). What it lacks
is a per-item verb: approve is all-or-nothing, so striking one hypothesis means a reject round-trip
through the planner.

- `HumanDecision` gains `keep: list[str] = []` and `drop: list[str] = []` (`graph/human.py`).
  `parse_resume` already maps dict keys; two fields.
- `hc_hypotheses_node`: on approve, hypotheses in `drop` (or not in `keep`, when `keep` is given)
  get `status="abandoned"` — the status already exists in `HypothesisStatus` and
  `planner_select_hypothesis` already filters on `status == "pending"`, so selection needs no change.
  Order of `keep` becomes priority order (rewrite `priority`).
- CLI: `amt approve <c> --keep H1,H3` / `--drop H2`. Two Typer options on the existing `approve`.

Reuse: `HypothesisStatus`, `parse_resume`, `pending` filter. **≈ +25 lines.**

### A2. Review every tabular shape once, at the same time, and stop re-opening it

This is the structural fix for log items 2, 3, 4 and the 8 gate cycles. Today `planner_feature_fit`
runs *per hypothesis, after selection*, and any Trainer config whose structural payload drifts is
routed back to the gate (`route_after_trainer_propose`). That edge is the bounce loop.

Change:

- After `HC_BACKLOG` approval, `planner_feature_fit` runs once for **all** approved tabular
  hypotheses (loop over `pending` tabular hypotheses inside the existing node; call the existing
  `propose_feature_fit` per hypothesis). State gains `feature_fit_proposals: dict[str, FeatureFitProposal]`
  replacing the singular `feature_fit_proposal`.
- `HC_FEATURE` becomes one gate showing all shapes with the A1 checklist verbs. Approving stores
  `approved_shapes: dict[str, (recipe, fit, hash)]`; dropping a shape abandons its hypothesis.
  Hypotheses whose shape the vocabulary cannot express die *here*, before any compute — which is
  exactly what the operator had to fake with the `notes` directive.
- `planner_select_hypothesis` copies the approved shape for the chosen hypothesis into the
  existing `approved_feature_recipe / approved_fit_spec / feature_fit_hash` keys. Downstream
  (`_bind_feature_fit`, `execute_training` refusal) is untouched.
- `_bind_feature_fit`: **clamp instead of bounce.** The approved structural payload always wins;
  the Trainer's structural ask is recorded in `last_interpretation`/notes for the record, not
  honored. Blank `objective`/`eval_metric` inherit the approved values (log item 2, "still worth
  doing"). Delete the `structural_change → re-propose` branch.
- Remove the edge `trainer_propose → planner_feature_fit` and `route_after_hc_feature_fit`'s
  redraft loop for the per-hypothesis case. `feature_fit_needed()` shrinks to "tabular family and
  no approved shape".
- Prompt `trainer_propose.md`: the "copy six fields verbatim" paragraph becomes one sentence
  ("structural fields are fixed; tune `booster_params`"), since code now enforces it.

Gate count for the operator's run drops from 1 + 4 (+ 8 bounces) to 2. The `notes` side channel
stops being load-bearing. Reuse: `propose_feature_fit`, `feature_fit_hash`, `HypothesisStatus`,
A1's verbs. **≈ −20 lines net** (deleted routing branches roughly pay for the loop and dict).

### A3. Make the backlog worth approving

The operator's "hypothesis formation could be better" and log items 7 and 8, in one place:

- **Tell the planner what exists.** Render the literal `TransformKind`, `CandidateStrategy`,
  `BASE_COLUMNS` values and the model registry's families into `planner_hypotheses.md` via the
  context dict `agents/base.py` already builds. Constants are importable from
  `schemas/feature_recipe.py`. Also include the baseline's validation metrics and the ProblemSpec's
  probe summary (item coverage, history-length distribution) so hypotheses are grounded in this
  data, not generic recsys lore.
- **Feasibility check before the human sees it.** `Hypothesis` gains
  `requires: list[str]` (transform kinds / external sources it needs). `planner_hypotheses_node`
  marks any hypothesis whose `requires` ⊄ vocabulary as `status="infeasible"` (one new
  `HypothesisStatus` literal) with the reason, and the gate renders it struck-through. Nothing
  infeasible enters selection.
- **Duplicate-shape detection.** At `HC_FEATURE`, compare each proposal's `feature_fit_hash` with
  every other proposal's and with `experiment_history`; annotate "same shape as H2". Both inputs
  are already in state. Same comparison in `trainer_interpret`'s context so bit-identical
  metrics are called "no-op retune" and count toward convergence, not iterations.
- Delete `FeatureRecipe.joins` / `RecipeSource.join_keys` and `_joins_reference_declared_sources`
  — validated but never executed (log item 7). Dead schema invites the planner to use it.

**≈ +40 lines, −30 lines.**

### A4. Gate payloads that answer the reviewer's questions

- Summary header above the JSON at every checkpoint, via a `summarize(payload)` sibling of the
  existing `checkpoint_document`: for `HC_FEATURE` — family, candidate strategy and `n`,
  **estimated candidate rows = validation `row_count` × `n`** (both already in state), the six fit
  fields, and any duplicate-shape note; for `HC_BACKLOG` — a Rich table (id, title, family,
  primary metric, expected effect, feasibility). Hypotheses stop being 8000 characters of JSON.
- Validate `item_attributes` table names at proposal time with the `sql_probe` the ProblemSpec
  node already receives (`deps.sql_probe`). A recipe pointing at a table that does not exist is
  rejected in the node with the error in the payload, not on the cluster after wheel upload.
- Raise `PREVIEW_CHARS` to 20 000 *and* put `candidate_generation` and `fit_spec` first in the
  dumped document (key order) so they are never below the fold.

Reuse: `checkpoint_document`, `deps.sql_probe`, `SplitMaterialization.validation.row_count`.
**≈ +45 lines.**

---

## B. Correctness holes the log left open (small, do first)

| # | Fix | Where | Δ |
|---|---|---|---|
| B1 | Drop `FIT_KNOBS` keys from `model`/`training` when a recipe is present — same partition loop that already handles `loss` | `jobs/models/tabular_common.py::resolve_cfg` | +3 |
| B2 | Blank `objective`/`eval_metric` inherit approved (folded into A2 clamp) | `graph/nodes/trainer.py::_bind_feature_fit` | +2 |
| B3 | Remove `preprocessing_present`, `features_available`, `train_inference_consistent` — never populated; verdict already comes from `checks` | `schemas/qa.py::Phase0Result`, executor mapping | −6 |
| B4 | Require non-empty `snapshot_id` and `dataset_version` in the existing contract `model_validator` | `schemas/evaluation_contract.py` | +4 |
| B5 | Read `settings.phase0_max_attempts` in routing/qa node; delete the module constant | `graph/routing.py`, `graph/nodes/qa.py` | 0 |
| B6 | Delete `route_after_qa_decision` (dead; `build.py` uses a plain edge) | `graph/routing.py` | −8 |
| B7 | Pass `extra_libraries` from a per-family `requires_pip` declared in `jobs/models/registry.py` through the executor's `_submit` — the parameter already exists on `submit_run` | `executor/databricks.py`, `registry.py` | +8 |
| B8 | `ensure_wheel` uninstalls older `amt-*` wheels from an all-purpose cluster (it already knows the current build stamp); retire `scripts/clear_cluster_wheels.py` | `databricks/wheel.py` | +10, −script |

---

## C. Operator ergonomics — drive it from one terminal

### C1. Progress you can watch

Replace `graph.invoke(...)` in `_resume`/`start`/`resume` with `graph.stream(..., stream_mode="updates")`
and print one line per node from the `Stage` value in the update. Same call site, same result;
the loop is the whole change. The executor prints the Databricks run page URL at submit
(`run_page_url` already exists in `databricks/jobs.py`) and the elapsed time at completion. The
twenty-minute silence becomes:

```
▸ planner_select_hypothesis   H2 lightgbm_ranker
▸ trainer_propose             iteration 1/6
▸ execute_training            https://…/#job/…/run/…   (cluster 0123-…)
  … 11m 42s  val/ndcg@10 = 0.0101
▸ trainer_interpret           no meaningful change → converged
```

**≈ +25 lines.**

### C2. `amt run <campaign>` — interactive session

One verb that loops `stream → checkpoint → prompt → resume` until the campaign is terminal or the
operator exits. At a checkpoint it prints the A4 summary, then `typer.prompt` for
`approve / reject <feedback> / drop H2,H4 / show / quit`. Everything it calls already exists
(`_resume`, `_report_position`, `checkpoint_document`); this is the loop plus a five-branch
`match`. This is the direct answer to "I had to rely on the Cursor agent to run the tool": the
non-interactive verbs remain for scripting, but a human no longer needs to know them.

**≈ +50 lines.**

### C3. Quiet, honest output

- Register `amt.schemas` with the checkpointer's serializer so the 17 `Deserializing unregistered
  type` lines disappear (`build.py::sqlite_checkpointer`; LangGraph says this becomes an error
  later anyway). +3 lines.
- `amt config`: `get_settings().model_dump()` with `*secret*`/`*key*`/`*token*` masked, plus the
  source of each value (`.env` vs shell) via `pydantic-settings`' `model_fields_set`. +20 lines.
- `amt doctor`: the runbook's manual smoke steps as one command — auth (`current_user.me`),
  warehouse reachable (`SELECT 1` through `ReadOnlySQL`), volume writable, cluster state,
  MLflow experiment root, LLM provider `structured_output` round-trip on a 3-field schema.
  Replaces §1.2's copy-paste line in the runbook. +40 lines.

### C4. Stop the state from lying

- `amt status` shows `gate_cycles` (count of `HC_*` interrupts resumed with `approved=False`) and
  flags `> 2` in yellow. The bounce loop was invisible because nothing counted it. +6 lines.
- When `_bind_feature_fit` clamps a Trainer ask (A2), `status` shows "trainer requested a
  structural change; kept approved shape" so a reviewer knows the model wanted more than it got.

---

## D. Prompts (cheap, do alongside A)

- `planner_hypotheses.md`: vocabulary block (A3); "each hypothesis must be distinguishable by
  `feature_fit_hash` from every other; if two reduce to the same shape, write one".
- `planner_feature_fit.md`: candidate-row budget rule — `n × validation_users` must be under a
  settings-driven cap (`AMT_MAX_CANDIDATE_ROWS`, default 50M); state it in the proposal.
- `trainer_propose.md`: shrink (A2).
- `trainer_interpret.md`: "if metrics are identical to a previous iteration, say `no_op` and
  recommend convergence".

---

## E. Sequencing

| Phase | Items | Why this order |
|---|---|---|
| 1 — half a day | B1–B8, C3 (warnings), C1 | Pure fixes and visibility; no behavior redesign, unblocks the next real run |
| 2 — one day | A1, A2, A4 | The gate redesign; A1's verbs are used by A2; A4 needs A2's dict payload |
| 3 — half a day | A3, D | Planner quality; prompts land last so they describe the code that exists |
| 4 — half a day | C2, C3 (`config`, `doctor`), C4 | Operator surface on top of the settled graph |

Verification for phase 2: extend `tests/integration/test_campaign.py` with one campaign of three
tabular hypotheses where the operator drops one at `HC_BACKLOG` and one at `HC_FEATURE`, and the
Trainer asks for a structural change on iteration 2 — assert exactly two interrupts, the change is
clamped, `execute_training` never sees an unapproved hash, and the dropped hypotheses read
`abandoned` in the conclusion.

Net code: roughly **+250 / −80 lines** across the four phases, no new modules except none —
`amt run`, `config` and `doctor` are commands in the existing `cli.py`.

---

## Deliberately not in this plan

Agreeing with the operator log: no recipe-language expansion, no feature store, no executor
generalization, no batching of `build_rows` until `n` needs it, no tree-wide reformat. Two more:

- **No web UI for checkpoints.** `amt run` covers the single-operator case; a UI is a project.
- **No per-iteration human gate.** The autonomous band between HC3 and HC4 is the point of the
  system; A2 reduces gates, it does not add them.
