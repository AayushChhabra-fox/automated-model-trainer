# Driving the `fyl-ctv-rerank-2026q3` campaign yourself

Everything below has been verified against the workspace. The runtime is proven
end to end on this data: the splits are materialized and the incumbent is
scored. What has **not** been driven yet is the agent half — HC1 through HC4 —
which is what this walks you through.

Budget roughly **3 hours** of wall clock and **$8–10** of Jobs Compute. Almost
all of it is trainer iterations; each job spends ~4.5 minutes just starting a
cluster.

---

## 0. Before anything

```bash
cd ~/automated-model-trainer
uv sync --extra web --extra jobs
```

Check the machine can do the job at all. Every failure here is one that would
otherwise surface twenty minutes into a job:

```bash
uv run amt doctor --tables \
  pnr_prod.silver.pr_video_impressions,product_api_dev.cursor_agent.hp_dash_home_container_daily
```

Six checks must read `ok`: auth, warehouse, volume writable, compute spec,
MLflow, LLM structured output — plus the two tables. If `compute spec` fails,
read what it says: an `AMT_EXISTING_CLUSTER_ID` in your shell or `.env` makes
the run untagged and unattributable, and a node type from the wrong cloud
(`Standard_DS3_v2` is Azure; this workspace is AWS) fails at cluster launch.

Then confirm what you are about to spend money on:

```bash
uv run amt config
```

Look at the **source** column. An `AMT_*` variable exported in your shell
silently shadows `.env`; that column is the only thing that tells you. You want
`node_type_id = i3.2xlarge`, `existing_cluster_id` empty, and
`catalog = product_api_dev`.

---

## 1. Start the campaign

Two surfaces, same graph, same checkpointer, same decisions. Use the browser —
it shows a progress log instead of twenty silent minutes.

```bash
uv run amt web
# http://127.0.0.1:8765
```

Click **new campaign**. Campaign id `fyl-ctv-rerank-2026q3`, budget
`3 hypotheses / 4 iterations / 240 compute minutes`, and paste this as the
problem. It is HC1, and every later stage is derived from it, so it is worth
reading before you paste it:

> Does a re-ranker beat the incumbent order of the Home Page "For You" rail
> (container `671107897793`) on that rail's own conversion metric, over the
> seven weeks to 2026-09-12, on CTV only?
>
> The rail is scored per render: one render of the container for one session at
> one instant is one evaluation unit, identified by
> `(session_id, page_id, parent_id, event_received)` in
> `pnr_prod.silver.pr_video_impressions`. The candidates are exactly the titles
> that render impressed — a re-ranker cannot surface anything else — and the
> incumbent's ranking is the slot each was shown in, the last element of
> `item_index` ("rail_rank,column"). A conversion is `conversion_flag = true`.
>
> CTV only: `parent_type = 'container'` and `client_name` in ROKU, FIRETV,
> TIZEN, LG, VIZIO, ANDROIDTV. Mobile and tvOS have different position
> mechanics and are separate campaigns. Exclude `item_index IS NULL` and
> `account_id = 'af44641e-a0ed-4d88-92d0-6fa68e0cbf1a'` (a schema example).
>
> Four tiles are visible on CTV, so `k = 4`. Position bias is severe — slot 0
> converts at 39%, slots 1/2/3 at 6.7/2.5/1.7% — so relevance must be
> position-weighted or a ranker that copies the logged order scores
> near-perfectly.
>
> The product number this has to move is the dashboard's Container CTR, daily,
> from `product_api_dev.cursor_agent.hp_dash_home_container_daily` where
> `container_id='671107897793' AND cohort='all' AND page_id='727846815041'`,
> columns `event_date` and `container_ctr_pct`.
>
> Window 2026-07-27 to 2026-09-13, train to 2026-08-30, validation to
> 2026-09-06, QA the last week. 62.6M impressions is too many for one node, so
> freeze a sample in the contract: `abs(hash(session_id)) % 10 = 0` before
> 2026-08-30 and `% 5 = 0` after.
>
> `days_since_air_date` is entirely NULL for this container — do not use it.
> `duration_seconds`, `content_type`, `genres_array`, `network_name` and
> `is_sportingevent` are fully populated.

CLI equivalent:

```bash
uv run amt start "<the text above>" -c fyl-ctv-rerank-2026q3 \
  --max-hypotheses 3 --max-iterations 4 --max-compute-minutes 240
```

The agent now runs up to eight read-only SQL probes and drafts a ProblemSpec.
That takes a minute or two and costs nothing but LLM tokens.

---

## 2. HC2 — the ProblemSpec

Check five things, in this order. The first is the one that matters:

1. `evaluation_mode` is `grouped_rerank`. If it says `next_item`, reject —
   everything downstream would answer the wrong question.
2. `baseline.kind` is `logged_order`.
3. `primary_offline_metric` is `ndcg@4`. Not `@10`: with ~3.5 candidates per
   render, `@10` is trivially 1.0.
4. Secondaries include `hitrate@1` — that is the slot-0 metric and the bridge
   to the dashboard's number.
5. The assumptions say the lift is *re-ordering* lift on the incumbent's own
   candidates.

Approve in the browser, or:

```bash
uv run amt approve fyl-ctv-rerank-2026q3
uv run amt reject fyl-ctv-rerank-2026q3 -m "evaluation_mode must be grouped_rerank: one render is one unit"
```

---

## 3. HC3 — the EvaluationContract, and the one gate worth being slow at

Approving HC3 **freezes the ground truth and immediately spends ~11 minutes of
cluster time** materializing 62.6M impressions into render-grain splits.
Everything after it is judged against what you approve here, and it cannot be
changed without starting over. Read it properly.

`amt show fyl-ctv-rerank-2026q3` prints the whole document. Check:

| field | must be |
|---|---|
| `evaluation_mode` | `grouped_rerank` |
| `relevance_weighting` | `pbm` |
| `dataset.group_key` | `[session_id, page_id, parent_id, event_received]` |
| `dataset.position_column` | `item_index` |
| `dataset.label_column` | `conversion_flag` |
| `dataset.label_positive_values` | `[true]` |
| `dataset.filters` | the container, `parent_type`, the six clients, both exclusions, **and the hash sampling** |
| `split_policy` | train_end `2026-08-30`, validation_end `2026-09-06`, qa_end `2026-09-13` |
| `primary_metric` | `ndcg@4` |
| `product_metric_source` | the dashboard table, `event_date`, `container_ctr_pct`, the three-way filter |
| `acceptance_criteria` | `min_proxy_correlation` 0.5, `min_proxy_days` 7, `max_cohort_regression` 0.02 |

`scripts/fyl_contract.py` is this contract written by hand — diff the agent's
draft against it if anything looks off.

Two traps worth knowing:

- **The sampling predicate.** Without it the job tries 62.6M impressions
  through single-node feature building. It must sample on `session_id`, not on
  the row, or one render lands split across two splits.
- **The boundaries.** `2026-08-30` and `2026-09-06` are Sundays and the last
  impression is Saturday `2026-09-12`, so validation and QA are whole Sun–Sat
  weeks. That also gives the dashboard series exactly 7 overlapping days, which
  is what `min_proxy_days` needs. Shift them by a day and the proxy check fails
  on a technicality after the whole campaign has run.

When you approve, materialization runs. When it comes back, sanity-check it
before going on — `amt status` shows the contract hash, and the split
references carry `row_count` and `positive_labels`. Expect roughly:

```
train       967,119 renders   386,520 with a conversion
validation  498,543 renders   203,779
qa          517,920 renders   192,591
```

**If any split shows `positive_labels = 0`, stop.** The label column is not
what the contract thinks it is, and every metric downstream would be computed
against an empty truth set. The job now refuses this outright, but check anyway.

Also check the propensities in `split_materialization.propensities`: slot 0 must
be 1.00 (the reference) and slot 1 should land near 0.11–0.18. If they are all
1.00 the weighting has silently become no weighting.

---

## 4. HC_BACKLOG — which three ideas get the budget

The planner proposes hypotheses; this gate decides which ones cost money. The
table above the JSON gives you id, family, metric, expected effect, cost and
status.

Look for:

- Anything marked `infeasible` — it asked for something the runtime cannot
  build, with the reason. It is already excluded; you do not need to act.
- Anything naming `days_since_air_date`. It is 100% NULL for this container.
  Drop it.
- Two hypotheses that are really the same idea. The next gate flags identical
  shapes, but striking a duplicate here saves the proposal round too.
- Position as an inference-time input. A model that scores on the slot it was
  given cannot be served. The recipe should use logged position at training
  time only, or not at all.

Edit the list rather than rejecting it — rejecting costs a full round trip
through the agent:

```bash
uv run amt approve fyl-ctv-rerank-2026q3 --keep hyp-1,hyp-3 --drop hyp-2
```

`--keep` also sets the order they run in, so put your best guess first: the
budget may not reach the third.

---

## 5. HC_FEATURE — every shape at once

One review for all tabular hypotheses. The table prices each one:

- **`candidates` and `est. rows`.** In this campaign the strategy must be
  `logged_impressions` — the candidate set is the impression set, so there is no
  `n` and no blow-up. If you see `popularity_topn n=200` against a 498k-row
  validation split, that is 99 million feature rows and the wrong framing
  entirely. Drop it.
- **The six fit fields** — objective, eval_metric, num_boost_round,
  early_stopping_rounds, negative_samples_per_positive, seed.
- **`same shape as …`** — two hypotheses that reduce to one `feature_fit_hash`
  are one experiment billed twice.
- **`no feature columns`** — nothing to fit.

Same verbs. Dropping a shape here abandons its hypothesis.

After this gate the campaign is autonomous until the first champion.

---

## 6. HC_CHAMPION — once per hypothesis

Each hypothesis's champion against the incumbent and against every previous
champion, with cohort deltas and the remaining budget. The incumbent's numbers
to beat, measured:

```
ndcg@4     0.86430          hitrate@1  0.69641          mrr@4  0.81487
hitrate@1 by client: ROKU 0.738 … ANDROIDTV 0.609
```

Three verbs:

```bash
uv run amt approve fyl-ctv-rerank-2026q3                  # continue
uv run amt approve fyl-ctv-rerank-2026q3 --drop hyp-3     # skip one
uv run amt approve fyl-ctv-rerank-2026q3 --freeze         # good enough, go to QA
```

`--freeze` is the one people forget. If the first hypothesis clearly beats the
incumbent and the second is a variation on it, freezing saves an hour of
cluster time and the evidence is no weaker.

---

## 7. QA, and then HC4

Freeze runs QA phase 0 (integrity on validation) then phase 1 (the QA week,
never read before now). Phase 1 does the proxy-sanity check **first**: it
correlates the incumbent's `ndcg@4` against the dashboard's daily Container CTR
across the seven QA days. If Spearman ρ < 0.5, the decision is `SPEC_REVIEW`
and no candidate number is read at all — the offline metric does not track the
product number for the model already in production, so it cannot certify a
replacement. That is a finding about the evaluation, not about the model, and
the fix is a click label rather than a different booster.

At HC4 you get the conclusion plus the experimentation report:

```bash
uv run amt report fyl-ctv-rerank-2026q3
```

Read it in the order it is printed. The proxy verdict first, then the effect in
**top-slot conversions per 1,000 renders** — the incumbent's figure there should
match what the dashboard shows, with the candidate's beside it as the
counterfactual — then the offline delta with its CI, then any regressed cohort,
then what an online test would cost. `/c/fyl-ctv-rerank-2026q3/report` in the
browser is the same thing as HTML.

```bash
uv run amt promote fyl-ctv-rerank-2026q3    # registers the model only if QA=PASS
```

---

## If something goes wrong

```bash
uv run amt status fyl-ctv-rerank-2026q3   # where it is, and what it has spent
uv run amt list                           # every campaign in the local DB
uv run amt show fyl-ctv-rerank-2026q3     # the pending document, in full
```

`status` has two lines worth watching. `gate_cycles` counts rejections — they
cost no compute, so nothing else moves when you are arguing with the planner,
and above 2 it is flagged. `trainer_asks_refused` appears when the Trainer
asked to change the approved shape and was clamped; the run is still valid, the
model just wanted more than it got.

A Databricks failure currently **raises out of the graph** rather than being
handed to the Trainer. The campaign is resumable — the checkpointer has
everything — but `amt resume` re-submits rather than re-polls, so you can pay
for the same job twice. Check the Databricks run page before resuming.

Nothing is lost by walking away. The campaign lives in
`~/.amt/checkpoints.sqlite`; `amt status` and the browser pick it up wherever
it stopped.
