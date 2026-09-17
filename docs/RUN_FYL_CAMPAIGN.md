# Driving the `lfy-ctv-rerank-2026q3` campaign yourself — Latest For You

> **Correction, 2026-09-16.** The first version of this guide targeted container
> `671107897793`. That rail is **Home Page For You (SPORTS/ENT Personas)** — the
> dashboard's pinned "FYL" rail, Container CTR ≈ 37 %. The rail this campaign is
> meant to improve is **Latest For You, `687599077569`** ("For You Vod (PNR with
> fallback) Sports"), Container CTR ≈ 2 %, rank #2 on the home page. Every
> number below was re-derived for `687599077569` from
> `pnr_prod.silver.pr_video_impressions` and the dashboard source. The runtime
> proven on the FYL data is the same runtime; only the contract changes. The
> file keeps its name so existing links resolve; `scripts/fyl_contract.py` needs
> a sibling `scripts/lfy_contract.py` with the constants in §3.1.

The runtime is proven end to end on this table: the grouped-rerank
materialization and the logged-order baseline ran on the FYL rail. What this
guide walks you through is driving the agent half — HC1 through HC4 — on the
right rail.

Budget roughly **3 hours** of wall clock and **$6–8** of Jobs Compute. Almost
all of it is trainer iterations; each job spends ~4.5 minutes just starting a
cluster. Materialization is shorter than FYL's 11 minutes: the table is
clustered by `parent_id` and this rail is 8.5M impressions over the window, not
62.6M.

---

## What this rail is (read once)

Latest For You on CTV, home page `727846815041`, weeks of 2026-07-27 → 09-12:

| | |
|---|---|
| Impressions | ≈ 1.5–2.3M rows/week; 99.9 % on the home page |
| Renders (eligible, see §3) | ≈ 250–350k/week; **3.4 candidates on average**, p90 4–5 |
| Distinct titles | **≈ 1,150/week.** 87 % `content_type = 'news'`, median age **1 day** at render time; sports clips (Football, Auto Racing, Baseball) most of the rest; a little old entertainment |
| Conversion (row) | ≈ 1.1 % CTV (Roku 1.2 %, FireTV 1.6 %, AndroidTV 1.6 %, Tizen 1.5 %, Vizio 1.9 %, LG 1.8 %); drifting down 1.6 % → 0.9 % across Aug–Sep |
| Renders with a conversion | **4–6 %** (Roku 2.5 %, FireTV 5.1 %, Vizio 5.9 %). Multi-conversion < 0.5 % |
| Position curve (slot → conv) | **0 → 2.5 %, 1 → 1.1 %, 2 → 0.7 %, 3 → 0.4 %**, 4+ ≈ 0.5–0.8 %. Slot 14 spikes to 1.4 % (end-of-rail tile) |
| Visible tiles | **4.** Slots 0–3 each ≈ 400k rows/week; slot 4 drops to 137k |
| Incumbent, measured | slot-0 conversions per render **2.0–2.6 %**; given a conversion, it is at slot 0 **44–47 %** of the time and within slots 0–3 **90–92 %** |
| Dashboard | Container CTR **2.00 %**, Item CTR 1.05 %, NDCG@3 2.82 %, rank on page #2 median (`item_index` first element = 1) |
| Columns you can use | `days_since_air_date` **is populated** here (0.01 % null — the opposite of FYL), `created_at`, `duration_seconds`, `content_type`, `genres_array`, `sport_name`, `league_name`, `network_name`, `series_id` (9 % null). `is_sportingevent` is 99.99 % null — do not use |

Three consequences that shape everything downstream:

1. **Position bias is real but mild** — slot 1 examines at ≈ 0.44 of slot 0, slot 3 at ≈ 0.18 (FYL was 0.12 / 0.03). `pbm` weighting still goes on, but the incumbent is beatable on unweighted numbers too. Expect the report's weighted and unweighted deltas to agree in sign.
2. **Items do not persist across splits.** A news clip is a day old at render time, so a title seen in QA week mostly did not exist in the train window. Item-id popularity features from train are useless here; the signal is in *attributes* (type, sport, recency, duration, network) and *account affinity* (what this account converted on before). Hypotheses that lean on `item_train_count` / `item_train_rank` will score like the baseline — drop them at `HC_BACKLOG`.
3. **Positives are sparse.** 74k converting renders in five train weeks. Do **not** hash-sample the train split the way FYL did; 1.27M renders at 100 % is the same order as the 967k FYL train split the runtime already handled, and a within-group ranking objective gets nothing from the 94 % of renders with no conversion anyway.

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
  pnr_prod.silver.pr_video_impressions,product_api_dev.cursor_agent.hp_dash_home_container_daily,product_api_prod.persona_driven_experience.6_1_user_to_cohorts
```

Seven checks run: auth, warehouse, volume writable, compute spec, MLflow, LLM
structured output, and the tables. Only `FAIL` exits non-zero — a node type
from the wrong cloud (`Standard_DS3_v2` is Azure; this workspace is AWS) is a
failure, because it dies at cluster launch twenty minutes in.

A `warn` is a finding you are allowed to accept. The one you will most likely
see is `AMT_EXISTING_CLUSTER_ID` set in your shell or `.env`, which pins the
campaign to a shared all-purpose cluster. That is a legitimate trade — a warm
cluster skips the ~4.5 minutes a job cluster spends starting, which over a
twelve-iteration campaign is about an hour — so `doctor` reads the cluster and
tells you exactly what you give up rather than refusing:

- An `amt=true` **cluster tag** makes its cost attributable to AMT in
  `system.billing.usage.custom_tags`. Per-run attribution (`amt_campaign_id`,
  `amt_role`, `amt_hypothesis_id`) cannot work on a shared cluster.
- It bills at the All-Purpose rate. Measured on this workspace, that is
  $0.650/DBU against $0.200 for Jobs Compute — **3.25×**, not the "about twice"
  rule of thumb.
- If its runtime is not an **ML** runtime, the booster families get
  pip-installed on every run and the installs pile up; run
  `scripts/clear_cluster_wheels.py` when they do.
- A **single node** is fine for the boosters; materialization of 8.5M rows is
  tolerable on one node for this rail, unlike FYL.

Leave `AMT_EXISTING_CLUSTER_ID` unset for a campaign whose cost you want to
attribute. Set it when you are iterating and want the fast loop.

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

Click **new campaign**. Campaign id `lfy-ctv-rerank-2026q3`, budget
`3 hypotheses / 4 iterations / 240 compute minutes`, and paste this as the
problem. It is HC1, and every later stage is derived from it, so read it before
you paste it:

> Does a re-ranker beat the incumbent order of the home-page **Latest For You**
> rail (container `687599077569`, "For You Vod (PNR with fallback) Sports") on
> that rail's own conversion metric, over the seven weeks to 2026-09-12, on CTV
> only?
>
> The rail is scored per render: one render of the container for one session at
> one instant is one evaluation unit, identified by
> `(session_id, page_id, parent_id, event_received)` in
> `pnr_prod.silver.pr_video_impressions`. The candidates are exactly the titles
> that render impressed — a re-ranker cannot surface anything else — and the
> incumbent's ranking is the slot each was shown in, the last element of
> `item_index` ("rail_rank,column"). A conversion is `conversion_flag = true`.
>
> CTV on the home page only: `parent_type = 'container'`,
> `page_id = '727846815041'`, `client_name` in ROKU, FIRETV, TIZEN, LG, VIZIO,
> ANDROIDTV. Web, Android phone, mobile (`D2C_MOB_LANDSCAPE_MIXED`) and tvOS
> (`D2C_LR_LANDSCAPE_MIXED`) have different position mechanics and are separate
> campaigns. Exclude `item_index IS NULL` and
> `account_id = 'af44641e-a0ed-4d88-92d0-6fa68e0cbf1a'` (a schema example).
> Renders with fewer than two distinct titles cannot be re-ranked and are
> dropped (about a third of Roku renders are single-tile partial renders).
>
> Four tiles are visible on CTV, so `k = 4`. Position bias is present but mild
> — slot 0 converts at 2.5 %, slots 1/2/3 at 1.1/0.7/0.4 % — so relevance is
> position-weighted (`pbm`, propensities per client from the train split).
>
> The rail is fresh content: 87 % news clips with a median age of one day, about
> 1,150 distinct titles a week. Titles in the QA week mostly did not exist in the
> train window, so item-identity popularity features cannot carry across splits.
> Signal must come from title attributes (`content_type`, `sport_name`,
> `days_since_air_date`, `duration_seconds`, `network_name`, `genres_array`)
> and from the account's own conversion history in the train window.
> `is_sportingevent` is entirely NULL for this container — do not use it.
>
> The product number this has to move is the dashboard's Container CTR, daily,
> from `product_api_dev.cursor_agent.hp_dash_home_container_daily` where
> `container_id='687599077569' AND cohort='all' AND page_id='727846815041'`,
> columns `event_date` and `container_ctr_pct`.
>
> Window 2026-07-27 to 2026-09-13, train to 2026-08-30, validation to
> 2026-09-06, QA the last week. No sampling: 1.27M train renders with 74k
> conversions is the whole signal there is.

CLI equivalent:

```bash
uv run amt start "<the text above>" -c lfy-ctv-rerank-2026q3 \
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
3. `primary_offline_metric` is `ndcg@4`. Not `@10`: with 3.4 candidates per
   render, `@10` is trivially 1.0.
4. Secondaries include `hitrate@1` — that is the slot-0 metric and the bridge
   to the dashboard's number.
5. The data-sources section says the container is **`687599077569`** and the
   page is `727846815041`. If the agent has drifted to `671107897793` (it is the
   id every dashboard config file mentions), reject with the correction.

Approve in the browser, or:

```bash
uv run amt approve lfy-ctv-rerank-2026q3
uv run amt reject lfy-ctv-rerank-2026q3 -m "container is 687599077569 (Latest For You), not 671107897793"
```

---

## 3. HC3 — the EvaluationContract, and the one gate worth being slow at

Approving HC3 **freezes the ground truth and immediately spends cluster time**
materializing render-grain splits. Everything after it is judged against what
you approve here, and it cannot be changed without starting over. Read it
properly.

`amt show lfy-ctv-rerank-2026q3` prints the whole document. Check:

| field | must be |
|---|---|
| `evaluation_mode` | `grouped_rerank` |
| `relevance_weighting` | `pbm` |
| `dataset.group_key` | `[session_id, page_id, parent_id, event_received]` |
| `dataset.position_column` | `item_index` |
| `dataset.label_column` | `conversion_flag` |
| `dataset.label_positive_values` | `[true]` |
| `dataset.filters` | `parent_id = '687599077569'`, `parent_type = 'container'`, `page_id = '727846815041'`, the six clients, both exclusions, **and no hash-sampling predicate** |
| `split_policy` | train_end `2026-08-30`, validation_end `2026-09-06`, qa_end `2026-09-13` |
| `primary_metric` | `ndcg@4` |
| `product_metric_source` | the dashboard table, `event_date`, `container_ctr_pct`, filter on **`687599077569`** |
| `acceptance_criteria` | `min_proxy_correlation` 0.5, `min_proxy_days` 7, `max_cohort_regression` 0.02 |
| `cohorts` | `client_name`, `history_length`, `content_type` of the slot-0 title |

### 3.1 The reference contract

`scripts/fyl_contract.py` is the FYL contract written by hand. Create
`scripts/lfy_contract.py` from it with exactly these changes, then diff the
agent's draft against it if anything looks off:

| constant | FYL value | Latest For You value |
|---|---|---|
| `CAMPAIGN` | `fyl-ctv-rerank-2026q3` | `lfy-ctv-rerank-2026q3` |
| `CONTAINER` | `671107897793` | `687599077569` |
| `FILTERS` | — | add `"page_id = '727846815041'"`; **delete** the sampling predicate |
| `TRAIN_SAMPLE` / `EVAL_SAMPLE` | `% 10 = 0` / `% 5 = 0` | remove both |
| `K` | 4 | 4 (unchanged) |
| `propensity_floor` | 0.02 | 0.05 (slot 3 is ≈ 0.18; nothing gets near the floor) |
| `population_eligibility` | "CTV renders of container 671107897793 … from the frozen session-hash sample" | "CTV home-page renders of container 687599077569 with at least two distinct impressed titles; no sampling" |
| `exclusions` | mobile/tvOS FORYOU types | `D2C_MOB_LANDSCAPE_MIXED`, `D2C_LR_LANDSCAPE_MIXED`, WEB and ANDROID clients, non-home pages, single-title renders |
| `product_metric_source.filter_sql` | `container_id = '671107897793' …` | `container_id = '687599077569' …` |
| `cohorts` | `client_name`, `history_length` | add `CohortDef(name="slot0_content_type", …)` |
| `CohortDef("client_name").description` | "Slot-0 conversion ranges 33–45 %" | "Row conversion 1.2–1.9 %; renders converting 2.5–5.9 %" |
| comment on `K` | slot-4 rate rises to 3.2 % | slot-4 rate rises to 0.73 % from slot 3's 0.43 % — same scrolled-population signature |

Everything else — group key, label, metrics, split boundaries, acceptance
criteria — is identical.

### 3.2 Two traps

- **The page filter.** Without `page_id = '727846815041'` the contract picks up
  a few hundred rows from other pages with a different rail rank and, more
  importantly, does not match the dashboard series the proxy check reads.
- **The boundaries.** `2026-08-30` and `2026-09-06` are Sundays and the last
  impression is Saturday `2026-09-12`, so validation and QA are whole Sun–Sat
  weeks. That also gives the dashboard series exactly 7 overlapping days, which
  is what `min_proxy_days` needs. Shift them by a day and the proxy check fails
  on a technicality after the whole campaign has run.

### 3.3 What materialization should return

`amt status` shows the contract hash, and the split references carry
`row_count` and `positive_labels`. Measured on the eligible population:

```
train       1,272,693 renders   4,493,985 rows    74,153 renders with a conversion
validation    269,891 renders     951,443 rows    11,864
qa            345,684 renders   1,253,236 rows    15,015
```

**If any split shows `positive_labels = 0`, stop.** The label column is not
what the contract thinks it is.

Check the propensities in `split_materialization.propensities`: slot 0 must be
1.00 (the reference) and slot 1 should land at **0.40–0.50**, slot 2 at
0.25–0.32, slot 3 at 0.15–0.20 — per client, with Roku a little steeper. If they
are all 1.00 the weighting has silently become no weighting; if slot 1 is below
0.2 you have the FYL rail's data.

---

## 4. HC_BACKLOG — which three ideas get the budget

The planner proposes hypotheses; this gate decides which ones cost money. The
table above the JSON gives you id, family, metric, expected effect, cost and
status.

What a good backlog for *this* rail looks like:

- **Attribute + recency ranker.** `content_type`, `sport_name`, `league_name`,
  `days_since_air_date`, `duration_seconds`, `network_name`, `genres_array`,
  `client_name`. This is the hypothesis most likely to move: "fresher and
  shorter news clips convert; the incumbent does not order by that."
- **Account affinity.** The account's share of train-window conversions by
  `content_type` / `sport_name`, matched against the candidate's type. This is
  the "personalisation the fallback is missing" hypothesis, and it needs a
  `ratio` over `count` transforms keyed on `account_id`.
- **Position-debiased objective.** Same features as the first, trained with the
  logged slot as a feature and scored with slot 0 for every candidate, or with
  a `pbm`-weighted objective. Useful as a contrast to the first, not as the
  first.

Drop, without a second thought:

- Anything relying on `item_train_count`, `item_train_rank` or any per-title
  statistic from the train window. Titles turn over daily; those features are
  NULL or stale for most of QA.
- Anything using `is_sportingevent` (all NULL) or position as an
  inference-time input (cannot be served).
- Two hypotheses that are really the same idea. The next gate flags identical
  shapes, but striking a duplicate here saves the proposal round too.

Edit the list rather than rejecting it — rejecting costs a full round trip
through the agent:

```bash
uv run amt approve lfy-ctv-rerank-2026q3 --keep hyp-1,hyp-3 --drop hyp-2
```

`--keep` also sets the order they run in, so put your best guess first: the
budget may not reach the third.

---

## 5. HC_FEATURE — every shape at once

One review for all tabular hypotheses. The table prices each one:

- **`candidates` and `est. rows`.** The strategy must be `logged_impressions` —
  the candidate set is the impression set, so there is no `n` and no blow-up.
  `popularity_topn` here is the wrong framing entirely; drop it.
- **The six fit fields** — objective, eval_metric, num_boost_round,
  early_stopping_rounds, negative_samples_per_positive, seed. With 4–6 % of
  renders converting, `lambdarank` on groups is the natural objective; a
  pointwise classifier will want `negative_samples_per_positive` well above the
  default or it spends its capacity on the 94 %.
- **`same shape as …`** — two hypotheses that reduce to one `feature_fit_hash`
  are one experiment billed twice.
- **`no feature columns`** — nothing to fit.
- **Where each feature's column comes from.** A render-grain split carries
  `candidates, positions, labels, weights, user_id, ts, client_name` and nothing
  else. Every per-title feature (`content_type`, `series_id`, `duration_seconds`,
  `genres`, …) must be backed by a `role: item_attributes` source — for this
  rail, `pnr_prod.silver.pr_item_feature`. A recipe that lists those names under
  a `split:` source is reading columns that do not exist; the job now refuses it
  ("feature sources … have no values for any item") instead of training on
  constants, but catch it here and save the cluster start. `client_name` is the
  one render-level feature. `days_since_air_date` and `promo_value` are not
  available in this mode. "Account history" does not exist in this mode either —
  `count … over: history` counts inside the render — so a shape built on it is a
  drop, not an approve.

Same verbs. Dropping a shape here abandons its hypothesis.

After this gate the campaign is autonomous until the first champion.

---

## 6. HC_CHAMPION — once per hypothesis

Each hypothesis's champion against the incumbent and against every previous
champion, with cohort deltas and the remaining budget. The incumbent's numbers
to beat, measured on the validation week over renders with a conversion:

```
hitrate@1 (unweighted)   0.465        hitrate@4   0.915
slot-0 conversions per 1,000 renders   ≈ 20.4  (dashboard Container CTR ≈ 2.0 %)
```

`pbm`-weighted `ndcg@4` and `hitrate@1` for the incumbent will print lower than
these — conversions at slots 1–3 are up-weighted — and the job prints both. A
candidate that beats the incumbent on the weighted primary but *loses* on
unweighted `hitrate@1` is moving later-slot conversions up at the expense of
slot-0 ones; read the cohort table before continuing.

Three verbs:

```bash
uv run amt approve lfy-ctv-rerank-2026q3                  # continue
uv run amt approve lfy-ctv-rerank-2026q3 --drop hyp-3     # skip one
uv run amt approve lfy-ctv-rerank-2026q3 --freeze         # good enough, go to QA
```

`--freeze` is the one people forget. If the first hypothesis clearly beats the
incumbent and the second is a variation on it, freezing saves an hour of
cluster time and the evidence is no weaker.

---

## 7. QA, and then HC4

Freeze runs QA phase 0 (integrity on validation) then phase 1 (the QA week,
never read before now). Phase 1 does the proxy-sanity check **first**: it
correlates the incumbent's `ndcg@4` against the dashboard's daily Container CTR
for `687599077569` across the seven QA days. If Spearman ρ < 0.5, the decision
is `SPEC_REVIEW` and no candidate number is read at all — the offline metric
does not track the product number for the model already in production, so it
cannot certify a replacement.

One thing to know before you read that verdict: the dashboard's CTR counts
`content_click`; our label counts click-then-playback within 90 s. On this rail
the incumbent's slot-0 *conversion* rate (2.0–2.6 %) sits at or above the
dashboard's slot-0 *click* CTR (2.0 %), which should not be possible if both
measured the same sessions — the dashboard applies a ≥ 10 s session filter and
a longer window. The proxy check is what decides whether that gap matters; if
it fails, the fix is a click label joined from `dtc_visitid_dim_plus`, not a
different booster.

At HC4 you get the conclusion plus the experimentation report:

```bash
uv run amt report lfy-ctv-rerank-2026q3
```

Read it in the order it is printed. The proxy verdict first, then the effect in
**slot-0 conversions per 1,000 renders** — the incumbent's figure there should
land near 20, with the candidate's beside it as the counterfactual — then the
offline delta with its CI, then any regressed cohort (watch `content_type =
sports`: it is 8 % of the rail and a news-trained ranker can bury it), then what
an online test would cost. At a 2 % base rate a relative lift of 10 % is a 0.2
point absolute move; the A/B section will ask for several weeks of this rail's
traffic. `/c/lfy-ctv-rerank-2026q3/report` in the browser is the same thing as
HTML.

```bash
uv run amt promote lfy-ctv-rerank-2026q3    # registers the model only if QA=PASS
```

---

## If something goes wrong

```bash
uv run amt status lfy-ctv-rerank-2026q3   # where it is, and what it has spent
uv run amt list                           # every campaign in the local DB
uv run amt show lfy-ctv-rerank-2026q3     # the pending document, in full
```

`status` has two lines worth watching. `gate_cycles` counts rejections — they
cost no compute, so nothing else moves when you are arguing with the planner,
and above 2 it is flagged. `trainer_asks_refused` appears when the Trainer
asked to change the approved shape and was clamped; the run is still valid, the
model just wanted more than it got.

**An approved shape turns out to be unexecutable** (a recipe reading columns the
split does not carry, an account-history feature in grouped mode). The gate is
final by design and the Trainer is clamped to it, so the way out is to reopen the
gate, not to resume:

```bash
uv run amt show   lfy-ctv-rerank-2026q3 --key approved_shapes   # see what was approved
uv run amt rewind lfy-ctv-rerank-2026q3 --to feature \
  -m "per-title features must come from pnr_prod.silver.pr_item_feature (role item_attributes); drop days_since_air_date and promo_value; there is no account history in grouped mode"
```

Nothing that ran is erased; the mid-flight hypothesis goes back to `pending` and
its budget slot comes back; the planner redrafts every shape from your message and
`HC_FEATURE` comes round again. `--to backlog` reopens the hypothesis list instead.
Both are refused once a champion is frozen — after that, a new campaign.

**QA failed on plumbing, not on the model** (Phase 0 could not reproduce because
the harness passed the model the wrong inputs; the conclusion came out
`INVALID_EXPERIMENT` with a `technical_defect`). SPEC Stage 9 allows exactly one
thing here: fix the defect and rerun Phase 0 *without touching the model*. That is
the only rewind allowed after the freeze:

```bash
uv run amt rewind lfy-ctv-rerank-2026q3 --to qa \
  -m "pyfunc signature dropped candidates/positions/context; rerun QA on the frozen artifact"
```

The frozen champion, its MLflow run and every training result stay as they are;
only `qa_*`, the conclusion and the experimentation report are cleared, and the
graph re-enters at Phase 0. A losing champion stays a losing champion — this
exists to get a *valid* verdict, not a different one.

A Databricks failure currently **raises out of the graph** rather than being
handed to the Trainer. The campaign is resumable — the checkpointer has
everything — but `amt resume` re-submits rather than re-polls, so you can pay
for the same job twice. Check the Databricks run page before resuming.

Nothing is lost by walking away. The campaign lives in
`~/.amt/checkpoints.sqlite`; `amt status` and the browser pick it up wherever
it stopped.

---

## Known limits of this campaign, stated up front

- **Roku singletons.** Under the exact-timestamp render key, 32 % of Roku
  renders carry one title and are dropped. Roku's consecutive impressions do
  share a timestamp (45k of 63k gaps are 0 ms on a sample day), so this is
  partial rendering, not a split key; but a 2-second window key would recover
  some of the 0.5–2 s stragglers. Follow-up, not v1.
- **Recency is coarse.** `days_since_air_date` is an integer and most of this
  rail is 0–1 days old. An `age_hours` transform (`event_received − created_at`)
  is the single vocabulary addition most likely to help this rail; it is ~10
  lines in `tabular_common` and worth doing before a second campaign.
- **Re-ordering lift only.** Candidates are the incumbent's. This measures
  whether the rail's 3–4 tiles are in the right order, not whether better tiles
  exist. That is the dashboard's Container CTR question exactly, and no more.
