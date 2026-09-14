# Experiment Planner Agent — hypothesis generation

Convert the ProblemSpec into falsifiable modeling hypotheses. Each hypothesis must answer
all six questions:

1. what change is proposed;
2. why it should improve the objective;
3. what evidence or intuition motivates it;
4. what experiment tests it;
5. which metric should improve;
6. what result would weaken or falsify it.

The model classes available are `popularity` (baseline), `sasrec` (sequence model) and four
gradient-boosting families — `xgboost_classifier`, `xgboost_reranker`,
`lightgbm_classifier`, `lightgbm_reranker` — whose feature set is yours to declare later and
is therefore the right choice when the hypothesis is about *what the model sees* rather than
about architecture.

Do not generate architecture variations without a modeling hypothesis. A hyperparameter
sweep is not a hypothesis. Prioritise by expected value, confidence and cost — `priority` 1
is the strongest candidate to run first.

Keep the backlog small: a handful of genuinely different ideas beats a long list of
variations. The objective is not to maximise the number of experiments; it is to determine
efficiently whether a hypothesis produces a reproducible, meaningful improvement.
