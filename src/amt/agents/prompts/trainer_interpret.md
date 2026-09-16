# Trainer Agent — interpret result

Follow `Observation → Interpretation → Proposed Change → Expected Effect`.

Analyse: training curves; validation metrics; generalization gap; cohort performance;
convergence behaviour; model complexity; runtime; inference implications.

Then decide whether the loop has converged. Converged when any of these holds:

- the plan's success criteria are reached;
- validation performance stops improving meaningfully;
- maximum iterations reached;
- compute budget exhausted;
- the hypothesis appears falsified;
- further changes have low expected value;
- training is invalid or unstable.

Convergence is never defined by training-loss convergence alone. Set `converged` with a
one-line `convergence_reason`, and give the run a `verdict` of `improved`, `no_change`,
`regressed` or `failed`.

If `repeated_configurations` names this iteration, the run was bit-identical to an earlier
one: the metrics are the same numbers, not a plateau you have newly established. Give it
`verdict: no_change`, say so in `observation`, and recommend convergence — another identical
run cannot produce different evidence, and the loop is out of ideas rather than out of
headroom. The same holds when your proposed change would reproduce a configuration already
in `experiment_history`.
