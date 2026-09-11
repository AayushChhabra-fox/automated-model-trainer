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
