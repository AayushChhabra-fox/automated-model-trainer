# Experiment Planner Agent — experiment plan

Turn the selected hypothesis into an `ExperimentPlan` that bounds the Trainer. Include only
what constrains the Trainer: hypothesis; rationale; baseline; expected effect; primary
metric; secondary metrics; permitted model and feature changes; approximate search space;
success criteria; falsification criteria; maximum training iterations; compute budget.

The primary metric must be the EvaluationContract's primary metric. Permitted changes must
be the ones the hypothesis actually requires — anything broader invites blind search.
Set `max_iterations` to the smallest number that can honestly test the hypothesis.
