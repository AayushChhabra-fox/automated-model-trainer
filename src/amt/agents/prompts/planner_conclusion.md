# Scientific Conclusion

Write the final report for the human checkpoint. Be concise and honest; a negative result is
a valid and valuable outcome.

Cover: Problem; Hypothesis; Experiments (only the ones that carry evidence); Best Model and
why it was selected; Validation Evidence versus baseline; QA Evidence; Cohort Effects;
Operational Considerations; Product Interpretation; Conclusion.

In the product interpretation you must distinguish expected impact from proven causal lift.
An offline improvement never proves online causal lift — say so explicitly.

Choose one recommendation:

- `proceed_to_ab_test`
- `shadow_or_canary`
- `supported_needs_more_validation`
- `hypothesis_rejected`
- `modeling_not_the_bottleneck`
- `evaluation_contract_requires_revision`

`report_markdown` holds the full readable report. Never hide a failed experiment, and never
claim an improvement the QA evidence does not support.
