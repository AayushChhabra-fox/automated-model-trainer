# Scientific Conclusion

Write the final report for the human checkpoint. Be concise and honest; a negative result is
a valid and valuable outcome.

Cover: Problem; Hypothesis; Experiments (only the ones that carry evidence); Best Model and
why it was selected; Validation Evidence versus baseline; QA Evidence; Cohort Effects;
Operational Considerations; Product Interpretation; Conclusion.

In the product interpretation you must distinguish expected impact from proven causal lift.
An offline improvement never proves online causal lift — say so explicitly.

`baseline_validation_metrics` is what the incumbent scored on the validation split, and
`validation_verdict` is the computed candidate-minus-incumbent delta for every hypothesis
champion. Report those numbers as given — the incumbent's number, the champion's, the
delta and the relative delta — in "Validation Evidence versus baseline". Do not say the
incumbent's metrics were not produced when they are in the context, and do not describe a
champion whose delta is negative as promising, provisional or the strongest candidate: it
lost to production, and the report says so in the first sentence of that section. A
negative validation delta is a rejected hypothesis whatever QA did or did not run; QA
failing on plumbing does not make a losing model "needs more validation".

Choose one recommendation:

- `proceed_to_ab_test`
- `shadow_or_canary`
- `supported_needs_more_validation`
- `hypothesis_rejected`
- `modeling_not_the_bottleneck`
- `evaluation_contract_requires_revision`

`report_markdown` holds the full readable report. Never hide a failed experiment, and never
claim an improvement the QA evidence does not support.
