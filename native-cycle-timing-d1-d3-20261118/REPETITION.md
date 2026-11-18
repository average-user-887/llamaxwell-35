# Repetition in the matched warm continuations

These counts describe one shared, unscored forced 3,072-output continuation from an 80K prompt. Lines were stripped and whitespace collapsed, then digit runs replaced with one marker. A duplicate is an additional occurrence in the indicated segment. Overlapping windows use 24 Unicode word tokens after digit normalization.

| Warm arm | Duplicate nonblank lines | Duplicate final-quarter lines | Trailing comment-only lines | Repeated 24-word windows | Python keyword-led lines |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4 serial on | 94/293 | 14/74 | 0 | 129/1,322 | 164 |
| Q4 depth three off | 49/260 | 4/65 | 33 | 30/1,385 | 102 |
| Q4 depth three attention | 101/298 | 8/75 | 0 | 96/1,308 | 158 |
| Q4 depth three combined | 68/253 | 26/64 | 0 | 28/1,198 | 106 |
| Q3 serial off | 73/283 | 16/71 | 0 | 46/1,318 | 170 |
| Q3 serial on | 73/283 | 16/71 | 0 | 46/1,318 | 170 |
| Q3 depth two off | 108/293 | 11/74 | 0 | 89/1,355 | 154 |
| Q3 depth two on | 109/311 | 18/78 | 0 | 71/1,359 | 179 |

The Q4 combined arm's final quarter had 40.6% duplicate normalized lines. It had fewer recurring windows overall than the attention-only arm and no trailing comment-only run. The Q4 OFF arm ended with 33 comment-only lines. All eight outputs had zero replacement or control characters and zero fence lines. The [complete counts](repetition.json) include all quarters and overall comment counts. Separate scored short arithmetic checks remain in their own receipts.
