# Current verifier baseline

- Cases: **80**
- Accuracy: **98.750%** (79/80)
- Macro-F1: **98.391%**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUPPORTED | 100.000% | 97.778% | 98.876% |
| CONTRADICTED | 92.857% | 100.000% | 96.296% |
| UNSUPPORTED | 100.000% | 100.000% | 100.000% |

## Confusion matrix

| Gold \\ Predicted | SUPPORTED | CONTRADICTED | UNSUPPORTED |
|---|---:|---:|---:|
| SUPPORTED | 44 | 1 | 0 |
| CONTRADICTED | 0 | 13 | 0 |
| UNSUPPORTED | 0 | 0 | 22 |

## Accuracy by category

| Category | Cases | Correct | Accuracy | Errors |
|---|---:|---:|---:|---:|
| compound_relations | 10 | 10 | 100.000% | 0 |
| contradiction | 10 | 10 | 100.000% | 0 |
| dates_numbers | 10 | 9 | 90.000% | 1 |
| direct_support | 10 | 10 | 100.000% | 0 |
| domains_coreference | 10 | 10 | 100.000% | 0 |
| entities_jurisdiction | 10 | 10 | 100.000% | 0 |
| paraphrase | 10 | 10 | 100.000% | 0 |
| unsupported_related | 10 | 10 | 100.000% | 0 |

## Failed cases

### NUM-08
- Category: `dates_numbers`
- Claim: The tower is 300 meters tall.
- Evidence: The tower is 250 meters tall.
- Gold: `SUPPORTED`
- Predicted: `CONTRADICTED`
- Confidence: `0.787`
- Failure reason: Numeric/date normalization is incomplete; only limited year/date rules apply.

## Top failure categories

1. `dates_numbers` — 1 errors
