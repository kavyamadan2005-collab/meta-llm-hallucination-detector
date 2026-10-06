# Current verifier baseline

- Cases: **80**
- Accuracy: **100.000%** (80/80)
- Macro-F1: **100.000%**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUPPORTED | 100.000% | 100.000% | 100.000% |
| CONTRADICTED | 100.000% | 100.000% | 100.000% |
| UNSUPPORTED | 100.000% | 100.000% | 100.000% |

## Confusion matrix

| Gold \\ Predicted | SUPPORTED | CONTRADICTED | UNSUPPORTED |
|---|---:|---:|---:|
| SUPPORTED | 45 | 0 | 0 |
| CONTRADICTED | 0 | 13 | 0 |
| UNSUPPORTED | 0 | 0 | 22 |

## Accuracy by category

| Category | Cases | Correct | Accuracy | Errors |
|---|---:|---:|---:|---:|
| compound_relations | 10 | 10 | 100.000% | 0 |
| contradiction | 10 | 10 | 100.000% | 0 |
| dates_numbers | 10 | 10 | 100.000% | 0 |
| direct_support | 10 | 10 | 100.000% | 0 |
| domains_coreference | 10 | 10 | 100.000% | 0 |
| entities_jurisdiction | 10 | 10 | 100.000% | 0 |
| paraphrase | 10 | 10 | 100.000% | 0 |
| unsupported_related | 10 | 10 | 100.000% | 0 |

## Failed cases

No failures.
## Top failure categories

