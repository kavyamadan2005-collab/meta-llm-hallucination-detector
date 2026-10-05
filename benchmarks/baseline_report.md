# Current verifier baseline

- Cases: **80**
- Accuracy: **63.750%** (51/80)
- Macro-F1: **58.727%**

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| SUPPORTED | 67.273% | 82.222% | 74.000% |
| CONTRADICTED | 100.000% | 46.154% | 63.158% |
| UNSUPPORTED | 42.105% | 36.364% | 39.024% |

## Confusion matrix

| Gold \\ Predicted | SUPPORTED | CONTRADICTED | UNSUPPORTED |
|---|---:|---:|---:|
| SUPPORTED | 37 | 0 | 8 |
| CONTRADICTED | 4 | 6 | 3 |
| UNSUPPORTED | 14 | 0 | 8 |

## Accuracy by category

| Category | Cases | Correct | Accuracy | Errors |
|---|---:|---:|---:|---:|
| compound_relations | 10 | 6 | 60.000% | 4 |
| contradiction | 10 | 5 | 50.000% | 5 |
| dates_numbers | 10 | 8 | 80.000% | 2 |
| direct_support | 10 | 10 | 100.000% | 0 |
| domains_coreference | 10 | 9 | 90.000% | 1 |
| entities_jurisdiction | 10 | 6 | 60.000% | 4 |
| paraphrase | 10 | 3 | 30.000% | 7 |
| unsupported_related | 10 | 4 | 40.000% | 6 |

## Failed cases

### PAR-01
- Category: `paraphrase`
- Claim: The company purchased the startup.
- Evidence: The firm acquired the young company.
- Gold: `SUPPORTED`
- Predicted: `UNSUPPORTED`
- Confidence: `0.600`
- Failure reason: Correct paraphrase lacks sufficient lexical/predicate overlap.

### PAR-02
- Category: `paraphrase`
- Claim: The scientist discovered the particle.
- Evidence: The researcher found the particle.
- Gold: `SUPPORTED`
- Predicted: `UNSUPPORTED`
- Confidence: `0.600`
- Failure reason: Correct paraphrase lacks sufficient lexical/predicate overlap.

### PAR-03
- Category: `paraphrase`
- Claim: The treaty ended the conflict.
- Evidence: The agreement brought the war to an end.
- Gold: `SUPPORTED`
- Predicted: `UNSUPPORTED`
- Confidence: `0.600`
- Failure reason: Correct paraphrase lacks sufficient lexical/predicate overlap.

### PAR-04
- Category: `paraphrase`
- Claim: The spacecraft entered orbit.
- Evidence: The probe was placed into orbit.
- Gold: `SUPPORTED`
- Predicted: `UNSUPPORTED`
- Confidence: `0.600`
- Failure reason: Correct paraphrase lacks sufficient lexical/predicate overlap.

### PAR-05
- Category: `paraphrase`
- Claim: The city increased its population.
- Evidence: The urban population grew.
- Gold: `SUPPORTED`
- Predicted: `UNSUPPORTED`
- Confidence: `0.562`
- Failure reason: Correct paraphrase lacks sufficient lexical/predicate overlap.

### PAR-09
- Category: `paraphrase`
- Claim: The team won the championship.
- Evidence: The team became champions.
- Gold: `SUPPORTED`
- Predicted: `UNSUPPORTED`
- Confidence: `0.600`
- Failure reason: Correct paraphrase lacks sufficient lexical/predicate overlap.

### PAR-10
- Category: `paraphrase`
- Claim: The algorithm sorts the list.
- Evidence: The procedure orders the list.
- Gold: `SUPPORTED`
- Predicted: `UNSUPPORTED`
- Confidence: `0.600`
- Failure reason: Correct paraphrase lacks sufficient lexical/predicate overlap.

### NUM-04
- Category: `dates_numbers`
- Claim: The launch occurred on March 5, 2022.
- Evidence: The launch occurred on March 6, 2022.
- Gold: `CONTRADICTED`
- Predicted: `UNSUPPORTED`
- Confidence: `0.810`
- Failure reason: Numeric/date normalization is incomplete; only limited year/date rules apply.

### NUM-06
- Category: `dates_numbers`
- Claim: The box contains 12 items.
- Evidence: The box contains 20 items.
- Gold: `CONTRADICTED`
- Predicted: `SUPPORTED`
- Confidence: `0.787`
- Failure reason: Numeric/date normalization is incomplete; only limited year/date rules apply.

### ENT-03
- Category: `entities_jurisdiction`
- Claim: Acme acquired Beta.
- Evidence: Beta acquired Acme.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.900`
- Failure reason: Entity or relation identity is not fully modeled; lexical evidence is insufficient or misleading.

### ENT-04
- Category: `entities_jurisdiction`
- Claim: Alpha founded Beta.
- Evidence: Alpha was founded by Beta.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.900`
- Failure reason: Entity or relation identity is not fully modeled; lexical evidence is insufficient or misleading.

### ENT-05
- Category: `entities_jurisdiction`
- Claim: Dr. Lee led the study.
- Evidence: Dr. Chen led the study.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.787`
- Failure reason: Entity or relation identity is not fully modeled; lexical evidence is insufficient or misleading.

### ENT-10
- Category: `entities_jurisdiction`
- Claim: The European Union adopted the regulation.
- Evidence: The European Union rejected the regulation.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.787`
- Failure reason: Entity or relation identity is not fully modeled; lexical evidence is insufficient or misleading.

### UNS-01
- Category: `unsupported_related`
- Claim: The ocean contains a hidden city.
- Evidence: The ocean contains many species.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.675`
- Failure reason: Lexical or semantic relevance was accepted without full entailment.

### UNS-02
- Category: `unsupported_related`
- Claim: The museum opened in 1890.
- Evidence: The museum contains paintings from 1890.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.750`
- Failure reason: Lexical or semantic relevance was accepted without full entailment.

### UNS-03
- Category: `unsupported_related`
- Claim: The engine uses hydrogen.
- Evidence: The engine uses gasoline.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.750`
- Failure reason: Lexical or semantic relevance was accepted without full entailment.

### UNS-05
- Category: `unsupported_related`
- Claim: The mountain is the tallest in Europe.
- Evidence: The mountain is located in Europe.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.750`
- Failure reason: Lexical or semantic relevance was accepted without full entailment.

### UNS-09
- Category: `unsupported_related`
- Claim: The school was founded by Maria.
- Evidence: The school was founded in 1920.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.750`
- Failure reason: Lexical or semantic relevance was accepted without full entailment.

### UNS-10
- Category: `unsupported_related`
- Claim: The device is safe for children.
- Evidence: The device is widely used by children.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.750`
- Failure reason: Lexical or semantic relevance was accepted without full entailment.

### CON-06
- Category: `contradiction`
- Claim: The sample was collected on July 10, 2021.
- Evidence: The sample was collected on July 11, 2021.
- Gold: `CONTRADICTED`
- Predicted: `UNSUPPORTED`
- Confidence: `0.810`
- Failure reason: The evidence and claim did not satisfy the current overlap and predicate rules.

### CON-07
- Category: `contradiction`
- Claim: The river is not in Canada.
- Evidence: The river is in Canada.
- Gold: `CONTRADICTED`
- Predicted: `UNSUPPORTED`
- Confidence: `0.900`
- Failure reason: The evidence and claim did not satisfy the current overlap and predicate rules.

### CON-08
- Category: `contradiction`
- Claim: The company has no offices in Berlin.
- Evidence: The company has offices in Berlin.
- Gold: `CONTRADICTED`
- Predicted: `SUPPORTED`
- Confidence: `0.900`
- Failure reason: Lexical or semantic relevance was accepted without full entailment.

### CON-09
- Category: `contradiction`
- Claim: The machine cannot produce electricity.
- Evidence: The machine can produce electricity.
- Gold: `CONTRADICTED`
- Predicted: `SUPPORTED`
- Confidence: `0.787`
- Failure reason: Lexical or semantic relevance was accepted without full entailment.

### CON-10
- Category: `contradiction`
- Claim: The report never mentioned the incident.
- Evidence: The report mentioned the incident.
- Gold: `CONTRADICTED`
- Predicted: `SUPPORTED`
- Confidence: `0.900`
- Failure reason: Lexical or semantic relevance was accepted without full entailment.

### CMP-02
- Category: `compound_relations`
- Claim: A and B signed the agreement.
- Evidence: A and B signed the agreement.
- Gold: `SUPPORTED`
- Predicted: `UNSUPPORTED`
- Confidence: `0.900`
- Failure reason: A single claim contains multiple propositions, but verification is not clause-aware.

### CMP-03
- Category: `compound_relations`
- Claim: The rover mapped the crater and collected samples.
- Evidence: The rover mapped the crater.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.720`
- Failure reason: A single claim contains multiple propositions, but verification is not clause-aware.

### CMP-05
- Category: `compound_relations`
- Claim: The treaty reduced tariffs and increased trade.
- Evidence: The treaty reduced tariffs.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.720`
- Failure reason: A single claim contains multiple propositions, but verification is not clause-aware.

### CMP-10
- Category: `compound_relations`
- Claim: The report lists causes and treatments.
- Evidence: The report lists causes.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.787`
- Failure reason: A single claim contains multiple propositions, but verification is not clause-aware.

### DOM-10
- Category: `domains_coreference`
- Claim: The company developed the vaccine.
- Evidence: The company developed the software.
- Gold: `UNSUPPORTED`
- Predicted: `SUPPORTED`
- Confidence: `0.750`
- Failure reason: Generic role/coreference wording is not semantically resolved by the verifier.

## Top failure categories

1. `paraphrase` — 7 errors
1. `unsupported_related` — 6 errors
1. `contradiction` — 5 errors
1. `compound_relations` — 4 errors
1. `entities_jurisdiction` — 4 errors
