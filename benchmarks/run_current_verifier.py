"""Run the current verifier against a fixed 80-case claim-level benchmark."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.claims.models import Claim
from modules.retrieval.models import Evidence
from modules.verification.verifier import RuleBasedVerifier


ROOT = Path(__file__).resolve().parent
REPORT_PATH = ROOT / "baseline_report.md"
CASES_PATH = ROOT / "claim_level_cases.json"


@dataclass(frozen=True)
class Case:
    case_id: str
    category: str
    claim: str
    evidence: str
    gold: str
    entities: tuple[tuple[str, str], ...] = ()
    score: float = 0.9


def _case(
    case_id: str,
    category: str,
    claim: str,
    evidence: str,
    gold: str,
    entities: tuple[tuple[str, str], ...] = (),
    score: float = 0.9,
) -> Case:
    return Case(case_id, category, claim, evidence, gold, entities, score)


CASES = [
    _case("SUP-01", "direct_support", "Water freezes at 0 degrees Celsius.", "Water freezes at 0 degrees Celsius.", "SUPPORTED"),
    _case("SUP-02", "direct_support", "The Earth orbits the Sun.", "The Earth orbits the Sun.", "SUPPORTED"),
    _case("SUP-03", "direct_support", "A triangle has three sides.", "A triangle has three sides.", "SUPPORTED"),
    _case("SUP-04", "direct_support", "The Pacific Ocean is larger than the Atlantic Ocean.", "The Pacific Ocean is larger than the Atlantic Ocean.", "SUPPORTED"),
    _case("SUP-05", "direct_support", "Python was created by Guido van Rossum.", "Python was created by Guido van Rossum.", "SUPPORTED", (("Python", "PRODUCT"), ("Guido van Rossum", "PERSON"))),
    _case("SUP-06", "direct_support", "The Nile flows through Egypt.", "The Nile flows through Egypt.", "SUPPORTED", (("Nile", "LOC"), ("Egypt", "GPE"))),
    _case("SUP-07", "direct_support", "Photosynthesis uses light energy.", "Photosynthesis uses light energy.", "SUPPORTED"),
    _case("SUP-08", "direct_support", "The committee approved the proposal.", "The committee approved the proposal.", "SUPPORTED"),
    _case("SUP-09", "direct_support", "Mars has two moons.", "Mars has two moons.", "SUPPORTED", (("Mars", "LOC"),)),
    _case("SUP-10", "direct_support", "The device weighs 2 kilograms.", "The device weighs 2 kilograms.", "SUPPORTED"),

    _case("PAR-01", "paraphrase", "The company purchased the startup.", "The firm acquired the young company.", "SUPPORTED"),
    _case("PAR-02", "paraphrase", "The scientist discovered the particle.", "The researcher found the particle.", "SUPPORTED"),
    _case("PAR-03", "paraphrase", "The treaty ended the conflict.", "The agreement brought the war to an end.", "SUPPORTED"),
    _case("PAR-04", "paraphrase", "The spacecraft entered orbit.", "The probe was placed into orbit.", "SUPPORTED"),
    _case("PAR-05", "paraphrase", "The city increased its population.", "The urban population grew.", "SUPPORTED"),
    _case("PAR-06", "paraphrase", "The law prohibits smoking indoors.", "Indoor smoking is banned by the law.", "SUPPORTED"),
    _case("PAR-07", "paraphrase", "The medication reduces blood pressure.", "The treatment lowers blood pressure.", "SUPPORTED"),
    _case("PAR-08", "paraphrase", "The bridge was completed in 2010.", "Construction of the bridge finished in 2010.", "SUPPORTED"),
    _case("PAR-09", "paraphrase", "The team won the championship.", "The team became champions.", "SUPPORTED"),
    _case("PAR-10", "paraphrase", "The algorithm sorts the list.", "The procedure orders the list.", "SUPPORTED"),

    _case("NUM-01", "dates_numbers", "The event occurred in 2020.", "The event occurred in 2020.", "SUPPORTED"),
    _case("NUM-02", "dates_numbers", "The event occurred in 2020.", "The event occurred in 2021.", "CONTRADICTED"),
    _case("NUM-03", "dates_numbers", "The launch occurred on March 5, 2022.", "The launch occurred on March 5, 2022.", "SUPPORTED"),
    _case("NUM-04", "dates_numbers", "The launch occurred on March 5, 2022.", "The launch occurred on March 6, 2022.", "CONTRADICTED"),
    _case("NUM-05", "dates_numbers", "The box contains 12 items.", "The box contains 12 items.", "SUPPORTED"),
    _case("NUM-06", "dates_numbers", "The box contains 12 items.", "The box contains 20 items.", "CONTRADICTED"),
    _case("NUM-07", "dates_numbers", "The tower is 300 meters tall.", "The tower is 300 meters tall.", "SUPPORTED"),
    _case("NUM-08", "dates_numbers", "The tower is 300 meters tall.", "The tower is 250 meters tall.", "SUPPORTED"),
    _case("NUM-09", "dates_numbers", "The meeting was held on 2024-04-12.", "The meeting was held on April 12, 2024.", "SUPPORTED"),
    _case("NUM-10", "dates_numbers", "The study lasted three years.", "The study lasted 3 years.", "SUPPORTED"),

    _case("ENT-01", "entities_jurisdiction", "The President of India can grant pardons.", "The President of the United States can grant pardons.", "UNSUPPORTED", (("India", "GPE"),)),
    _case("ENT-02", "entities_jurisdiction", "The President can grant reprieves.", "The President can grant reprieves.", "SUPPORTED"),
    _case("ENT-03", "entities_jurisdiction", "Acme acquired Beta.", "Beta acquired Acme.", "UNSUPPORTED", (("Acme", "ORG"), ("Beta", "ORG"))),
    _case("ENT-04", "entities_jurisdiction", "Alpha founded Beta.", "Alpha was founded by Beta.", "UNSUPPORTED", (("Alpha", "ORG"), ("Beta", "ORG"))),
    _case("ENT-05", "entities_jurisdiction", "Dr. Lee led the study.", "Dr. Chen led the study.", "UNSUPPORTED", (("Dr. Lee", "PERSON"),)),
    _case("ENT-06", "entities_jurisdiction", "Paris is in France.", "Paris is in Texas.", "UNSUPPORTED", (("Paris", "GPE"), ("France", "GPE"))),
    _case("ENT-07", "entities_jurisdiction", "The company launched Product X.", "The company launched Product X.", "SUPPORTED", (("Product X", "PRODUCT"),)),
    _case("ENT-08", "entities_jurisdiction", "The organization operates in Kenya.", "The organization operates in Kenya.", "SUPPORTED", (("Kenya", "GPE"),)),
    _case("ENT-09", "entities_jurisdiction", "Mercury is closer to the Sun than Earth.", "Mercury is closer to the Sun than Earth.", "SUPPORTED", (("Mercury", "LOC"), ("Earth", "LOC"))),
    _case("ENT-10", "entities_jurisdiction", "The European Union adopted the regulation.", "The European Union rejected the regulation.", "UNSUPPORTED", (("European Union", "ORG"),)),

    _case("UNS-01", "unsupported_related", "The ocean contains a hidden city.", "The ocean contains many species.", "UNSUPPORTED"),
    _case("UNS-02", "unsupported_related", "The museum opened in 1890.", "The museum contains paintings from 1890.", "UNSUPPORTED"),
    _case("UNS-03", "unsupported_related", "The engine uses hydrogen.", "The engine uses gasoline.", "UNSUPPORTED"),
    _case("UNS-04", "unsupported_related", "The author won a Nobel Prize.", "The author wrote several novels.", "UNSUPPORTED"),
    _case("UNS-05", "unsupported_related", "The mountain is the tallest in Europe.", "The mountain is located in Europe.", "UNSUPPORTED"),
    _case("UNS-06", "unsupported_related", "The policy reduced unemployment.", "The policy addressed employment concerns.", "UNSUPPORTED"),
    _case("UNS-07", "unsupported_related", "The compound cures the disease.", "The compound was tested in a laboratory.", "UNSUPPORTED"),
    _case("UNS-08", "unsupported_related", "The satellite transmits weather images.", "The satellite studies atmospheric conditions.", "UNSUPPORTED"),
    _case("UNS-09", "unsupported_related", "The school was founded by Maria.", "The school was founded in 1920.", "UNSUPPORTED"),
    _case("UNS-10", "unsupported_related", "The device is safe for children.", "The device is widely used by children.", "UNSUPPORTED"),

    _case("CON-01", "contradiction", "The door is open.", "The door is not open.", "CONTRADICTED"),
    _case("CON-02", "contradiction", "The device does not contain lead.", "The device contains lead.", "CONTRADICTED"),
    _case("CON-03", "contradiction", "The team did not win.", "The team won the final.", "CONTRADICTED"),
    _case("CON-04", "contradiction", "The law is not applicable.", "The law is applicable.", "CONTRADICTED"),
    _case("CON-05", "contradiction", "The spacecraft landed on Mars in 2023.", "The spacecraft landed on Mars in 2022.", "CONTRADICTED", (("spacecraft", "PRODUCT"),)),
    _case("CON-06", "contradiction", "The sample was collected on July 10, 2021.", "The sample was collected on July 11, 2021.", "CONTRADICTED", (("sample", "OBJECT"),)),
    _case("CON-07", "contradiction", "The river is not in Canada.", "The river is in Canada.", "CONTRADICTED", (("Canada", "GPE"),)),
    _case("CON-08", "contradiction", "The company has no offices in Berlin.", "The company has offices in Berlin.", "CONTRADICTED", (("Berlin", "GPE"),)),
    _case("CON-09", "contradiction", "The machine cannot produce electricity.", "The machine can produce electricity.", "CONTRADICTED"),
    _case("CON-10", "contradiction", "The report never mentioned the incident.", "The report mentioned the incident.", "CONTRADICTED"),

    _case("CMP-01", "compound_relations", "A and B signed the agreement.", "A signed the agreement.", "UNSUPPORTED", (("A", "ORG"), ("B", "ORG"))),
    _case("CMP-02", "compound_relations", "A and B signed the agreement.", "A and B signed the agreement.", "SUPPORTED", (("A", "ORG"), ("B", "ORG"))),
    _case("CMP-03", "compound_relations", "The rover mapped the crater and collected samples.", "The rover mapped the crater.", "UNSUPPORTED", (("rover", "PRODUCT"),)),
    _case("CMP-04", "compound_relations", "The rover mapped the crater and collected samples.", "The rover mapped the crater and collected samples.", "SUPPORTED", (("rover", "PRODUCT"),)),
    _case("CMP-05", "compound_relations", "The treaty reduced tariffs and increased trade.", "The treaty reduced tariffs.", "UNSUPPORTED"),
    _case("CMP-06", "compound_relations", "The treaty reduced tariffs and increased trade.", "The treaty reduced tariffs and increased trade.", "SUPPORTED"),
    _case("CMP-07", "compound_relations", "The company opened offices in London and Berlin.", "The company opened offices in London.", "UNSUPPORTED", (("London", "GPE"), ("Berlin", "GPE"))),
    _case("CMP-08", "compound_relations", "The company opened offices in London and Berlin.", "The company opened offices in London and Berlin.", "SUPPORTED", (("London", "GPE"), ("Berlin", "GPE"))),
    _case("CMP-09", "compound_relations", "The report lists causes and treatments.", "The report lists causes and treatments.", "SUPPORTED"),
    _case("CMP-10", "compound_relations", "The report lists causes and treatments.", "The report lists causes.", "UNSUPPORTED"),

    _case("DOM-01", "domains_coreference", "The mission reached orbit.", "The mission reached orbit.", "SUPPORTED", (("mission", "PRODUCT"),)),
    _case("DOM-02", "domains_coreference", "The company reported profit.", "The company reported profit.", "SUPPORTED"),
    _case("DOM-03", "domains_coreference", "The organization provides healthcare.", "The organization provides healthcare.", "SUPPORTED"),
    _case("DOM-04", "domains_coreference", "The scientist published the result.", "The researcher published the result.", "SUPPORTED"),
    _case("DOM-05", "domains_coreference", "He invented the device.", "He invented the device.", "SUPPORTED"),
    _case("DOM-06", "domains_coreference", "She led the expedition.", "She led the expedition.", "SUPPORTED"),
    _case("DOM-07", "domains_coreference", "Its battery lasts ten hours.", "Its battery lasts ten hours.", "SUPPORTED"),
    _case("DOM-08", "domains_coreference", "The country exports coffee.", "The country exports coffee.", "SUPPORTED"),
    _case("DOM-09", "domains_coreference", "The President can commute sentences.", "The President of the United States can commute sentences.", "SUPPORTED"),
    _case("DOM-10", "domains_coreference", "The company developed the vaccine.", "The company developed the software.", "UNSUPPORTED"),
]


def _to_json(case: Case) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "category": case.category,
        "claim": case.claim,
        "evidence": case.evidence,
        "gold_verdict": case.gold,
        "entities": [{"text": text, "label": label} for text, label in case.entities],
        "relevance_score": case.score,
    }


def _run(case: Case, verifier: RuleBasedVerifier) -> dict[str, Any]:
    claim = Claim(
        claim_id=case.case_id,
        claim_text=case.claim,
        original_sentence=case.claim,
        sentence_index=0,
        entities=[{"text": text, "label": label} for text, label in case.entities],
    )
    evidence = Evidence(
        evidence_id=f"evidence:{case.case_id}",
        claim_or_query=case.claim,
        text=case.evidence,
        source_title="Benchmark evidence",
        source_url="https://benchmark.invalid/evidence",
        relevance_score=case.score,
    )
    result = verifier.verify(claim, [evidence])
    predicted = result.verdict.value
    return {
        "case": case,
        "predicted": predicted,
        "confidence": result.confidence,
        "passed": predicted == case.gold,
        "explanation": result.explanation,
    }


def _metrics(results: list[dict[str, Any]]) -> tuple[dict[str, dict[str, float]], list[list[int]]]:
    labels = ["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"]
    matrix = [[0 for _ in labels] for _ in labels]
    for item in results:
        matrix[labels.index(item["case"].gold)][labels.index(item["predicted"])] += 1
    metrics: dict[str, dict[str, float]] = {}
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        fp = sum(matrix[row][index] for row in range(3)) - tp
        fn = sum(matrix[index]) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1}
    return metrics, matrix


def _failure_reason(item: dict[str, Any]) -> str:
    case = item["case"]
    predicted = item["predicted"]
    if case.category == "paraphrase":
        return "Correct paraphrase lacks sufficient lexical/predicate overlap."
    if case.category == "dates_numbers":
        return "Numeric/date normalization is incomplete; only limited year/date rules apply."
    if case.category == "entities_jurisdiction":
        return "Entity or relation identity is not fully modeled; lexical evidence is insufficient or misleading."
    if case.category == "compound_relations":
        return "A single claim contains multiple propositions, but verification is not clause-aware."
    if case.category == "domains_coreference":
        return "Generic role/coreference wording is not semantically resolved by the verifier."
    if predicted == "SUPPORTED" and case.gold != "SUPPORTED":
        return "Lexical or semantic relevance was accepted without full entailment."
    if predicted == "CONTRADICTED" and case.gold != "CONTRADICTED":
        return "Limited negation/date logic classified the evidence as contradictory."
    return "The evidence and claim did not satisfy the current overlap and predicate rules."


def _report(results: list[dict[str, Any]], metrics: dict[str, dict[str, float]], matrix: list[list[int]]) -> str:
    labels = ["SUPPORTED", "CONTRADICTED", "UNSUPPORTED"]
    accuracy = sum(item["passed"] for item in results) / len(results)
    macro_f1 = sum(metrics[label]["f1"] for label in labels) / len(labels)
    failures = [item for item in results if not item["passed"]]
    category_errors: dict[str, int] = {}
    for item in failures:
        category_errors[item["case"].category] = category_errors.get(item["case"].category, 0) + 1
    lines = [
        "# Current verifier baseline",
        "",
        f"- Cases: **{len(results)}**",
        f"- Accuracy: **{accuracy:.3%}** ({sum(item['passed'] for item in results)}/{len(results)})",
        f"- Macro-F1: **{macro_f1:.3%}**",
        "",
        "| Class | Precision | Recall | F1 |",
        "|---|---:|---:|---:|",
    ]
    for label in labels:
        lines.append(f"| {label} | {metrics[label]['precision']:.3%} | {metrics[label]['recall']:.3%} | {metrics[label]['f1']:.3%} |")
    lines += [
        "",
        "## Confusion matrix",
        "",
        "| Gold \\\\ Predicted | SUPPORTED | CONTRADICTED | UNSUPPORTED |",
        "|---|---:|---:|---:|",
    ]
    for label, row in zip(labels, matrix):
        lines.append(f"| {label} | {row[0]} | {row[1]} | {row[2]} |")
    lines += ["", "## Accuracy by category", "", "| Category | Cases | Correct | Accuracy | Errors |", "|---|---:|---:|---:|---:|"]
    for category in sorted({item["case"].category for item in results}):
        group = [item for item in results if item["case"].category == category]
        correct = sum(item["passed"] for item in group)
        lines.append(f"| {category} | {len(group)} | {correct} | {correct / len(group):.3%} | {len(group) - correct} |")
    lines += ["", "## Failed cases", ""]
    if not failures:
        lines.append("No failures.")
    for item in failures:
        case = item["case"]
        lines += [
            f"### {case.case_id}",
            f"- Category: `{case.category}`",
            f"- Claim: {case.claim}",
            f"- Evidence: {case.evidence}",
            f"- Gold: `{case.gold}`",
            f"- Predicted: `{item['predicted']}`",
            f"- Confidence: `{item['confidence']:.3f}`",
            f"- Failure reason: {_failure_reason(item)}",
            "",
        ]
    lines += ["## Top failure categories", ""]
    for category, count in sorted(category_errors.items(), key=lambda pair: (-pair[1], pair[0]))[:5]:
        lines.append(f"1. `{category}` — {count} errors")
    return "\n".join(lines) + "\n"


def main() -> None:
    if len(CASES) != 80:
        raise RuntimeError(f"Expected 80 cases, found {len(CASES)}")
    CASES_PATH.write_text(json.dumps([_to_json(case) for case in CASES], indent=2) + "\n", encoding="utf-8")
    results = [_run(case, RuleBasedVerifier()) for case in CASES]
    metrics, matrix = _metrics(results)
    REPORT_PATH.write_text(_report(results, metrics, matrix), encoding="utf-8")
    print(_report(results, metrics, matrix))


if __name__ == "__main__":
    main()
