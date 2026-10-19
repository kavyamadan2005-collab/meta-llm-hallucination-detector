"""Claim graph construction and serialization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from modules.claims.models import Claim
from modules.verification.models import VerificationResult


class ClaimGraph:
    """Directed entity graph with claim provenance on nodes and edges."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()
        self._claims: dict[str, str] = {}

    @staticmethod
    def _key(entity_text: str) -> str:
        return re.sub(r"\s+", " ", entity_text).strip().casefold()

    def add_entity(self, text: str, entity_type: str, claim_id: str | None = None) -> str:
        """Add or reuse an entity node and return its stable graph key."""
        key = self._key(text)
        if not key:
            raise ValueError("entity text must not be empty")
        if key not in self.graph:
            self.graph.add_node(key, text=text.strip(), entity_type=entity_type, claim_ids=[])
        elif entity_type and not self.graph.nodes[key].get("entity_type"):
            self.graph.nodes[key]["entity_type"] = entity_type
        if claim_id and claim_id not in self.graph.nodes[key]["claim_ids"]:
            self.graph.nodes[key]["claim_ids"].append(claim_id)
        return key

    def add_relationship(
        self,
        source: str,
        target: str,
        relationship_type: str,
        claim_id: str | None = None,
    ) -> None:
        """Add a directed relationship and preserve all claim provenance."""
        if not relationship_type.strip():
            raise ValueError("relationship_type must not be empty")
        attributes = self.graph.get_edge_data(source, target, default={})
        if not attributes:
            self.graph.add_edge(source, target, relationship_type=relationship_type, claim_ids=[])
            attributes = self.graph[source][target]
        if claim_id and claim_id not in attributes["claim_ids"]:
            attributes["claim_ids"].append(claim_id)

    def add_claim(self, claim: Claim) -> None:
        """Add claim entities and only relationships supported by simple patterns."""
        self._claims[claim.claim_id] = claim.claim_text
        relation_entities = self._relation_entities(claim.claim_text)
        if relation_entities:
            source_text, target_text, relationship_type = relation_entities
            source = self.add_entity(source_text, "DOMAIN_ENTITY", claim.claim_id)
            target = self.add_entity(target_text, "DOMAIN_ENTITY", claim.claim_id)
            self.add_relationship(
                source,
                target,
                relationship_type,
                claim.claim_id,
            )
            return
        entities = [
            self.add_entity(entity.text, entity.label, claim.claim_id)
            for entity in claim.entities
        ]
        self._add_inferred_relationships(claim, entities)

    def add_claims(self, claims: Iterable[Claim]) -> None:
        """Add multiple claims to this graph."""
        for claim in claims:
            self.add_claim(claim)

    def _add_inferred_relationships(self, claim: Claim, entities: list[str]) -> None:
        if len(entities) < 2:
            return
        sentence = claim.claim_text
        by_match = re.search(r"\b(?:launched|created|founded|developed)\s+(.+?)\s+by\s+", sentence, re.I)
        in_match = re.search(r"\b(?:launched|created|founded|developed)\s+(.+?)\s+in\s+", sentence, re.I)
        if by_match:
            self.add_relationship(entities[0], entities[1], "launched_by", claim.claim_id)
            if len(entities) >= 3:
                self.add_relationship(entities[0], entities[2], "launched_in", claim.claim_id)
        elif in_match:
            if len(entities) >= 3:
                self.add_relationship(entities[1], entities[0], "launched_by", claim.claim_id)
                self.add_relationship(entities[1], entities[2], "launched_in", claim.claim_id)
            else:
                self.add_relationship(entities[0], entities[1], "launched_in", claim.claim_id)
        elif len(entities) == 2:
            verb_match = re.search(
                r"\b(is|was|are|were|has|have|built|launched|created|founded|"
                r"orbits?|revolves?\s+around|depends?\s+on|located\s+in|"
                r"capital\s+of|wrote|developed)\b",
                sentence,
                re.I,
            )
            if verb_match:
                relationship = re.sub(r"\s+", "_", verb_match.group(1).lower())
                self.add_relationship(
                    entities[0],
                    entities[1],
                    relationship,
                    claim.claim_id,
                )

    @staticmethod
    def _relation_entities(
        sentence: str,
    ) -> tuple[str, str, str] | None:
        """Extract only explicit, capitalized subject-predicate-object wording."""
        capital_match = re.search(
            r"\b(?P<source>[A-Z][A-Za-z0-9-]*)\s+is\s+"
            r"(?:not\s+)?(?:the\s+)?capital(?:\s+city)?\s+of\s+"
            r"(?:the\s+)?(?P<target>[A-Z][A-Za-z0-9-]*)\b",
            sentence,
        )
        if capital_match:
            return (
                capital_match.group("source"),
                capital_match.group("target"),
                "capital_of",
            )

        reverse_capital_match = re.search(
            r"\b(?:the\s+)?capital(?:\s+city)?\s+of\s+"
            r"(?:the\s+)?(?P<target>[A-Z][A-Za-z0-9-]*)\s+is\s+"
            r"(?:the\s+)?(?P<source>[A-Z][A-Za-z0-9-]*)\b",
            sentence,
        )
        if reverse_capital_match:
            return (
                reverse_capital_match.group("source"),
                reverse_capital_match.group("target"),
                "capital_of",
            )

        settled_match = re.search(
            r"\b(?P<target>[A-Z][A-Za-z0-9-]*)\s+is\s+mentioned\s+"
            r"as\s+a\s+place\s+where\s+(?:some\s+)?"
            r"(?P<source>[A-Z][A-Za-z0-9-]*)s?\s+have\s+settled\b",
            sentence,
        )
        if settled_match:
            return (
                settled_match.group("source"),
                settled_match.group("target"),
                "settled_in",
            )

        match = re.search(
            r"\b(?P<source>(?:the\s+)?[A-Z][A-Za-z0-9-]*)\s+"
            r"(?P<predicate>orbits?|revolves?\s+around|depends?\s+on|"
            r"located\s+in|wrote|developed)\s+"
            r"(?:the\s+)?(?P<target>[A-Z][A-Za-z0-9-]*)\b",
            sentence,
        )
        if match:
            source = re.sub(r"^the\s+", "", match.group("source"), flags=re.I)
            return (
                source,
                match.group("target"),
                re.sub(r"\s+", "_", match.group("predicate").casefold()),
            )

        generic_match = re.search(
            r"\b(?P<source>(?:the\s+)?[A-Z][A-Za-z0-9-]*)\s+"
            r"(?P<predicate>is|was|are|were|has|have)\s+"
            r"(?:the\s+)?(?P<target>[A-Z][A-Za-z0-9-]*)\b",
            sentence,
        )
        if not generic_match:
            return None
        source = re.sub(r"^the\s+", "", generic_match.group("source"), flags=re.I)
        target = generic_match.group("target")
        predicate = re.sub(
            r"\s+",
            "_",
            generic_match.group("predicate").casefold(),
        )
        return source, target, predicate

    def annotate_verification(
        self,
        results: Iterable[VerificationResult],
    ) -> None:
        """Attach claim verdicts and evidence provenance to relationship edges."""
        by_claim = {result.claim_id: result for result in results}
        for source, target, attributes in self.graph.edges(data=True):
            claim_ids = attributes.get("claim_ids", [])
            claim_results = [
                by_claim[claim_id]
                for claim_id in claim_ids
                if claim_id in by_claim
            ]
            if not claim_results:
                continue

            if any(result.verdict.value == "CONTRADICTED" for result in claim_results):
                status = "CONTRADICTED"
            elif any(
                result.verdict.value == "INSUFFICIENT_EVIDENCE"
                for result in claim_results
            ):
                status = "INSUFFICIENT_EVIDENCE"
            else:
                status = "SUPPORTED"

            evidence = []
            for result in claim_results:
                for item in [
                    *result.supporting_evidence,
                    *result.contradicting_evidence,
                ]:
                    serialized = item.model_dump(mode="json")
                    if serialized not in evidence:
                        evidence.append(serialized)

            attributes["verification_status"] = status
            attributes["claim_texts"] = [
                self._claims[claim_id]
                for claim_id in claim_ids
                if claim_id in self._claims
            ]
            attributes["evidence"] = evidence
            attributes["confidence"] = max(
                result.confidence for result in claim_results
            )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible graph data, including claim text provenance."""
        return {
            "nodes": [
                {"id": node_id, **dict(attributes)}
                for node_id, attributes in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "id": f"relationship-{index + 1}",
                    "source": source,
                    "target": target,
                    **dict(attributes),
                }
                for index, (source, target, attributes)
                in enumerate(self.graph.edges(data=True))
            ],
            "claims": [
                {"claim_id": claim_id, "claim_text": text}
                for claim_id, text in self._claims.items()
            ],
        }

    def export_graphml(self, path: str | Path) -> None:
        """Export the graph in GraphML format."""
        graph = self.graph.copy()
        for _, attributes in graph.nodes(data=True):
            attributes["claim_ids"] = ",".join(attributes.get("claim_ids", []))
        for _, _, attributes in graph.edges(data=True):
            attributes["claim_ids"] = ",".join(attributes.get("claim_ids", []))
        nx.write_graphml(graph, Path(path))