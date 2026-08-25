"""Claim graph construction and serialization."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable

import networkx as nx

from modules.claims.models import Claim


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
            verb_match = re.search(r"\b(is|was|are|were|has|have|built|launched|created|founded)\b", sentence, re.I)
            if verb_match:
                self.add_relationship(entities[0], entities[1], verb_match.group(1).lower(), claim.claim_id)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible graph data, including claim text provenance."""
        return {
            "nodes": [
                {"id": node_id, **dict(attributes)}
                for node_id, attributes in self.graph.nodes(data=True)
            ],
            "edges": [
                {"source": source, "target": target, **dict(attributes)}
                for source, target, attributes in self.graph.edges(data=True)
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