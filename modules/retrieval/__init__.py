"""Evidence retrieval and local knowledge-base indexing."""

from modules.retrieval.external import ExternalRetrieverError, WikipediaRetriever
from modules.retrieval.curated import CuratedFactRetriever
from modules.retrieval.hybrid import HybridRetriever
from modules.retrieval.local import LocalFaissRetriever
from modules.retrieval.models import DocumentChunk, Evidence
from modules.retrieval.multi import MultiSourceRetriever
from modules.retrieval.openalex import OpenAlexRetriever
from modules.retrieval.web import (
	DevelopmentWebSearchProvider,
	TavilyWebSearchProvider,
	WebSearchProvider,
	WebSearchResult,
	WebSearchRetriever,
)

__all__ = [
	"DocumentChunk",
	"Evidence",
	"ExternalRetrieverError",
	"CuratedFactRetriever",
	"HybridRetriever",
	"LocalFaissRetriever",
	"MultiSourceRetriever",
	"OpenAlexRetriever",
	"DevelopmentWebSearchProvider",
	"TavilyWebSearchProvider",
	"WebSearchProvider",
	"WebSearchResult",
	"WebSearchRetriever",
	"WikipediaRetriever",
]
