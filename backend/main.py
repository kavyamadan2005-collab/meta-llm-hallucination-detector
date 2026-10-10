"""FastAPI application entry point for the project foundation."""

from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.core.config import get_settings
from backend.core.logging import configure_logging
from backend.api.analysis import AnalysisComponents, create_analysis_components
from backend.api.schemas import AnalyzeRequest, AnalyzeResponse, OverallResult
from modules.graph import ClaimGraph
from modules.llm import LLMProviderError

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def _cached_analysis_components() -> AnalysisComponents:
    """Construct production analysis services once per process."""
    return create_analysis_components(settings)


def get_analysis_components(request: AnalyzeRequest) -> AnalysisComponents:
    """Provide analysis services through FastAPI dependency injection."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        return _cached_analysis_components()
    except (RuntimeError, ValueError, TypeError) as exc:
        logger.warning("Analysis component initialization failed: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Analysis pipeline unavailable") from exc


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    """Return the current service health status."""
    return {"status": "ok", "environment": settings.environment}


@app.post("/analyze", response_model=AnalyzeResponse, tags=["analysis"])
def analyze(
    request: AnalyzeRequest,
    components: AnalysisComponents = Depends(get_analysis_components),
) -> AnalyzeResponse:
    """Run generation, claim analysis, retrieval, and verification in order."""
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")
    try:
        llm_response = components.provider.generate(question)
        claims = components.extractor.extract(llm_response.response_text)
        graph = ClaimGraph()
        graph.add_claims(claims)
        verification_results = components.verifier.verify_claims(claims, components.retrieval_top_k)
        contradicted = sum(
            result.verdict.value == "CONTRADICTED"
            for result in verification_results
        )
        insufficient = sum(
            result.verdict.value == "INSUFFICIENT_EVIDENCE"
            for result in verification_results
        )
        if contradicted:
            overall_status = "HALLUCINATION_DETECTED"
            overall_explanation = "One or more claims contradict the retrieved evidence."
        elif insufficient or not verification_results:
            overall_status = "INCONCLUSIVE"
            overall_explanation = "One or more claims lack sufficient evidence for verification."
        else:
            overall_status = "VERIFIED"
            overall_explanation = "All extracted claims are supported by the retrieved evidence."
        overall_confidence = (
            sum(result.confidence for result in verification_results)
            / len(verification_results)
            if verification_results
            else 0.0
        )
        return AnalyzeResponse(
            question=question,
            llm_response=llm_response,
            claims=claims,
            graph=graph.to_dict(),
            verification_results=verification_results,
            overall_result=OverallResult(
                status=overall_status,
                confidence=overall_confidence,
                explanation=overall_explanation,
            ),
        )
    except LLMProviderError as exc:
        logger.warning("Analysis provider failure: %s", str(exc))
        raise HTTPException(status_code=502, detail="LLM provider failed") from exc
    except (RuntimeError, ValueError, TypeError) as exc:
        logger.warning("Analysis pipeline failure: %s", type(exc).__name__)
        raise HTTPException(status_code=503, detail="Analysis pipeline unavailable") from exc


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Return a consistent JSON response for expected HTTP errors."""
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log unexpected errors without exposing internal details to clients."""
    logger.exception("Unhandled exception while processing %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
