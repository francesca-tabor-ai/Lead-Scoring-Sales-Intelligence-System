"""
FastAPI Route Handlers
Competitor Intelligence & Market Gap Finder
"""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse

from api.schemas import (
    RunPipelineRequest, PipelineResponse, OpportunityResponse,
    ClusterResponse, HealthResponse, UpdateWeightsRequest
)
from agents import (
    Orchestrator, ScraperAgent, FeatureExtractionAgent,
    GapDetectionAgent, OpportunityScoringAgent
)
from agents.scraper import ScrapeTarget
from agents.scorer import ScoringOutput
from config.settings import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory result cache (production: use Redis)
_last_result: Optional[ScoringOutput] = None
_current_weights = {
    "alpha": settings.ALPHA,
    "beta": settings.BETA,
    "gamma": settings.GAMMA,
}


# ─── Health ──────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        timestamp=datetime.utcnow(),
    )


# ─── Pipeline ────────────────────────────────────────────────────────────────

@router.post("/pipeline/run", response_model=PipelineResponse, tags=["Pipeline"])
async def run_pipeline(request: RunPipelineRequest):
    """
    Execute the full multi-agent pipeline:
    Scrape → Extract → Gap Detect → Score → Return ranked opportunities
    """
    global _last_result

    if not request.targets:
        raise HTTPException(status_code=400, detail="At least one scrape target is required.")

    # Build ScrapeTargets
    targets = [
        ScrapeTarget(
            url=t.url,
            source_type=t.source_type,
            product_name=t.product_name,
            competitor_name=t.competitor_name,
            max_reviews=t.max_reviews,
            extra=t.extra,
        )
        for t in request.targets
    ]

    # Instantiate agents
    agents = [
        ScraperAgent(),
        FeatureExtractionAgent(),
        GapDetectionAgent(
            complaint_rate_threshold=request.complaint_rate_threshold,
            competition_density_threshold=request.competition_density_threshold,
        ),
        OpportunityScoringAgent(
            alpha=request.alpha,
            beta=request.beta,
            gamma=request.gamma,
        ),
    ]

    orchestrator = Orchestrator(agents=agents, stop_on_failure=True)

    try:
        result = orchestrator.execute(targets)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {str(e)}")

    if not result.success:
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {result.error}")

    scoring: ScoringOutput = result.data
    _last_result = scoring

    # Build response
    top_opps = [
        OpportunityResponse(**opp.to_dict())
        for opp in scoring.opportunities
    ]

    all_clusters = []
    for opp in scoring.opportunities:
        all_clusters.append(ClusterResponse(
            cluster_index=opp.cluster_index,
            label=opp.label,
            top_keywords=opp.top_keywords,
            volume=opp.volume,
            mention_freq=opp.demand_raw,
            complaint_freq=opp.complaint_rate,
            avg_rating=0.0,
            sentiment_mean=0.0,
            gap_severity=opp.gap_severity,
            is_underserved=opp.is_underserved,
            competition_density=opp.competition_raw,
            complaint_rate=opp.complaint_rate,
        ))

    underserved_count = sum(1 for o in scoring.opportunities if o.is_underserved)

    return PipelineResponse(
        status="success",
        message=f"Pipeline complete. {underserved_count} underserved gaps identified.",
        total_reviews=scoring.total_reviews,
        total_clusters=len(scoring.opportunities),
        underserved_gaps=underserved_count,
        top_opportunities=top_opps,
        all_clusters=all_clusters,
        weights=scoring.weights,
        run_at=datetime.utcnow(),
    )


# ─── Opportunities ───────────────────────────────────────────────────────────

@router.get("/opportunities", response_model=List[OpportunityResponse], tags=["Opportunities"])
async def get_opportunities(top_k: int = 50, underserved_only: bool = False):
    """Get ranked opportunities from the last pipeline run."""
    if _last_result is None:
        raise HTTPException(status_code=404, detail="No pipeline results available. Run /pipeline/run first.")

    opps = _last_result.opportunities
    if underserved_only:
        opps = [o for o in opps if o.is_underserved]

    return [OpportunityResponse(**o.to_dict()) for o in opps[:top_k]]


@router.get("/opportunities/{cluster_index}", response_model=OpportunityResponse, tags=["Opportunities"])
async def get_opportunity_by_cluster(cluster_index: int):
    """Get detailed opportunity for a specific cluster."""
    if _last_result is None:
        raise HTTPException(status_code=404, detail="No pipeline results available.")

    for opp in _last_result.opportunities:
        if opp.cluster_index == cluster_index:
            return OpportunityResponse(**opp.to_dict())

    raise HTTPException(status_code=404, detail=f"Cluster {cluster_index} not found.")


@router.get("/opportunities/{cluster_index}/evidence", tags=["Opportunities"])
async def get_cluster_evidence(cluster_index: int, limit: int = 20):
    """Get representative reviews (evidence) for a cluster."""
    if _last_result is None:
        raise HTTPException(status_code=404, detail="No pipeline results available.")

    for opp in _last_result.opportunities:
        if opp.cluster_index == cluster_index:
            return {
                "cluster_index": cluster_index,
                "label": opp.label,
                "reviews": opp.representative_reviews[:limit],
                "competitor_coverage": [
                    {
                        "competitor": c.competitor_name,
                        "covers": c.covers,
                        "smoothed_share": round(c.smoothed_share, 4),
                        "evidence_count": c.evidence_count,
                    }
                    for c in opp.competitor_coverages
                ],
            }
    raise HTTPException(status_code=404, detail=f"Cluster {cluster_index} not found.")


# ─── Scoring Weights ─────────────────────────────────────────────────────────

@router.get("/score/weights", tags=["Configuration"])
async def get_weights():
    """Get current scoring weights."""
    return _current_weights


@router.post("/score/weights", tags=["Configuration"])
async def update_weights(request: UpdateWeightsRequest):
    """Update scoring weights (α, β, γ)."""
    global _current_weights
    _current_weights = {
        "alpha": request.alpha,
        "beta": request.beta,
        "gamma": request.gamma,
    }
    # Update settings in memory
    settings.ALPHA = request.alpha
    settings.BETA = request.beta
    settings.GAMMA = request.gamma
    return {"status": "updated", "weights": _current_weights}


# ─── Summary ─────────────────────────────────────────────────────────────────

@router.get("/summary", tags=["Pipeline"])
async def get_summary():
    """Get a text summary of the last run."""
    if _last_result is None:
        raise HTTPException(status_code=404, detail="No pipeline results available.")
    return {
        "summary": _last_result.summary(),
        "total_opportunities": len(_last_result.opportunities),
        "underserved": sum(1 for o in _last_result.opportunities if o.is_underserved),
        "weights": _last_result.weights,
    }
