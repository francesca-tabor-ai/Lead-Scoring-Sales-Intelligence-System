"""
Pipeline runner — wires all four agents together and returns a PipelineResult.

Architecture:
  ScraperAgent → FeatureExtractionAgent → GapDetectionAgent → OpportunityScoringAgent
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.base import Orchestrator
from agents.scraper import ScraperAgent, ScrapeTarget
from agents.extractor import FeatureExtractionAgent
from agents.gap_detector import GapDetectionAgent
from agents.scorer import OpportunityScoringAgent, ScoringOutput
from models.schemas import (
    PipelineResult, OpportunityScore, GapResult,
    ClusterMetrics, CompetitorCoverage,
)
from config.settings import settings

logger = logging.getLogger(__name__)


def run_pipeline_native(
    targets: List[Dict[str, Any]],
    alpha: float = settings.ALPHA,
    beta:  float = settings.BETA,
    gamma: float = settings.GAMMA,
    complaint_threshold: Optional[float] = None,
    competition_threshold: Optional[float] = None,
) -> ScoringOutput:
    """
    Runs the four-agent pipeline and returns the raw ScoringOutput.
    """
    scrape_targets = [
        ScrapeTarget(
            url=t.get("url_or_id", ""),
            source_type=t.get("source", "mock"),
            product_name=t.get("product_id", "unknown"),
            competitor_name=t.get("competitor", "unknown"),
            max_reviews=t.get("max_reviews", settings.MAX_REVIEWS_PER_PRODUCT),
            extra=t.get("extra", {}),
        )
        for t in targets
    ]

    pipeline = Orchestrator([
        ScraperAgent(),
        FeatureExtractionAgent(),
        GapDetectionAgent(
            complaint_rate_threshold=complaint_threshold,
            competition_density_threshold=competition_threshold,
        ),
        OpportunityScoringAgent(alpha=alpha, beta=beta, gamma=gamma),
    ])

    result = pipeline.execute(scrape_targets)
    if not result.success:
        raise RuntimeError(f"Pipeline failed: {result.error}")

    logger.info(pipeline.summary())
    return result.data


def run_pipeline(
    targets: List[Dict[str, Any]],
    category: str = "general",
    mock: bool = True,
    scoring_weights: Optional[Dict[str, float]] = None,
    k_min: int = 3,
    k_max: int = 10,
    complaint_threshold: float = settings.COMPLAINT_RATE_THRESHOLD,
    competition_threshold: float = settings.COMPETITION_DENSITY_THRESHOLD,
    n_bootstrap: int = 0,
    verbose: bool = True,
) -> PipelineResult:
    """
    End-to-end pipeline execution returning a PipelineResult.

    Parameters
    ----------
    targets : list[dict]
        Each dict: {competitor, product_id, source, url_or_id, max_reviews?}
    category : str
        Label for this run.
    mock : bool
        If True, all sources are overridden to 'mock'.
    scoring_weights : dict, optional
        {"alpha": ..., "beta": ..., "gamma": ...}
    """
    weights = scoring_weights or {}
    alpha = weights.get("alpha", settings.ALPHA)
    beta  = weights.get("beta",  settings.BETA)
    gamma = weights.get("gamma", settings.GAMMA)

    if mock:
        for t in targets:
            t["source"] = "mock"

    scoring: ScoringOutput = run_pipeline_native(
        targets=targets,
        alpha=alpha, beta=beta, gamma=gamma,
        complaint_threshold=complaint_threshold,
        competition_threshold=competition_threshold,
    )

    # ── Convert to unified PipelineResult ─────────────────────────────
    opportunities: List[OpportunityScore] = [
        OpportunityScore(
            cluster_id=o.cluster_index,
            label=o.label,
            demand_raw=o.demand_raw,
            competition_raw=o.competition_raw,
            neg_sentiment_raw=o.neg_sentiment_score,
            demand_q=o.demand_score,
            competition_q=o.competition_score,
            neg_sentiment_q=o.neg_sentiment_score,
            raw_score=o.raw_score,
            confidence=o.confidence,
            final_score=o.final_score,
            rank=o.rank,
            ci_low=0.0,
            ci_high=0.0,
            components_json={
                **o.components,
                "is_underserved": o.is_underserved,
                "gap_severity": o.gap_severity,
                "recommended_action": o.recommended_action,
            },
        )
        for o in scoring.opportunities
    ]

    gaps: List[GapResult] = [
        GapResult(
            cluster_id=o.cluster_index,
            label=o.label,
            complaint_rate=o.complaint_rate,
            competition_density=o.competition_raw,
            gap_severity=o.gap_severity,
            is_underserved=o.is_underserved,
            demand=o.demand_raw,
            sentiment_mean=1.0 - 2.0 * o.neg_sentiment_score,
            competitor_coverages=[
                CompetitorCoverage(
                    cluster_id=o.cluster_index,
                    competitor=c.competitor_name,
                    raw_share=c.raw_share,
                    smoothed_share=c.smoothed_share,
                    covers=c.covers,
                )
                for c in (o.competitor_coverages or [])
            ],
        )
        for o in scoring.opportunities
    ]

    cluster_metrics: List[ClusterMetrics] = [
        ClusterMetrics(
            cluster_id=o.cluster_index,
            label=o.label,
            size=o.volume,
            top_terms=o.top_keywords[:8],
            representative_reviews=o.representative_reviews[:3],
            mention_freq=o.demand_raw,
            complaint_rate=o.complaint_rate,
            complaint_intensity=0.0,
            avg_rating=0.0,
            sentiment_mean=1.0 - 2.0 * o.neg_sentiment_score,
            model_version=scoring.model_version,
        )
        for o in scoring.opportunities
    ]

    return PipelineResult(
        run_id=str(uuid.uuid4()),
        category=category,
        total_reviews=scoring.total_reviews,
        n_clusters=len(cluster_metrics),
        n_gaps=sum(1 for g in gaps if g.is_underserved),
        opportunities=opportunities,
        gaps=gaps,
        cluster_metrics=cluster_metrics,
        model_version=scoring.model_version,
        executed_at=datetime.utcnow(),
    )
