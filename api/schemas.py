"""
Pydantic schemas for API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ─── Request Schemas ─────────────────────────────────────────────────────────

class ScrapeTargetRequest(BaseModel):
    url: str
    source_type: str = Field("mock", description="g2 | trustpilot | app_store | mock")
    product_name: str
    competitor_name: str
    max_reviews: int = Field(200, ge=1, le=1000)
    extra: Dict[str, Any] = {}


class RunPipelineRequest(BaseModel):
    targets: List[ScrapeTargetRequest]
    alpha: float = Field(0.5, ge=0, le=1)
    beta: float  = Field(0.3, ge=0, le=1)
    gamma: float = Field(0.2, ge=0, le=1)
    complaint_rate_threshold: float = Field(0.30, ge=0, le=1)
    competition_density_threshold: float = Field(0.20, ge=0, le=1)
    category: str = "default"


class UpdateWeightsRequest(BaseModel):
    alpha: float = Field(..., ge=0, le=1)
    beta: float  = Field(..., ge=0, le=1)
    gamma: float = Field(..., ge=0, le=1)


# ─── Response Schemas ────────────────────────────────────────────────────────

class OpportunityResponse(BaseModel):
    rank: int
    cluster_index: int
    label: str
    top_keywords: List[str]
    final_score: float
    raw_score: float
    confidence: float
    complaint_rate: float
    complaint_pct: str
    competition_density_pct: str
    demand_raw: float
    volume: int
    gap_severity: float
    is_underserved: bool
    components: Dict[str, float]
    recommended_action: str
    representative_reviews: List[str]


class ClusterResponse(BaseModel):
    cluster_index: int
    label: str
    top_keywords: List[str]
    volume: int
    mention_freq: float
    complaint_freq: float
    avg_rating: float
    sentiment_mean: float
    gap_severity: float
    is_underserved: bool
    competition_density: float
    complaint_rate: float


class PipelineResponse(BaseModel):
    status: str
    message: str
    total_reviews: int
    total_clusters: int
    underserved_gaps: int
    top_opportunities: List[OpportunityResponse]
    all_clusters: List[ClusterResponse]
    weights: Dict[str, float]
    run_at: datetime


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
