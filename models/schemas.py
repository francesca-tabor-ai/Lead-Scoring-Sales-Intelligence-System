"""
Core data models / schemas for the Competitor Intelligence system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime


# ---------------------------------------------------------------------------
# Raw ingestion
# ---------------------------------------------------------------------------

@dataclass
class Review:
    review_id: str
    product_id: str
    competitor: str
    source: str                     # e.g. "g2", "app_store", "amazon", "mock"
    rating: float                   # 1–5
    text: str
    date: Optional[datetime] = None
    locale: str = "en"
    author_hash: Optional[str] = None
    url: Optional[str] = None
    # enriched fields (filled by FeatureExtractionAgent)
    cleaned_text: Optional[str] = None
    language: Optional[str] = None
    sentiment_score: Optional[float] = None   # –1 to +1
    complaint_flag: Optional[int] = None       # 0 or 1
    aspects: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    cluster_id: Optional[int] = None
    distance_to_centroid: Optional[float] = None


# ---------------------------------------------------------------------------
# Cluster-level artefacts
# ---------------------------------------------------------------------------

@dataclass
class ClusterMetrics:
    cluster_id: int
    label: str                          # human-readable theme
    size: int                           # |C_k|
    top_terms: List[str]                # TF-IDF top terms
    representative_reviews: List[str]   # nearest-centroid review texts
    mention_freq: float                 # |C_k| / N
    complaint_rate: float               # complaints / |C_k|
    complaint_intensity: float          # avg max(0, –sent)
    avg_rating: float
    sentiment_mean: float
    model_version: str = "v1"
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CompetitorCoverage:
    cluster_id: int
    competitor: str
    raw_share: float
    smoothed_share: float
    covers: bool            # smoothed_share > tau_cov


@dataclass
class GapResult:
    cluster_id: int
    label: str
    complaint_rate: float
    competition_density: float      # fraction of competitors covering cluster
    gap_severity: float             # continuous [0,1]
    is_underserved: bool
    demand: float
    sentiment_mean: float
    competitor_coverages: List[CompetitorCoverage] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Opportunity scoring
# ---------------------------------------------------------------------------

@dataclass
class OpportunityScore:
    cluster_id: int
    label: str
    demand_raw: float
    competition_raw: float
    neg_sentiment_raw: float
    demand_q: float             # quantile-normalised
    competition_q: float
    neg_sentiment_q: float
    raw_score: float            # before confidence scaling
    confidence: float           # 1 – exp(–n / eta)
    final_score: float          # confidence * raw_score
    rank: int
    ci_low: float = 0.0
    ci_high: float = 0.0
    components_json: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pipeline run summary
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    run_id: str
    category: str
    total_reviews: int
    n_clusters: int
    n_gaps: int
    opportunities: List[OpportunityScore]
    gaps: List[GapResult]
    cluster_metrics: List[ClusterMetrics]
    model_version: str
    executed_at: datetime = field(default_factory=datetime.utcnow)
