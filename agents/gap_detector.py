"""
Gap Detection Agent
--------------------
Identifies underserved clusters using:

  Underserved = High Complaint Frequency AND Low Competition Density

Mathematical model:
  - Competition Density = smoothed share of competitors that "cover" a cluster
  - Complaint Rate = fraction of cluster reviews that are complaints
  - Continuous gap severity via logistic(a*z(CR) - b*z(CD) + c*z(Demand))

Input:  ExtractorOutput
Output: GapDetectionOutput
"""

import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

from agents.base import Agent
from agents.extractor import ExtractorOutput, ClusterSummary, EnrichedReview
from config.settings import settings

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class CompetitorCoverage:
    """Coverage stats for one competitor × one cluster."""
    competitor_name: str
    cluster_index: int
    raw_share: float           # n_jk / n_j
    smoothed_share: float      # (n_jk + m*p_k) / (n_j + m)
    covers: bool               # smoothed_share > threshold
    evidence_count: int        # review count in this cluster for this competitor


@dataclass
class GapCluster:
    """An underserved cluster with gap metrics."""
    cluster_index: int
    label: str
    top_keywords: List[str]

    # Core gap metrics
    complaint_rate: float           # 0-1
    complaint_intensity: float      # avg magnitude of negative sentiment
    competition_density: float      # 0-1 (fraction of competitors covering)
    demand: float                   # mention_freq (0-1)
    volume: int

    # Gap scores
    gap_severity: float             # continuous [0, 1]
    is_underserved: bool            # hard threshold decision

    # Coverage breakdown
    competitor_coverages: List[CompetitorCoverage] = field(default_factory=list)

    # Cluster stats
    avg_rating: float = 0.0
    sentiment_mean: float = 0.0
    representative_reviews: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "cluster_index": self.cluster_index,
            "label": self.label,
            "top_keywords": self.top_keywords,
            "complaint_rate": round(self.complaint_rate, 4),
            "complaint_intensity": round(self.complaint_intensity, 4),
            "competition_density": round(self.competition_density, 4),
            "demand": round(self.demand, 4),
            "volume": self.volume,
            "gap_severity": round(self.gap_severity, 4),
            "is_underserved": self.is_underserved,
            "avg_rating": round(self.avg_rating, 2),
            "sentiment_mean": round(self.sentiment_mean, 4),
            "competitor_coverage_pct": round(self.competition_density * 100, 1),
            "complaint_pct": round(self.complaint_rate * 100, 1),
            "competitors_covering": [
                c.competitor_name for c in self.competitor_coverages if c.covers
            ],
        }


@dataclass
class GapDetectionOutput:
    """Full output of the GapDetectionAgent."""
    all_clusters: List[GapCluster]
    underserved_clusters: List[GapCluster]
    total_reviews: int
    total_competitors: int
    complaint_rate_threshold: float
    competition_density_threshold: float

    @property
    def gap_count(self) -> int:
        return len(self.underserved_clusters)

    def summary(self) -> str:
        lines = [
            f"Gap Detection Results:",
            f"  Total clusters analyzed: {len(self.all_clusters)}",
            f"  Underserved gaps found: {self.gap_count}",
            f"  Complaint rate threshold: {self.complaint_rate_threshold:.0%}",
            f"  Competition density threshold: {self.competition_density_threshold:.0%}",
            "",
        ]
        for i, gap in enumerate(self.underserved_clusters[:5], 1):
            lines.append(
                f"  {i}. [{gap.label}] "
                f"Complaint={gap.complaint_rate:.0%} "
                f"Coverage={gap.competition_density:.0%} "
                f"Severity={gap.gap_severity:.2f}"
            )
        return "\n".join(lines)


# ─── Coverage Calculation ─────────────────────────────────────────────────────


def compute_competition_density(
    cluster_index: int,
    enriched_reviews: List[EnrichedReview],
    smoothing_m: int = settings.SMOOTHING_STRENGTH,
    coverage_threshold: float = settings.COVERAGE_SHARE_THRESHOLD,
) -> Tuple[float, List[CompetitorCoverage]]:
    """
    Compute how many competitors "cover" a cluster using smoothed share:

        smoothed_share_jk = (n_jk + m * p_k) / (n_j + m)

    coverage_density = (# competitors covering cluster k) / J

    Returns (density, [CompetitorCoverage])
    """
    total_in_cluster = sum(1 for r in enriched_reviews if r.cluster_index == cluster_index)
    total_reviews = len(enriched_reviews)

    if total_reviews == 0:
        return 0.0, []

    p_k = total_in_cluster / total_reviews  # global cluster proportion

    # Group by competitor
    competitor_counts: Dict[str, Dict] = defaultdict(lambda: {"total": 0, "in_cluster": 0})
    for r in enriched_reviews:
        competitor_counts[r.competitor_name]["total"] += 1
        if r.cluster_index == cluster_index:
            competitor_counts[r.competitor_name]["in_cluster"] += 1

    J = len(competitor_counts)
    if J == 0:
        return 0.0, []

    coverages = []
    covering_count = 0

    for comp_name, counts in competitor_counts.items():
        n_j = counts["total"]
        n_jk = counts["in_cluster"]

        raw_share = n_jk / max(n_j, 1)
        smoothed_share = (n_jk + smoothing_m * p_k) / (n_j + smoothing_m)
        covers = smoothed_share > coverage_threshold

        if covers:
            covering_count += 1

        coverages.append(CompetitorCoverage(
            competitor_name=comp_name,
            cluster_index=cluster_index,
            raw_share=raw_share,
            smoothed_share=smoothed_share,
            covers=covers,
            evidence_count=n_jk,
        ))

    density = covering_count / J
    return density, coverages


# ─── Gap Severity Score ───────────────────────────────────────────────────────


def logistic(x: float) -> float:
    """Standard logistic / sigmoid function."""
    return 1.0 / (1.0 + math.exp(-x))


def z_score(value: float, mean: float, std: float) -> float:
    """Standardize to z-score; returns 0 if std is zero."""
    if std == 0:
        return 0.0
    return (value - mean) / std


def compute_gap_severity(
    complaint_rate: float,
    competition_density: float,
    demand: float,
    cr_mean: float, cr_std: float,
    cd_mean: float, cd_std: float,
    d_mean: float,  d_std: float,
    a: float = 1.5,
    b: float = 1.0,
    c: float = 0.8,
) -> float:
    """
    Continuous gap severity score in [0, 1]:

      GapSeverity = σ(a*z(CR) - b*z(CD) + c*z(Demand))

    High CR + Low CD + High Demand → high severity.
    """
    z_cr = z_score(complaint_rate, cr_mean, cr_std)
    z_cd = z_score(competition_density, cd_mean, cd_std)
    z_d = z_score(demand, d_mean, d_std)

    raw = a * z_cr - b * z_cd + c * z_d
    return logistic(raw)


# ─── GapDetectionAgent ───────────────────────────────────────────────────────


class GapDetectionAgent(Agent):
    """
    Agent 3: Gap Detection

    For each cluster:
      1. Compute complaint rate (from ClusterSummary)
      2. Compute competition density via smoothed share
      3. Apply hard underserved threshold
      4. Compute continuous gap severity score
    """

    def __init__(
        self,
        complaint_rate_threshold: Optional[float] = None,
        competition_density_threshold: Optional[float] = None,
    ):
        super().__init__(name="GapDetectionAgent")
        self.cr_threshold = complaint_rate_threshold or settings.COMPLAINT_RATE_THRESHOLD
        self.cd_threshold = competition_density_threshold or settings.COMPETITION_DENSITY_THRESHOLD

    def run(self, extractor_output: ExtractorOutput) -> GapDetectionOutput:
        clusters: List[ClusterSummary] = extractor_output.clusters
        reviews: List[EnrichedReview] = extractor_output.enriched_reviews

        if not clusters:
            raise ValueError("No clusters provided to GapDetectionAgent")

        self.logger.info(
            f"Analyzing {len(clusters)} clusters across "
            f"{len(set(r.competitor_name for r in reviews))} competitors..."
        )

        # Collect raw metrics for normalization
        complaint_rates = [c.complaint_freq for c in clusters]
        demands = [c.mention_freq for c in clusters]

        # First pass: compute competition density for each cluster
        gap_clusters: List[GapCluster] = []
        comp_densities: List[float] = []

        for cluster in clusters:
            comp_density, coverages = compute_competition_density(
                cluster_index=cluster.cluster_index,
                enriched_reviews=reviews,
            )
            comp_densities.append(comp_density)

            gap_clusters.append(GapCluster(
                cluster_index=cluster.cluster_index,
                label=cluster.label,
                top_keywords=cluster.top_keywords,
                complaint_rate=cluster.complaint_freq,
                complaint_intensity=cluster.complaint_intensity,
                competition_density=comp_density,
                demand=cluster.mention_freq,
                volume=cluster.volume,
                gap_severity=0.0,  # computed next
                is_underserved=False,
                competitor_coverages=coverages,
                avg_rating=cluster.avg_rating,
                sentiment_mean=cluster.sentiment_mean,
                representative_reviews=cluster.representative_reviews,
            ))

        # Compute normalization stats
        cr_arr = np.array(complaint_rates)
        cd_arr = np.array(comp_densities)
        d_arr = np.array(demands)

        cr_mean, cr_std = float(cr_arr.mean()), float(cr_arr.std()) or 1e-6
        cd_mean, cd_std = float(cd_arr.mean()), float(cd_arr.std()) or 1e-6
        d_mean,  d_std  = float(d_arr.mean()),  float(d_arr.std())  or 1e-6

        # Second pass: compute gap severity + underserved flag
        total_competitors = len(set(r.competitor_name for r in reviews))

        for gc in gap_clusters:
            gc.gap_severity = compute_gap_severity(
                gc.complaint_rate, gc.competition_density, gc.demand,
                cr_mean, cr_std, cd_mean, cd_std, d_mean, d_std,
            )
            gc.is_underserved = (
                gc.complaint_rate > self.cr_threshold
                and gc.competition_density < self.cd_threshold
            )

        # Sort all clusters by gap severity desc
        gap_clusters.sort(key=lambda g: g.gap_severity, reverse=True)
        underserved = [g for g in gap_clusters if g.is_underserved]

        self.logger.info(
            f"Gap detection complete: {len(underserved)} underserved clusters "
            f"out of {len(gap_clusters)}"
        )
        for i, g in enumerate(underserved[:5], 1):
            self.logger.info(
                f"  {i}. [{g.label}] "
                f"CR={g.complaint_rate:.0%} CD={g.competition_density:.0%} "
                f"Sev={g.gap_severity:.3f}"
            )

        return GapDetectionOutput(
            all_clusters=gap_clusters,
            underserved_clusters=underserved,
            total_reviews=len(reviews),
            total_competitors=total_competitors,
            complaint_rate_threshold=self.cr_threshold,
            competition_density_threshold=self.cd_threshold,
        )
