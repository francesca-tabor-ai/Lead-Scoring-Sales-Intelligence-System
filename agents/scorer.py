"""
Opportunity Scoring Agent
--------------------------
Ranks market gaps by opportunity score:

  Score_k = α * Q(Demand_k) - β * Q(Competition_k) + γ * Q(NegSent_k)

With confidence scaling for small clusters:

  FinalScore_k = Conf_k * Score_k
  Conf_k = 1 - exp(-n_k / η)

All inputs are quantile-normalized to [0,1] for robustness against outliers.

Input:  GapDetectionOutput
Output: ScoringOutput
"""

import math
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from agents.base import Agent
from agents.gap_detector import GapDetectionOutput, GapCluster
from config.settings import settings

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class ScoredOpportunity:
    """A scored, ranked opportunity (one per cluster)."""
    rank: int
    cluster_index: int
    label: str
    top_keywords: List[str]

    # Normalized component scores [0, 1]
    demand_score: float
    competition_score: float
    neg_sentiment_score: float
    confidence: float

    # Raw metrics
    demand_raw: float
    competition_raw: float
    complaint_rate: float
    volume: int

    # Final scores
    raw_score: float
    final_score: float             # confidence-adjusted
    gap_severity: float
    is_underserved: bool

    # Explainability
    components: Dict[str, float] = field(default_factory=dict)
    representative_reviews: List[str] = field(default_factory=list)
    competitor_coverages: List[Any] = field(default_factory=list)
    recommended_action: str = ""

    def to_dict(self) -> Dict:
        return {
            "rank": self.rank,
            "cluster_index": self.cluster_index,
            "label": self.label,
            "top_keywords": self.top_keywords,
            "final_score": round(self.final_score, 4),
            "raw_score": round(self.raw_score, 4),
            "confidence": round(self.confidence, 4),
            "demand_score": round(self.demand_score, 4),
            "competition_score": round(self.competition_score, 4),
            "neg_sentiment_score": round(self.neg_sentiment_score, 4),
            "demand_raw": round(self.demand_raw, 4),
            "competition_raw": round(self.competition_raw, 4),
            "complaint_rate": round(self.complaint_rate, 4),
            "complaint_pct": f"{self.complaint_rate * 100:.1f}%",
            "competition_density_pct": f"{self.competition_raw * 100:.1f}%",
            "volume": self.volume,
            "gap_severity": round(self.gap_severity, 4),
            "is_underserved": self.is_underserved,
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "recommended_action": self.recommended_action,
            "representative_reviews": self.representative_reviews[:2],
        }


@dataclass
class ScoringOutput:
    """Final output of the full pipeline."""
    opportunities: List[ScoredOpportunity]
    weights: Dict[str, float]
    model_version: str = "v1"
    total_reviews: int = 0
    total_competitors: int = 0

    @property
    def top_opportunity(self) -> Optional[ScoredOpportunity]:
        return self.opportunities[0] if self.opportunities else None

    def summary(self) -> str:
        lines = ["=== TOP MARKET OPPORTUNITIES ===", ""]
        for opp in self.opportunities[:10]:
            flag = "🎯" if opp.is_underserved else "  "
            lines.append(
                f"  {flag} #{opp.rank:>2} [{opp.label:<35}] "
                f"Score={opp.final_score:.3f} "
                f"Complaint={opp.complaint_rate:.0%} "
                f"Coverage={opp.competition_raw:.0%} "
                f"Vol={opp.volume}"
            )
        return "\n".join(lines)

    def to_dict_list(self) -> List[Dict]:
        return [o.to_dict() for o in self.opportunities]


# ─── Normalization Utilities ─────────────────────────────────────────────────


def quantile_normalize(values: List[float]) -> List[float]:
    """
    Quantile rank normalization: maps values to [0, 1] rank positions.
    Robust to heavy tails and outliers.
    """
    if len(values) == 0:
        return []
    arr = np.array(values)
    n = len(arr)
    ranks = arr.argsort().argsort()  # double argsort = rank
    return list(ranks / max(n - 1, 1))


def minmax_normalize(values: List[float]) -> List[float]:
    """Min-max normalization to [0, 1]."""
    arr = np.array(values)
    v_min, v_max = arr.min(), arr.max()
    if v_max == v_min:
        return [0.5] * len(values)
    return list((arr - v_min) / (v_max - v_min))


def confidence_factor(n: int, eta: float = settings.CONFIDENCE_ETA) -> float:
    """
    Exponential confidence factor:
      Conf = 1 - exp(-n / η)
    Small clusters (n << η) get penalized; large clusters → 1.0.
    """
    return 1.0 - math.exp(-n / eta)


# ─── Action Recommendations ──────────────────────────────────────────────────


def generate_recommendation(opp: "ScoredOpportunity") -> str:
    """Generate a plain-English recommended action based on scores."""
    score = opp.final_score
    cr = opp.complaint_rate
    cd = opp.competition_raw
    label = opp.label

    if score >= 0.7 and cr >= 0.35 and cd < 0.20:
        return (
            f"🚀 HIGH PRIORITY: '{label}' is a strong whitespace opportunity. "
            f"{cr:.0%} complaint rate with only {cd:.0%} competitor coverage. "
            "Prioritize in next sprint."
        )
    elif score >= 0.5 and cr >= 0.25:
        return (
            f"📈 MEDIUM PRIORITY: '{label}' shows meaningful pain with limited competition. "
            "Consider adding to roadmap backlog."
        )
    elif cd >= 0.5:
        return (
            f"⚠️ COMPETITIVE: '{label}' is widely covered by competitors. "
            "Feature parity may be required; focus on differentiation."
        )
    else:
        return (
            f"👀 MONITOR: '{label}' shows early signals but insufficient volume. "
            "Track over next quarter."
        )


# ─── OpportunityScoringAgent ─────────────────────────────────────────────────


class OpportunityScoringAgent(Agent):
    """
    Agent 4: Opportunity Scoring

    Weighted model:
      Score_k = α * Q(Demand) - β * Q(Competition) + γ * Q(NegSentiment)
      FinalScore_k = Confidence_k * Score_k

    Weights (α, β, γ) configurable via settings or at init.
    """

    def __init__(
        self,
        alpha: Optional[float] = None,
        beta: Optional[float] = None,
        gamma: Optional[float] = None,
        eta: Optional[float] = None,
    ):
        super().__init__(name="OpportunityScoringAgent")
        self.alpha = alpha if alpha is not None else settings.ALPHA
        self.beta  = beta  if beta  is not None else settings.BETA
        self.gamma = gamma if gamma is not None else settings.GAMMA
        self.eta   = eta   if eta   is not None else settings.CONFIDENCE_ETA

    def run(self, gap_output: GapDetectionOutput) -> ScoringOutput:
        clusters: List[GapCluster] = gap_output.all_clusters

        if not clusters:
            raise ValueError("No clusters provided to OpportunityScoringAgent")

        self.logger.info(
            f"Scoring {len(clusters)} clusters "
            f"(α={self.alpha}, β={self.beta}, γ={self.gamma})"
        )

        # Extract raw values
        demands      = [c.demand for c in clusters]
        competitions = [c.competition_density for c in clusters]
        # NegSentiment: (1 - sentiment_mean) / 2  scaled to [0, 1]
        neg_sents    = [(1.0 - c.sentiment_mean) / 2.0 for c in clusters]
        volumes      = [c.volume for c in clusters]

        # Quantile normalize
        q_demand  = quantile_normalize(demands)
        q_comp    = quantile_normalize(competitions)
        q_negsent = quantile_normalize(neg_sents)

        scored: List[ScoredOpportunity] = []

        for i, cluster in enumerate(clusters):
            qd = q_demand[i]
            qc = q_comp[i]
            qn = q_negsent[i]

            raw_score = self.alpha * qd - self.beta * qc + self.gamma * qn
            # Normalize raw score to [0, 1] (theoretical range: [-β, α+γ])
            raw_score_norm = (raw_score + self.beta) / (self.alpha + self.beta + self.gamma)
            raw_score_norm = max(0.0, min(1.0, raw_score_norm))

            conf = confidence_factor(cluster.volume, self.eta)
            final = conf * raw_score_norm

            opp = ScoredOpportunity(
                rank=0,  # set after sorting
                cluster_index=cluster.cluster_index,
                label=cluster.label,
                top_keywords=cluster.top_keywords,
                demand_score=qd,
                competition_score=qc,
                neg_sentiment_score=qn,
                confidence=conf,
                demand_raw=cluster.demand,
                competition_raw=cluster.competition_density,
                complaint_rate=cluster.complaint_rate,
                volume=cluster.volume,
                raw_score=raw_score_norm,
                final_score=final,
                gap_severity=cluster.gap_severity,
                is_underserved=cluster.is_underserved,
                components={
                    "alpha * demand": round(self.alpha * qd, 4),
                    "beta * competition": round(self.beta * qc, 4),
                    "gamma * neg_sentiment": round(self.gamma * qn, 4),
                    "confidence": round(conf, 4),
                },
                representative_reviews=cluster.representative_reviews,
                competitor_coverages=cluster.competitor_coverages,
            )
            scored.append(opp)

        # Sort by final_score descending, assign ranks
        scored.sort(key=lambda o: o.final_score, reverse=True)
        for rank, opp in enumerate(scored, 1):
            opp.rank = rank
            opp.recommended_action = generate_recommendation(opp)

        self.logger.info("=== TOP 5 OPPORTUNITIES ===")
        for opp in scored[:5]:
            flag = "🎯" if opp.is_underserved else "  "
            self.logger.info(
                f"  {flag} #{opp.rank} [{opp.label}] "
                f"Score={opp.final_score:.3f} "
                f"CR={opp.complaint_rate:.0%} "
                f"Cov={opp.competition_raw:.0%}"
            )

        return ScoringOutput(
            opportunities=scored,
            weights={"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma},
            model_version="v1",
            total_reviews=gap_output.total_reviews,
            total_competitors=gap_output.total_competitors,
        )
