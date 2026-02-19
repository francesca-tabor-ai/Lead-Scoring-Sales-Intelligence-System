"""
Opportunity Scoring Agent
-------------------------
Scores each cluster using:

  Score_k = α·Q(Demand_k) − β·Q(Competition_k) + γ·Q(NegSent_k)

  Final_Score_k = Confidence_k · Score_k
  Confidence_k  = 1 − exp(−n_k / η)

Includes bootstrap confidence intervals.

Input  : tuple(list[Review], list[ClusterMetrics], list[GapResult])
Output : list[OpportunityScore]   (ranked, highest score first)
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

from .base import Agent
from ..models import ClusterMetrics, GapResult, OpportunityScore, Review

logger = logging.getLogger(__name__)


# ── quantile normalisation ─────────────────────────────────────────────────

def _quantile_rank(values: List[float]) -> List[float]:
    """Map values to their quantile rank in [0, 1]."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    ranks = arr.argsort().argsort().astype(float)
    return list(ranks / max(n - 1, 1))


def _minmax(values: List[float]) -> List[float]:
    arr = np.array(values, dtype=float)
    lo, hi = arr.min(), arr.max()
    if hi == lo:
        return [0.5] * len(values)
    return list((arr - lo) / (hi - lo))


# ── bootstrap CI ──────────────────────────────────────────────────────────

def _bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n_boot: int = 200,
    alpha: float = 0.05,
    rng: np.random.Generator = None,
) -> Tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(42)
    boot_stats = []
    for _ in range(n_boot):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_stats.append(stat_fn(sample))
    boot_stats = np.array(boot_stats)
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return lo, hi


# ── Main Agent ─────────────────────────────────────────────────────────────

class OpportunityScoringAgent(Agent):
    """
    Parameters
    ----------
    alpha, beta, gamma : float
        Weights for Demand, Competition, NegSentiment components.
    confidence_eta : float
        η in Confidence = 1 − exp(−n / η).  Larger η → slower ramp-up.
    n_bootstrap : int
        Bootstrap iterations for CIs. 0 to skip.
    use_quantile_norm : bool
        True = quantile normalization (robust to outliers).
        False = min-max normalization.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        confidence_eta: float = 30.0,
        n_bootstrap: int = 200,
        use_quantile_norm: bool = True,
        verbose: bool = True,
    ):
        super().__init__(name="OpportunityScoringAgent", verbose=verbose)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eta = confidence_eta
        self.n_bootstrap = n_bootstrap
        self.norm_fn = _quantile_rank if use_quantile_norm else _minmax

    # ------------------------------------------------------------------
    def run(
        self,
        payload: Tuple[List[Review], List[ClusterMetrics], List[GapResult]],
    ) -> List[OpportunityScore]:

        reviews, cluster_metrics, gaps = payload
        self._log(f"Scoring {len(gaps)} gap candidates…")

        if not gaps:
            return []

        # Map cluster_id → metrics for quick lookup
        metrics_map: Dict[int, ClusterMetrics] = {
            cm.cluster_id: cm for cm in cluster_metrics
        }

        # ── raw component values ───────────────────────────────────────
        demands     = [g.demand for g in gaps]
        competitions = [g.competition_density for g in gaps]
        # NegSentiment = (1 − sent_mean) / 2  ∈ [0, 1]
        neg_sents   = [(1.0 - g.sentiment_mean) / 2.0 for g in gaps]

        # ── normalise ──────────────────────────────────────────────────
        d_norm = self.norm_fn(demands)
        c_norm = self.norm_fn(competitions)
        s_norm = self.norm_fn(neg_sents)

        # ── per-cluster raw scores ─────────────────────────────────────
        raw_scores = [
            self.alpha * d_norm[i]
            - self.beta * c_norm[i]
            + self.gamma * s_norm[i]
            for i in range(len(gaps))
        ]

        # clip to [0, 1] after weighting
        raw_scores_arr = np.clip(raw_scores, 0.0, 1.0)

        # ── confidence scaling ─────────────────────────────────────────
        cluster_sizes = [
            metrics_map[g.cluster_id].size if g.cluster_id in metrics_map else 1
            for g in gaps
        ]
        confidences = [
            float(1.0 - np.exp(-n / self.eta)) for n in cluster_sizes
        ]
        final_scores = [
            float(conf * raw) for conf, raw in zip(confidences, raw_scores_arr)
        ]

        # ── bootstrap CIs on final_score per cluster ──────────────────
        rng = np.random.default_rng(42)
        scored: List[OpportunityScore] = []

        for i, gap in enumerate(gaps):
            cm = metrics_map.get(gap.cluster_id)

            ci_low, ci_high = 0.0, 0.0
            if self.n_bootstrap > 0 and cm and cm.size > 1:
                # Collect member sentiment scores for bootstrap
                member_sents = np.array(
                    [
                        r.sentiment_score or 0.0
                        for r in reviews
                        if r.cluster_id == gap.cluster_id
                           and r.sentiment_score is not None
                    ]
                )
                if len(member_sents) > 1:
                    # boot the mean sentiment → derive neg_sent → final score
                    def _stat(arr):
                        ns = (1.0 - arr.mean()) / 2.0
                        raw = self.alpha * d_norm[i] - self.beta * c_norm[i] + self.gamma * ns
                        raw = float(np.clip(raw, 0.0, 1.0))
                        return confidences[i] * raw

                    ci_low, ci_high = _bootstrap_ci(member_sents, _stat, self.n_bootstrap, rng=rng)

            scored.append(
                OpportunityScore(
                    cluster_id=gap.cluster_id,
                    label=gap.label,
                    demand_raw=demands[i],
                    competition_raw=competitions[i],
                    neg_sentiment_raw=neg_sents[i],
                    demand_q=d_norm[i],
                    competition_q=c_norm[i],
                    neg_sentiment_q=s_norm[i],
                    raw_score=float(raw_scores_arr[i]),
                    confidence=confidences[i],
                    final_score=final_scores[i],
                    rank=0,  # assigned below
                    ci_low=max(0.0, ci_low),
                    ci_high=min(1.0, ci_high),
                    components_json={
                        "alpha": self.alpha,
                        "beta": self.beta,
                        "gamma": self.gamma,
                        "demand_q": d_norm[i],
                        "competition_q": c_norm[i],
                        "neg_sentiment_q": s_norm[i],
                        "confidence_eta": self.eta,
                        "n_reviews": cluster_sizes[i],
                        "is_underserved": gap.is_underserved,
                        "gap_severity": gap.gap_severity,
                    },
                )
            )

        # ── rank ───────────────────────────────────────────────────────
        scored.sort(key=lambda s: s.final_score, reverse=True)
        for rank, s in enumerate(scored, start=1):
            s.rank = rank

        self._log(
            "Top opportunity: cluster_id=%s label=%r score=%.4f",
            scored[0].cluster_id,
            scored[0].label,
            scored[0].final_score,
        )
        return scored

    # ------------------------------------------------------------------
    def update_weights(self, alpha: float, beta: float, gamma: float) -> None:
        """Allow admin to re-tune weights at runtime."""
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._log(f"Weights updated: α={alpha}, β={beta}, γ={gamma}")
