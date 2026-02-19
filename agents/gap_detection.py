"""
Gap Detection Agent
-------------------
Identifies underserved market clusters using:
  - Complaint Rate  > τ_c   (default 0.30)
  - Competition Density < τ_d  (default 0.25)

Also computes a continuous GapSeverity score for ranking.

Input  : tuple(list[Review], list[ClusterMetrics])
Output : tuple(list[Review], list[ClusterMetrics], list[GapResult])
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .base import Agent
from ..models import ClusterMetrics, CompetitorCoverage, GapResult, Review

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _standardize(values: List[float]) -> List[float]:
    arr = np.array(values, dtype=float)
    mu, sigma = arr.mean(), arr.std()
    if sigma == 0:
        return [0.0] * len(values)
    return list((arr - mu) / sigma)


class GapDetectionAgent(Agent):
    """
    Parameters
    ----------
    complaint_threshold : float
        τ_c — minimum complaint rate to be "high demand".
    competition_threshold : float
        τ_d — maximum competition density to qualify as "underserved".
    coverage_threshold : float
        τ_cov — share threshold above which a competitor "covers" a cluster.
    smoothing_strength : float
        m — Laplace smoothing strength for share estimation.
    gap_weights : tuple(a, b, c)
        Weights (a, b, c) in the continuous GapSeverity formula:
        σ(a·z(ComplaintRate) - b·z(CompDensity) + c·z(Demand))
    """

    def __init__(
        self,
        complaint_threshold: float = 0.30,
        competition_threshold: float = 0.25,
        coverage_threshold: float = 0.10,
        smoothing_strength: float = 5.0,
        gap_weights: Tuple[float, float, float] = (1.5, 1.0, 0.5),
        verbose: bool = True,
    ):
        super().__init__(name="GapDetectionAgent", verbose=verbose)
        self.tau_c = complaint_threshold
        self.tau_d = competition_threshold
        self.tau_cov = coverage_threshold
        self.m = smoothing_strength
        self.a, self.b, self.c = gap_weights

    # ------------------------------------------------------------------
    def run(
        self,
        payload: Tuple[List[Review], List[ClusterMetrics]],
    ) -> Tuple[List[Review], List[ClusterMetrics], List[GapResult]]:

        reviews, cluster_metrics = payload
        self._log(f"Analysing {len(cluster_metrics)} clusters for {len(reviews)} reviews…")

        # ── competitor coverage per cluster ────────────────────────────
        coverages = self._compute_coverages(reviews, cluster_metrics)

        # ── global cluster proportion (for smoothing) ──────────────────
        n_total = len(reviews)
        global_cluster_prop: Dict[int, float] = {
            cm.cluster_id: cm.size / n_total for cm in cluster_metrics
        }

        gaps: List[GapResult] = []
        # collect raw values for standardization
        complaint_rates, comp_densities, demands = [], [], []

        for cm in cluster_metrics:
            cov_list = coverages.get(cm.cluster_id, [])
            n_competitors = len(set(c.competitor for c in cov_list)) if cov_list else 1
            covers_count = sum(1 for c in cov_list if c.covers)
            comp_density = covers_count / max(n_competitors, 1)

            complaint_rates.append(cm.complaint_rate)
            comp_densities.append(comp_density)
            demands.append(cm.mention_freq)

        if not cluster_metrics:
            return reviews, cluster_metrics, []

        z_cr = _standardize(complaint_rates)
        z_cd = _standardize(comp_densities)
        z_d = _standardize(demands)

        for idx, cm in enumerate(cluster_metrics):
            cov_list = coverages.get(cm.cluster_id, [])
            n_competitors = len(set(c.competitor for c in cov_list)) if cov_list else 1
            covers_count = sum(1 for c in cov_list if c.covers)
            comp_density = covers_count / max(n_competitors, 1)

            is_underserved = (
                cm.complaint_rate > self.tau_c
                and comp_density < self.tau_d
            )

            # continuous gap severity
            severity_logit = (
                self.a * z_cr[idx]
                - self.b * z_cd[idx]
                + self.c * z_d[idx]
            )
            gap_severity = float(_sigmoid(severity_logit))

            gaps.append(
                GapResult(
                    cluster_id=cm.cluster_id,
                    label=cm.label,
                    complaint_rate=cm.complaint_rate,
                    competition_density=comp_density,
                    gap_severity=gap_severity,
                    is_underserved=is_underserved,
                    demand=cm.mention_freq,
                    sentiment_mean=cm.sentiment_mean,
                    competitor_coverages=cov_list,
                )
            )

        n_underserved = sum(1 for g in gaps if g.is_underserved)
        self._log(
            f"Detected {n_underserved} underserved clusters "
            f"({len(gaps)} total analysed)."
        )
        return reviews, cluster_metrics, gaps

    # ------------------------------------------------------------------
    def _compute_coverages(
        self,
        reviews: List[Review],
        cluster_metrics: List[ClusterMetrics],
    ) -> Dict[int, List[CompetitorCoverage]]:
        """
        For each cluster k and competitor j:
          smoothed_share = (n_jk + m * p_k) / (n_j + m)
          covers = smoothed_share > tau_cov
        """
        # Build count maps
        n_jk: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        n_j: Dict[str, int] = defaultdict(int)

        for r in reviews:
            if r.cluster_id is None:
                continue
            n_jk[r.competitor][r.cluster_id] += 1
            n_j[r.competitor] += 1

        n_total = len(reviews)
        global_prop: Dict[int, float] = {
            cm.cluster_id: cm.size / max(n_total, 1)
            for cm in cluster_metrics
        }

        result: Dict[int, List[CompetitorCoverage]] = defaultdict(list)

        for competitor, cluster_counts in n_jk.items():
            nj = n_j[competitor]
            for cm in cluster_metrics:
                cid = cm.cluster_id
                njk = cluster_counts.get(cid, 0)
                pk = global_prop.get(cid, 0.0)
                raw = njk / max(nj, 1)
                smoothed = (njk + self.m * pk) / (nj + self.m)
                result[cid].append(
                    CompetitorCoverage(
                        cluster_id=cid,
                        competitor=competitor,
                        raw_share=raw,
                        smoothed_share=smoothed,
                        covers=smoothed > self.tau_cov,
                    )
                )

        return result
