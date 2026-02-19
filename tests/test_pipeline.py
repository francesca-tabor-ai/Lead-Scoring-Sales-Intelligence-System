"""
End-to-end pipeline tests using mock data.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agents.scraper import ScraperAgent, ScrapeTarget, MockScraper, deduplicate_reviews
from agents.extractor import FeatureExtractionAgent, clean_text, blend_sentiment
from agents.gap_detector import GapDetectionAgent, compute_competition_density
from agents.scorer import OpportunityScoringAgent, quantile_normalize, confidence_factor
from agents.base import Orchestrator


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_targets():
    return [
        ScrapeTarget(url="https://mock.com/a", source_type="mock",
                     product_name="ProductA", competitor_name="CompA", max_reviews=100),
        ScrapeTarget(url="https://mock.com/b", source_type="mock",
                     product_name="ProductB", competitor_name="CompB", max_reviews=100),
        ScrapeTarget(url="https://mock.com/c", source_type="mock",
                     product_name="ProductC", competitor_name="CompC", max_reviews=100),
    ]


@pytest.fixture
def raw_reviews(mock_targets):
    agent = ScraperAgent()
    return agent.run(mock_targets)


@pytest.fixture
def extractor_output(raw_reviews):
    agent = FeatureExtractionAgent()
    return agent.run(raw_reviews)


@pytest.fixture
def gap_output(extractor_output):
    agent = GapDetectionAgent()
    return agent.run(extractor_output)


@pytest.fixture
def scoring_output(gap_output):
    agent = OpportunityScoringAgent()
    return agent.run(gap_output)


# ─── Scraper Tests ────────────────────────────────────────────────────────────

class TestScraperAgent:
    def test_mock_scraper_returns_reviews(self, mock_targets):
        agent = ScraperAgent()
        reviews = agent.run(mock_targets)
        assert len(reviews) > 0

    def test_reviews_have_required_fields(self, raw_reviews):
        for r in raw_reviews:
            assert r.text, "Review text is empty"
            assert r.competitor_name, "Competitor name missing"
            assert r.product_name, "Product name missing"
            assert r.source == "mock"

    def test_deduplication_removes_exact_duplicates(self):
        from agents.scraper import RawReview
        from datetime import datetime
        duplicate = RawReview(
            text="This is the exact same review text",
            rating=3.0,
            review_date=datetime.utcnow(),
            author_hash="abc123",
            source="mock",
            product_name="Prod",
            competitor_name="Comp",
            url="https://test.com",
        )
        reviews = [duplicate, duplicate, duplicate]
        deduped = deduplicate_reviews(reviews, threshold=0.95)
        assert len(deduped) == 1

    def test_multiple_competitors_represented(self, raw_reviews):
        competitors = set(r.competitor_name for r in raw_reviews)
        assert len(competitors) >= 2


# ─── Extractor Tests ──────────────────────────────────────────────────────────

class TestFeatureExtractionAgent:
    def test_enrichment_produces_sentiment(self, extractor_output):
        for r in extractor_output.enriched_reviews[:10]:
            assert -1.0 <= r.sentiment_score <= 1.0
            assert r.sentiment_label in ("positive", "negative", "neutral")

    def test_clusters_are_produced(self, extractor_output):
        assert len(extractor_output.clusters) >= 2

    def test_cluster_metrics_populated(self, extractor_output):
        for c in extractor_output.clusters:
            assert c.volume > 0
            assert 0.0 <= c.complaint_freq <= 1.0
            assert 0.0 <= c.mention_freq <= 1.0
            assert len(c.top_keywords) > 0

    def test_clean_text_removes_urls(self):
        text = "Check https://example.com for more info!!!"
        cleaned = clean_text(text)
        assert "https" not in cleaned
        assert "!!!" not in cleaned

    def test_blend_sentiment_with_rating(self):
        blended = blend_sentiment(0.5, 1.0, lam=0.65)
        assert -1.0 <= blended <= 1.0

    def test_blend_sentiment_without_rating(self):
        blended = blend_sentiment(-0.8, None)
        assert blended == -0.8


# ─── Gap Detector Tests ───────────────────────────────────────────────────────

class TestGapDetectionAgent:
    def test_gap_output_has_clusters(self, gap_output):
        assert len(gap_output.all_clusters) >= 2

    def test_gap_severity_in_range(self, gap_output):
        for gc in gap_output.all_clusters:
            assert 0.0 <= gc.gap_severity <= 1.0, f"Gap severity out of range: {gc.gap_severity}"

    def test_underserved_criteria_correct(self, gap_output):
        for gc in gap_output.underserved_clusters:
            assert gc.complaint_rate > gap_output.complaint_rate_threshold, (
                f"Underserved cluster complaint rate too low: {gc.complaint_rate:.2f}"
            )
            assert gc.competition_density < gap_output.competition_density_threshold, (
                f"Underserved cluster competition too high: {gc.competition_density:.2f}"
            )

    def test_competition_density_calculation(self, extractor_output):
        reviews = extractor_output.enriched_reviews
        if reviews:
            cluster_0 = extractor_output.clusters[0]
            density, coverages = compute_competition_density(
                cluster_index=cluster_0.cluster_index,
                enriched_reviews=reviews,
            )
            assert 0.0 <= density <= 1.0
            assert len(coverages) > 0


# ─── Scorer Tests ─────────────────────────────────────────────────────────────

class TestOpportunityScoringAgent:
    def test_scores_in_range(self, scoring_output):
        for opp in scoring_output.opportunities:
            assert 0.0 <= opp.final_score <= 1.0, f"Score out of range: {opp.final_score}"

    def test_ranks_are_sequential(self, scoring_output):
        ranks = [o.rank for o in scoring_output.opportunities]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_sorted_by_score_desc(self, scoring_output):
        scores = [o.final_score for o in scoring_output.opportunities]
        assert scores == sorted(scores, reverse=True)

    def test_recommendations_generated(self, scoring_output):
        for opp in scoring_output.opportunities:
            assert opp.recommended_action, "Recommendation missing"

    def test_quantile_normalize(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        normed = quantile_normalize(values)
        assert len(normed) == 5
        assert all(0.0 <= v <= 1.0 for v in normed)
        assert normed[-1] == 1.0
        assert normed[0] == 0.0

    def test_confidence_factor_bounds(self):
        assert confidence_factor(0) == pytest.approx(0.0, abs=0.01)
        assert confidence_factor(1000) > 0.99
        assert 0.0 < confidence_factor(50) < 1.0


# ─── Full Pipeline Test ───────────────────────────────────────────────────────

class TestFullPipeline:
    def test_full_pipeline_end_to_end(self, mock_targets):
        orchestrator = Orchestrator(
            agents=[
                ScraperAgent(),
                FeatureExtractionAgent(),
                GapDetectionAgent(),
                OpportunityScoringAgent(),
            ],
            stop_on_failure=True,
        )
        result = orchestrator.execute(mock_targets)
        assert result.success, f"Pipeline failed: {result.error}"
        assert result.data is not None
        scoring = result.data
        assert len(scoring.opportunities) >= 2
        assert scoring.total_reviews > 0

    def test_pipeline_custom_weights(self, mock_targets):
        orchestrator = Orchestrator(
            agents=[
                ScraperAgent(),
                FeatureExtractionAgent(),
                GapDetectionAgent(complaint_rate_threshold=0.20),
                OpportunityScoringAgent(alpha=0.6, beta=0.2, gamma=0.2),
            ]
        )
        result = orchestrator.execute(mock_targets)
        assert result.success

    def test_empty_targets_handled(self):
        agent = ScraperAgent()
        reviews = agent.run([])
        assert reviews == []
