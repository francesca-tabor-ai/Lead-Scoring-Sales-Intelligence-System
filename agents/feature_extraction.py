"""
Feature Extraction Agent
------------------------
Converts raw reviews into structured feature signals:
  1. Text cleaning & language detection
  2. Sentiment scoring (rating-blended)
  3. Aspect extraction (keyword-based or model-based)
  4. Sentence embeddings
  5. K-means clustering with silhouette-based K selection
  6. Cluster labelling via discriminative TF-IDF terms

Input  : list[Review]
Output : tuple(list[Review], list[ClusterMetrics])
         Reviews are enriched in-place with:
           cleaned_text, language, sentiment_score, complaint_flag,
           aspects, embedding, cluster_id, distance_to_centroid
"""

from __future__ import annotations

import logging
import re
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import Agent
from ..models import Review, ClusterMetrics

logger = logging.getLogger(__name__)

# ── optional NLP deps ──────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import normalize
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    logger.warning("scikit-learn not installed — clustering will use random assignments.")

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    logger.warning(
        "sentence-transformers not installed — falling back to TF-IDF vectors for embeddings."
    )

try:
    from textblob import TextBlob
    _HAS_TEXTBLOB = True
except ImportError:
    _HAS_TEXTBLOB = False

try:
    import langdetect
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False


# ── constants ──────────────────────────────────────────────────────────────

COMPLAINT_LEXICON = {
    "crash", "broken", "slow", "bug", "error", "terrible", "awful",
    "horrible", "useless", "frustrating", "frustration", "issue",
    "problem", "fail", "failure", "fails", "missing", "impossible",
    "nightmare", "confusing", "confused", "confusion", "clunky", "outdated",
    "difficult", "complicated", "poor", "bad", "worse", "worst", "hate",
    "disappointed", "disappointing", "unusable", "doesn't work", "not working",
    "painful", "annoying", "annoyed", "laggy", "lag", "timeout", "times out",
    "expensive", "overpriced", "limited", "incomplete", "unreliable",
}

ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "checkout_ux": ["checkout", "cart", "purchase", "payment", "pay", "buy"],
    "api_performance": ["api", "latency", "response time", "rate limit", "endpoint", "sdk"],
    "mobile_experience": ["mobile", "app", "android", "ios", "smartphone", "phone"],
    "customer_support": ["support", "customer service", "help desk", "ticket", "reply", "response"],
    "onboarding": ["onboarding", "setup", "documentation", "docs", "guide", "tutorial", "getting started"],
    "dashboard_ux": ["dashboard", "ui", "interface", "navigate", "navigation", "menu"],
    "search": ["search", "filter", "find", "results", "query"],
    "pricing": ["price", "pricing", "cost", "expensive", "cheap", "affordable", "subscription"],
    "analytics": ["analytics", "report", "reporting", "export", "data", "insights", "metrics"],
    "integrations": ["integration", "connect", "stripe", "zapier", "webhook", "third-party"],
}


# ── helpers ────────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", " ", text)             # strip URLs
    text = re.sub(r"[^a-z0-9\s'.,!?-]", " ", text)  # keep basic punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _detect_language(text: str) -> str:
    if _HAS_LANGDETECT:
        try:
            return langdetect.detect(text)
        except Exception:
            pass
    return "en"


def _rating_sentiment(rating: float) -> float:
    """Normalise star rating to [–1, +1]."""
    return (rating - 3.0) / 2.0


def _text_sentiment(text: str) -> float:
    """Returns sentiment in [–1, +1]."""
    if _HAS_TEXTBLOB:
        return TextBlob(text).sentiment.polarity
    # Simple lexicon fallback
    neg_words = {"bad", "terrible", "awful", "horrible", "broken", "fail",
                 "crash", "slow", "useless", "frustrating", "confusing",
                 "painful", "annoying", "laggy", "expensive"}
    pos_words = {"good", "great", "excellent", "amazing", "love", "perfect",
                 "easy", "fast", "reliable", "helpful", "simple", "clean"}
    tokens = set(text.lower().split())
    neg = len(tokens & neg_words)
    pos = len(tokens & pos_words)
    total = neg + pos
    if total == 0:
        return 0.0
    return (pos - neg) / total


def _extract_aspects(text: str) -> List[str]:
    found = []
    t = text.lower()
    for aspect, kws in ASPECT_KEYWORDS.items():
        if any(kw in t for kw in kws):
            found.append(aspect)
    return found


def _blend_sentiment(
    rating: float, text: str, lam: float = 0.6
) -> float:
    """
    Blended: λ·text_sentiment + (1–λ)·rating_sentiment
    """
    ts = _text_sentiment(text)
    rs = _rating_sentiment(rating)
    return lam * ts + (1 - lam) * rs


def _is_complaint(sentiment: float, aspects: List[str], tau: float = -0.1) -> int:
    return int(sentiment < tau)


# ── embedding layer ────────────────────────────────────────────────────────

class _EmbeddingModel:
    """Wrapper that falls back gracefully to TF-IDF if sentence-transformers unavailable."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._st_model = None
        self._tfidf = None
        self._tfidf_matrix = None

        if _HAS_ST:
            logger.info("Loading SentenceTransformer: %s", model_name)
            self._st_model = SentenceTransformer(model_name)
        else:
            logger.info("Using TF-IDF as embedding fallback.")

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        if self._st_model is not None:
            vecs = self._st_model.encode(texts, show_progress_bar=False, batch_size=64)
            return normalize(np.array(vecs))
        else:
            if not _HAS_SKLEARN:
                # Pure random fallback (testing only)
                rng = np.random.default_rng(42)
                return rng.standard_normal((len(texts), 50))
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize
            self._tfidf = TfidfVectorizer(max_features=512, min_df=1)
            mat = self._tfidf.fit_transform(texts).toarray()
            return normalize(mat)


# ── clustering ─────────────────────────────────────────────────────────────

def _choose_k(embeddings: np.ndarray, k_range: range) -> int:
    if not _HAS_SKLEARN:
        return min(5, len(embeddings))
    best_k, best_score = k_range.start, -1.0
    for k in k_range:
        if k >= len(embeddings):
            break
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        try:
            score = silhouette_score(embeddings, labels)
        except Exception:
            score = -1.0
        logger.debug("  K=%d  silhouette=%.4f", k, score)
        if score > best_score:
            best_score, best_k = score, k
    logger.info("Best K=%d (silhouette=%.4f)", best_k, best_score)
    return best_k


def _kmeans(embeddings: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    if not _HAS_SKLEARN:
        rng = np.random.default_rng(42)
        labels = rng.integers(0, k, size=len(embeddings))
        centroids = np.zeros((k, embeddings.shape[1]))
        return labels, centroids

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km.cluster_centers_


def _top_tfidf_terms(
    texts: List[str], indices: List[int], all_indices: List[int], n: int = 8
) -> List[str]:
    if not _HAS_SKLEARN or not texts:
        return []
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_texts = [texts[i] for i in indices]
    other_texts = [texts[i] for i in all_indices if i not in set(indices)]

    all_texts = cluster_texts + other_texts
    labels = [1] * len(cluster_texts) + [0] * len(other_texts)

    if len(set(labels)) < 2 or len(all_texts) < 2:
        tfidf = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
        mat = tfidf.fit_transform(cluster_texts)
        scores = np.asarray(mat.mean(axis=0)).flatten()
        top_idx = scores.argsort()[::-1][:n]
        return [tfidf.get_feature_names_out()[i] for i in top_idx]

    tfidf = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    mat = tfidf.fit_transform(all_texts).toarray()
    cluster_mean = mat[: len(cluster_texts)].mean(axis=0)
    other_mean = mat[len(cluster_texts) :].mean(axis=0) if other_texts else np.zeros_like(cluster_mean)
    discriminative = cluster_mean - other_mean
    top_idx = discriminative.argsort()[::-1][:n]
    return [tfidf.get_feature_names_out()[i] for i in top_idx]


# ── Main Agent ─────────────────────────────────────────────────────────────

class FeatureExtractionAgent(Agent):
    """
    Parameters
    ----------
    k_min, k_max : int
        Range of K to search for optimal clustering.
    sentiment_threshold : float
        Sentiment below this → complaint.
    sentiment_lambda : float
        Blend weight for text vs rating sentiment.
    embedding_model : str
        sentence-transformers model name.
    model_version : str
        Artefact versioning tag.
    """

    def __init__(
        self,
        k_min: int = 3,
        k_max: int = 12,
        sentiment_threshold: float = -0.1,
        sentiment_lambda: float = 0.6,
        embedding_model: str = "all-MiniLM-L6-v2",
        model_version: str = "v1",
        verbose: bool = True,
    ):
        super().__init__(name="FeatureExtractionAgent", verbose=verbose)
        self.k_min = k_min
        self.k_max = k_max
        self.tau = sentiment_threshold
        self.lam = sentiment_lambda
        self.model_version = model_version
        self._embedder = _EmbeddingModel(embedding_model)

    # ------------------------------------------------------------------
    def run(self, reviews: List[Review]) -> Tuple[List[Review], List[ClusterMetrics]]:
        self._log(f"Processing {len(reviews)} reviews…")

        # 1. Enrich
        for r in reviews:
            r.cleaned_text = _clean_text(r.text)
            r.language = _detect_language(r.text)
            r.sentiment_score = _blend_sentiment(r.rating, r.text, self.lam)
            r.aspects = _extract_aspects(r.text)
            r.complaint_flag = _is_complaint(r.sentiment_score, r.aspects, self.tau)

        # 2. Embed
        cleaned = [r.cleaned_text for r in reviews]
        self._log("Generating embeddings…")
        embeddings = self._embedder.fit_transform(cleaned)

        # 3. Cluster
        k = _choose_k(embeddings, range(self.k_min, self.k_max + 1))
        labels, centroids = _kmeans(embeddings, k)

        for i, r in enumerate(reviews):
            r.cluster_id = int(labels[i])
            c = centroids[int(labels[i])]
            r.distance_to_centroid = float(np.linalg.norm(embeddings[i] - c))
            r.embedding = embeddings[i].tolist()

        # 4. Build cluster metrics
        cluster_metrics = self._build_cluster_metrics(reviews, embeddings, centroids, k)
        self._log(f"Built {len(cluster_metrics)} cluster metrics.")
        return reviews, cluster_metrics

    # ------------------------------------------------------------------
    def _build_cluster_metrics(
        self,
        reviews: List[Review],
        embeddings: np.ndarray,
        centroids: np.ndarray,
        k: int,
    ) -> List[ClusterMetrics]:
        n = len(reviews)
        all_indices = list(range(n))
        metrics = []

        for cid in range(k):
            member_idx = [i for i, r in enumerate(reviews) if r.cluster_id == cid]
            if not member_idx:
                continue

            members = [reviews[i] for i in member_idx]

            # Statistics
            ratings = [r.rating for r in members]
            sentiments = [r.sentiment_score for r in members if r.sentiment_score is not None]
            complaints = [r.complaint_flag for r in members if r.complaint_flag is not None]

            complaint_rate = float(np.mean(complaints)) if complaints else 0.0
            complaint_intensity = float(
                np.mean([max(0.0, -(s or 0.0)) for s in sentiments])
            ) if sentiments else 0.0
            avg_rating = float(np.mean(ratings)) if ratings else 3.0
            sentiment_mean = float(np.mean(sentiments)) if sentiments else 0.0
            mention_freq = len(members) / n

            # Representative reviews (nearest centroid)
            c = centroids[cid]
            sorted_by_dist = sorted(member_idx, key=lambda i: np.linalg.norm(embeddings[i] - c))
            rep_reviews = [reviews[i].text for i in sorted_by_dist[:3]]

            # Top TF-IDF terms
            top_terms = _top_tfidf_terms(
                [r.cleaned_text or "" for r in reviews],
                member_idx,
                all_indices,
            )

            # Auto-label from top terms + aspects
            label = self._auto_label(top_terms, members)

            metrics.append(
                ClusterMetrics(
                    cluster_id=cid,
                    label=label,
                    size=len(members),
                    top_terms=top_terms,
                    representative_reviews=rep_reviews,
                    mention_freq=mention_freq,
                    complaint_rate=complaint_rate,
                    complaint_intensity=complaint_intensity,
                    avg_rating=avg_rating,
                    sentiment_mean=sentiment_mean,
                    model_version=self.model_version,
                )
            )

        return metrics

    # ------------------------------------------------------------------
    @staticmethod
    def _auto_label(top_terms: List[str], members: List[Review]) -> str:
        """Produce a readable label from top TF-IDF terms and aspect co-occurrences."""
        if not top_terms:
            return "general_feedback"

        # Count aspect presence
        aspect_count: Dict[str, int] = defaultdict(int)
        for r in members:
            for a in (r.aspects or []):
                aspect_count[a] += 1

        if aspect_count:
            dominant_aspect = max(aspect_count, key=aspect_count.get)
            return dominant_aspect

        # Fallback to top TF-IDF term
        return top_terms[0].replace(" ", "_")
