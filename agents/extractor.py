"""
Feature Extraction Agent
-------------------------
Converts raw reviews into structured feature signals using:
  - Text cleaning & normalization
  - Sentiment scoring (VADER + rating blend)
  - Sentence-transformer embeddings
  - TF-IDF (for explainability / keyword themes)
  - K-Means clustering with automatic K selection (silhouette)
  - Cluster labeling via TF-IDF differential scores
  - Aspect extraction via keyword heuristics

Input:  List[RawReview]
Output: ExtractorOutput (clusters, enriched_reviews, cluster_metrics)
"""

import re
import math
import logging
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from datetime import datetime
from collections import Counter, defaultdict

logger = logging.getLogger(__name__)

# ─── Optional heavy imports (graceful fallback) ───────────────────────────────
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False
    logger.warning("sentence-transformers not installed — using TF-IDF fallback embeddings")

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    logger.warning("scikit-learn not installed — clustering will be unavailable")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
    _HAS_VADER = True
except ImportError:
    _HAS_VADER = False
    logger.warning("vaderSentiment not installed — using rating-only sentiment")

from agents.base import Agent
from agents.scraper import RawReview
from config.settings import settings


# ─── Data Structures ──────────────────────────────────────────────────────────


@dataclass
class EnrichedReview:
    """A RawReview enriched with NLP signals."""
    review_id: str
    text: str
    cleaned_text: str
    rating: Optional[float]
    review_date: Optional[datetime]
    author_hash: str
    source: str
    product_name: str
    competitor_name: str
    url: str
    language: str
    sentiment_score: float        # blended [-1, 1]
    sentiment_label: str          # positive / negative / neutral
    is_complaint: bool
    aspects: List[str]
    embedding: Optional[np.ndarray] = None
    cluster_index: Optional[int] = None
    distance_to_centroid: Optional[float] = None


@dataclass
class ClusterSummary:
    """Aggregated statistics for one cluster."""
    cluster_index: int
    label: str
    top_keywords: List[str]
    volume: int
    mention_freq: float           # fraction of all reviews
    complaint_freq: float         # fraction that are complaints
    complaint_intensity: float    # avg(-sentiment) for complaints
    avg_rating: float
    sentiment_mean: float
    centroid: Optional[np.ndarray] = None
    representative_reviews: List[str] = field(default_factory=list)


@dataclass
class ExtractorOutput:
    """Output bundle from the FeatureExtractionAgent."""
    enriched_reviews: List[EnrichedReview]
    clusters: List[ClusterSummary]
    k: int
    silhouette: float
    inertia: float
    model_version: str
    category: str = "default"


# ─── Text Preprocessing ──────────────────────────────────────────────────────


_STOPWORDS = {
    "the", "a", "an", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "with", "this", "that", "they", "we", "i", "you",
    "my", "your", "their", "our", "be", "was", "were", "are", "has", "have",
    "had", "not", "no", "so", "if", "as", "from", "by", "can", "do", "did",
    "will", "would", "could", "should", "just", "very", "also", "more",
    "than", "been", "there", "which", "who", "what", "when", "how",
}

_ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "checkout_ux": ["checkout", "cart", "payment", "button", "form", "flow"],
    "api_speed": ["api", "latency", "timeout", "slow", "response time", "rate limit"],
    "mobile_app": ["mobile", "app", "ios", "android", "crash", "freeze"],
    "customer_support": ["support", "help desk", "ticket", "response", "customer service"],
    "pricing": ["price", "cost", "expensive", "billing", "plan", "subscription"],
    "onboarding": ["setup", "onboarding", "documentation", "docs", "tutorial", "guide"],
    "integrations": ["integration", "connect", "sync", "import", "plugin", "webhook"],
    "reporting": ["report", "dashboard", "analytics", "export", "chart", "filter"],
    "performance": ["performance", "speed", "fast", "slow", "loading", "lag"],
    "reliability": ["bug", "crash", "error", "downtime", "outage", "reliability"],
}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s'-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_aspects(text: str) -> List[str]:
    found = []
    text_lower = text.lower()
    for aspect, keywords in _ASPECT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(aspect)
    return found


def detect_language(text: str) -> str:
    """Lightweight language detection — English heuristic."""
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return "en"


# ─── Sentiment ───────────────────────────────────────────────────────────────


def rating_sentiment(rating: Optional[float]) -> Optional[float]:
    """Map 1-5 star rating to [-1, 1]."""
    if rating is None:
        return None
    return (rating - 3.0) / 2.0


def text_sentiment(text: str) -> float:
    """VADER compound score, or 0.0 fallback."""
    if _HAS_VADER:
        return _vader.polarity_scores(text)["compound"]
    return 0.0


def blend_sentiment(
    text_sent: float,
    rating_sent: Optional[float],
    lam: float = 0.65,
) -> float:
    """
    Blended sentiment: λ * text_sentiment + (1-λ) * rating_sentiment.
    If no rating, use text only.
    """
    if rating_sent is None:
        return text_sent
    return lam * text_sent + (1 - lam) * rating_sent


def sentiment_label(score: float) -> str:
    if score > 0.1:
        return "positive"
    if score < -0.1:
        return "negative"
    return "neutral"


# ─── Embeddings ──────────────────────────────────────────────────────────────


class EmbeddingModel:
    """Wraps sentence-transformers or falls back to TF-IDF."""

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self.model_name = model_name
        self._sbert = None
        self._tfidf = None
        self._tfidf_matrix = None
        self._tfidf_texts = None

        if _HAS_SBERT:
            logger.info(f"Loading sentence-transformer: {model_name}")
            self._sbert = SentenceTransformer(model_name)
            logger.info("Embedding model loaded.")
        else:
            logger.info("Using TF-IDF fallback embeddings (install sentence-transformers for better quality)")

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._sbert is not None:
            return self._sbert.encode(texts, show_progress_bar=False, batch_size=64)
        # TF-IDF fallback
        if not _HAS_SKLEARN:
            # Very basic bag-of-words fallback
            return self._bow_fallback(texts)
        if self._tfidf is None:
            self._tfidf = TfidfVectorizer(max_features=512, stop_words="english")
            matrix = self._tfidf.fit_transform(texts)
            return matrix.toarray()
        matrix = self._tfidf.transform(texts)
        return matrix.toarray()

    def _bow_fallback(self, texts: List[str]) -> np.ndarray:
        vocab: Dict[str, int] = {}
        for t in texts:
            for w in t.split():
                if w not in vocab:
                    vocab[w] = len(vocab)
        vectors = []
        for t in texts:
            v = [0.0] * len(vocab)
            for w in t.split():
                if w in vocab:
                    v[vocab[w]] += 1.0
            norm = math.sqrt(sum(x**2 for x in v)) or 1.0
            vectors.append([x / norm for x in v])
        return np.array(vectors)

    @property
    def version(self) -> str:
        return self.model_name if self._sbert else "tfidf-fallback"


# ─── Clustering ──────────────────────────────────────────────────────────────


def choose_k(
    embeddings: np.ndarray,
    k_min: int = 2,
    k_max: int = 20,
    min_samples: int = 5,
) -> Tuple[int, float, float]:
    """
    Choose K via silhouette score.
    Returns (best_k, best_silhouette, inertia_at_best_k).
    """
    if not _HAS_SKLEARN:
        return 5, 0.0, 0.0

    n = len(embeddings)
    k_max = min(k_max, n - 1, settings.MAX_CLUSTERS)
    k_min = max(k_min, 2)

    if n < k_min * min_samples:
        k = max(2, n // min_samples)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        sil = silhouette_score(embeddings, labels) if len(set(labels)) > 1 else 0.0
        return k, float(sil), float(km.inertia_)

    best_k, best_sil, best_inertia = k_min, -1.0, 0.0
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        if len(set(labels)) < 2:
            continue
        sil = silhouette_score(embeddings, labels, sample_size=min(1000, n))
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_inertia = float(km.inertia_)

    return best_k, float(best_sil), best_inertia


def cluster_embeddings(
    embeddings: np.ndarray, k: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run K-Means. Returns (labels, centroids, distances).
    """
    if not _HAS_SKLEARN:
        labels = np.zeros(len(embeddings), dtype=int)
        centroid = embeddings.mean(axis=0, keepdims=True)
        dists = np.linalg.norm(embeddings - centroid, axis=1)
        return labels, centroid, dists

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    centroids = km.cluster_centers_
    dists = np.linalg.norm(embeddings - centroids[labels], axis=1)
    return labels, centroids, dists


# ─── Cluster Labeling ─────────────────────────────────────────────────────────


def label_cluster(
    cluster_texts: List[str],
    all_texts: List[str],
    top_n: int = 10,
) -> Tuple[str, List[str]]:
    """
    Generate a human-readable label for a cluster using TF-IDF differential.
    Returns (label_string, top_keywords).
    """
    if not _HAS_SKLEARN or not cluster_texts:
        # Fallback: frequency count
        words = Counter()
        for t in cluster_texts:
            words.update(w for w in t.split() if w not in _STOPWORDS and len(w) > 3)
        top = [w for w, _ in words.most_common(top_n)]
        return " / ".join(top[:3]), top

    tfidf = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1, 2))
    all_matrix = tfidf.fit_transform(all_texts)
    vocab = tfidf.get_feature_names_out()

    cluster_idx = [i for i, t in enumerate(all_texts) if t in set(cluster_texts)]
    if not cluster_idx:
        cluster_idx = list(range(len(cluster_texts)))
        other_idx = []
    else:
        other_idx = [i for i in range(len(all_texts)) if i not in set(cluster_idx)]

    cluster_mean = np.asarray(all_matrix[cluster_idx].mean(axis=0)).flatten()
    other_mean = (
        np.asarray(all_matrix[other_idx].mean(axis=0)).flatten()
        if other_idx
        else np.zeros_like(cluster_mean)
    )

    differential = cluster_mean - other_mean
    top_indices = differential.argsort()[::-1][:top_n]
    top_keywords = [vocab[i] for i in top_indices if differential[i] > 0]

    if not top_keywords:
        top_keywords = [vocab[i] for i in top_indices[:top_n]]

    label = " / ".join(top_keywords[:3]) if top_keywords else f"cluster_{len(top_keywords)}"
    return label, top_keywords


# ─── FeatureExtractionAgent ───────────────────────────────────────────────────


class FeatureExtractionAgent(Agent):
    """
    Agent 2: Feature Extraction

    Pipeline:
      1. Clean text + detect language
      2. Compute blended sentiment
      3. Extract aspects
      4. Generate embeddings
      5. Cluster (auto-select K via silhouette)
      6. Label each cluster with TF-IDF differential keywords
      7. Compute per-cluster metrics
    """

    def __init__(self):
        super().__init__(name="FeatureExtractionAgent")
        self._embedding_model: Optional[EmbeddingModel] = None

    @property
    def embedding_model(self) -> EmbeddingModel:
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel()
        return self._embedding_model

    def _enrich(self, review: RawReview, idx: int) -> EnrichedReview:
        cleaned = clean_text(review.text)
        lang = "en"  # skip heavy detection for speed; can enable per-review
        t_sent = text_sentiment(cleaned)
        r_sent = rating_sentiment(review.rating)
        blended = blend_sentiment(t_sent, r_sent)
        label = sentiment_label(blended)
        is_complaint = blended < settings.SENTIMENT_THRESHOLD
        aspects = extract_aspects(review.text)

        return EnrichedReview(
            review_id=hashlib.sha256(f"{review.author_hash}{review.text[:50]}".encode()).hexdigest()[:12],
            text=review.text,
            cleaned_text=cleaned,
            rating=review.rating,
            review_date=review.review_date,
            author_hash=review.author_hash,
            source=review.source,
            product_name=review.product_name,
            competitor_name=review.competitor_name,
            url=review.url,
            language=lang,
            sentiment_score=blended,
            sentiment_label=label,
            is_complaint=is_complaint,
            aspects=aspects,
        )

    def _compute_cluster_metrics(
        self,
        cluster_index: int,
        members: List[EnrichedReview],
        total_reviews: int,
        all_cleaned: List[str],
    ) -> Tuple[ClusterSummary, str, List[str]]:
        volume = len(members)
        mention_freq = volume / max(total_reviews, 1)
        complaints = [r for r in members if r.is_complaint]
        complaint_freq = len(complaints) / max(volume, 1)
        complaint_intensity = (
            float(np.mean([-r.sentiment_score for r in complaints]))
            if complaints else 0.0
        )
        ratings = [r.rating for r in members if r.rating is not None]
        avg_rating = float(np.mean(ratings)) if ratings else 0.0
        sentiments = [r.sentiment_score for r in members]
        sentiment_mean = float(np.mean(sentiments)) if sentiments else 0.0

        cluster_texts = [r.cleaned_text for r in members]
        label, top_keywords = label_cluster(cluster_texts, all_cleaned)

        # Representative reviews: 3 closest to centroid (already sorted by dist)
        rep_reviews = [r.text for r in members[:3]]

        summary = ClusterSummary(
            cluster_index=cluster_index,
            label=label,
            top_keywords=top_keywords,
            volume=volume,
            mention_freq=mention_freq,
            complaint_freq=complaint_freq,
            complaint_intensity=complaint_intensity,
            avg_rating=avg_rating,
            sentiment_mean=sentiment_mean,
            representative_reviews=rep_reviews,
        )
        return summary

    def run(self, reviews: List[RawReview]) -> ExtractorOutput:
        if not reviews:
            raise ValueError("No reviews provided to FeatureExtractionAgent")

        self.logger.info(f"Enriching {len(reviews)} reviews...")

        # Step 1: Enrich all reviews
        enriched: List[EnrichedReview] = [
            self._enrich(r, i) for i, r in enumerate(reviews)
        ]

        # Step 2: Language filter
        if settings.LANGUAGE_FILTER:
            enriched = [r for r in enriched if r.language == settings.LANGUAGE_FILTER]
            self.logger.info(
                f"After language filter ({settings.LANGUAGE_FILTER}): {len(enriched)} reviews"
            )

        if len(enriched) < 10:
            self.logger.warning("Very few reviews — using minimal clustering (k=2)")

        # Step 3: Embeddings
        self.logger.info("Generating embeddings...")
        cleaned_texts = [r.cleaned_text for r in enriched]
        embeddings = self.embedding_model.encode(cleaned_texts)
        embeddings = normalize(embeddings) if _HAS_SKLEARN else embeddings

        for r, emb in zip(enriched, embeddings):
            r.embedding = emb

        # Step 4: Choose K and cluster
        self.logger.info("Selecting optimal K...")
        best_k, best_sil, inertia = choose_k(
            embeddings,
            k_min=2,
            k_max=min(settings.MAX_CLUSTERS, len(enriched) // settings.MIN_CLUSTER_SIZE),
        )
        self.logger.info(f"  K={best_k}, silhouette={best_sil:.3f}, inertia={inertia:.1f}")

        labels, centroids, distances = cluster_embeddings(embeddings, best_k)

        for r, lbl, dist in zip(enriched, labels, distances):
            r.cluster_index = int(lbl)
            r.distance_to_centroid = float(dist)

        # Step 5: Build cluster summaries
        cluster_members: Dict[int, List[EnrichedReview]] = defaultdict(list)
        for r in enriched:
            cluster_members[r.cluster_index].append(r)

        # Sort members by distance (closest = most representative)
        for k in cluster_members:
            cluster_members[k].sort(key=lambda r: r.distance_to_centroid or float("inf"))

        clusters: List[ClusterSummary] = []
        for ci in range(best_k):
            members = cluster_members.get(ci, [])
            if not members:
                continue
            summary = self._compute_cluster_metrics(ci, members, len(enriched), cleaned_texts)
            if centroids is not None and len(centroids) > ci:
                summary.centroid = centroids[ci]
            clusters.append(summary)

        self.logger.info(f"Extraction complete: {best_k} clusters, {len(enriched)} enriched reviews")

        return ExtractorOutput(
            enriched_reviews=enriched,
            clusters=clusters,
            k=best_k,
            silhouette=best_sil,
            inertia=inertia,
            model_version=self.embedding_model.version,
        )
