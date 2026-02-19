"""
Review Scraper Agent
---------------------
Collects competitor product reviews from multiple sources.

Supported sources (v1):
  - Generic HTML page scraping
  - G2 (public reviews)
  - Trustpilot
  - App Store (iTunes lookup API)
  - Mock/demo mode for development

Architecture:
  ScraperAgent.run(config) -> List[RawReview]
"""

import hashlib
import time
import random
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from agents.base import Agent
from config.settings import settings

logger = logging.getLogger(__name__)


# ─── Data Structures ────────────────────────────────────────────────────────


@dataclass
class ScrapeTarget:
    """One product / URL to scrape."""
    url: str
    source_type: str          # "g2" | "trustpilot" | "app_store" | "generic" | "mock"
    product_name: str
    competitor_name: str
    max_reviews: int = 200
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RawReview:
    """Normalized raw review record."""
    text: str
    rating: Optional[float]
    review_date: Optional[datetime]
    author_hash: str
    source: str
    product_name: str
    competitor_name: str
    url: str
    locale: str = "en"


# ─── Per-Source Scrapers ──────────────────────────────────────────────────────


class BaseScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.USER_AGENT})

    def _get(self, url: str, **kwargs) -> requests.Response:
        """HTTP GET with retry + exponential backoff."""
        for attempt in range(settings.MAX_RETRIES):
            try:
                resp = self.session.get(url, timeout=settings.REQUEST_TIMEOUT, **kwargs)
                resp.raise_for_status()
                return resp
            except requests.RequestException as e:
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"Attempt {attempt+1} failed for {url}: {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"Failed to fetch {url} after {settings.MAX_RETRIES} attempts")

    def _hash_author(self, author: str) -> str:
        return hashlib.sha256(author.encode()).hexdigest()[:16]

    def _clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def scrape(self, target: ScrapeTarget) -> List[RawReview]:
        raise NotImplementedError


class G2Scraper(BaseScraper):
    """Scrapes G2.com review pages."""

    def scrape(self, target: ScrapeTarget) -> List[RawReview]:
        reviews = []
        page = 1
        base_url = target.url.rstrip("/")

        while len(reviews) < target.max_reviews:
            url = f"{base_url}?page={page}"
            try:
                resp = self._get(url)
            except RuntimeError:
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            review_cards = soup.find_all("div", class_=re.compile(r"paper.*review", re.I))

            if not review_cards:
                # Fallback: look for itemprop="reviewBody"
                review_cards = soup.find_all(attrs={"itemprop": "reviewBody"})

            if not review_cards:
                logger.info(f"G2: No reviews found on page {page}, stopping.")
                break

            for card in review_cards:
                text_el = card.find(attrs={"itemprop": "reviewBody"}) or card
                text = self._clean_text(text_el.get_text())
                if len(text) < 20:
                    continue

                rating_el = card.find(attrs={"itemprop": "ratingValue"})
                rating = float(rating_el["content"]) if rating_el else None

                date_el = card.find("time")
                review_date = None
                if date_el and date_el.get("datetime"):
                    try:
                        review_date = datetime.fromisoformat(date_el["datetime"][:10])
                    except ValueError:
                        pass

                author_el = card.find(attrs={"itemprop": "author"})
                author = author_el.get_text() if author_el else "unknown"

                reviews.append(RawReview(
                    text=text,
                    rating=rating,
                    review_date=review_date,
                    author_hash=self._hash_author(author),
                    source="g2",
                    product_name=target.product_name,
                    competitor_name=target.competitor_name,
                    url=url,
                ))

            page += 1
            time.sleep(settings.SCRAPE_DELAY_SECONDS)

        return reviews[:target.max_reviews]


class TrustpilotScraper(BaseScraper):
    """Scrapes Trustpilot review pages."""

    def scrape(self, target: ScrapeTarget) -> List[RawReview]:
        reviews = []
        page = 1
        base_url = target.url.rstrip("/")

        while len(reviews) < target.max_reviews:
            url = f"{base_url}?page={page}"
            try:
                resp = self._get(url)
            except RuntimeError:
                break

            soup = BeautifulSoup(resp.text, "html.parser")
            cards = soup.find_all("article", class_=re.compile(r"review", re.I))

            if not cards:
                break

            for card in cards:
                body = card.find("p", class_=re.compile(r"review-content", re.I))
                if not body:
                    body = card.find("p")
                if not body:
                    continue
                text = self._clean_text(body.get_text())
                if len(text) < 20:
                    continue

                star_el = card.find(attrs={"data-service-review-rating": True})
                rating = None
                if star_el:
                    try:
                        rating = float(star_el["data-service-review-rating"])
                    except (ValueError, KeyError):
                        pass

                date_el = card.find("time")
                review_date = None
                if date_el and date_el.get("datetime"):
                    try:
                        review_date = datetime.fromisoformat(date_el["datetime"][:10])
                    except ValueError:
                        pass

                author_el = card.find(class_=re.compile(r"consumer-info", re.I))
                author = author_el.get_text() if author_el else "unknown"

                reviews.append(RawReview(
                    text=text,
                    rating=rating,
                    review_date=review_date,
                    author_hash=self._hash_author(author),
                    source="trustpilot",
                    product_name=target.product_name,
                    competitor_name=target.competitor_name,
                    url=url,
                ))

            page += 1
            time.sleep(settings.SCRAPE_DELAY_SECONDS)

        return reviews[:target.max_reviews]


class AppStoreScraper(BaseScraper):
    """Fetches App Store reviews via iTunes RSS API (no HTML scraping needed)."""

    RSS_TEMPLATE = (
        "https://itunes.apple.com/{country}/rss/customerreviews/"
        "page={page}/id={app_id}/sortby=mostrecent/json"
    )

    def scrape(self, target: ScrapeTarget) -> List[RawReview]:
        app_id = target.extra.get("app_id") or target.sku_or_app_id if hasattr(target, "sku_or_app_id") else None
        country = target.extra.get("country", "us")
        if not app_id:
            # Try to extract from URL
            m = re.search(r"/id(\d+)", target.url)
            app_id = m.group(1) if m else None
        if not app_id:
            logger.warning("AppStoreScraper: no app_id provided, skipping.")
            return []

        reviews = []
        for page in range(1, 11):  # iTunes RSS only provides up to 10 pages
            if len(reviews) >= target.max_reviews:
                break
            url = self.RSS_TEMPLATE.format(country=country, page=page, app_id=app_id)
            try:
                resp = self._get(url)
                data = resp.json()
            except Exception as e:
                logger.warning(f"AppStoreScraper page {page} failed: {e}")
                break

            entries = data.get("feed", {}).get("entry", [])
            if not entries:
                break
            # First entry is the app metadata, skip it
            for entry in entries[1:]:
                text = entry.get("content", {}).get("label", "")
                if len(text) < 20:
                    continue
                rating_raw = entry.get("im:rating", {}).get("label")
                rating = float(rating_raw) if rating_raw else None
                date_str = entry.get("updated", {}).get("label", "")
                review_date = None
                try:
                    review_date = datetime.fromisoformat(date_str[:10])
                except Exception:
                    pass
                author = entry.get("author", {}).get("name", {}).get("label", "unknown")

                reviews.append(RawReview(
                    text=text,
                    rating=rating,
                    review_date=review_date,
                    author_hash=self._hash_author(author),
                    source="app_store",
                    product_name=target.product_name,
                    competitor_name=target.competitor_name,
                    url=target.url,
                ))
            time.sleep(settings.SCRAPE_DELAY_SECONDS * 0.5)

        return reviews[:target.max_reviews]


class MockScraper(BaseScraper):
    """
    Generates synthetic review data for development/demo.
    Produces realistic complaint distributions across themes.
    """

    THEMES = [
        ("checkout_ux", 0.42, ["checkout", "payment", "cart", "confusing", "slow", "form", "button"]),
        ("api_speed", 0.35, ["api", "slow", "timeout", "response", "latency", "rate limit"]),
        ("mobile_app", 0.55, ["mobile", "app", "crash", "freeze", "ios", "android", "update"]),
        ("customer_support", 0.48, ["support", "help", "response", "ticket", "slow", "unhelpful"]),
        ("pricing", 0.28, ["price", "expensive", "cost", "billing", "charge", "plan"]),
        ("onboarding", 0.33, ["setup", "onboarding", "documentation", "confusing", "tutorial"]),
        ("integrations", 0.31, ["integration", "connect", "api", "plugin", "sync", "import"]),
        ("reporting", 0.25, ["report", "dashboard", "analytics", "export", "chart", "filter"]),
    ]

    POSITIVE_PHRASES = [
        "Great product overall!", "Really impressed with the features.",
        "Easy to use and reliable.", "Best in class for our use case.",
        "Excellent value for money.", "The team is very responsive.",
        "Highly recommend this tool.", "Works exactly as expected.",
    ]

    COMPLAINT_TEMPLATES = [
        "The {theme} is really frustrating. {detail} Makes it hard to use.",
        "I keep having issues with {theme}. {detail} Very disappointed.",
        "{detail} This is a major problem with their {theme}.",
        "Wish they would fix the {theme} issue. {detail}",
        "Not happy with {theme}. {detail} Thinking of switching.",
        "The {theme} needs a lot of work. {detail} Rating them 2 stars.",
    ]

    def scrape(self, target: ScrapeTarget) -> List[RawReview]:
        import random
        from datetime import timedelta
        random.seed(hash(target.product_name) % 2**31)

        reviews = []
        n = min(target.max_reviews, 300)

        for i in range(n):
            # Pick a theme weighted by complaint rate
            theme_name, complaint_rate, keywords = random.choice(self.THEMES)

            is_negative = random.random() < complaint_rate
            rating = random.choice([1, 2]) if is_negative else random.choice([4, 5])

            if is_negative:
                detail = f"The {random.choice(keywords)} doesn't work properly."
                template = random.choice(self.COMPLAINT_TEMPLATES)
                text = template.format(theme=theme_name.replace("_", " "), detail=detail)
            else:
                text = random.choice(self.POSITIVE_PHRASES)

            days_ago = random.randint(1, 730)
            review_date = datetime.utcnow() - timedelta(days=days_ago)

            reviews.append(RawReview(
                text=text,
                rating=float(rating),
                review_date=review_date,
                author_hash=self._hash_author(f"user_{i}_{target.competitor_name}"),
                source="mock",
                product_name=target.product_name,
                competitor_name=target.competitor_name,
                url=target.url,
            ))

        return reviews


# ─── Deduplication ───────────────────────────────────────────────────────────


def deduplicate_reviews(reviews: List[RawReview], threshold: float = 0.90) -> List[RawReview]:
    """
    Remove near-duplicate reviews using Jaccard similarity on word sets.
    Fast O(n²) pass — swap for MinHash/LSH on large corpora.
    """
    unique = []
    seen_sets = []

    for review in reviews:
        words = set(review.text.lower().split())
        is_dup = False
        for seen in seen_sets:
            intersection = len(words & seen)
            union = len(words | seen)
            if union > 0 and intersection / union >= threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(review)
            seen_sets.append(words)

    logger.info(f"Deduplication: {len(reviews)} → {len(unique)} reviews")
    return unique


# ─── ScraperAgent ────────────────────────────────────────────────────────────


_SCRAPER_MAP = {
    "g2": G2Scraper,
    "trustpilot": TrustpilotScraper,
    "app_store": AppStoreScraper,
    "mock": MockScraper,
}


class ScraperAgent(Agent):
    """
    Agent 1: Review Scraper

    Input:  List[ScrapeTarget]
    Output: List[RawReview]  (deduplicated)
    """

    def __init__(self):
        super().__init__(name="ScraperAgent")
        self._scrapers: Dict[str, BaseScraper] = {}

    def _get_scraper(self, source_type: str) -> BaseScraper:
        if source_type not in self._scrapers:
            cls = _SCRAPER_MAP.get(source_type, MockScraper)
            self._scrapers[source_type] = cls()
        return self._scrapers[source_type]

    def run(self, targets: List[ScrapeTarget]) -> List[RawReview]:
        all_reviews: List[RawReview] = []

        for target in targets:
            self.logger.info(
                f"Scraping [{target.source_type}] {target.product_name} "
                f"@ {target.competitor_name} — {target.url}"
            )
            scraper = self._get_scraper(target.source_type)
            try:
                reviews = scraper.scrape(target)
                self.logger.info(
                    f"  → {len(reviews)} reviews scraped from {target.competitor_name}"
                )
                all_reviews.extend(reviews)
            except Exception as e:
                self.logger.error(f"  ❌ Scraping failed for {target.url}: {e}")

        deduped = deduplicate_reviews(all_reviews)
        self.logger.info(f"Total: {len(deduped)} unique reviews across {len(targets)} targets")
        return deduped
