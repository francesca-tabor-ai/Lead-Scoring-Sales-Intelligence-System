"""
Configuration & Settings
Competitor Intelligence & Market Gap Finder
"""

from pydantic import BaseModel
from typing import Optional
import os


class Settings(BaseModel):
    # App
    APP_NAME: str = "Competitor Intelligence & Market Gap Finder"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "sqlite:///./competitor_intel.db"

    # Scraping
    SCRAPE_DELAY_SECONDS: float = 1.5
    MAX_REVIEWS_PER_PRODUCT: int = 500
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    USER_AGENT: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    # NLP / Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_CLUSTERS: int = 20
    MIN_CLUSTER_SIZE: int = 5
    SENTIMENT_THRESHOLD: float = -0.2
    LANGUAGE_FILTER: Optional[str] = "en"

    # Gap Detection thresholds
    # COMPETITION_DENSITY_THRESHOLD: fraction of competitors that "cover" a cluster.
    # 0.80 means a cluster is underserved if <80% of competitors address it —
    # practical for real-world scenarios where most clusters are partially covered.
    COMPLAINT_RATE_THRESHOLD: float = 0.30
    COMPETITION_DENSITY_THRESHOLD: float = 0.80
    COVERAGE_SHARE_THRESHOLD: float = 0.10
    SMOOTHING_STRENGTH: int = 5

    # Opportunity scoring weights
    ALPHA: float = 0.5
    BETA: float = 0.3
    GAMMA: float = 0.2
    CONFIDENCE_ETA: float = 50.0

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000


settings = Settings()
