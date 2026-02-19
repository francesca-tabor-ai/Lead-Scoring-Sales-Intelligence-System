"""
SQLAlchemy ORM Models
Competitor Intelligence & Market Gap Finder
"""

from sqlalchemy import (
    Column, Integer, Float, String, Text, Boolean,
    DateTime, ForeignKey, JSON, Index
)
from sqlalchemy.orm import relationship, declarative_base
from datetime import datetime

Base = declarative_base()


class Competitor(Base):
    __tablename__ = "competitor"

    competitor_id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False, unique=True)
    category = Column(String(255))
    region = Column(String(100), default="global")
    website = Column(String(500))
    created_at = Column(DateTime, default=datetime.utcnow)

    products = relationship("Product", back_populates="competitor", cascade="all, delete-orphan")


class Product(Base):
    __tablename__ = "product"

    product_id = Column(Integer, primary_key=True, autoincrement=True)
    competitor_id = Column(Integer, ForeignKey("competitor.competitor_id"), nullable=False)
    name = Column(String(255), nullable=False)
    sku_or_app_id = Column(String(255))
    pricing_tier = Column(String(100))
    source_url = Column(String(1000))
    source_type = Column(String(50))  # amazon, g2, app_store, trustpilot, etc.
    created_at = Column(DateTime, default=datetime.utcnow)

    competitor = relationship("Competitor", back_populates="products")
    reviews = relationship("Review", back_populates="product", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_product_competitor", "competitor_id"),)


class Review(Base):
    __tablename__ = "review"

    review_id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, ForeignKey("product.product_id"), nullable=False)
    source = Column(String(100))
    rating = Column(Float)
    text = Column(Text, nullable=False)
    review_date = Column(DateTime)
    locale = Column(String(20), default="en")
    author_hash = Column(String(64))
    url = Column(String(1000))
    is_duplicate = Column(Boolean, default=False)
    scraped_at = Column(DateTime, default=datetime.utcnow)

    product = relationship("Product", back_populates="reviews")
    enrichment = relationship("ReviewEnrichment", back_populates="review", uselist=False)
    cluster_memberships = relationship("ClusterMembership", back_populates="review")

    __table_args__ = (
        Index("ix_review_product", "product_id"),
        Index("ix_review_date", "review_date"),
    )


class ReviewEnrichment(Base):
    __tablename__ = "review_enrichment"

    enrichment_id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(Integer, ForeignKey("review.review_id"), nullable=False, unique=True)
    cleaned_text = Column(Text)
    language = Column(String(20))
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    is_complaint = Column(Boolean, default=False)
    aspects = Column(JSON)
    embedding_vector = Column(JSON)
    processed_at = Column(DateTime, default=datetime.utcnow)

    review = relationship("Review", back_populates="enrichment")


class ClusterRun(Base):
    __tablename__ = "cluster_run"

    run_id = Column(Integer, primary_key=True, autoincrement=True)
    model_version = Column(String(100))
    k = Column(Integer)
    silhouette_score = Column(Float)
    inertia = Column(Float)
    num_reviews = Column(Integer)
    category = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)

    clusters = relationship("Cluster", back_populates="run", cascade="all, delete-orphan")


class Cluster(Base):
    __tablename__ = "cluster"

    cluster_id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("cluster_run.run_id"), nullable=False)
    cluster_index = Column(Integer)
    label = Column(String(255))
    top_keywords = Column(JSON)
    centroid_vector = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)

    run = relationship("ClusterRun", back_populates="clusters")
    memberships = relationship("ClusterMembership", back_populates="cluster")
    metrics = relationship("ClusterMetrics", back_populates="cluster", uselist=False)
    competitor_coverage = relationship("ClusterCompetitorCoverage", back_populates="cluster")
    opportunity = relationship("Opportunity", back_populates="cluster", uselist=False)

    __table_args__ = (Index("ix_cluster_run", "run_id"),)


class ClusterMembership(Base):
    __tablename__ = "cluster_membership"

    membership_id = Column(Integer, primary_key=True, autoincrement=True)
    review_id = Column(Integer, ForeignKey("review.review_id"), nullable=False)
    cluster_id = Column(Integer, ForeignKey("cluster.cluster_id"), nullable=False)
    distance_to_centroid = Column(Float)
    assigned_at = Column(DateTime, default=datetime.utcnow)

    review = relationship("Review", back_populates="cluster_memberships")
    cluster = relationship("Cluster", back_populates="memberships")

    __table_args__ = (
        Index("ix_membership_cluster", "cluster_id"),
        Index("ix_membership_review", "review_id"),
    )


class ClusterMetrics(Base):
    __tablename__ = "cluster_metrics"

    metrics_id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(Integer, ForeignKey("cluster.cluster_id"), nullable=False, unique=True)
    mention_freq = Column(Float)
    complaint_freq = Column(Float)
    complaint_intensity = Column(Float)
    avg_rating = Column(Float)
    sentiment_mean = Column(Float)
    volume = Column(Integer)
    volume_trend = Column(Float)
    updated_at = Column(DateTime, default=datetime.utcnow)

    cluster = relationship("Cluster", back_populates="metrics")


class ClusterCompetitorCoverage(Base):
    __tablename__ = "cluster_competitor_coverage"

    coverage_id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(Integer, ForeignKey("cluster.cluster_id"), nullable=False)
    competitor_id = Column(Integer, ForeignKey("competitor.competitor_id"), nullable=False)
    raw_share = Column(Float)
    smoothed_share = Column(Float)
    covers = Column(Boolean, default=False)
    evidence_count = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow)

    cluster = relationship("Cluster", back_populates="competitor_coverage")

    __table_args__ = (
        Index("ix_coverage_cluster_competitor", "cluster_id", "competitor_id"),
    )


class Opportunity(Base):
    __tablename__ = "opportunity"

    opportunity_id = Column(Integer, primary_key=True, autoincrement=True)
    cluster_id = Column(Integer, ForeignKey("cluster.cluster_id"), nullable=False, unique=True)
    opportunity_score = Column(Float)
    raw_score = Column(Float)
    confidence = Column(Float)
    demand = Column(Float)
    competition = Column(Float)
    neg_sentiment = Column(Float)
    gap_severity = Column(Float)
    is_underserved = Column(Boolean, default=False)
    rank = Column(Integer)
    components_json = Column(JSON)
    model_version = Column(String(100))
    updated_at = Column(DateTime, default=datetime.utcnow)

    cluster = relationship("Cluster", back_populates="opportunity")

    __table_args__ = (Index("ix_opportunity_score", "opportunity_score"),)
