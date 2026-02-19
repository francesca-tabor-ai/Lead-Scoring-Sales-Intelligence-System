# 🔍 Competitor Intelligence & Market Gap Finder

> AI-powered multi-agent system that transforms competitor reviews into ranked product opportunities.

---

## 🧠 What It Does

| Agent | Role | Output |
|-------|------|--------|
| **ScraperAgent** | Collects reviews from G2, Trustpilot, App Store, or mock | List of raw reviews |
| **FeatureExtractionAgent** | NLP enrichment + K-Means clustering | Enriched reviews + cluster themes |
| **GapDetectionAgent** | Identifies underserved clusters | Gap severity scores |
| **OpportunityScoringAgent** | Ranks opportunities by weighted model | Final scored + ranked gaps |

**Core Formula:**
```
Score_k = α·Q(Demand_k) − β·Q(Competition_k) + γ·Q(NegSentiment_k)
FinalScore_k = Confidence_k × Score_k
```

Default weights: α=0.5, β=0.3, γ=0.2

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run demo (mock data)
```bash
python run.py --mode demo
```

### 3. Start Dashboard (Streamlit)
```bash
python run.py --mode dashboard
# → http://localhost:8501
```

### 4. Start API (FastAPI)
```bash
python run.py --mode api
# → http://localhost:8000/docs
```

### 5. Run with Docker
```bash
docker-compose up
```

---

## 📁 Project Structure

```
competitor_intelligence/
├── agents/
│   ├── base.py            # Abstract Agent + Orchestrator
│   ├── scraper.py         # Review Scraper Agent
│   ├── extractor.py       # Feature Extraction Agent (NLP + Clustering)
│   ├── gap_detector.py    # Gap Detection Agent
│   └── scorer.py         # Opportunity Scoring Agent
├── api/
│   ├── main.py            # FastAPI app entry point
│   ├── routes.py          # API route handlers
│   └── schemas.py         # Pydantic request/response models
├── dashboard/
│   └── app.py             # Streamlit dashboard
├── db/
│   ├── models.py          # SQLAlchemy ORM models
│   └── database.py        # DB session management
├── config/
│   └── settings.py        # Centralized configuration
├── tests/
│   └── test_pipeline.py   # Pytest test suite
├── run.py                 # CLI runner
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/pipeline/run` | Run full pipeline |
| `GET` | `/api/v1/opportunities` | Get ranked opportunities |
| `GET` | `/api/v1/opportunities/{id}` | Get single cluster detail |
| `GET` | `/api/v1/opportunities/{id}/evidence` | Get review evidence |
| `GET` | `/api/v1/score/weights` | Get current weights |
| `POST` | `/api/v1/score/weights` | Update α, β, γ |
| `GET` | `/api/v1/summary` | Pipeline summary |
| `GET` | `/api/v1/health` | Health check |

### Example API call:
```bash
curl -X POST http://localhost:8000/api/v1/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "targets": [
      {"url": "https://example.com", "source_type": "mock",
       "product_name": "ProductA", "competitor_name": "CompA", "max_reviews": 200}
    ],
    "alpha": 0.5, "beta": 0.3, "gamma": 0.2
  }'
```

---

## 📊 Dashboard Sections

1. **KPI Cards** — Reviews analyzed, clusters found, market gaps identified
2. **Top Opportunity Card** — Gauge chart + recommendation
3. **Opportunity Ranking Table** — All clusters ranked by score
4. **Cluster Map** — Complaint Rate vs Competition Density scatter
5. **Complaint Heatmap** — Per-cluster complaint bars
6. **Score Breakdown** — Stacked bar showing α·Demand − β·Competition + γ·Sentiment
7. **Feature Gap Explorer** — Drill-down with evidence + competitor coverage

---

## ⚙️ Configuration

All settings in `config/settings.py` or via `.env`:

| Key | Default | Description |
|-----|---------|-------------|
| `ALPHA` | `0.5` | Demand weight |
| `BETA` | `0.3` | Competition weight |
| `GAMMA` | `0.2` | Sentiment weight |
| `COMPLAINT_RATE_THRESHOLD` | `0.30` | Min complaint rate for gap |
| `COMPETITION_DENSITY_THRESHOLD` | `0.20` | Max coverage for gap |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `MAX_CLUSTERS` | `20` | Max K for clustering |

---

## 🧪 Run Tests
```bash
python -m pytest tests/ -v
```

---

## 🗺️ Architecture

```
List[ScrapeTarget]
       │
       ▼
 ┌─────────────┐
 │ ScraperAgent│  ← HTML, RSS, Mock
 └──────┬──────┘
        │ List[RawReview]
        ▼
 ┌──────────────────────┐
 │ FeatureExtractionAgent│  ← Sentiment, Embeddings, K-Means
 └──────────┬───────────┘
            │ ExtractorOutput
            ▼
 ┌───────────────────┐
 │ GapDetectionAgent │  ← Coverage density, Gap severity
 └────────┬──────────┘
          │ GapDetectionOutput
          ▼
 ┌────────────────────────┐
 │ OpportunityScoringAgent│  ← Weighted score + confidence
 └────────────┬───────────┘
              │ ScoringOutput
              ▼
     API / Dashboard
```
