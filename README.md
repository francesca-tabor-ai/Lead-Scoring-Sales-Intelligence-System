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

## ☁️ Deploying to AWS Lambda

The app uses **sentence-transformers** and **scikit-learn**, so the full dependency set is ~7.5 GB. Lambda’s **500 MB limit** applies to **zip** deployments and to **runtime dependency installation**. You must deploy as a **Lambda container image** (up to **10 GB**) — do not use zip or “Install dependencies at runtime.”

### Fix the “500 MB” error

Use **only** one of these; anything else (zip upload, “Build from source”, “Install dependencies at runtime”, or a pipeline that packages code as zip) will hit the limit:

| ✅ Use | ❌ Do not use |
|--------|----------------|
| **GitHub Action** below (container image) | Lambda “Create function” → “Build from source” / “Deploy from GitHub” (zip) |
| **SAM**: `sam build` then `sam deploy` | Zip upload or “runtime dependency installation” |
| **Docker** build + push to ECR + create Lambda from image | Any tool that deploys a .zip of your code + deps |

### Deploy with GitHub Actions (container image)

1. In the repo: **Settings → Secrets and variables → Actions**. Add:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
2. Optional: **Variables** → `SAM_STACK_NAME` (e.g. `lead-scoring-api`), or the workflow uses `lead-scoring-api`.
3. Push to `main` or run **Actions → “Deploy Lambda (container image)” → Run workflow**.

The workflow runs `sam build` (builds the image from `Dockerfile.lambda`) and `sam deploy` with `--resolve-image-repos`, so the function uses the container image and 10 GB ephemeral storage.

### Deploy with AWS SAM (local)

The repo includes a SAM template that builds and deploys the API as a **container image** with 10 GB ephemeral storage.

1. **Install** [AWS SAM CLI](https://docs.aws.amazon.com/serverless-application-model/latest/developerguide/install-sam-cli.html) and ensure **Docker** is running.

2. **Build** (builds the Lambda container image from `Dockerfile.lambda`):
   ```bash
   sam build
   ```

3. **Deploy** (first time use `--guided` to set stack name, region, etc.):
   ```bash
   sam deploy --guided
   ```
   Then for later updates:
   ```bash
   sam deploy
   ```

4. After deploy, the **API URL** is in the stack outputs (`LeadScoringApiUrl`). Use it like:
   ```text
   https://<id>.lambda-url.<region>.on.aws/
   https://<id>.lambda-url.<region>.on.aws/api/v1/health
   https://<id>.lambda-url.<region>.on.aws/docs
   ```

The template sets: **PackageType: Image**, **EphemeralStorage: 10240 MB**, **MemorySize: 4096**, **Timeout: 900**.

### Deploy with Docker + ECR (manual)

1. Build: `docker build -f Dockerfile.lambda -t lead-scoring-api .`
2. Tag and push the image to Amazon ECR in your region.
3. Create (or update) a Lambda function from that image. Set **Handler** to `api.lambda_handler.handler`, **Ephemeral storage** to 10 GB, and add a **Function URL** if needed.

### Why not zip?

Zip (or “runtime dependency installation”) is limited to 500 MB. This app’s dependencies are ~7.5 GB, so deployment must use a **container image**.

**If you still see “Total dependency size exceeds Lambda ephemeral storage limit (500 MB)”:** your deployment is using a **zip** or **runtime dependency installation**, not a container image. Switch to **SAM** (`sam build` then `sam deploy`) so the template’s `PackageType: Image` and `Dockerfile.lambda` are used, or deploy the Docker image to Lambda manually. Do not use “Deploy from zip” or options that install dependencies at runtime.

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
