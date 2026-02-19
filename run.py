#!/usr/bin/env python3
"""
Quick CLI runner for the Competitor Intelligence pipeline.

Usage:
    python run.py                    # Demo with mock data
    python run.py --mode api         # Start FastAPI server
    python run.py --mode dashboard   # Start Streamlit dashboard
    python run.py --mode demo        # Demo pipeline run
"""

import sys
import os
import argparse
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run")


def demo():
    """Run a demo pipeline with mock data and print results."""
    from agents import (
        Orchestrator, ScraperAgent, FeatureExtractionAgent,
        GapDetectionAgent, OpportunityScoringAgent
    )
    from agents.scraper import ScrapeTarget
    from db.database import init_db

    init_db()

    print("\n" + "="*70)
    print("  🔍 COMPETITOR INTELLIGENCE & MARKET GAP FINDER — DEMO RUN")
    print("="*70 + "\n")

    # Demo targets (5 fake competitors, mock data)
    targets = [
        ScrapeTarget(url="https://mock.com/a", source_type="mock",
                     product_name="ProjectManagementPro", competitor_name="Asana-Alt",
                     max_reviews=150),
        ScrapeTarget(url="https://mock.com/b", source_type="mock",
                     product_name="CRMCloud", competitor_name="Salesforce-Alt",
                     max_reviews=150),
        ScrapeTarget(url="https://mock.com/c", source_type="mock",
                     product_name="DevToolsPlatform", competitor_name="Jira-Alt",
                     max_reviews=150),
        ScrapeTarget(url="https://mock.com/d", source_type="mock",
                     product_name="MarketingHub", competitor_name="HubSpot-Alt",
                     max_reviews=100),
        ScrapeTarget(url="https://mock.com/e", source_type="mock",
                     product_name="DataPipeline", competitor_name="Segment-Alt",
                     max_reviews=100),
    ]

    orchestrator = Orchestrator(
        agents=[
            ScraperAgent(),
            FeatureExtractionAgent(),
            GapDetectionAgent(
                complaint_rate_threshold=0.28,
                competition_density_threshold=0.25,
            ),
            OpportunityScoringAgent(alpha=0.5, beta=0.3, gamma=0.2),
        ],
        stop_on_failure=True,
    )

    print("🤖 Running pipeline...\n")
    result = orchestrator.execute(targets)

    if not result.success:
        print(f"\n❌ Pipeline failed: {result.error}")
        sys.exit(1)

    scoring = result.data
    print(orchestrator.summary())
    print()
    print(scoring.summary())
    print()

    # Print detailed top 3
    print("\n" + "─"*70)
    print("  📋 TOP 3 DETAILED OPPORTUNITIES")
    print("─"*70)
    for opp in scoring.opportunities[:3]:
        print(f"\n{'🎯' if opp.is_underserved else '📊'} #{opp.rank} — {opp.label.upper()}")
        print(f"   Score:          {opp.final_score:.3f} (confidence: {opp.confidence:.0%})")
        print(f"   Complaint Rate: {opp.complaint_rate:.0%}")
        print(f"   Coverage:       {opp.competition_raw:.0%}")
        print(f"   Volume:         {opp.volume} reviews")
        print(f"   Keywords:       {', '.join(opp.top_keywords[:5])}")
        print(f"   Action:         {opp.recommended_action}")
        if opp.representative_reviews:
            print(f"   Example:        \"{opp.representative_reviews[0][:100]}...\"")

    print("\n" + "="*70)
    print(f"  ✅ Analysis complete!")
    print(f"  💡 {sum(1 for o in scoring.opportunities if o.is_underserved)} market gaps identified")
    print(f"  🌐 Start dashboard: streamlit run dashboard/app.py")
    print(f"  🔌 Start API:       uvicorn api.main:app --reload --port 8000")
    print("="*70 + "\n")


def start_api():
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


def start_dashboard():
    import subprocess
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        os.path.join(os.path.dirname(__file__), "dashboard", "app.py"),
        "--server.port", "8501",
    ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Competitor Intelligence & Market Gap Finder")
    parser.add_argument(
        "--mode",
        choices=["demo", "api", "dashboard"],
        default="demo",
        help="Run mode: demo | api | dashboard",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        demo()
    elif args.mode == "api":
        start_api()
    elif args.mode == "dashboard":
        start_dashboard()
