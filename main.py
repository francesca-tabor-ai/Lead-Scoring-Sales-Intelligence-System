"""
Entry point for the Competitor Intelligence & Market Gap Finder.

Usage:
  # Run a quick demo (mock data, no installs needed beyond core deps):
  python main.py demo

  # Start the FastAPI server:
  python main.py api

  # Start the Streamlit dashboard:
  python main.py dashboard

  # Run tests:
  python main.py test
"""

from __future__ import annotations

import sys
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger("main")


def demo():
    """
    End-to-end demo run with mock data.
    Prints a formatted report to stdout.
    """
    from utils.pipeline import run_pipeline

    logger.info("=== Competitor Intelligence — Demo Run ===")

    targets = [
        {"competitor": "CompA", "product_id": "prod_a", "source": "mock", "url_or_id": ""},
        {"competitor": "CompB", "product_id": "prod_b", "source": "mock", "url_or_id": ""},
        {"competitor": "CompC", "product_id": "prod_c", "source": "mock", "url_or_id": ""},
    ]

    result = run_pipeline(
        targets=targets,
        category="SaaS CRM",
        mock=True,
        scoring_weights={"alpha": 0.5, "beta": 0.3, "gamma": 0.2},
        k_min=3,
        k_max=8,
        complaint_threshold=0.30,
        competition_threshold=0.25,
        n_bootstrap=200,
        verbose=True,
    )

    # ── Print report ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPETITOR INTELLIGENCE REPORT")
    print("=" * 70)
    print(f"  Run ID     : {result.run_id}")
    print(f"  Category   : {result.category}")
    print(f"  Reviews    : {result.total_reviews}")
    print(f"  Clusters   : {result.n_clusters}")
    print(f"  Gaps Found : {result.n_gaps}")
    print(f"  Timestamp  : {result.executed_at.isoformat()}")
    print("=" * 70)

    # Cluster Summary
    print("\n📊 CLUSTER SUMMARY")
    print("-" * 70)
    for cm in sorted(result.cluster_metrics, key=lambda c: c.complaint_rate, reverse=True):
        print(
            f"  [{cm.cluster_id}] {cm.label:<28} "
            f"n={cm.size:>3}  complaint={cm.complaint_rate:.1%}  "
            f"rating={cm.avg_rating:.1f}  sent={cm.sentiment_mean:+.2f}"
        )
        if cm.top_terms:
            print(f"       Keywords: {', '.join(cm.top_terms[:5])}")

    # Gap Detection Results
    print("\n⚠️  DETECTED MARKET GAPS (underserved only)")
    print("-" * 70)
    underserved = [g for g in result.gaps if g.is_underserved]
    if underserved:
        for g in sorted(underserved, key=lambda x: x.gap_severity, reverse=True):
            print(
                f"  ✅ [{g.cluster_id}] {g.label:<28} "
                f"complaint={g.complaint_rate:.1%}  "
                f"comp_density={g.competition_density:.1%}  "
                f"severity={g.gap_severity:.3f}"
            )
    else:
        print("  No underserved gaps detected with current thresholds.")

    # Top Opportunities
    print("\n🏆 TOP OPPORTUNITIES")
    print("-" * 70)
    for opp in result.opportunities[:5]:
        gap = next((g for g in result.gaps if g.cluster_id == opp.cluster_id), None)
        underserved_tag = " ✅ UNDERSERVED" if (gap and gap.is_underserved) else ""
        print(
            f"  #{opp.rank}  {opp.label:<28} "
            f"score={opp.final_score:.4f}  "
            f"conf={opp.confidence:.2f}  "
            f"CI=[{opp.ci_low:.2f},{opp.ci_high:.2f}]"
            f"{underserved_tag}"
        )
        print(
            f"      demand_q={opp.demand_q:.2f}  "
            f"competition_q={opp.competition_q:.2f}  "
            f"neg_sent_q={opp.neg_sentiment_q:.2f}"
        )

    # Top recommendation
    if result.opportunities:
        top = result.opportunities[0]
        top_gap = next((g for g in result.gaps if g.cluster_id == top.cluster_id), None)
        print("\n" + "=" * 70)
        print("  💡 TOP RECOMMENDATION")
        print("=" * 70)
        print(f"  Feature Theme   : {top.label.replace('_', ' ').title()}")
        if top_gap:
            print(f"  Complaint Rate  : {top_gap.complaint_rate:.1%}")
            print(f"  Comp. Density   : {top_gap.competition_density:.1%}")
        print(f"  Opportunity Score: {top.final_score:.4f}")
        print(f"  Confidence      : {top.confidence:.2%}")
        print(f"  → Prioritize this feature in your next sprint.")
        print("=" * 70)

    return result


def start_api():
    """Start the FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        print("Error: install uvicorn first:  pip install uvicorn")
        sys.exit(1)
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)


def start_dashboard():
    """Launch the Streamlit dashboard."""
    import subprocess
    import os
    dashboard_path = os.path.join(os.path.dirname(__file__), "dashboard", "app.py")
    subprocess.run(["streamlit", "run", dashboard_path], check=True)


def run_tests():
    """Run pytest."""
    import subprocess
    result = subprocess.run(
        ["pytest", "tests/", "-v", "--tb=short"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    import os
    command = sys.argv[1] if len(sys.argv) > 1 else "demo"

    if command == "demo":
        demo()
    elif command == "api":
        start_api()
    elif command == "dashboard":
        start_dashboard()
    elif command == "test":
        run_tests()
    else:
        print(f"Unknown command: {command}")
        print("Usage: python main.py [demo|api|dashboard|test]")
        sys.exit(1)
