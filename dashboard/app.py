"""
Streamlit Dashboard
Competitor Intelligence & Market Gap Finder

Sections:
  1. Sidebar — pipeline configuration
  2. Overview KPIs
  3. Opportunity Ranking Table
  4. Cluster Map (2D UMAP/PCA scatter)
  5. Complaint Heatmap (cluster × competitor)
  6. Feature Gap Explorer (drill-down)
  7. Raw Review Evidence
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import logging

from agents import (
    Orchestrator, ScraperAgent, FeatureExtractionAgent,
    GapDetectionAgent, OpportunityScoringAgent
)
from agents.scraper import ScrapeTarget
from agents.scorer import ScoringOutput, ScoredOpportunity
from config.settings import settings

logger = logging.getLogger(__name__)

# ─── Page Config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Competitor Intelligence & Market Gap Finder",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 20px;
        color: white;
        text-align: center;
        margin: 5px;
    }
    .gap-badge {
        background: #ff4b4b;
        color: white;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: bold;
    }
    .opportunity-badge {
        background: #00cc88;
        color: white;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 12px;
    }
    h1 { color: #2d3748; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ─── State Management ────────────────────────────────────────────────────────

def get_state():
    if "scoring_output" not in st.session_state:
        st.session_state.scoring_output = None
    if "enriched_reviews" not in st.session_state:
        st.session_state.enriched_reviews = None
    return st.session_state


# ─── Pipeline Runner ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def run_pipeline_cached(
    targets_json: str,
    alpha: float, beta: float, gamma: float,
    cr_threshold: float, cd_threshold: float,
):
    """Run the full pipeline (cached to avoid re-runs on widget changes)."""
    import json
    targets_data = json.loads(targets_json)
    targets = [
        ScrapeTarget(
            url=t["url"],
            source_type=t["source_type"],
            product_name=t["product_name"],
            competitor_name=t["competitor_name"],
            max_reviews=t.get("max_reviews", 200),
        )
        for t in targets_data
    ]

    # Run pipeline
    scraper = ScraperAgent()
    extractor = FeatureExtractionAgent()
    gap_detector = GapDetectionAgent(
        complaint_rate_threshold=cr_threshold,
        competition_density_threshold=cd_threshold,
    )
    scorer = OpportunityScoringAgent(alpha=alpha, beta=beta, gamma=gamma)

    orchestrator = Orchestrator(
        agents=[scraper, extractor, gap_detector, scorer],
        stop_on_failure=True,
    )
    result = orchestrator.execute(targets)
    if not result.success:
        raise RuntimeError(result.error)
    return result.data


# ─── Sidebar ─────────────────────────────────────────────────────────────────

def render_sidebar() -> Dict:
    with st.sidebar:
        st.image("https://img.icons8.com/nolan/96/000000/spy.png", width=60)
        st.title("🔍 Configuration")
        st.divider()

        # ── Competitors ──────────────────────────────────────────────────────
        st.subheader("📋 Competitors to Analyze")
        st.caption("Each row = one product/competitor to scrape")

        default_targets = [
            {"competitor": "CompetitorA", "product": "ProductA", "url": "https://example.com/a", "source": "mock"},
            {"competitor": "CompetitorB", "product": "ProductB", "url": "https://example.com/b", "source": "mock"},
            {"competitor": "CompetitorC", "product": "ProductC", "url": "https://example.com/c", "source": "mock"},
            {"competitor": "CompetitorD", "product": "ProductD", "url": "https://example.com/d", "source": "mock"},
        ]

        targets_df = pd.DataFrame(default_targets)
        edited = st.data_editor(
            targets_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "source": st.column_config.SelectboxColumn(
                    options=["mock", "g2", "trustpilot", "app_store"]
                )
            },
        )

        max_reviews = st.slider("Max reviews per product", 50, 500, 200, step=50)

        st.divider()

        # ── Scoring Weights ──────────────────────────────────────────────────
        st.subheader("⚖️ Scoring Weights")
        st.caption("α + β + γ should sum to 1.0")
        alpha = st.slider("α Demand weight", 0.0, 1.0, settings.ALPHA, step=0.05)
        beta  = st.slider("β Competition weight", 0.0, 1.0, settings.BETA, step=0.05)
        gamma = st.slider("γ Sentiment weight", 0.0, 1.0, settings.GAMMA, step=0.05)

        total = alpha + beta + gamma
        if abs(total - 1.0) > 0.05:
            st.warning(f"Weights sum to {total:.2f} (ideally 1.0)")

        st.divider()

        # ── Gap Thresholds ───────────────────────────────────────────────────
        st.subheader("🎯 Gap Detection Thresholds")
        cr_threshold = st.slider("Min complaint rate", 0.1, 0.7, 0.30, step=0.05,
                                  format="%.0f%%", help="Minimum complaint rate to flag a gap")
        cd_threshold = st.slider("Max competition density", 0.05, 0.50, 0.20, step=0.05,
                                  format="%.0f%%", help="Maximum competitor coverage to flag a gap")

        st.divider()
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    return {
        "targets": edited,
        "max_reviews": max_reviews,
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "cr_threshold": cr_threshold / 100 if cr_threshold > 1 else cr_threshold,
        "cd_threshold": cd_threshold / 100 if cd_threshold > 1 else cd_threshold,
        "run": run_button,
    }


# ─── KPI Cards ───────────────────────────────────────────────────────────────

def render_kpis(scoring: ScoringOutput):
    underserved = [o for o in scoring.opportunities if o.is_underserved]
    top = scoring.top_opportunity

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("📊 Total Reviews", f"{scoring.total_reviews:,}")
    with col2:
        st.metric("🗂️ Clusters Found", len(scoring.opportunities))
    with col3:
        st.metric("🎯 Market Gaps", len(underserved), delta=f"+{len(underserved)} opportunities")
    with col4:
        st.metric("🏆 Top Score", f"{top.final_score:.3f}" if top else "—")
    with col5:
        avg_complaint = np.mean([o.complaint_rate for o in scoring.opportunities])
        st.metric("⚠️ Avg Complaint Rate", f"{avg_complaint:.0%}")


# ─── Opportunity Table ───────────────────────────────────────────────────────

def render_opportunity_table(scoring: ScoringOutput):
    st.subheader("📈 Opportunity Ranking")

    rows = []
    for opp in scoring.opportunities:
        rows.append({
            "Rank": opp.rank,
            "Feature Cluster": opp.label,
            "Score": opp.final_score,
            "Complaint %": f"{opp.complaint_rate:.0%}",
            "Coverage %": f"{opp.competition_raw:.0%}",
            "Volume": opp.volume,
            "Gap?": "🎯 YES" if opp.is_underserved else "—",
            "Confidence": f"{opp.confidence:.0%}",
        })

    df = pd.DataFrame(rows)

    # Color rows: underserved = red highlight
    def highlight_gap(row):
        if "YES" in str(row.get("Gap?", "")):
            return ["background-color: #fff0f0"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df.style.apply(highlight_gap, axis=1),
        use_container_width=True,
        hide_index=True,
    )


# ─── Cluster Map ─────────────────────────────────────────────────────────────

def render_cluster_map(scoring: ScoringOutput):
    st.subheader("🗺️ Cluster Map (Opportunity Space)")

    df = pd.DataFrame([{
        "label": opp.label,
        "complaint_rate": opp.complaint_rate * 100,
        "competition_density": opp.competition_raw * 100,
        "final_score": opp.final_score,
        "volume": opp.volume,
        "is_underserved": "🎯 Gap" if opp.is_underserved else "Covered",
        "rank": opp.rank,
    } for opp in scoring.opportunities])

    fig = px.scatter(
        df,
        x="competition_density",
        y="complaint_rate",
        size="volume",
        color="is_underserved",
        hover_name="label",
        hover_data={"final_score": ":.3f", "rank": True, "volume": True},
        color_discrete_map={"🎯 Gap": "#ff4b4b", "Covered": "#4a90e2"},
        size_max=60,
        title="Complaint Rate vs Competition Density — Bubble = Volume",
        labels={
            "competition_density": "Competition Density % (higher = more crowded)",
            "complaint_rate": "Complaint Rate % (higher = more pain)",
        },
    )

    # Add quadrant lines
    fig.add_hline(
        y=30, line_dash="dash", line_color="orange",
        annotation_text="Complaint threshold 30%", annotation_position="bottom right"
    )
    fig.add_vline(
        x=20, line_dash="dash", line_color="green",
        annotation_text="Coverage threshold 20%", annotation_position="top right"
    )

    # Annotate gap quadrant
    fig.add_annotation(
        x=5, y=65, text="🎯 HIGH OPPORTUNITY ZONE",
        showarrow=False, font=dict(size=12, color="red"),
        bgcolor="rgba(255,235,235,0.8)",
    )

    fig.update_layout(height=500, legend_title="Cluster Type")
    st.plotly_chart(fig, use_container_width=True)


# ─── Complaint Heatmap ───────────────────────────────────────────────────────

def render_complaint_heatmap(scoring: ScoringOutput):
    st.subheader("🔥 Complaint Heatmap by Cluster")

    # Build bar chart of complaint rates
    df = pd.DataFrame([{
        "Cluster": f"#{opp.rank} {opp.label[:25]}",
        "Complaint Rate": opp.complaint_rate * 100,
        "Is Gap": opp.is_underserved,
    } for opp in scoring.opportunities]).sort_values("Complaint Rate", ascending=True)

    fig = px.bar(
        df,
        x="Complaint Rate",
        y="Cluster",
        orientation="h",
        color="Is Gap",
        color_discrete_map={True: "#ff4b4b", False: "#4a90e2"},
        title="Complaint Rate per Cluster (red = underserved gap)",
        labels={"Complaint Rate": "Complaint Rate (%)"},
    )
    fig.add_vline(x=30, line_dash="dash", line_color="orange",
                  annotation_text="30% threshold")
    fig.update_layout(height=max(300, len(df) * 35), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ─── Scoring Breakdown ───────────────────────────────────────────────────────

def render_scoring_breakdown(scoring: ScoringOutput):
    st.subheader("🔬 Score Component Breakdown")

    df = pd.DataFrame([{
        "Cluster": f"#{opp.rank} {opp.label[:20]}",
        "α·Demand": opp.components.get("alpha * demand", 0),
        "-β·Competition": -opp.components.get("beta * competition", 0),
        "γ·Sentiment": opp.components.get("gamma * neg_sentiment", 0),
        "Final Score": opp.final_score,
    } for opp in scoring.opportunities[:15]])

    fig = go.Figure()
    components = ["α·Demand", "-β·Competition", "γ·Sentiment"]
    colors = ["#00cc88", "#ff4b4b", "#ffa500"]

    for comp, color in zip(components, colors):
        fig.add_trace(go.Bar(
            name=comp, x=df["Cluster"], y=df[comp],
            marker_color=color, opacity=0.8,
        ))

    fig.add_trace(go.Scatter(
        name="Final Score", x=df["Cluster"], y=df["Final Score"],
        mode="markers+lines",
        marker=dict(size=8, color="#764ba2"),
        line=dict(color="#764ba2", width=2),
    ))

    fig.update_layout(
        barmode="relative",
        title="Score Decomposition (top 15 clusters)",
        xaxis_tickangle=-45,
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# ─── Feature Gap Explorer ─────────────────────────────────────────────────────

def render_gap_explorer(scoring: ScoringOutput):
    st.subheader("🔎 Feature Gap Explorer")

    underserved = [o for o in scoring.opportunities if o.is_underserved]
    all_opps = scoring.opportunities

    selected_label = st.selectbox(
        "Select a cluster to explore:",
        options=[f"#{o.rank} {o.label}" for o in all_opps],
    )

    selected_rank = int(selected_label.split(" ")[0].replace("#", ""))
    opp = next((o for o in all_opps if o.rank == selected_rank), None)

    if not opp:
        st.warning("Cluster not found.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Score", f"{opp.final_score:.3f}")
    col2.metric("Complaint Rate", f"{opp.complaint_rate:.0%}")
    col3.metric("Competitor Coverage", f"{opp.competition_raw:.0%}")
    col4.metric("Volume", opp.volume)

    if opp.is_underserved:
        st.error(f"🎯 **MARKET GAP DETECTED** — {opp.recommended_action}")
    else:
        st.info(f"ℹ️ {opp.recommended_action}")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("**🏷️ Top Keywords**")
        kw_cols = st.columns(min(5, len(opp.top_keywords)))
        for i, kw in enumerate(opp.top_keywords[:5]):
            kw_cols[i].markdown(f"`{kw}`")

        st.markdown("**📊 Score Components**")
        comp_df = pd.DataFrame([
            {"Component": k, "Value": v}
            for k, v in opp.components.items()
        ])
        fig = px.bar(comp_df, x="Component", y="Value",
                     color="Value", color_continuous_scale="RdYlGn")
        fig.update_layout(height=220, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("**🗣️ Representative Reviews**")
        for i, review in enumerate(opp.representative_reviews[:3], 1):
            sentiment = "🔴" if opp.complaint_rate > 0.3 else "🟡"
            st.markdown(f"{sentiment} *\"{review[:200]}\"*")
            st.divider()

    # Competitor coverage breakdown
    if opp.competitor_coverages:
        st.markdown("**🏢 Competitor Coverage for this Cluster**")
        cov_df = pd.DataFrame([{
            "Competitor": c.competitor_name,
            "Covers?": "✅" if c.covers else "❌",
            "Review Share": f"{c.raw_share:.1%}",
            "Evidence Count": c.evidence_count,
        } for c in opp.competitor_coverages])
        st.dataframe(cov_df, use_container_width=True, hide_index=True)


# ─── Output Example Card ──────────────────────────────────────────────────────

def render_top_opportunity_card(scoring: ScoringOutput):
    top = scoring.top_opportunity
    if not top:
        return

    st.subheader("🏆 Top Opportunity")
    with st.container():
        col1, col2 = st.columns([2, 1])
        with col1:
            badge = "🎯 **MARKET GAP**" if top.is_underserved else "📊 **OPPORTUNITY**"
            st.markdown(f"### {badge}: {top.label}")
            st.markdown(f"**Keywords:** `{'` · `'.join(top.top_keywords[:5])}`")
            st.markdown(f"**Action:** {top.recommended_action}")
        with col2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=top.final_score * 100,
                title={"text": "Opportunity Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#764ba2"},
                    "steps": [
                        {"range": [0, 40], "color": "#e8f5e9"},
                        {"range": [40, 70], "color": "#fff9c4"},
                        {"range": [70, 100], "color": "#ffebee"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 70,
                    },
                },
            ))
            fig.update_layout(height=200)
            st.plotly_chart(fig, use_container_width=True)


# ─── Main App ─────────────────────────────────────────────────────────────────

def main():
    st.title("🔍 Competitor Intelligence & Market Gap Finder")
    st.caption("AI-powered multi-agent system for competitive analysis and product opportunity scoring")

    config = render_sidebar()
    state = get_state()

    if config["run"]:
        with st.spinner("🤖 Running multi-agent pipeline..."):
            try:
                import json
                targets_list = [
                    {
                        "url": row["url"],
                        "source_type": row["source"],
                        "product_name": row["product"],
                        "competitor_name": row["competitor"],
                        "max_reviews": config["max_reviews"],
                    }
                    for _, row in config["targets"].iterrows()
                    if row.get("competitor") and row.get("url")
                ]
                if not targets_list:
                    st.error("No valid targets configured.")
                    return

                scoring = run_pipeline_cached(
                    targets_json=json.dumps(targets_list),
                    alpha=config["alpha"],
                    beta=config["beta"],
                    gamma=config["gamma"],
                    cr_threshold=config["cr_threshold"],
                    cd_threshold=config["cd_threshold"],
                )
                state.scoring_output = scoring
                st.success(f"✅ Analysis complete! {scoring.total_reviews:,} reviews analyzed across {len(scoring.opportunities)} clusters.")
            except Exception as e:
                st.error(f"❌ Pipeline failed: {e}")
                return

    scoring: Optional[ScoringOutput] = state.scoring_output

    if scoring is None:
        st.info(
            "👈 **Configure your competitors in the sidebar and click 'Run Analysis'** "
            "to start the pipeline.\n\n"
            "The system will scrape reviews, extract feature clusters, detect market gaps, "
            "and score opportunities."
        )

        # Show example output
        st.subheader("📋 Example Output")
        example_data = {
            "Feature": ["Mobile Checkout Simplification", "API Speed & Reliability", "Onboarding Flow", "Reporting Dashboard"],
            "Complaint Rate": ["42%", "35%", "33%", "25%"],
            "Competition Coverage": ["18%", "40%", "22%", "55%"],
            "Opportunity Score": [0.78, 0.43, 0.61, 0.29],
            "Status": ["🎯 GAP", "—", "🎯 GAP", "—"],
        }
        st.dataframe(pd.DataFrame(example_data), hide_index=True, use_container_width=True)
        return

    # ── Render dashboard sections ────────────────────────────────────────────
    render_kpis(scoring)
    st.divider()

    render_top_opportunity_card(scoring)
    st.divider()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Opportunity Ranking",
        "🗺️ Cluster Map",
        "🔥 Complaint Heatmap",
        "🔬 Score Breakdown",
        "🔎 Gap Explorer",
    ])

    with tab1:
        render_opportunity_table(scoring)
    with tab2:
        render_cluster_map(scoring)
    with tab3:
        render_complaint_heatmap(scoring)
    with tab4:
        render_scoring_breakdown(scoring)
    with tab5:
        render_gap_explorer(scoring)


if __name__ == "__main__":
    main()
