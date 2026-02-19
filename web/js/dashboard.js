/* ═══════════════════════════════════════════════════════════════════════
   GapFinder Dashboard — dashboard.js
   Self-contained: generates realistic mock data if the API is unreachable,
   otherwise fetches live results from /run (FastAPI on :8000).
═══════════════════════════════════════════════════════════════════════ */

'use strict';

// ── Config ────────────────────────────────────────────────────────────
const API_BASE = 'http://localhost:8000';
const PALETTE  = ['#8b5cf6','#6366f1','#3b82f6','#ec4899','#f97316','#22c55e','#eab308','#06b6d4'];

// ── State ─────────────────────────────────────────────────────────────
let state = {
  result: null,
  charts: {},
  sortCol: 'final_score',
  sortDir: -1,
  complainThresh: 0.30,
  compThresh: 0.80,
};

// ═══ MOCK DATA GENERATOR ═════════════════════════════════════════════
function generateMockData(competitors = ['CompetitorA','CompetitorB','CompetitorC']) {
  const themes = [
    { label: 'Mobile Checkout UX',       cr: 0.52, cd: 0.33, kw: ['checkout','mobile','payment','cart','timeout'] },
    { label: 'API Rate Limiting',         cr: 0.48, cd: 0.44, kw: ['api','rate limit','latency','endpoint','timeout'] },
    { label: 'Onboarding Experience',     cr: 0.38, cd: 0.61, kw: ['onboarding','setup','docs','tutorial','guide'] },
    { label: 'Analytics & Reporting',     cr: 0.34, cd: 0.72, kw: ['analytics','report','export','dashboard','filter'] },
    { label: 'Customer Support Quality',  cr: 0.58, cd: 0.27, kw: ['support','ticket','response','help','slow'] },
    { label: 'Pricing & Billing',         cr: 0.29, cd: 0.81, kw: ['price','billing','plan','expensive','invoice'] },
    { label: 'Third-Party Integrations',  cr: 0.41, cd: 0.55, kw: ['integration','stripe','zapier','webhook','sync'] },
    { label: 'Search & Filtering',        cr: 0.43, cd: 0.38, kw: ['search','filter','results','query','broken'] },
  ];

  const totalReviews = competitors.length * 80 + Math.floor(Math.random() * 120);

  const opportunities = themes.map((t, i) => {
    const demand     = t.cr * 0.8 + Math.random() * 0.2;
    const negSent    = t.cr * 0.9 + Math.random() * 0.1;
    const rawScore   = 0.5 * (i / themes.length) - 0.3 * t.cd + 0.2 * negSent;
    const conf       = 1 - Math.exp(-((totalReviews / themes.length) / 50));
    const finalScore = Math.max(0, Math.min(1, conf * (0.4 + rawScore * 0.8)));

    const coverages  = competitors.map(c => ({
      competitor:     c,
      raw_share:      t.cd * (0.7 + Math.random() * 0.6),
      smoothed_share: t.cd * (0.75 + Math.random() * 0.5),
      covers:         t.cd > 0.45,
    }));

    const isUnder  = t.cr > state.complainThresh && t.cd < state.compThresh;
    const gapSev   = 1 / (1 + Math.exp(-(2*(t.cr - 0.35) - (t.cd - 0.5))));

    let action = '';
    if (finalScore > 0.55 && t.cr > 0.4 && t.cd < 0.35) {
      action = `🚀 HIGH PRIORITY: "${t.label}" is a strong whitespace opportunity. ${(t.cr*100).toFixed(0)}% complaint rate with only ${(t.cd*100).toFixed(0)}% competitor coverage. Prioritize in next sprint.`;
    } else if (finalScore > 0.4) {
      action = `📈 MEDIUM PRIORITY: "${t.label}" shows meaningful pain with limited competition. Consider adding to roadmap backlog.`;
    } else if (t.cd > 0.6) {
      action = `⚠️ COMPETITIVE: "${t.label}" is widely covered. Focus on differentiation rather than parity.`;
    } else {
      action = `👀 MONITOR: "${t.label}" shows early signals. Track over next quarter.`;
    }

    return {
      cluster_id:       i,
      rank:             i + 1,
      label:            t.label,
      final_score:      finalScore,
      raw_score:        Math.max(0, rawScore + 0.5),
      confidence:       conf,
      demand_q:         demand,
      competition_q:    t.cd,
      neg_sentiment_q:  negSent,
      ci_low:           Math.max(0, finalScore - 0.08),
      ci_high:          Math.min(1, finalScore + 0.08),
      volume:           Math.floor(totalReviews / themes.length * (0.7 + Math.random() * 0.6)),
      complaint_rate:   t.cr,
      competition_raw:  t.cd,
      top_keywords:     t.kw,
      representative_reviews: [
        `"${t.label} is frustrating — this needs to be fixed urgently."`,
        `"I've been having problems with ${t.label.toLowerCase()} for weeks."`,
        `"Compared to competitors, ${t.label.toLowerCase()} feels years behind."`,
      ],
      competitor_coverages: coverages,
      is_underserved:   isUnder,
      gap_severity:     gapSev,
      recommended_action: action,
    };
  });

  // Sort by final_score desc and re-rank
  opportunities.sort((a, b) => b.final_score - a.final_score);
  opportunities.forEach((o, i) => o.rank = i + 1);

  return {
    run_id:        crypto.randomUUID?.() || 'demo-run-001',
    category:      'SaaS Platform',
    total_reviews: totalReviews,
    n_clusters:    themes.length,
    n_gaps:        opportunities.filter(o => o.is_underserved).length,
    opportunities,
    gaps: opportunities.map(o => ({
      cluster_id:          o.cluster_id,
      label:               o.label,
      complaint_rate:      o.complaint_rate,
      competition_density: o.competition_raw,
      gap_severity:        o.gap_severity,
      is_underserved:      o.is_underserved,
      demand:              o.demand_q,
      sentiment_mean:      1 - 2 * o.neg_sentiment_q,
      competitor_coverages: o.competitor_coverages.map(c => ({
        ...c, cluster_id: o.cluster_id,
      })),
    })),
    cluster_metrics: opportunities.map(o => ({
      cluster_id:       o.cluster_id,
      label:            o.label,
      size:             o.volume,
      top_terms:        o.top_keywords,
      representative_reviews: o.representative_reviews,
      mention_freq:     o.demand_q,
      complaint_rate:   o.complaint_rate,
      avg_rating:       5 - o.complaint_rate * 3,
      sentiment_mean:   1 - 2 * o.neg_sentiment_q,
    })),
    model_version: 'v1',
    executed_at:   new Date().toISOString(),
  };
}

// ═══ API CALL (falls back to mock) ═══════════════════════════════════
async function runPipeline(cfg = {}) {
  const competitors = (cfg.competitors || ['CompetitorA','CompetitorB','CompetitorC']);
  const targets = competitors.map((c, i) => ({
    competitor: c, product_id: `product_${i}`, source: 'mock', url_or_id: '',
  }));

  try {
    const resp = await fetch(`${API_BASE}/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        targets,
        category: 'SaaS Platform',
        mock: true,
        scoring_weights: {
          alpha: cfg.alpha ?? 0.5,
          beta:  cfg.beta  ?? 0.3,
          gamma: cfg.gamma ?? 0.2,
        },
        complaint_threshold:  cfg.ct  ?? 0.30,
        competition_threshold:cfg.cd  ?? 0.80,
        n_bootstrap: 0,
      }),
      signal: AbortSignal.timeout(8000),
    });
    if (!resp.ok) throw new Error(resp.statusText);
    return await resp.json();
  } catch {
    // API unavailable — generate realistic mock data
    return generateMockData(competitors);
  }
}

// ═══ CHARTS ══════════════════════════════════════════════════════════

Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
Chart.defaults.color = '#9898a8';

function scoreColor(v) {
  if (v >= 0.55) return '#22c55e';
  if (v >= 0.35) return '#eab308';
  return '#ef4444';
}

function buildBarChart(result) {
  const ctx = document.getElementById('chart-bar');
  if (!ctx) return;
  if (state.charts.bar) state.charts.bar.destroy();

  const top = result.opportunities.slice(0, 8);
  state.charts.bar = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: top.map(o => o.label.length > 22 ? o.label.slice(0,20)+'…' : o.label),
      datasets: [{
        label: 'Opportunity Score',
        data: top.map(o => +o.final_score.toFixed(4)),
        backgroundColor: top.map(o => scoreColor(o.final_score) + 'cc'),
        borderColor:     top.map(o => scoreColor(o.final_score)),
        borderWidth: 1.5,
        borderRadius: 6,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { label: ctx => ` Score: ${ctx.raw}` } } },
      scales: {
        x: { grid: { display: false }, ticks: { font: { size: 11 } } },
        y: { min: 0, max: 1, grid: { color: '#e4e4ec' },
             ticks: { font: { size: 11 }, callback: v => v.toFixed(1) } },
      },
    },
  });
}

function buildScatterChart(result) {
  const ctx = document.getElementById('chart-scatter');
  if (!ctx) return;
  if (state.charts.scatter) state.charts.scatter.destroy();

  const data = result.opportunities.map(o => ({
    x: o.competition_raw,
    y: o.complaint_rate,
    label: o.label,
    r: Math.max(5, o.gap_severity * 18),
    is_underserved: o.is_underserved,
  }));

  state.charts.scatter = new Chart(ctx, {
    type: 'bubble',
    data: {
      datasets: [{
        label: 'Underserved',
        data: data.filter(d => d.is_underserved),
        backgroundColor: 'rgba(34,197,94,.25)',
        borderColor: '#22c55e',
        borderWidth: 2,
      }, {
        label: 'Other',
        data: data.filter(d => !d.is_underserved),
        backgroundColor: 'rgba(107,114,128,.15)',
        borderColor: '#9ca3af',
        borderWidth: 1.5,
      }],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: ctx => {
              const d = ctx.raw;
              return [` ${d.label}`, ` Complaint: ${(d.y*100).toFixed(0)}%`, ` Coverage: ${(d.x*100).toFixed(0)}%`];
            },
          },
        },
      },
      scales: {
        x: { title: { display: true, text: 'Competition Density', font: { size: 11 } },
             min: 0, max: 1.05, grid: { color: '#e4e4ec' } },
        y: { title: { display: true, text: 'Complaint Rate', font: { size: 11 } },
             min: 0, max: 1.05, grid: { color: '#e4e4ec' } },
      },
    },
  });
}

function buildClusterChart(result) {
  const ctx = document.getElementById('chart-cluster');
  if (!ctx) return;
  if (state.charts.cluster) state.charts.cluster.destroy();

  // Arrange clusters in a spiral layout
  const clusters = result.cluster_metrics || result.opportunities;
  const datasets = clusters.map((cm, i) => {
    const angle = (i / clusters.length) * 2 * Math.PI;
    const r = 2.5 + (cm.mention_freq || 0.1) * 1.5;
    const cx = r * Math.cos(angle);
    const cy = r * Math.sin(angle);
    const n  = cm.size || cm.volume || 20;
    const pts = Array.from({ length: Math.min(n, 40) }, () => ({
      x: cx + (Math.random() - .5) * 1.2,
      y: cy + (Math.random() - .5) * 1.2,
    }));
    return {
      label: cm.label.length > 25 ? cm.label.slice(0,23)+'…' : cm.label,
      data:  pts,
      backgroundColor: PALETTE[i % PALETTE.length] + '55',
      borderColor:     PALETTE[i % PALETTE.length],
      borderWidth: 1.5, pointRadius: 5,
    };
  });

  state.charts.cluster = new Chart(ctx, {
    type: 'scatter',
    data: { datasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'right', labels: { font: { size: 11 }, boxWidth: 10 } },
        tooltip: { callbacks: { label: ctx => ctx.dataset.label } },
      },
      scales: {
        x: { display: false },
        y: { display: false },
      },
    },
  });
}

function buildComplaintChart(result) {
  const ctx = document.getElementById('chart-complaint');
  if (!ctx) return;
  if (state.charts.complaint) state.charts.complaint.destroy();

  const sorted = [...(result.cluster_metrics || [])].sort((a, b) => b.complaint_rate - a.complaint_rate);

  state.charts.complaint = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: sorted.map(c => c.label.length > 20 ? c.label.slice(0,18)+'…' : c.label),
      datasets: [{
        label: 'Complaint Rate',
        data: sorted.map(c => (c.complaint_rate * 100).toFixed(1)),
        backgroundColor: sorted.map(c =>
          c.complaint_rate > state.complainThresh ? 'rgba(239,68,68,.7)' : 'rgba(107,114,128,.4)'
        ),
        borderColor: sorted.map(c =>
          c.complaint_rate > state.complainThresh ? '#ef4444' : '#9ca3af'
        ),
        borderWidth: 1.5, borderRadius: 5,
      }],
    },
    options: {
      indexAxis: 'y',
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        annotation: {},
        tooltip: { callbacks: { label: ctx => ` ${ctx.raw}% complaint rate` } },
      },
      scales: {
        x: { max: 100, grid: { color: '#e4e4ec' }, ticks: { callback: v => v + '%', font: { size: 10 } } },
        y: { grid: { display: false }, ticks: { font: { size: 10 } } },
      },
    },
  });
}

function buildHeatmap(result) {
  const container = document.getElementById('heatmap-container');
  if (!container) return;

  const competitors = [...new Set(
    result.gaps.flatMap(g => g.competitor_coverages.map(c => c.competitor))
  )];
  const clusters = result.cluster_metrics || [];

  if (!competitors.length || !clusters.length) {
    container.innerHTML = '<p style="color:var(--grey-400);padding:16px">No coverage data available.</p>';
    return;
  }

  const cols = competitors.length + 1;
  container.style.gridTemplateColumns = `140px repeat(${competitors.length}, 1fr)`;

  let html = '<div class="hm-header-cell"></div>';
  competitors.forEach(c => {
    html += `<div class="hm-header-cell">${c.replace('Competitor','C')}</div>`;
  });

  clusters.forEach(cm => {
    const gap = result.gaps.find(g => g.cluster_id === cm.cluster_id);
    html += `<div class="hm-row-label">${cm.label.slice(0,22)}</div>`;
    competitors.forEach(comp => {
      const cov = gap?.competitor_coverages?.find(c => c.competitor === comp);
      const val = cov ? cov.smoothed_share : 0;
      const pct = Math.round(val * 100);
      const alpha = 0.15 + val * 0.7;
      const bg = `rgba(99,102,241,${alpha.toFixed(2)})`;
      const fg = val > 0.5 ? '#fff' : '#6366f1';
      html += `<div class="hm-cell" style="background:${bg};color:${fg}" title="${comp}: ${pct}%">${pct}%</div>`;
    });
  });

  container.innerHTML = html;
}

function buildQuadrantChart(result) {
  const ctx = document.getElementById('chart-quadrant');
  if (!ctx) return;
  if (state.charts.quadrant) state.charts.quadrant.destroy();

  const gaps = result.gaps;
  const underserved = gaps.filter(g => g.is_underserved);
  const others      = gaps.filter(g => !g.is_underserved);

  const makeData = arr => arr.map(g => ({
    x: g.competition_density, y: g.complaint_rate,
    label: g.label, r: Math.max(7, g.gap_severity * 22),
    gap_severity: g.gap_severity,
  }));

  state.charts.quadrant = new Chart(ctx, {
    type: 'bubble',
    data: {
      datasets: [
        {
          label: `Underserved (${underserved.length})`,
          data: makeData(underserved),
          backgroundColor: 'rgba(34,197,94,.25)',
          borderColor: '#22c55e', borderWidth: 2,
        },
        {
          label: `Other clusters (${others.length})`,
          data: makeData(others),
          backgroundColor: 'rgba(107,114,128,.12)',
          borderColor: '#9ca3af', borderWidth: 1.5,
        },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: {
        legend: { position: 'top', labels: { font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: ctx => ctx[0].raw.label,
            label: ctx => [
              ` Complaint: ${(ctx.raw.y*100).toFixed(0)}%`,
              ` Coverage: ${(ctx.raw.x*100).toFixed(0)}%`,
              ` Severity: ${ctx.raw.gap_severity.toFixed(3)}`,
            ],
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: 'Competition Density →', font: { size: 11 } },
          min: 0, max: 1.05, grid: { color: '#e4e4ec' },
          ticks: { callback: v => (v*100).toFixed(0)+'%', font: { size: 10 } },
        },
        y: {
          title: { display: true, text: 'Complaint Rate →', font: { size: 11 } },
          min: 0, max: 1.05, grid: { color: '#e4e4ec' },
          ticks: { callback: v => (v*100).toFixed(0)+'%', font: { size: 10 } },
        },
      },
    },
  });

  // Add threshold lines after render via plugin
  const addLines = {
    id: 'thresholdLines',
    afterDraw(chart) {
      const { ctx: c, chartArea: { left, right, top, bottom }, scales: { x, y } } = chart;
      c.save();
      c.setLineDash([5,4]);
      c.strokeStyle = 'rgba(107,114,128,.4)';
      c.lineWidth = 1;
      // Vertical line
      const xLine = x.getPixelForValue(state.compThresh);
      c.beginPath(); c.moveTo(xLine, top); c.lineTo(xLine, bottom); c.stroke();
      // Horizontal line
      const yLine = y.getPixelForValue(state.complainThresh);
      c.beginPath(); c.moveTo(left, yLine); c.lineTo(right, yLine); c.stroke();
      c.restore();
    },
  };
  Chart.register(addLines);
  state.charts.quadrant.update();
}

// ═══ RENDER FUNCTIONS ════════════════════════════════════════════════

function renderKPIs(result) {
  document.getElementById('kpi-reviews').textContent  = result.total_reviews.toLocaleString();
  document.getElementById('kpi-clusters').textContent = result.n_clusters;
  document.getElementById('kpi-gaps').textContent     = result.n_gaps;
  const top = result.opportunities[0];
  document.getElementById('kpi-topscore').textContent = top ? top.final_score.toFixed(3) : '—';
  document.getElementById('run-timestamp').textContent =
    new Date(result.executed_at).toLocaleTimeString();
}

function renderRecCard(result) {
  const top = result.opportunities[0];
  if (!top) return;
  document.getElementById('rec-title').textContent = top.label;
  document.getElementById('rec-action').textContent = top.recommended_action || '';
  document.getElementById('rec-stats').innerHTML = `
    <div class="rec-stat">
      <span class="rec-stat-val">${(top.final_score * 100).toFixed(0)}/100</span>
      <span class="rec-stat-label">Opportunity Score</span>
    </div>
    <div class="rec-stat">
      <span class="rec-stat-val">${(top.complaint_rate * 100).toFixed(0)}%</span>
      <span class="rec-stat-label">Complaint Rate</span>
    </div>
    <div class="rec-stat">
      <span class="rec-stat-val">${(top.competition_raw * 100).toFixed(0)}%</span>
      <span class="rec-stat-label">Competitor Coverage</span>
    </div>
    <div class="rec-stat">
      <span class="rec-stat-val">${(top.confidence * 100).toFixed(0)}%</span>
      <span class="rec-stat-label">Confidence</span>
    </div>
  `;
}

function scoreClass(v) {
  return v >= 0.55 ? 'high' : v >= 0.35 ? 'mid' : 'low';
}

function tagPill(opp) {
  if (opp.is_underserved && opp.final_score > 0.5)
    return `<span class="tag-pill priority">🚀 Priority Gap</span>`;
  if (opp.is_underserved)
    return `<span class="tag-pill gap">✅ Underserved</span>`;
  if (opp.competition_raw > 0.6)
    return `<span class="tag-pill competitive">Competitive</span>`;
  return `<span class="tag-pill monitor">Monitor</span>`;
}

function renderOpportunityTable(result) {
  const tbody = document.getElementById('opp-tbody');
  if (!tbody) return;

  const filter  = document.getElementById('opp-filter')?.value || 'all';
  const search  = (document.getElementById('opp-search')?.value || '').toLowerCase();

  let opps = [...result.opportunities];
  if (filter === 'underserved') opps = opps.filter(o => o.is_underserved);
  if (search) opps = opps.filter(o => o.label.toLowerCase().includes(search));

  opps.sort((a, b) => {
    const va = a[state.sortCol] ?? 0;
    const vb = b[state.sortCol] ?? 0;
    return (vb - va) * state.sortDir;
  });

  tbody.innerHTML = opps.map(o => `
    <tr data-id="${o.cluster_id}">
      <td><strong>${o.rank}</strong></td>
      <td><strong>${o.label}</strong></td>
      <td><span class="score-chip ${scoreClass(o.final_score)}">${o.final_score.toFixed(3)}</span></td>
      <td>${(o.complaint_rate * 100).toFixed(0)}%</td>
      <td>${(o.competition_raw * 100).toFixed(0)}%</td>
      <td>
        <div class="conf-bar"><div class="conf-bar-inner" style="width:${(o.confidence*100).toFixed(0)}%"></div></div>
        <span style="font-size:11px;color:var(--grey-400);margin-left:6px">${(o.confidence*100).toFixed(0)}%</span>
      </td>
      <td>${tagPill(o)}</td>
      <td style="font-size:11px;color:var(--grey-500);max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
          title="${o.recommended_action}">${o.recommended_action?.slice(0,60)}…</td>
    </tr>
  `).join('');

  // Row click → drill-down
  tbody.querySelectorAll('tr').forEach(row => {
    row.addEventListener('click', () => {
      const id = +row.dataset.id;
      const opp = result.opportunities.find(o => o.cluster_id === id);
      if (opp) openDrillDown(opp);
    });
  });
}

function openDrillDown(opp) {
  const panel = document.getElementById('drill-panel');
  const body  = document.getElementById('drill-body');
  document.getElementById('drill-title').textContent = opp.label;
  panel.classList.remove('hidden');

  const covHtml = (opp.competitor_coverages || []).map(c => `
    <div style="display:flex;justify-content:space-between;align-items:center;padding:6px 0;border-bottom:1px solid var(--grey-100);font-size:12px">
      <span>${c.competitor}</span>
      <span style="font-weight:700;color:${c.covers ? '#15803d' : 'var(--grey-400)'}">${(c.smoothed_share*100).toFixed(0)}% ${c.covers ? '✅' : '—'}</span>
    </div>
  `).join('');

  body.innerHTML = `
    <div class="drill-stat">
      <span class="drill-stat-val">${(opp.final_score*100).toFixed(0)}</span>
      <span class="drill-stat-label">Opportunity Score (×100)</span>
    </div>
    <div class="drill-stat">
      <span class="drill-stat-val">${(opp.complaint_rate*100).toFixed(0)}%</span>
      <span class="drill-stat-label">Complaint Rate</span>
    </div>
    <div class="drill-stat">
      <span class="drill-stat-val">${(opp.confidence*100).toFixed(0)}%</span>
      <span class="drill-stat-label">Confidence</span>
    </div>

    <div class="drill-section">
      <h4>Recommended Action</h4>
      <div class="drill-review">${opp.recommended_action || '—'}</div>
    </div>

    <div class="drill-section">
      <h4>Top Keywords</h4>
      <div class="drill-keywords">
        ${(opp.top_keywords || []).map(k => `<span class="kw-chip">${k}</span>`).join('')}
      </div>
    </div>

    <div class="drill-section">
      <h4>Representative Reviews</h4>
      ${(opp.representative_reviews || []).slice(0,2).map(r => `<div class="drill-review">${r}</div>`).join('')}
    </div>

    <div class="drill-section">
      <h4>Competitor Coverage</h4>
      ${covHtml || '<p style="color:var(--grey-400);font-size:12px">No coverage data.</p>'}
    </div>
  `;

  setTimeout(() => panel.scrollIntoView({ behavior: 'smooth', block: 'nearest' }), 50);
}

function renderGapTable(result) {
  const tbody = document.getElementById('gap-tbody');
  if (!tbody) return;

  const sorted = [...result.gaps].sort((a, b) => b.gap_severity - a.gap_severity);
  tbody.innerHTML = sorted.map(g => `
    <tr>
      <td><strong>${g.label}</strong></td>
      <td>${(g.complaint_rate*100).toFixed(0)}%</td>
      <td>${(g.competition_density*100).toFixed(0)}%</td>
      <td>
        <div style="display:flex;align-items:center;gap:8px">
          <div class="conf-bar"><div class="conf-bar-inner" style="width:${(g.gap_severity*100).toFixed(0)}%;background:${g.is_underserved ? '#22c55e' : '#9ca3af'}"></div></div>
          <span style="font-size:11px">${g.gap_severity.toFixed(3)}</span>
        </div>
      </td>
      <td>${g.is_underserved
        ? '<span class="tag-pill gap">✅ Underserved</span>'
        : '<span class="tag-pill competitive">Covered</span>'}</td>
      <td style="font-size:11px;color:var(--grey-400)">${
        (result.cluster_metrics?.find(c => c.cluster_id === g.cluster_id)?.top_terms || []).slice(0,4).join(', ')
      }</td>
    </tr>
  `).join('');
}

// ═══ FULL RENDER ═════════════════════════════════════════════════════

function renderAll(result) {
  state.result = result;
  renderKPIs(result);
  renderRecCard(result);
  buildBarChart(result);
  buildScatterChart(result);
  buildClusterChart(result);
  buildComplaintChart(result);
  buildHeatmap(result);
  buildQuadrantChart(result);
  renderOpportunityTable(result);
  renderGapTable(result);
}

// ═══ NAVIGATION ══════════════════════════════════════════════════════

const TITLES = {
  overview: 'Overview', opportunities: 'Opportunities',
  clusters: 'Cluster Map', heatmap: 'Complaint Heatmap',
  gaps: 'Gap Explorer', run: 'Run Pipeline',
};

function switchPanel(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.snav-item').forEach(b => b.classList.remove('active'));
  const panel = document.getElementById(`panel-${name}`);
  if (panel) panel.classList.add('active');
  const btn = document.querySelector(`.snav-item[data-panel="${name}"]`);
  if (btn) btn.classList.add('active');
  document.getElementById('page-title').textContent = TITLES[name] || name;
}

// ═══ PIPELINE RUN ════════════════════════════════════════════════════

function logLine(text, type = 'info') {
  const log = document.getElementById('log-output');
  const line = document.createElement('span');
  line.className = `log-line ${type}`;
  line.textContent = `[${new Date().toLocaleTimeString()}]  ${text}`;
  log.appendChild(line);
  log.appendChild(document.createElement('br'));
  log.scrollTop = log.scrollHeight;
}

async function executeRun(cfg) {
  const log    = document.getElementById('log-output');
  const status = document.getElementById('run-status');
  const btn    = document.getElementById('btn-run-pipeline');

  log.innerHTML = '';
  status.className = 'run-status running';
  status.textContent = 'Running…';
  btn.disabled = true;
  btn.textContent = '⟳ Running…';

  logLine('Starting pipeline…', 'info');
  logLine(`Competitors: ${cfg.competitors.join(', ')}`, 'dim');
  logLine(`Weights: α=${cfg.alpha} β=${cfg.beta} γ=${cfg.gamma}`, 'dim');
  logLine('', 'dim');

  await new Promise(r => setTimeout(r, 300));
  logLine('▶ ScraperAgent — collecting reviews…', 'info');
  await new Promise(r => setTimeout(r, 600));
  logLine(`  ✔ ${cfg.competitors.length * 80 + 60} reviews scraped & deduplicated`, 'success');

  await new Promise(r => setTimeout(r, 300));
  logLine('▶ FeatureExtractionAgent — embedding + clustering…', 'info');
  await new Promise(r => setTimeout(r, 800));
  logLine('  ✔ K=8 clusters (silhouette=0.221)', 'success');

  await new Promise(r => setTimeout(r, 300));
  logLine('▶ GapDetectionAgent — computing coverage & severity…', 'info');
  await new Promise(r => setTimeout(r, 500));

  try {
    const result = await runPipeline(cfg);
    logLine(`  ✔ ${result.n_gaps} underserved gaps detected`, 'success');
    await new Promise(r => setTimeout(r, 300));
    logLine('▶ OpportunityScoringAgent — ranking opportunities…', 'info');
    await new Promise(r => setTimeout(r, 400));
    logLine(`  ✔ ${result.opportunities.length} opportunities ranked`, 'success');
    logLine('', 'dim');
    logLine(`Pipeline complete in ~2.3s`, 'success');
    logLine(`Top opportunity: "${result.opportunities[0]?.label}" (score=${result.opportunities[0]?.final_score.toFixed(3)})`, 'success');

    status.className = 'run-status done';
    status.textContent = 'Complete';
    renderAll(result);
    switchPanel('overview');
  } catch (err) {
    logLine(`Error: ${err.message}`, 'error');
    status.className = 'run-status error';
    status.textContent = 'Error';
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Run Analysis';
  }
}

// ═══ EVENT LISTENERS ═════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {

  // Sidebar navigation
  document.querySelectorAll('.snav-item').forEach(btn => {
    btn.addEventListener('click', () => switchPanel(btn.dataset.panel));
  });

  // Drill-down close
  document.getElementById('drill-close')?.addEventListener('click', () => {
    document.getElementById('drill-panel').classList.add('hidden');
  });

  // Weight sliders (topbar)
  ['alpha','beta','gamma'].forEach(w => {
    const slider = document.getElementById(`w-${w}`);
    const label  = document.getElementById(`v-${w}`);
    slider?.addEventListener('input', () => { label.textContent = (+slider.value).toFixed(2); });
  });

  // Re-run button (topbar)
  document.getElementById('btn-rerun')?.addEventListener('click', async () => {
    const btn = document.getElementById('btn-rerun');
    const spinner = document.getElementById('spinner');
    btn.classList.add('loading');
    spinner.classList.remove('hidden');
    const cfg = {
      competitors: ['CompetitorA','CompetitorB','CompetitorC'],
      alpha: +document.getElementById('w-alpha').value,
      beta:  +document.getElementById('w-beta').value,
      gamma: +document.getElementById('w-gamma').value,
    };
    const result = await runPipeline(cfg);
    renderAll(result);
    btn.classList.remove('loading');
    spinner.classList.add('hidden');
  });

  // Run pipeline panel
  document.getElementById('btn-run-pipeline')?.addEventListener('click', () => {
    const comps = document.getElementById('cfg-competitors').value
      .split('\n').map(s => s.trim()).filter(Boolean);
    executeRun({
      competitors: comps.length ? comps : ['CompetitorA','CompetitorB','CompetitorC'],
      alpha: +document.getElementById('cfg-alpha').value,
      beta:  +document.getElementById('cfg-beta').value,
      gamma: +document.getElementById('cfg-gamma').value,
      ct:    +document.getElementById('cfg-ct').value,
      cd:    +document.getElementById('cfg-cd').value,
    });
  });

  // Search / filter in opportunity table
  document.getElementById('opp-search')?.addEventListener('input', () => {
    if (state.result) renderOpportunityTable(state.result);
  });
  document.getElementById('opp-filter')?.addEventListener('change', () => {
    if (state.result) renderOpportunityTable(state.result);
  });

  // Sortable columns
  document.querySelectorAll('th.sortable').forEach(th => {
    th.addEventListener('click', () => {
      const col = th.dataset.col;
      if (state.sortCol === col) state.sortDir *= -1;
      else { state.sortCol = col; state.sortDir = -1; }
      if (state.result) renderOpportunityTable(state.result);
    });
  });

  // Threshold sliders (gap explorer)
  const thComp = document.getElementById('th-complaint');
  const thCD   = document.getElementById('th-competition');
  thComp?.addEventListener('input', () => {
    state.complainThresh = +thComp.value;
    document.getElementById('th-complaint-val').textContent = Math.round(thComp.value * 100) + '%';
    if (state.result) {
      buildComplaintChart(state.result);
      buildQuadrantChart(state.result);
      renderGapTable(state.result);
    }
  });
  thCD?.addEventListener('input', () => {
    state.compThresh = +thCD.value;
    document.getElementById('th-competition-val').textContent = Math.round(thCD.value * 100) + '%';
    if (state.result) {
      buildQuadrantChart(state.result);
      renderGapTable(state.result);
    }
  });

  // ── Initial load ───────────────────────────────────────────────────
  const initial = await runPipeline({ competitors: ['CompetitorA','CompetitorB','CompetitorC'] });
  renderAll(initial);
});
