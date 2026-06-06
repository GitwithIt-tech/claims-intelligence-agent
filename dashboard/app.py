"""
dashboard/app.py - Premium Claims Intelligence Agent UI
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text
from config.setting import db_settings
import time

API_URL = "http://localhost:8000"
engine  = create_engine(db_settings.url)

st.set_page_config(
    page_title="ARIA — Claims Intelligence",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@400;500;600&display=swap" rel="stylesheet">

<style>
:root {
    --bg-void:     #080c14;
    --bg-deep:     #0d1220;
    --bg-surface:  #111827;
    --bg-elevated: #1a2235;
    --bg-hover:    #1f2d42;
    --accent-cyan:  #00d4ff;
    --accent-green: #00ff9d;
    --accent-amber: #ffb020;
    --accent-red:   #ff4560;
    --accent-blue:  #4d8af0;
    --text-primary:   #f0f4ff;
    --text-secondary: #8896b3;
    --text-muted:     #4a5568;
    --border-dim:   rgba(255,255,255,0.06);
    --border-glow:  rgba(0,212,255,0.3);
    --font-display: 'Syne', sans-serif;
    --font-mono:    'DM Mono', monospace;
    --font-body:    'Instrument Sans', sans-serif;
}

/* ── Reset ── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg-void) !important;
    color: var(--text-primary) !important;
}

.stApp { background: var(--bg-void) !important; }
.main .block-container { padding: 1.5rem 2rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-deep) !important;
    border-right: 1px solid var(--border-dim) !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

/* ── Hide defaults ── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background: var(--bg-elevated) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s ease !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 3px rgba(0,212,255,0.12) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue)) !important;
    color: var(--bg-void) !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-display) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,212,255,0.35) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 14px !important;
    padding: 1.2rem 1.4rem !important;
}
[data-testid="stMetricLabel"] {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-secondary) !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}

/* ── Selectbox / Radio ── */
[data-testid="stRadio"] label {
    color: var(--text-secondary) !important;
    font-size: 0.9rem !important;
}

/* ── Plotly charts ── */
.js-plotly-plot { border-radius: 14px !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent-cyan) !important; }

/* ── Divider ── */
hr { border-color: var(--border-dim) !important; }

/* ── Alert boxes ── */
.stSuccess { background: rgba(0,255,157,0.08) !important; border: 1px solid rgba(0,255,157,0.2) !important; border-radius: 10px !important; }
.stError   { background: rgba(255,69,96,0.08) !important; border: 1px solid rgba(255,69,96,0.2) !important; border-radius: 10px !important; }
.stWarning { background: rgba(255,176,32,0.08) !important; border: 1px solid rgba(255,176,32,0.2) !important; border-radius: 10px !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-dim) !important;
    border-radius: 12px !important;
}

/* ── Custom components ── */
.aria-logo {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 1.8rem 1.5rem 1rem;
}
.aria-logo-hex {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));
    clip-path: polygon(50% 0%, 100% 25%, 100% 75%, 50% 100%, 0% 75%, 0% 25%);
    display: flex; align-items: center; justify-content: center;
    animation: hexPulse 3s ease-in-out infinite;
}
@keyframes hexPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,212,255,0); }
    50% { box-shadow: 0 0 20px 4px rgba(0,212,255,0.3); }
}
.aria-logo-text { font-family: var(--font-display); font-size: 1.3rem; font-weight: 800; letter-spacing: 0.05em; color: var(--text-primary); }
.aria-logo-sub  { font-family: var(--font-mono); font-size: 0.65rem; color: var(--text-muted); letter-spacing: 0.1em; text-transform: uppercase; margin-top: 1px; }

.nav-item {
    display: flex; align-items: center; gap: 12px;
    padding: 0.75rem 1.5rem;
    margin: 2px 0.75rem;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.15s ease;
    font-size: 0.9rem;
    color: var(--text-secondary);
    border: 1px solid transparent;
}
.nav-item:hover { background: var(--bg-hover); color: var(--text-primary); }
.nav-item.active {
    background: rgba(0,212,255,0.1);
    color: var(--accent-cyan);
    border-color: rgba(0,212,255,0.2);
}

.page-header {
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--border-dim);
}
.page-title {
    font-family: var(--font-display);
    font-size: 1.8rem;
    font-weight: 800;
    color: var(--text-primary);
    margin: 0;
    letter-spacing: -0.02em;
}
.page-subtitle {
    font-size: 0.9rem;
    color: var(--text-secondary);
    margin-top: 4px;
    font-family: var(--font-mono);
}

.stat-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-radius: 16px;
    padding: 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s ease, transform 0.2s ease;
}
.stat-card:hover { border-color: rgba(0,212,255,0.25); transform: translateY(-2px); }
.stat-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent-cyan), transparent);
}
.stat-label { font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px; }
.stat-value { font-family: var(--font-display); font-size: 2rem; font-weight: 800; color: var(--text-primary); line-height: 1; }
.stat-sub   { font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-secondary); margin-top: 6px; }

.chat-container {
    display: flex; flex-direction: column; gap: 16px;
    max-height: 520px; overflow-y: auto;
    padding: 1rem;
    background: var(--bg-deep);
    border: 1px solid var(--border-dim);
    border-radius: 16px;
    margin-bottom: 1rem;
}
.msg-user {
    display: flex; justify-content: flex-end;
    animation: slideInRight 0.3s ease;
}
.msg-agent {
    display: flex; justify-content: flex-start;
    animation: slideInLeft 0.3s ease;
}
@keyframes slideInRight { from { opacity: 0; transform: translateX(20px); } to { opacity: 1; transform: translateX(0); } }
@keyframes slideInLeft  { from { opacity: 0; transform: translateX(-20px); } to { opacity: 1; transform: translateX(0); } }

.bubble-user {
    background: linear-gradient(135deg, var(--accent-blue), #2563eb);
    color: white;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 70%;
    font-size: 0.9rem;
    line-height: 1.5;
}
.bubble-agent {
    background: var(--bg-elevated);
    border: 1px solid var(--border-dim);
    color: var(--text-primary);
    padding: 14px 18px;
    border-radius: 4px 18px 18px 18px;
    max-width: 80%;
    font-size: 0.9rem;
    line-height: 1.6;
}
.bubble-agent-header {
    display: flex; align-items: center; gap: 8px;
    margin-bottom: 8px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-dim);
}
.agent-tag {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    padding: 2px 8px;
    border-radius: 20px;
    font-weight: 500;
}
.tag-sql  { background: rgba(77,138,240,0.15); color: var(--accent-blue); border: 1px solid rgba(77,138,240,0.3); }
.tag-rag  { background: rgba(0,255,157,0.1);  color: var(--accent-green); border: 1px solid rgba(0,255,157,0.25); }
.tag-ml   { background: rgba(0,212,255,0.1);  color: var(--accent-cyan);  border: 1px solid rgba(0,212,255,0.25); }
.tag-both { background: rgba(255,176,32,0.1); color: var(--accent-amber); border: 1px solid rgba(255,176,32,0.25); }

.claim-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-radius: 20px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
}
.claim-card::after {
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(0,212,255,0.08), transparent 70%);
}

.risk-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 14px;
    border-radius: 20px;
    font-family: var(--font-mono);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.risk-high   { background: rgba(255,69,96,0.12);  color: var(--accent-red);   border: 1px solid rgba(255,69,96,0.3); }
.risk-medium { background: rgba(255,176,32,0.12); color: var(--accent-amber); border: 1px solid rgba(255,176,32,0.3); }
.risk-low    { background: rgba(0,255,157,0.1);   color: var(--accent-green); border: 1px solid rgba(0,255,157,0.25); }

.pulse-dot {
    width: 8px; height: 8px; border-radius: 50%;
    animation: pulseDot 1.5s ease-in-out infinite;
    display: inline-block;
}
.pulse-green { background: var(--accent-green); }
.pulse-red   { background: var(--accent-red); }
@keyframes pulseDot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.7); }
}

.chart-card {
    background: var(--bg-surface);
    border: 1px solid var(--border-dim);
    border-radius: 16px;
    padding: 1.2rem;
}
.chart-title {
    font-family: var(--font-display);
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    margin-bottom: 1rem;
    letter-spacing: -0.01em;
}

.thinking-dots span {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent-cyan);
    margin: 0 2px;
    animation: thinking 1.2s ease-in-out infinite;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes thinking {
    0%, 80%, 100% { transform: scale(0.6); opacity: 0.4; }
    40% { transform: scale(1); opacity: 1; }
}

.sidebar-section-label {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    padding: 0.5rem 1.5rem 0.25rem;
    margin-top: 0.5rem;
}

.live-indicator {
    display: inline-flex; align-items: center; gap: 6px;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--accent-green);
    letter-spacing: 0.06em;
}

.score-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 10px 0;
    border-bottom: 1px solid var(--border-dim);
}
.score-row:last-child { border-bottom: none; }
.score-label { font-size: 0.85rem; color: var(--text-secondary); }
.score-bar-wrap { flex: 1; margin: 0 16px; height: 4px; background: var(--bg-hover); border-radius: 4px; }
.score-bar { height: 4px; border-radius: 4px; transition: width 0.8s ease; }
.score-val { font-family: var(--font-mono); font-size: 0.85rem; font-weight: 500; min-width: 45px; text-align: right; }

.example-chip {
    display: inline-block;
    background: var(--bg-elevated);
    border: 1px solid var(--border-dim);
    border-radius: 20px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: var(--text-secondary);
    cursor: pointer;
    transition: all 0.15s ease;
    margin: 4px;
    font-family: var(--font-body);
}
.example-chip:hover { border-color: var(--accent-cyan); color: var(--accent-cyan); background: rgba(0,212,255,0.05); }
</style>
""", unsafe_allow_html=True)

# ── Data helpers ──────────────────────────────────────────────────────────────
PLOT_THEME = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#8896b3", size=11),
    margin=dict(t=20, b=20, l=10, r=10),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.06)"),
)

@st.cache_data(ttl=60)
def get_stats():
    try:
        return requests.get(f"{API_URL}/claims/stats", timeout=10).json()
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_claims_data():
    with engine.connect() as conn:
        return pd.read_sql(text("SELECT * FROM claims"), conn)

def ask_question(q):
    try:
        r = requests.post(f"{API_URL}/query", json={"question": q}, timeout=120)
        return (r.json(), None) if r.status_code == 200 else (None, r.json().get("detail"))
    except Exception as e:
        return None, str(e)

def get_claim(cid):
    try:
        r = requests.get(f"{API_URL}/claim/{cid}", timeout=60)
        return (r.json(), None) if r.status_code == 200 else (None, r.json().get("detail"))
    except Exception as e:
        return None, str(e)

def tag_html(route):
    classes = {"sql":"tag-sql","rag":"tag-rag","ml":"tag-ml","both_sql_rag":"tag-both"}
    labels  = {"sql":"SQL Agent","rag":"RAG Agent","ml":"ML Agent","both_sql_rag":"SQL + RAG"}
    cls = classes.get(route, "tag-sql")
    lbl = labels.get(route, route.upper())
    return f'<span class="agent-tag {cls}">{lbl}</span>'

# ── Session state ─────────────────────────────────────────────────────────────
if "messages"      not in st.session_state: st.session_state.messages = []
if "current_page"  not in st.session_state: st.session_state.current_page = "dashboard"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="aria-logo">
        <div class="aria-logo-hex"></div>
        <div>
            <div class="aria-logo-text">ARIA</div>
            <div class="aria-logo-sub">Claims Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">Navigation</div>', unsafe_allow_html=True)

    pages = {
        "dashboard": ("⬡", "Dashboard"),
        "chat":      ("◈", "Ask ARIA"),
        "claim":     ("◎", "Claim Analyser"),
        "analytics": ("◻", "Analytics"),
    }

    page = st.radio(
        "nav",
        list(pages.keys()),
        format_func=lambda k: f"{pages[k][0]}  {pages[k][1]}",
        label_visibility="collapsed",
        key="nav_radio",
    )

    st.markdown('<div class="sidebar-section-label" style="margin-top:1.5rem;">System</div>', unsafe_allow_html=True)

    try:
        h = requests.get(f"{API_URL}/health", timeout=3).json()
        db_ok = h.get("db") == "connected"
        st.markdown(f"""
        <div style="padding:0.6rem 1.5rem;">
            <div class="live-indicator">
                <span class="pulse-dot pulse-green"></span> API ONLINE
            </div>
            <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);margin-top:4px;">
                DB: {"connected" if db_ok else "disconnected"}
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception:
        st.markdown("""
        <div style="padding:0.6rem 1.5rem;">
            <div style="font-family:var(--font-mono);font-size:0.7rem;color:#ff4560;">
                <span class="pulse-dot pulse-red"></span> API OFFLINE
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:absolute;bottom:1.5rem;left:0;right:0;padding:0 1.5rem;">
        <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);line-height:1.8;border-top:1px solid rgba(255,255,255,0.06);padding-top:1rem;">
            LangGraph · Groq LLaMA 3.3<br>
            ChromaDB · XGBoost · FastAPI<br>
            PostgreSQL · Streamlit
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "dashboard":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Operations Dashboard</div>
        <div class="page-subtitle">Real-time claims intelligence overview</div>
    </div>
    """, unsafe_allow_html=True)

    stats = get_stats()
    df    = get_claims_data()

    if stats:
        c1,c2,c3,c4,c5 = st.columns(5)
        cards = [
            (c1, "TOTAL CLAIMS",     f"{stats['total_claims']:,}",       "All time"),
            (c2, "OPEN CLAIMS",      f"{stats['open_claims']:,}",         "Active"),
            (c3, "FRAUD FLAGGED",    f"{stats['fraud_flagged']:,}",       f"{stats['fraud_flagged']/stats['total_claims']*100:.1f}% rate"),
            (c4, "LITIGATION RISK",  f"{stats['litigation_flagged']:,}",  f"{stats['litigation_flagged']/stats['total_claims']*100:.1f}% rate"),
            (c5, "AVG CLAIM",        f"£{stats['avg_claim_amount']:,.0f}", f"~{stats['avg_resolution_days']:.0f} days avg"),
        ]
        for col, label, value, sub in cards:
            col.markdown(f"""
            <div class="stat-card">
                <div class="stat-label">{label}</div>
                <div class="stat-value">{value}</div>
                <div class="stat-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-card"><div class="chart-title">Claims by Type</div>', unsafe_allow_html=True)
        tc = df["claim_type"].value_counts().reset_index()
        tc.columns = ["type","count"]
        fig = px.bar(tc, x="type", y="count",
                     color="count", color_continuous_scale=["#1a2235","#00d4ff"])
        fig.update_layout(**PLOT_THEME, coloraxis_showscale=False, showlegend=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card"><div class="chart-title">Fraud Score by Region</div>', unsafe_allow_html=True)
        rf = df.groupby("region")["fraud_score"].mean().sort_values(ascending=True).reset_index()
        fig = px.bar(rf, x="fraud_score", y="region", orientation="h",
                     color="fraud_score", color_continuous_scale=["#1a2235","#ff4560"])
        fig.update_layout(**PLOT_THEME, coloraxis_showscale=False, showlegend=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="chart-card"><div class="chart-title">Status Distribution</div>', unsafe_allow_html=True)
        sc = df["status"].value_counts().reset_index()
        sc.columns = ["status","count"]
        colors = {"open":"#00d4ff","closed":"#00ff9d","in_review":"#ffb020","litigated":"#ff4560"}
        fig = px.pie(sc, values="count", names="status",
                     color="status", color_discrete_map=colors,
                     hole=0.55)
        fig.update_layout(**PLOT_THEME, showlegend=True,
                          legend=dict(font=dict(color="#8896b3",size=11)))
        fig.update_traces(textposition="outside", textinfo="percent+label",
                          marker=dict(line=dict(color="#080c14",width=2)))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="chart-card"><div class="chart-title">Litigation Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="litigation_score", nbins=40,
                           color_discrete_sequence=["#4d8af0"])
        fig.update_layout(**PLOT_THEME, showlegend=False)
        fig.update_traces(marker_line_width=0, opacity=0.85)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHAT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "chat":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Ask ARIA</div>
        <div class="page-subtitle">Natural language queries across all data sources</div>
    </div>
    """, unsafe_allow_html=True)

    # Example chips
    examples = [
        "Which regions have the highest fraud scores?",
        "What does AXA policy say about claim time limits?",
        "Average resolution time for litigated claims?",
        "Score claim CLM0000100",
        "Top adjusters by high-risk claims",
        "What are vehicle claim procedures?",
    ]

    st.markdown("**Quick queries:**")
    chip_cols = st.columns(3)
    for i, ex in enumerate(examples):
        if chip_cols[i % 3].button(ex, key=f"chip_{i}",
                                    use_container_width=True):
            st.session_state.messages.append({"role":"user","content":ex})
            with st.spinner(""):
                result, err = ask_question(ex)
            if result:
                st.session_state.messages.append({
                    "role":    "agent",
                    "content": result["answer"],
                    "route":   result["route"],
                    "sources": result.get("sources",[]),
                })
            st.rerun()

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    # Chat history
    if st.session_state.messages:
        chat_html = '<div class="chat-container">'
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                chat_html += f'<div class="msg-user"><div class="bubble-user">{msg["content"]}</div></div>'
            else:
                route  = msg.get("route","sql")
                tag    = tag_html(route)
                answer = msg["content"].replace("\n","<br>")
                chat_html += f'''
                <div class="msg-agent">
                    <div class="bubble-agent">
                        <div class="bubble-agent-header">
                            {tag}
                            <span style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);">ARIA</span>
                        </div>
                        {answer}
                    </div>
                </div>'''
        chat_html += '</div>'
        st.markdown(chat_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="chat-container" style="align-items:center;justify-content:center;min-height:200px;">
            <div style="text-align:center;color:var(--text-muted);">
                <div style="font-size:2rem;margin-bottom:0.5rem;">◈</div>
                <div style="font-family:var(--font-mono);font-size:0.8rem;">Ask ARIA anything about your claims</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Input
    with st.form("chat_form", clear_on_submit=True):
        c1, c2 = st.columns([6,1])
        question = c1.text_input("q", placeholder="Ask anything...", label_visibility="collapsed")
        submitted = c2.form_submit_button("Send", use_container_width=True)

    if submitted and question.strip():
        st.session_state.messages.append({"role":"user","content":question})
        with st.spinner("ARIA is thinking..."):
            result, err = ask_question(question)
        if result:
            st.session_state.messages.append({
                "role":    "agent",
                "content": result["answer"],
                "route":   result["route"],
                "sources": result.get("sources",[]),
            })
        elif err:
            st.session_state.messages.append({
                "role":"agent","content":f"Error: {err}","route":"sql","sources":[]
            })
        st.rerun()

    if st.session_state.messages:
        if st.button("Clear conversation", use_container_width=False):
            st.session_state.messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# CLAIM ANALYSER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "claim":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Claim Analyser</div>
        <div class="page-subtitle">AI-powered risk scoring and policy context</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([4,1])
    claim_id = c1.text_input("claim_input", placeholder="Enter Claim ID — e.g. CLM0000042",
                              label_visibility="collapsed")
    analyse  = c2.button("Analyse →", type="primary", use_container_width=True)

    st.markdown("""
    <div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted);margin-top:6px;">
        Try: CLM0000001 · CLM0000042 · CLM0000100 · CLM0001234
    </div>
    """, unsafe_allow_html=True)

    if analyse and claim_id:
        with st.spinner(f"Running full AI analysis on {claim_id.upper()}..."):
            result, error = get_claim(claim_id)

        if error:
            st.error(f"**Error:** {error}")
        elif result:
            tier = result["risk_tier"].lower()
            tier_class = f"risk-{tier}"
            tier_dot   = {"high":"🔴","medium":"🟡","low":"🟢"}.get(tier,"⚪")

            st.markdown(f"""
            <div class="claim-card">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:1.5rem;">
                    <div>
                        <div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted);margin-bottom:6px;">CLAIM ID</div>
                        <div style="font-family:var(--font-display);font-size:1.6rem;font-weight:800;color:var(--text-primary);">{result['claim_id']}</div>
                    </div>
                    <div class="risk-badge {tier_class}">{tier_dot} {tier.upper()} RISK</div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:1rem;">
                    <div>
                        <div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);margin-bottom:4px;">TYPE</div>
                        <div style="font-weight:600;color:var(--text-primary);text-transform:capitalize;">{result['claim_type']}</div>
                    </div>
                    <div>
                        <div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);margin-bottom:4px;">AMOUNT</div>
                        <div style="font-weight:600;color:var(--text-primary);">£{result['claim_amount']:,.2f}</div>
                    </div>
                    <div>
                        <div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);margin-bottom:4px;">STATUS</div>
                        <div style="font-weight:600;color:var(--text-primary);text-transform:capitalize;">{result['status']}</div>
                    </div>
                    <div>
                        <div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);margin-bottom:4px;">REGION</div>
                        <div style="font-weight:600;color:var(--text-primary);">{result['region']}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="chart-card">', unsafe_allow_html=True)
                st.markdown('<div class="chart-title">Risk Scores</div>', unsafe_allow_html=True)

                scores = [
                    ("Fraud Score",      result["fraud_score"],      "#ff4560" if result["fraud_flag"] else "#00ff9d"),
                    ("Litigation Score", result["litigation_score"],  "#ff4560" if result["litigation_flag"] else "#00ff9d"),
                ]
                for label, score, color in scores:
                    pct = score * 100
                    flag = "⚠ FLAGGED" if score > 0.5 else "✓ OK"
                    flag_color = "#ff4560" if score > 0.5 else "#00ff9d"
                    st.markdown(f"""
                    <div class="score-row">
                        <div class="score-label">{label}</div>
                        <div class="score-bar-wrap">
                            <div class="score-bar" style="width:{pct}%;background:{color};"></div>
                        </div>
                        <div class="score-val" style="color:{flag_color};">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown(f"""
                <div style="margin-top:1.2rem;padding-top:1rem;border-top:1px solid var(--border-dim);">
                    <div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);margin-bottom:6px;">RESOLUTION FORECAST</div>
                    <div style="font-family:var(--font-display);font-size:2rem;font-weight:800;color:var(--accent-cyan);">
                        {result['resolution_days_forecast']} <span style="font-size:1rem;color:var(--text-secondary);">days</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=[result["fraud_score"]*100,
                       result["litigation_score"]*100,
                       min(result["resolution_days_forecast"]/365*100, 100),
                       result["claim_amount"]/100000*100 if result["claim_amount"] < 100000 else 100,
                       50],
                    theta=["Fraud", "Litigation", "Resolution\nComplexity", "Claim\nSeverity", "Overall"],
                    fill="toself",
                    fillcolor="rgba(0,212,255,0.12)",
                    line=dict(color="#00d4ff", width=2),
                    name="Risk Profile",
                ))
                fig.update_layout(
                    **PLOT_THEME,
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0,100],
                                        gridcolor="rgba(255,255,255,0.06)",
                                        linecolor="rgba(255,255,255,0.06)",
                                        tickfont=dict(color="#4a5568",size=9)),
                        angularaxis=dict(gridcolor="rgba(255,255,255,0.06)",
                                         linecolor="rgba(255,255,255,0.06)",
                                         tickfont=dict(color="#8896b3",size=10)),
                    ),
                    showlegend=False,
                    height=280,
                )
                st.markdown('<div class="chart-card"><div class="chart-title">Risk Radar</div>', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
                st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("📄 Relevant Policy Context", expanded=False):
                st.markdown(f"""
                <div style="font-size:0.9rem;line-height:1.7;color:var(--text-secondary);">
                    {result['policy_context']}
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "analytics":
    st.markdown("""
    <div class="page-header">
        <div class="page-title">Analytics</div>
        <div class="page-subtitle">Deep dive into claims patterns and trends</div>
    </div>
    """, unsafe_allow_html=True)

    df = get_claims_data()
    df["claim_date"] = pd.to_datetime(df["claim_date"])

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="chart-card"><div class="chart-title">Avg Fraud Score by Claim Type</div>', unsafe_allow_html=True)
        ft = df.groupby("claim_type")["fraud_score"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(ft, x="claim_type", y="fraud_score",
                     color="fraud_score", color_continuous_scale=["#1a2235","#ff4560"])
        fig.update_layout(**PLOT_THEME, coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card"><div class="chart-title">Resolution Days by Status</div>', unsafe_allow_html=True)
        colors_map = {"open":"#00d4ff","closed":"#00ff9d","in_review":"#ffb020","litigated":"#ff4560"}
        fig = px.box(df, x="status", y="resolution_days",
                     color="status", color_discrete_map=colors_map)
        fig.update_layout(**PLOT_THEME, showlegend=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="chart-card" style="margin:1rem 0;"><div class="chart-title">Claims Volume Over Time</div>', unsafe_allow_html=True)
    monthly = df.groupby(df["claim_date"].dt.to_period("M")).size().reset_index()
    monthly.columns = ["Month","Claims"]
    monthly["Month"] = monthly["Month"].astype(str)
    fig = px.area(monthly, x="Month", y="Claims",
                  color_discrete_sequence=["#00d4ff"])
    fig.update_traces(fill="tozeroy", fillcolor="rgba(0,212,255,0.08)", line_width=2)
    fig.update_layout(**PLOT_THEME, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
    st.markdown('</div>', unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="chart-card"><div class="chart-title">Fraud vs Litigation by Region</div>', unsafe_allow_html=True)
        rs = df.groupby("region").agg(
            fraud_rate=("fraud_flag","mean"),
            litigation_rate=("litigation_flag","mean"),
            count=("claim_id","count")
        ).reset_index()
        fig = px.scatter(rs, x="fraud_rate", y="litigation_rate",
                         text="region", size="count", size_max=30,
                         color="fraud_rate",
                         color_continuous_scale=["#00ff9d","#ff4560"])
        fig.update_traces(textposition="top center",
                          textfont=dict(color="#8896b3",size=10),
                          marker_line_width=0)
        fig.update_layout(**PLOT_THEME, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="chart-card"><div class="chart-title">Claim Amount Distribution (95th pct)</div>', unsafe_allow_html=True)
        dff = df[df["claim_amount"] < df["claim_amount"].quantile(0.95)]
        fig = px.histogram(dff, x="claim_amount", nbins=50,
                           color_discrete_sequence=["#4d8af0"])
        fig.update_layout(**PLOT_THEME, showlegend=False)
        fig.update_traces(marker_line_width=0, opacity=0.85)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)