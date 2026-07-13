# This is the main entry point of the application; it builds and runs the Streamlit web app that allows users to interact with the airline passenger forecasting model.

"""
====================================================
app.py  —  AirFlow Forecaster  (Animated Edition)
Project: Airline Passenger Forecasting
====================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ── src modules (unchanged) ───────────────────────────────────────────────────
from src.data_loader        import DataLoader
from src.preprocessing      import Preprocessor
from src.sequence_generator import SequenceGenerator
from src.train_test_split   import TimeSeriesSplit
from src.model              import ModelBuilder

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AirFlow Forecaster",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
DATA_PATH   = "data/airline_passengers.csv"
MODEL_PATH  = "models/lstm_models.keras"
SCALER_PATH = "models/scaler.pkl"

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS + ANIMATIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;900&display=swap');

/* ── Keyframes ── */
@keyframes fadeInUp   { from{opacity:0;transform:translateY(28px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeInLeft { from{opacity:0;transform:translateX(-30px)} to{opacity:1;transform:translateX(0)} }
@keyframes glowPulse  {
  0%,100%{ box-shadow:0 0 10px rgba(56,189,248,.25),0 4px 20px rgba(56,189,248,.08); }
  50%    { box-shadow:0 0 28px rgba(56,189,248,.6), 0 8px 40px rgba(56,189,248,.2);  }
}
@keyframes borderGlow {
  0%,100%{ border-color:rgba(56,189,248,.25);  }
  50%    { border-color:rgba(129,140,248,.8);   }
}
@keyframes shimmer {
  0%  { background-position:-1000px 0; }
  100%{ background-position: 1000px 0; }
}
@keyframes float {
  0%,100%{ transform:translateY(0px);  }
  50%    { transform:translateY(-7px); }
}
@keyframes spinSlow { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
@keyframes gradientShift {
  0%  { background-position:0%   50%; }
  50% { background-position:100% 50%; }
  100%{ background-position:0%   50%; }
}
@keyframes progressShimmer {
  0%  { background-position:-200px 0; }
  100%{ background-position:calc(200px + 100%) 0; }
}

/* ── Base ── */
html,body,[class*="css"]{ font-family:'Outfit',sans-serif!important; }
.stApp {
  background:radial-gradient(ellipse at 20% 20%, rgba(14,165,233,.07) 0%, transparent 50%),
             radial-gradient(ellipse at 80% 80%, rgba(232,121,249,.05) 0%, transparent 50%),
             #020817 !important;
  color:#e2e8f0!important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#080d1c 0%,#050910 100%)!important;
  border-right:1px solid rgba(56,189,248,.18)!important;
}
[data-testid="stSidebar"] *{ color:#cbd5e1!important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
  background:rgba(8,13,28,.85)!important; border-radius:14px!important;
  padding:5px!important; border:1px solid rgba(56,189,248,.18)!important;
  backdrop-filter:blur(14px)!important;
  animation:fadeInUp .6s ease forwards;
}
.stTabs [data-baseweb="tab"]{
  background:transparent!important; color:#475569!important;
  border-radius:10px!important; padding:9px 24px!important;
  font-weight:600!important; border:none!important;
  transition:all .3s cubic-bezier(.4,0,.2,1)!important; font-size:14px!important;
}
.stTabs [data-baseweb="tab"]:hover{ color:#94a3b8!important; background:rgba(56,189,248,.06)!important; }
.stTabs [aria-selected="true"]{
  background:linear-gradient(135deg,#0ea5e9,#6366f1,#e879f9)!important;
  background-size:200% 200%!important; color:#fff!important;
  animation:gradientShift 3s ease infinite!important;
  box-shadow:0 0 22px rgba(14,165,233,.5),0 4px 16px rgba(99,102,241,.3)!important;
}

/* ── Buttons ── */
.stButton>button{
  width:100%!important;
  background:linear-gradient(135deg,#0ea5e9,#6366f1,#e879f9)!important;
  background-size:200% 200%!important;
  animation:gradientShift 3s ease infinite!important;
  color:#fff!important; border:none!important; border-radius:14px!important;
  padding:13px 0!important; font-size:15px!important; font-weight:700!important;
  letter-spacing:.04em!important;
  box-shadow:0 0 25px rgba(14,165,233,.4),0 4px 16px rgba(99,102,241,.3)!important;
  transition:all .3s!important;
}
.stButton>button:hover{
  transform:translateY(-3px) scale(1.02)!important;
  box-shadow:0 0 42px rgba(14,165,233,.65),0 8px 28px rgba(99,102,241,.45)!important;
}
.stButton>button:active{ transform:scale(.96)!important; }

/* ── Progress bar ── */
.stProgress>div>div>div{
  background:linear-gradient(90deg,#0ea5e9,#6366f1,#e879f9,#0ea5e9)!important;
  background-size:200% 100%!important;
  animation:progressShimmer 1.4s linear infinite!important;
}

/* ── Section headers ── */
.section-hdr{
  font-size:18px; font-weight:700; color:#f1f5f9;
  margin:28px 0 14px; display:flex; align-items:center; gap:10px;
  animation:fadeInLeft .5s ease forwards;
}
.section-hdr::after{
  content:''; flex:1; height:1px;
  background:linear-gradient(90deg,rgba(56,189,248,.5),rgba(129,140,248,.25),transparent);
}

/* ── Dataframe ── */
[data-testid="stDataFrame"]{
  border:1px solid rgba(56,189,248,.14)!important;
  border-radius:14px!important; overflow:hidden!important;
  animation:fadeInUp .6s ease forwards;
}

/* ── Select ── */
[data-baseweb="select"]>div{
  background:rgba(10,15,30,.85)!important;
  border-color:rgba(56,189,248,.22)!important; border-radius:10px!important;
}

/* ── Download button ── */
[data-testid="stDownloadButton"]>button{
  background:linear-gradient(135deg,#0f172a,#1e293b)!important;
  border:1px solid rgba(56,189,248,.35)!important; color:#38bdf8!important;
  animation:borderGlow 2s ease-in-out infinite!important;
}

/* ── Hide chrome ── */
#MainMenu,footer,header{ visibility:hidden; }
label{ color:#94a3b8!important; font-size:13px!important; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PARTICLE CANVAS (floating stars + shooting star + connection lines)
# ═══════════════════════════════════════════════════════════════════════════════
components.html("""
<canvas id="cvs" style="position:fixed;top:0;left:0;width:100vw;height:100vh;
  pointer-events:none;z-index:0;opacity:.5;"></canvas>
<script>
(function(){
  const c=document.getElementById('cvs'),ctx=c.getContext('2d');
  let W=c.width=innerWidth, H=c.height=innerHeight;
  addEventListener('resize',()=>{W=c.width=innerWidth;H=c.height=innerHeight;});
  const PAL=['56,189,248','129,140,248','232,121,249','14,165,233','167,139,250'];
  const pts=Array.from({length:110},()=>({
    x:Math.random()*W, y:Math.random()*H,
    r:Math.random()*1.7+.3,
    vx:(Math.random()-.5)*.22, vy:(Math.random()-.5)*.22,
    rgb:PAL[0|Math.random()*PAL.length],
    ph:Math.random()*6.28
  }));
  let sh={on:false,x:0,y:0,life:0};
  setInterval(()=>{sh={on:true,x:Math.random()*W*.6,y:Math.random()*H*.35,life:0};},4200);
  function frame(){
    ctx.clearRect(0,0,W,H);
    if(sh.on){
      const t=sh.life/45,a=t<.5?t*2:(1-t)*2;
      const grd=ctx.createLinearGradient(sh.x-100*t,sh.y-40*t,sh.x+60*t,sh.y+20*t);
      grd.addColorStop(0,'rgba(255,255,255,0)');
      grd.addColorStop(1,`rgba(255,255,255,${a*.9})`);
      ctx.strokeStyle=grd; ctx.lineWidth=1.8; ctx.shadowBlur=10; ctx.shadowColor='#fff';
      ctx.beginPath(); ctx.moveTo(sh.x-100*t,sh.y-40*t); ctx.lineTo(sh.x+60*t,sh.y+20*t);
      ctx.stroke(); ctx.shadowBlur=0;
      if(++sh.life>45) sh.on=false;
    }
    pts.forEach(p=>{
      p.x+=p.vx; p.y+=p.vy; p.ph+=.022;
      if(p.x<0)p.x=W; if(p.x>W)p.x=0;
      if(p.y<0)p.y=H; if(p.y>H)p.y=0;
      const a=(.3+.4*Math.sin(p.ph));
      ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,6.28);
      ctx.fillStyle=`rgba(${p.rgb},${a})`;
      ctx.shadowColor=`rgba(${p.rgb},${a*.8})`; ctx.shadowBlur=5;
      ctx.fill(); ctx.shadowBlur=0;
    });
    for(let i=0;i<pts.length;i++) for(let j=i+1;j<pts.length;j++){
      const dx=pts[i].x-pts[j].x, dy=pts[i].y-pts[j].y, d=Math.hypot(dx,dy);
      if(d<80){
        ctx.strokeStyle=`rgba(56,189,248,${(1-d/80)*.07})`;
        ctx.lineWidth=.45; ctx.beginPath();
        ctx.moveTo(pts[i].x,pts[i].y); ctx.lineTo(pts[j].x,pts[j].y); ctx.stroke();
      }
    }
    requestAnimationFrame(frame);
  }
  frame();
})();
</script>
""", height=0)


# ═══════════════════════════════════════════════════════════════════════════════
# ANIMATED HEADER — flying plane + wave + badges
# ═══════════════════════════════════════════════════════════════════════════════
components.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@700;900&display=swap');
@keyframes planeFly{
  0%  {transform:translateX(-150px) translateY(8px)  rotate(-6deg);opacity:0}
  8%  {opacity:1}
  88% {opacity:1}
  100%{transform:translateX(110vw)  translateY(-12px) rotate(4deg);opacity:0}
}
@keyframes gradShift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes waveMove{0%{transform:translateX(0)}100%{transform:translateX(-50%)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes glowText{0%,100%{filter:drop-shadow(0 0 18px rgba(56,189,248,.55))}
  50%{filter:drop-shadow(0 0 38px rgba(232,121,249,.7))}}
.wrap{
  text-align:center;padding:30px 0 0;position:relative;overflow:hidden;
  background:linear-gradient(135deg,rgba(14,165,233,.07),rgba(129,140,248,.04),rgba(232,121,249,.06));
  border-bottom:1px solid rgba(56,189,248,.11);
}
.plane{position:absolute;top:14px;font-size:36px;
  animation:planeFly 7.5s ease-in-out infinite;
  filter:drop-shadow(0 0 14px rgba(56,189,248,.95));
}
.title{font-family:'Outfit',sans-serif;font-size:54px;font-weight:900;letter-spacing:-2px;
  background:linear-gradient(135deg,#38bdf8 0%,#818cf8 45%,#e879f9 100%);
  background-size:200% 200%;
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
  animation:gradShift 4s ease infinite,fadeUp .8s ease forwards,glowText 4s ease-in-out infinite;
  margin:0;line-height:1.1;
}
.sub{font-family:'Outfit',sans-serif;font-size:14.5px;color:#475569;
  margin:10px 0 0;letter-spacing:.09em;animation:fadeUp 1s ease .2s both;}
.badges{margin:12px 0 0;animation:fadeUp 1s ease .4s both;}
.badge{display:inline-block;margin:0 5px;padding:4px 15px;border-radius:99px;
  font-size:11px;font-weight:700;letter-spacing:.09em;}
.b1{background:rgba(14,165,233,.14);border:1px solid rgba(14,165,233,.35);color:#38bdf8;}
.b2{background:rgba(129,140,248,.14);border:1px solid rgba(129,140,248,.35);color:#818cf8;}
.b3{background:rgba(232,121,249,.14);border:1px solid rgba(232,121,249,.35);color:#e879f9;}
.divbar{height:3px;width:160px;margin:16px auto 0;border-radius:99px;
  background:linear-gradient(90deg,#38bdf8,#818cf8,#e879f9);
  background-size:200% 200%;animation:gradShift 3s ease infinite;}
.wave-wrap{position:relative;overflow:hidden;height:36px;margin-top:18px;}
.wave{position:absolute;bottom:0;left:0;width:200%;height:36px;
  background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 1440 36'%3E%3Cpath fill='rgba(56,189,248,0.06)' d='M0,18 C360,36 720,0 1080,18 C1260,27 1350,13 1440,18 L1440,36 L0,36 Z'/%3E%3C/svg%3E");
  background-size:50% 100%;animation:waveMove 5s linear infinite;}
</style>
<div class="wrap">
  <div class="plane">✈</div>
  <h1 class="title">✈ AirFlow Forecaster</h1>
  <p class="sub">Deep Learning Time-Series Forecasting &nbsp;·&nbsp; Airline Passenger Dataset 1949–1960</p>
  <div class="badges">
    <span class="badge b1">⚡ LSTM</span>
    <span class="badge b2">🔷 GRU</span>
    <span class="badge b3">🔵 RNN</span>
  </div>
  <div class="divbar"></div>
  <div class="wave-wrap"><div class="wave"></div></div>
</div>
""", height=222)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def dark_layout(height=360):
    return dict(
        paper_bgcolor="rgba(2,8,23,0)",
        plot_bgcolor ="rgba(2,8,23,0)",
        font=dict(family="Outfit", color="#94a3b8", size=12),
        height=height,
        margin=dict(l=44, r=20, t=40, b=44),
        xaxis=dict(gridcolor="rgba(255,255,255,.035)", zeroline=False,
                   linecolor="rgba(255,255,255,.07)", showgrid=True,
                   tickfont=dict(size=11)),
        yaxis=dict(gridcolor="rgba(255,255,255,.035)", zeroline=False,
                   linecolor="rgba(255,255,255,.07)", showgrid=True,
                   tickfont=dict(size=11)),
        legend=dict(bgcolor="rgba(8,13,28,.85)",
                    bordercolor="rgba(56,189,248,.2)", borderwidth=1,
                    font=dict(size=12), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="rgba(8,13,28,.96)",
                        bordercolor="rgba(56,189,248,.45)",
                        font_size=13, font_family="Outfit"),
    )


def kpi_card(icon, label, value, sub, delay="0s"):
    uid = label.replace(" ", "_").replace("/", "_")
    return f"""
<style>
@keyframes fadeInUp__{uid}{{from{{opacity:0;transform:translateY(26px)}}to{{opacity:1;transform:translateY(0)}}}}
@keyframes glowP__{uid}{{0%,100%{{box-shadow:0 0 10px rgba(56,189,248,.2),0 4px 18px rgba(56,189,248,.07)}}
  50%{{box-shadow:0 0 28px rgba(56,189,248,.55),0 8px 36px rgba(56,189,248,.18)}}}}
@keyframes gradS__{uid}{{0%{{background-position:0% 50%}}50%{{background-position:100% 50%}}100%{{background-position:0% 50%}}}}
@keyframes floatK__{uid}{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-6px)}}}}
@keyframes spinK__{uid}{{from{{transform:rotate(0deg)}}to{{transform:rotate(360deg)}}}}
</style>
<div style="
  background:linear-gradient(135deg,rgba(12,20,40,.94),rgba(6,12,26,.96));
  border:1px solid rgba(56,189,248,.2);border-radius:18px;padding:22px 16px;
  text-align:center;backdrop-filter:blur(16px);position:relative;overflow:hidden;
  animation:fadeInUp__{uid} .7s ease {delay} both, glowP__{uid} 3.5s ease-in-out infinite;
  transition:transform .25s,box-shadow .25s;cursor:default;"
  onmouseover="this.style.transform='translateY(-6px) scale(1.03)'"
  onmouseout="this.style.transform='translateY(0) scale(1)'">
  <div style="position:absolute;top:-50%;left:-50%;width:200%;height:200%;
    background:conic-gradient(transparent 260deg,rgba(56,189,248,.035) 360deg);
    animation:spinK__{uid} 8s linear infinite;pointer-events:none;"></div>
  <div style="font-size:28px;animation:floatK__{uid} 3.5s ease-in-out infinite {delay};">{icon}</div>
  <div style="font-size:10.5px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;
              color:#334155;margin:7px 0;">{label}</div>
  <div style="font-size:36px;font-weight:900;
    background:linear-gradient(135deg,#38bdf8,#818cf8,#e879f9);background-size:200% 200%;
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    animation:gradS__{uid} 4s ease infinite;">{value}</div>
  <div style="font-size:11.5px;color:#334155;margin-top:5px;">{sub}</div>
</div>"""


@st.cache_data(show_spinner=False)
def load_data():
    return DataLoader(DATA_PATH).load_data()


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    components.html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@700;900&display=swap');
@keyframes floatLogo{0%,100%{transform:translateY(0)}50%{transform:translateY(-8px)}}
@keyframes gradS{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes fadeLeft{from{opacity:0;transform:translateX(-18px)}to{opacity:1;transform:translateX(0)}}
</style>
<div style="text-align:center;padding:14px 0 18px;animation:fadeLeft .6s ease forwards;">
  <div style="font-size:46px;filter:drop-shadow(0 0 16px rgba(56,189,248,.9));
              animation:floatLogo 3s ease-in-out infinite;">✈</div>
  <div style="font-family:'Outfit',sans-serif;font-size:19px;font-weight:700;
    background:linear-gradient(135deg,#38bdf8,#818cf8);background-size:200% 200%;
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
    animation:gradS 3s ease infinite;margin-top:5px;">Model Controls</div>
  <div style="font-size:11px;color:#334155;margin-top:3px;letter-spacing:.06em;">Configure &amp; Train</div>
</div>
""", height=112)

    st.markdown("---")

    model_type = st.selectbox(
        "🤖  Architecture",
        ["lstm", "gru", "rnn"],
        format_func=lambda x: {
            "lstm": "⚡ LSTM – Long Short-Term Memory",
            "gru":  "🔷 GRU  – Gated Recurrent Unit",
            "rnn":  "🔵 RNN  – Simple Recurrent",
        }[x],
    )
    seq_len     = st.slider("📏  Sequence Length (months)", 6, 24, 12, 1)
    train_split = st.slider("✂️  Train Split %", 60, 90, 80, 5) / 100
    epochs      = st.slider("🔁  Epochs", 10, 300, 100, 10)
    batch_size  = st.selectbox("📦  Batch Size", [8, 16, 32, 64], index=1)

    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("🚀  Run Forecast")
    st.markdown("---")

    components.html("""
<style>
@keyframes borderGlow{0%,100%{border-color:rgba(56,189,248,.18)}50%{border-color:rgba(129,140,248,.55)}}
@keyframes fadeInUp{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
</style>
<div style="border:1px solid rgba(56,189,248,.18);border-radius:12px;padding:13px 15px;
  animation:borderGlow 2.5s ease-in-out infinite,fadeInUp .8s ease forwards;
  background:rgba(8,13,28,.7);font-family:'Outfit',sans-serif;">
  <div style="font-size:10.5px;color:#334155;font-weight:700;letter-spacing:.1em;margin-bottom:4px;">📂 DATASET</div>
  <div style="font-size:12px;color:#475569;">Airline Passengers 1949-1960</div>
  <div style="font-size:10.5px;color:#334155;font-weight:700;letter-spacing:.1em;margin:10px 0 4px;">💾 MODEL PATH</div>
  <div style="font-size:11px;color:#475569;">models/lstm_models.keras</div>
  <div style="font-size:10.5px;color:#334155;font-weight:700;letter-spacing:.1em;margin:10px 0 4px;">⚖️ SCALER PATH</div>
  <div style="font-size:11px;color:#475569;">models/scaler.pkl</div>
</div>
""", height=160)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner("🛫  Loading dataset…"):
    df = load_data()


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["📊  Data Overview", "🧠  Train & Evaluate", "🔮  Forecast"])


# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — DATA OVERVIEW
# ───────────────────────────────────────────────────────────────────────────────
with tab1:

    growth = int(((df["Passengers"].iloc[-1] / df["Passengers"].iloc[0]) - 1) * 100)
    kpis = [
        ("📋", "Total Records",   str(len(df)),
         "Monthly observations",                                       "0s"),
        ("📅", "Date Range",      f"{df.index.year.min()}–{df.index.year.max()}",
         "12 full years",                                             ".12s"),
        ("🏆", "Peak Passengers", f"{int(df['Passengers'].max()):,}",
         f"Avg {int(df['Passengers'].mean()):,}/mo",                  ".24s"),
        ("📈", "Overall Growth",  f"{growth}%",
         "1949 → 1960",                                               ".36s"),
    ]
    cols = st.columns(4)
    for col, (icon, lbl, val, sub, delay) in zip(cols, kpis):
        with col:
            st.markdown(kpi_card(icon, lbl, val, sub, delay), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Historical trend
    st.markdown('<div class="section-hdr">📈 Historical Passenger Trend</div>',
                unsafe_allow_html=True)
    fig_h = go.Figure()
    fig_h.add_trace(go.Scatter(
        x=df.index, y=df["Passengers"], mode="lines",
        name="Passengers",
        line=dict(color="#38bdf8", width=3, shape="spline", smoothing=1.2),
        fill="tozeroy", fillcolor="rgba(56,189,248,.07)",
        hovertemplate="<b>%{x|%b %Y}</b><br>✈ %{y:,} passengers<extra></extra>",
    ))
    peak_idx = df["Passengers"].idxmax()
    fig_h.add_trace(go.Scatter(
        x=[peak_idx], y=[df["Passengers"].max()],
        mode="markers+text",
        marker=dict(size=15, color="#e879f9", symbol="star",
                    line=dict(color="#fff", width=2)),
        text=[f"  Peak: {int(df['Passengers'].max()):,}"],
        textposition="top right",
        textfont=dict(color="#e879f9", size=12, family="Outfit"),
        name="Peak", showlegend=False,
    ))
    fig_h.update_layout(**dark_layout(360), yaxis_title="Passengers/month")
    st.plotly_chart(fig_h, use_container_width=True)

    # Rolling average
    st.markdown('<div class="section-hdr">📉 12-Month Rolling Average</div>',
                unsafe_allow_html=True)
    roll = df["Passengers"].rolling(12, center=True).mean()
    fig_r2 = go.Figure()
    fig_r2.add_trace(go.Scatter(
        x=df.index, y=df["Passengers"], mode="lines", name="Raw",
        line=dict(color="rgba(56,189,248,.4)", width=1.5),
    ))
    fig_r2.add_trace(go.Scatter(
        x=df.index, y=roll, mode="lines", name="12-mo Avg",
        line=dict(color="#e879f9", width=3, shape="spline", smoothing=1.3),
        fill="tonexty", fillcolor="rgba(232,121,249,.05)",
        hovertemplate="<b>%{x|%b %Y}</b><br>Rolling: %{y:,.1f}<extra></extra>",
    ))
    fig_r2.update_layout(**dark_layout(290), yaxis_title="Passengers")
    st.plotly_chart(fig_r2, use_container_width=True)

    # Seasonal heatmap
    st.markdown('<div class="section-hdr">🗓️ Seasonal Heatmap — Month × Year</div>',
                unsafe_allow_html=True)
    df_h2 = df.copy()
    df_h2["Year"] = df_h2.index.year
    df_h2["Mon"]  = df_h2.index.month
    pivot = df_h2.pivot_table(index="Mon", columns="Year",
                               values="Passengers", aggfunc="mean")
    mnames = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig_hm = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(c) for c in pivot.columns],
        y=mnames,
        colorscale=[[0,"#020817"],[0.2,"#0e3a5e"],[0.45,"#0ea5e9"],
                    [0.7,"#818cf8"],[1.0,"#e879f9"]],
        hovertemplate="<b>%{y} %{x}</b><br>✈ %{z:,.0f}<extra></extra>",
        showscale=True, zsmooth="best",
        colorbar=dict(tickfont=dict(color="#64748b", size=11),
                      outlinecolor="rgba(56,189,248,.14)",
                      bgcolor="rgba(8,13,28,.6)"),
    ))
    fig_hm.update_layout(**dark_layout(310))
    st.plotly_chart(fig_hm, use_container_width=True)

    # Annual bar + raw table
    col_a, col_b = st.columns([1.1, 1])
    with col_a:
        st.markdown('<div class="section-hdr">📅 Year-over-Year Averages</div>',
                    unsafe_allow_html=True)
        ann = df.groupby(df.index.year)["Passengers"].agg(["mean", "max"]).reset_index()
        ann.columns = ["Year", "Avg", "Max"]
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=ann["Year"].astype(str), y=ann["Avg"], name="Avg",
            marker=dict(color=ann["Avg"],
                        colorscale=[[0,"#0ea5e9"],[.5,"#818cf8"],[1,"#e879f9"]],
                        showscale=False, opacity=.88,
                        line=dict(color="rgba(0,0,0,0)")),
            hovertemplate="<b>%{x}</b><br>Avg: %{y:.0f}<extra></extra>",
        ))
        fig_bar.add_trace(go.Scatter(
            x=ann["Year"].astype(str), y=ann["Max"],
            mode="lines+markers", name="Peak",
            line=dict(color="#e879f9", width=2),
            marker=dict(size=7, color="#e879f9",
                        line=dict(color="#020817", width=1.5)),
            hovertemplate="Peak: %{y:,}<extra></extra>",
        ))
        fig_bar.update_layout(**dark_layout(300), yaxis_title="Passengers")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-hdr">📋 Dataset Explorer</div>',
                    unsafe_allow_html=True)
        display = (df.rename_axis("Month").reset_index()
                     .assign(Month=lambda d: d["Month"].dt.strftime("%b %Y"))
                     .set_index("Month"))
        st.dataframe(display, use_container_width=True, height=300)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — TRAIN & EVALUATE
# ───────────────────────────────────────────────────────────────────────────────
with tab2:

    if not run_btn:
        components.html("""
<style>
@keyframes fadeInUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
@keyframes bGlow{0%,100%{border-color:rgba(14,165,233,.25)}50%{border-color:rgba(14,165,233,.7)}}
</style>
<div style="background:rgba(14,165,233,.055);border:1px solid rgba(14,165,233,.25);
  border-left:4px solid #0ea5e9;border-radius:12px;padding:15px 20px;
  animation:fadeInUp .5s ease forwards,bGlow 2s ease-in-out infinite;
  font-family:'Outfit',sans-serif;font-size:14px;color:#94a3b8;line-height:1.6;">
  ⚙️ &nbsp;Configure model parameters in the sidebar, then click
  <strong style="color:#38bdf8;">🚀 Run Forecast</strong> to train and evaluate.
</div>
""", height=65)

        st.markdown('<div class="section-hdr">🏗️ Architecture Preview</div>',
                    unsafe_allow_html=True)
        arch = {
            "Layer":       ["Input", model_type.upper(), "Dropout",
                             "Dense (ReLU)", "Dense (Output)"],
            "Shape/Units": [f"({seq_len}, 1)", "64 units", "20%",
                             "32 units", "1 unit"],
            "Activation":  ["—", "tanh / sigmoid", "—", "ReLU", "Linear"],
            "Est. Params": ["—", "~17,000", "—", "~2,080", "33"],
        }
        st.dataframe(pd.DataFrame(arch), use_container_width=True, hide_index=True)

    else:
        prog = st.progress(0, text="🛫  Initialising pipeline…")

        prog.progress(10, text="⚖️  MinMaxScaler fitting…")
        preprocessor = Preprocessor()
        scaled_df    = preprocessor.scale_data(df)

        prog.progress(25, text="🔗  Generating sliding-window sequences…")
        gen  = SequenceGenerator(sequence_length=seq_len)
        X, y = gen.create_sequences(scaled_df)

        prog.progress(40, text="✂️  Train / test split…")
        splitter = TimeSeriesSplit(train_size=train_split)
        X_train, X_test, y_train, y_test = splitter.split(X, y)

        prog.progress(55, text=f"🏗️  Building {model_type.upper()} model…")
        builder = ModelBuilder(model_type=model_type, input_shape=(seq_len, 1))
        model   = builder.build_model()

        prog.progress(65, text=f"🧠  Training — {epochs} epochs, batch {batch_size}…")
        history = model.fit(
            X_train, y_train,
            epochs=epochs, batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=0,
        )

        prog.progress(90, text="💾  Saving model & scaler…")
        model.save(MODEL_PATH)

        prog.progress(100, text="✅  Complete!")
        prog.empty()
        st.success("🎉  Model trained and saved successfully!", icon="✅")

        st.session_state.update({
            "model": model, "scaler": preprocessor.scaler,
            "X_test": X_test, "y_test": y_test,
            "history": history, "scaled_df": scaled_df,
            "seq_len": seq_len, "train_split": train_split, "df": df,
        })

        # Evaluation metrics
        scaler    = preprocessor.scaler
        y_pred_sc = model.predict(X_test, verbose=0)
        y_pred    = scaler.inverse_transform(y_pred_sc)
        y_true    = scaler.inverse_transform(y_test)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae  = float(mean_absolute_error(y_true, y_pred))
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2   = float(1 - ss_res / ss_tot)

        m_cols = st.columns(4)
        for col, (icon, lbl, val, sub, delay) in zip(m_cols, [
            ("📉", "RMSE",      f"{rmse:.2f}",  "Root Mean Sq. Error",  "0s"),
            ("📐", "MAE",       f"{mae:.2f}",   "Mean Absolute Error",  ".1s"),
            ("🎯", "MAPE",      f"{mape:.1f}%", "Mean Abs % Error",     ".2s"),
            ("📊", "R² Score",  f"{r2:.4f}",    "Model fit quality",    ".3s"),
        ]):
            with col:
                st.markdown(kpi_card(icon, lbl, val, sub, delay), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Loss curves
        st.markdown('<div class="section-hdr">📉 Training & Validation Loss</div>',
                    unsafe_allow_html=True)
        ep = list(range(1, epochs + 1))
        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(
            x=ep, y=history.history["loss"],
            mode="lines", name="Train Loss",
            line=dict(color="#38bdf8", width=2.5, shape="spline", smoothing=1.2),
            fill="tozeroy", fillcolor="rgba(56,189,248,.05)",
            hovertemplate="Epoch %{x}<br>Train: %{y:.5f}<extra></extra>",
        ))
        fig_l.add_trace(go.Scatter(
            x=ep, y=history.history["val_loss"],
            mode="lines", name="Val Loss",
            line=dict(color="#e879f9", width=2.5, dash="dot", shape="spline"),
            hovertemplate="Epoch %{x}<br>Val: %{y:.5f}<extra></extra>",
        ))
        fig_l.update_layout(**dark_layout(310),
                             yaxis_title="MSE Loss", xaxis_title="Epoch")
        st.plotly_chart(fig_l, use_container_width=True)

        # MAE curves
        if "mae" in history.history:
            st.markdown('<div class="section-hdr">📈 MAE over Epochs</div>',
                        unsafe_allow_html=True)
            fig_mae = go.Figure()
            fig_mae.add_trace(go.Scatter(
                x=ep, y=history.history["mae"],
                mode="lines", name="Train MAE",
                line=dict(color="#818cf8", width=2.5, shape="spline"),
                fill="tozeroy", fillcolor="rgba(129,140,248,.05)",
            ))
            if "val_mae" in history.history:
                fig_mae.add_trace(go.Scatter(
                    x=ep, y=history.history["val_mae"],
                    mode="lines", name="Val MAE",
                    line=dict(color="#f97316", width=2.5, dash="dot", shape="spline"),
                ))
            fig_mae.update_layout(**dark_layout(280),
                                   yaxis_title="MAE", xaxis_title="Epoch")
            st.plotly_chart(fig_mae, use_container_width=True)

        # Actual vs Predicted
        st.markdown('<div class="section-hdr">🎯 Test Set: Actual vs Predicted</div>',
                    unsafe_allow_html=True)
        split_idx  = int(len(df) * train_split)
        test_dates = df.index[split_idx + seq_len:]
        n          = min(len(test_dates), len(y_true), len(y_pred))
        td         = test_dates[:n]
        yt         = y_true[:n].flatten()
        yp         = y_pred[:n].flatten()

        fig_t = go.Figure()
        err = np.abs(yt - yp)
        fig_t.add_trace(go.Scatter(
            x=list(td) + list(td[::-1]),
            y=list(yt + err) + list((yt - err)[::-1]),
            fill="toself", fillcolor="rgba(249,115,22,.06)",
            line=dict(color="rgba(0,0,0,0)"),
            name="Error Band", hoverinfo="skip",
        ))
        fig_t.add_trace(go.Scatter(
            x=td, y=yt, mode="lines", name="Actual",
            line=dict(color="#38bdf8", width=3, shape="spline", smoothing=1.1),
            hovertemplate="<b>%{x|%b %Y}</b><br>Actual: <b>%{y:,}</b><extra></extra>",
        ))
        fig_t.add_trace(go.Scatter(
            x=td, y=yp, mode="lines+markers", name="Predicted",
            line=dict(color="#f97316", width=2.5, dash="dot", shape="spline"),
            marker=dict(size=6, color="#f97316",
                        line=dict(color="#020817", width=1.5)),
            hovertemplate="<b>%{x|%b %Y}</b><br>Predicted: <b>%{y:,}</b><extra></extra>",
        ))
        fig_t.update_layout(**dark_layout(340), yaxis_title="Passengers")
        st.plotly_chart(fig_t, use_container_width=True)

        # Residuals
        st.markdown('<div class="section-hdr">📊 Residuals Analysis</div>',
                    unsafe_allow_html=True)
        residuals = yt - yp
        fig_res = make_subplots(rows=1, cols=2,
                                 subplot_titles=["Residuals over Time",
                                                 "Distribution"])
        fig_res.add_trace(go.Scatter(
            x=td, y=residuals, mode="lines+markers",
            marker=dict(size=5, color="#818cf8",
                        line=dict(color="#020817", width=1)),
            line=dict(color="#818cf8", width=2, shape="spline"),
            name="Residual",
        ), row=1, col=1)
        fig_res.add_hline(y=0, line_dash="dash",
                           line_color="rgba(255,255,255,.15)", row=1, col=1)
        fig_res.add_trace(go.Histogram(
            x=residuals, nbinsx=14,
            marker=dict(color="rgba(129,140,248,.6)",
                        line=dict(color="#818cf8", width=1.2)),
            name="Freq",
        ), row=1, col=2)
        fig_res.update_layout(
            paper_bgcolor="rgba(2,8,23,0)", plot_bgcolor="rgba(2,8,23,0)",
            font=dict(family="Outfit", color="#94a3b8", size=12),
            height=285, showlegend=False,
            margin=dict(l=44, r=20, t=44, b=44),
            hoverlabel=dict(bgcolor="rgba(8,13,28,.96)",
                            bordercolor="rgba(56,189,248,.4)",
                            font_size=13, font_family="Outfit"),
        )
        fig_res.update_xaxes(gridcolor="rgba(255,255,255,.035)",
                              linecolor="rgba(255,255,255,.07)")
        fig_res.update_yaxes(gridcolor="rgba(255,255,255,.035)",
                              linecolor="rgba(255,255,255,.07)")
        st.plotly_chart(fig_res, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — FORECAST
# ───────────────────────────────────────────────────────────────────────────────
with tab3:

    horizon = st.slider("🔭  Future Forecast Horizon (months)", 6, 48, 24, 6,
                        key="horiz_sl")

    model_ready = False

    if "model" in st.session_state:
        model     = st.session_state["model"]
        scaler    = st.session_state["scaler"]
        scaled_df = st.session_state["scaled_df"]
        df_full   = st.session_state["df"]
        seq_len_f = st.session_state["seq_len"]
        model_ready = True

    elif os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            from tensorflow.keras.models import load_model as _lm
            model      = _lm(MODEL_PATH)
            scaler     = joblib.load(SCALER_PATH)
            pp2        = Preprocessor()
            pp2.scaler = scaler
            scaled_df  = pp2.scale_data(df)
            df_full    = df
            seq_len_f  = seq_len
            model_ready = True
            st.info("💾  Pre-trained model loaded from disk.", icon="📂")
        except Exception as e:
            st.error(f"Could not load saved model: {e}")

    else:
        components.html("""
<style>
@keyframes fadeInUp{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:translateY(0)}}
@keyframes bGlow{0%,100%{border-color:rgba(14,165,233,.2)}50%{border-color:rgba(232,121,249,.5)}}
</style>
<div style="background:rgba(14,165,233,.05);border:1px solid rgba(14,165,233,.2);
  border-left:4px solid #0ea5e9;border-radius:12px;padding:15px 20px;
  animation:fadeInUp .5s ease,bGlow 2s ease-in-out infinite;
  font-family:'Outfit',sans-serif;font-size:14px;color:#94a3b8;">
  🔮 No trained model found. Go to
  <strong style="color:#38bdf8;">🧠 Train &amp; Evaluate</strong>
  and click <strong style="color:#38bdf8;">🚀 Run Forecast</strong> first.
</div>""", height=68)

    if model_ready:
        with st.spinner("🔮  Generating future predictions…"):
            vals      = scaled_df.values.flatten().tolist()
            last_seq  = vals[-seq_len_f:]
            fut_sc    = []
            for _ in range(horizon):
                inp = np.array(last_seq[-seq_len_f:]).reshape(1, seq_len_f, 1)
                p   = model.predict(inp, verbose=0)[0][0]
                fut_sc.append(p)
                last_seq.append(p)
            future_preds = scaler.inverse_transform(
                np.array(fut_sc).reshape(-1, 1)
            ).flatten()
            future_dates = pd.date_range(
                start=df_full.index[-1] + pd.DateOffset(months=1),
                periods=horizon, freq="MS",
            )

        upper = future_preds * 1.15
        lower = future_preds * 0.85

        # Forecast KPI cards
        fc_cols = st.columns(4)
        pi = future_preds.argmax()
        ni = future_preds.argmin()
        for col, (icon, lbl, val, sub, delay) in zip(fc_cols, [
            ("🔭", "Horizon",      f"{horizon} mo",
             "Future months",                           "0s"),
            ("📈", "Forecast Peak", f"{int(future_preds.max()):,}",
             future_dates[pi].strftime("%b %Y"),        ".1s"),
            ("📉", "Forecast Min",  f"{int(future_preds.min()):,}",
             future_dates[ni].strftime("%b %Y"),        ".2s"),
            ("📊", "Monthly Avg",  f"{int(future_preds.mean()):,}",
             "avg passengers",                          ".3s"),
        ]):
            with col:
                st.markdown(kpi_card(icon, lbl, val, sub, delay), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Full forecast chart
        st.markdown('<div class="section-hdr">🔮 Full Forecast Chart</div>',
                    unsafe_allow_html=True)
        fig_fc = go.Figure()
        fig_fc.add_trace(go.Scatter(
            x=df_full.index, y=df_full["Passengers"],
            mode="lines", name="Historical",
            line=dict(color="#38bdf8", width=2.5, shape="spline", smoothing=1.1),
            fill="tozeroy", fillcolor="rgba(56,189,248,.055)",
            hovertemplate="<b>%{x|%b %Y}</b><br>%{y:,} passengers<extra></extra>",
        ))
        fig_fc.add_trace(go.Scatter(
            x=list(future_dates) + list(future_dates[::-1]),
            y=list(upper) + list(lower[::-1]),
            fill="toself", fillcolor="rgba(232,121,249,.09)",
            line=dict(color="rgba(0,0,0,0)"),
            name="±15% Confidence", hoverinfo="skip",
        ))
        fig_fc.add_trace(go.Scatter(
            x=future_dates, y=future_preds,
            mode="lines+markers", name="Forecast",
            line=dict(color="#e879f9", width=3, dash="dot", shape="spline"),
            marker=dict(size=8, color="#e879f9",
                        line=dict(color="#020817", width=2)),
            hovertemplate="<b>%{x|%b %Y}</b><br>🔮 %{y:,.0f}<extra></extra>",
        ))
        # Use add_shape instead of add_vline — add_vline computes mean of ALL
        # x-axis datetimes (incl. 1949-1960) producing negative Unix timestamps
        # that crash datetime.fromtimestamp() on Windows.
        divider_x = df_full.index[-1].strftime("%Y-%m-%d")
        fig_fc.add_shape(
            type="line",
            x0=divider_x, x1=divider_x,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="rgba(255,255,255,.18)", dash="dash", width=1.5),
        )
        fig_fc.add_annotation(
            x=divider_x, y=1,
            xref="x", yref="paper",
            text="← History | Forecast →",
            showarrow=False,
            font=dict(color="#475569", size=11, family="Outfit"),
            xanchor="left", yanchor="top",
            xshift=6,
        )

        fig_fc.update_layout(**dark_layout(450), yaxis_title="Passengers")
        st.plotly_chart(fig_fc, use_container_width=True)

        # Forecast table + bar
        fc_df = pd.DataFrame({
            "Month":        future_dates.strftime("%b %Y"),
            "Forecast":     future_preds.astype(int),
            "Lower (−15%)": lower.astype(int),
            "Upper (+15%)": upper.astype(int),
        })

        tbl_c, bar_c = st.columns([1.1, 1])
        with tbl_c:
            st.markdown('<div class="section-hdr" style="font-size:16px;">📋 Forecast Table</div>',
                        unsafe_allow_html=True)
            st.dataframe(fc_df.set_index("Month"),
                         use_container_width=True, height=340)

        with bar_c:
            st.markdown('<div class="section-hdr" style="font-size:16px;">📊 Forecast Bar</div>',
                        unsafe_allow_html=True)
            fig_fb = go.Figure(go.Bar(
                x=fc_df["Month"], y=fc_df["Forecast"],
                marker=dict(
                    color=fc_df["Forecast"],
                    colorscale=[[0,"#6366f1"],[.5,"#818cf8"],[1,"#e879f9"]],
                    showscale=False, opacity=.88,
                    line=dict(color="rgba(0,0,0,0)")),
                text=fc_df["Forecast"],
                texttemplate="%{text:,}",
                textposition="outside",
                textfont=dict(color="#94a3b8", size=9),
                hovertemplate="<b>%{x}</b><br>🔮 %{y:,}<extra></extra>",
            ))
            fig_fb.update_layout(**dark_layout(340),
                                  yaxis_title="Passengers",
                                  xaxis_tickangle=-50)
            st.plotly_chart(fig_fb, use_container_width=True)

        # Download
        st.markdown("<br>", unsafe_allow_html=True)
        csv_bytes = fc_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download Forecast CSV",
            data=csv_bytes,
            file_name="airline_forecast.csv",
            mime="text/csv",
            use_container_width=True,
        )