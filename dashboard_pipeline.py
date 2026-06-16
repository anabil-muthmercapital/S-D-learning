#!/usr/bin/env python
# =============================================================================
# dashboard_pipeline.py — Interactive "watch the pipeline run" dashboard
# =============================================================================
# بيستورد نفس دوال build_dataset — اللي بتشوفه هنا = نفس اللي بيتحط في الـ dataset.
# التشغيل:  cd /Users/an/Desktop/S-D-learning && streamlit run dashboard_pipeline.py
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.base_detector import detect_bases, find_base_clusters
from utils.config import (
    CHART_BG,
    CHART_GRID,
    COLOR_BEAR,
    COLOR_BULL,
    HTF_REF,
    WATCHLIST,
)
from utils.data_loader import load_enriched_timeframes
from utils.freshness import add_freshness
from utils.htf_range import add_curve_score
from utils.labeler import DEFAULT_HTF_LTF_MAP, label_zones
from utils.legs_formation import detect_formations
from utils.sets_scoring import add_sets_score
from utils.time_scoring import add_time_score
from utils.trend_alignment import add_trend_score
from utils.zone_detector import detect_zones

st.set_page_config(page_title="OTA S&D — Pipeline Live", page_icon="📈", layout="wide")

COLOR_DOJI_SAFE = "#ffd700"
LTFS = ["1h", "4h", "1d"]
REQUIRED_TFS = {
    "1h": {"1h", "4h", "1d"},
    "4h": {"4h", "1d", "1wk"},
    "1d": {"1d", "1wk"},
}
PIPELINE_STEPS = [
    "1. Candles",
    "2. Base Detection",
    "3. Formations",
    "4. Zones + Departure",
    "5. Scoring",
    "6. Labeling",
]


def _asset_class(symbol: str) -> str:
    for cls, syms in WATCHLIST.items():
        if symbol in syms:
            return cls
    return "unknown"


@st.cache_data(show_spinner=False)
def run_pipeline(symbol: str, ltf: str, max_hold: int = 60) -> dict:
    data = load_enriched_timeframes(symbol)
    ltf_df = data[ltf]
    passed_clusters, failed_clusters = detect_bases(ltf_df)
    all_clusters = find_base_clusters(ltf_df)
    formations = detect_formations(ltf_df, passed_clusters)
    zones, rejected = detect_zones(ltf_df, formations)
    if zones:
        add_freshness(ltf_df, zones)
        add_time_score(zones)
        curve_ref_tf = HTF_REF.get(ltf, "1d")
        curve_ref_df = data.get(curve_ref_tf)
        if curve_ref_df is None:
            curve_ref_df = data.get("1d")
        if curve_ref_df is not None:
            add_curve_score(zones, curve_ref_df, ltf_df.index)
        add_trend_score(zones, ltf_df)
        add_sets_score(zones)
        itf_name, htf_name = DEFAULT_HTF_LTF_MAP[ltf]
        itf_df = data.get(itf_name)
        htf_df = data.get(htf_name) if htf_name else None
        label_zones(zones, ltf_df, itf_df, htf_df, max_hold_bars=max_hold)
    return {
        "ltf_df": ltf_df,
        "all_clusters": all_clusters,
        "passed_clusters": passed_clusters,
        "failed_clusters": failed_clusters,
        "formations": formations,
        "zones": zones,
        "rejected": rejected,
    }


def _base_candles(df: pd.DataFrame, highlight_base: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        increasing_line_color=COLOR_BULL, decreasing_line_color=COLOR_BEAR, name="Price"))
    if highlight_base and "is_base" in df.columns:
        bb = df[df["is_base"]]
        if not bb.empty:
            fig.add_trace(go.Scatter(
                x=bb.index, y=bb["high"] * 1.001, mode="markers",
                marker=dict(symbol="triangle-down", size=7, color=COLOR_DOJI_SAFE),
                name="base candle", hoverinfo="skip"))
    fig.update_layout(
        paper_bgcolor=CHART_BG, plot_bgcolor=CHART_BG, font=dict(color="#d1d4dc"),
        xaxis=dict(gridcolor=CHART_GRID, rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor=CHART_GRID), height=620,
        margin=dict(l=10, r=10, t=30, b=10), showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)"))
    return fig


def _add_zone_box(fig, df, zone, color, label, extend_bars=30):
    bs, be = zone["start"], zone["end"]
    x0 = df.index[bs]
    end_pos = min(be + extend_bars, len(df) - 1)
    x1 = df.index[end_pos]
    prox, dist = zone["proximal"], zone["distal"]
    top, bot = max(prox, dist), min(prox, dist)
    fig.add_shape(type="rect", x0=x0, x1=x1, y0=bot, y1=top,
                  line=dict(color=color, width=1), fillcolor=color, opacity=0.15, layer="below")
    fig.add_shape(type="line", x0=x0, x1=x1, y0=prox, y1=prox, line=dict(color=color, width=2))
    fig.add_shape(type="line", x0=x0, x1=x1, y0=dist, y1=dist,
                  line=dict(color=color, width=1, dash="dash"))
    fig.add_annotation(x=x0, y=top, text=label, showarrow=False,
                       font=dict(color=color, size=11), xanchor="left", yanchor="bottom")


st.sidebar.title("⚙️ Controls")
all_symbols = [s for syms in WATCHLIST.values() for s in syms]
default_sym = all_symbols.index("USDJPY=X") if "USDJPY=X" in all_symbols else 0
symbol = st.sidebar.selectbox("Symbol", all_symbols, index=default_sym)
ltf = st.sidebar.selectbox("Timeframe", LTFS, index=2)
st.sidebar.markdown("---")
st.sidebar.markdown("### Pipeline step")
step = st.sidebar.radio("Show up to:", PIPELINE_STEPS, index=0, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("كل خطوة بتبني على اللي قبلها — نفس ترتيب build_dataset.py. اللي بتشوفه = نفس اللي بيتحط في الـ dataset.")

st.title("📈 OTA S&D — الـ Pipeline بيشتغل قدامك")
st.caption(f"**{symbol}** · **{ltf}** · asset class: **{_asset_class(symbol)}**")

try:
    with st.spinner(f"بشغّل الـ pipeline على {symbol} {ltf} ..."):
        result = run_pipeline(symbol, ltf)
except Exception as exc:  # noqa: BLE001
    st.error(f"فشل تشغيل الـ pipeline: {type(exc).__name__}: {exc}")
    st.info(f"اتأكد إن الرمز عنده CSVs في data/raw/ والفريمات المطلوبة موجودة ({sorted(REQUIRED_TFS[ltf])}).")
    st.stop()

df = result["ltf_df"]
view = df.tail(250)
step_idx = PIPELINE_STEPS.index(step)

cols = st.columns(len(PIPELINE_STEPS))
for i, (c, s) in enumerate(zip(cols, PIPELINE_STEPS)):
    (c.success if i <= step_idx else c.info)(s.split(". ")[1])
st.markdown("---")

if step_idx == 0:
    st.subheader("1️⃣ الشموع الخام + كشف شموع الـ base")
    st.markdown("كل شمعة جسمها صغير (body_ratio ≤ 0.5) بتتعلّم كـ **base candle** (المثلثات الصفرا).")
    st.plotly_chart(_base_candles(view, highlight_base=True), use_container_width=True)
    n_base = int(view["is_base"].sum()) if "is_base" in view.columns else 0
    st.metric("شموع base في آخر 250 شمعة", n_base)

elif step_idx == 1:
    st.subheader("2️⃣ كشف الـ Base Clusters")
    st.markdown("الـ base candles المتراصة بتتجمّع في **clusters** بتعدّي **min_count** و **compactness** (≤ 2.5 ATR). 🟢 عدّى · 🔴 اترفض.")
    fig = _base_candles(view, highlight_base=False)
    vstart = len(df) - len(view)
    p_n = f_n = 0
    for c in result["passed_clusters"]:
        if c["end"] >= vstart:
            sub = df.iloc[c["start"]: c["end"] + 1]
            fig.add_shape(type="rect", x0=df.index[c["start"]], x1=df.index[c["end"]],
                          y0=sub["low"].min(), y1=sub["high"].max(),
                          line=dict(color=COLOR_BULL, width=1.5), fillcolor=COLOR_BULL, opacity=0.2)
            p_n += 1
    for c in result["failed_clusters"]:
        if c["end"] >= vstart:
            sub = df.iloc[c["start"]: c["end"] + 1]
            fig.add_shape(type="rect", x0=df.index[c["start"]], x1=df.index[c["end"]],
                          y0=sub["low"].min(), y1=sub["high"].max(),
                          line=dict(color=COLOR_BEAR, width=1, dash="dot"), fillcolor=COLOR_BEAR, opacity=0.1)
            f_n += 1
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 عدّت (عرض)", p_n)
    c2.metric("🔴 اترفضت (عرض)", f_n)
    c3.metric("إجمالي عدّت", len(result["passed_clusters"]))

elif step_idx == 2:
    st.subheader("3️⃣ الـ Formations — اتجاه المنطقة")
    st.markdown("بنقيس الـ **leg-out**: لأعلى → 🟢 **Demand**، لأسفل → 🔴 **Supply**. اللي فشل في leg_strength أو clean_departure بيترفض.")
    fig = _base_candles(view, highlight_base=False)
    vstart = len(df) - len(view)
    d_n = s_n = 0
    for f in result["formations"]:
        if f["end"] < vstart:
            continue
        color = COLOR_BULL if f["zone_type"] == "demand" else COLOR_BEAR
        sub = df.iloc[f["start"]: f["end"] + 1]
        fig.add_shape(type="rect", x0=df.index[f["start"]], x1=df.index[f["end"]],
                      y0=sub["low"].min(), y1=sub["high"].max(),
                      line=dict(color=color, width=1.5), fillcolor=color, opacity=0.2)
        fig.add_annotation(x=df.index[f["start"]], y=sub["high"].max(), text=f["formation"],
                           showarrow=False, font=dict(color=color, size=10), xanchor="left", yanchor="bottom")
        if f["zone_type"] == "demand":
            d_n += 1
        else:
            s_n += 1
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Demand", d_n)
    c2.metric("🔴 Supply", s_n)
    c3.metric("إجمالي", len(result["formations"]))

elif step_idx == 3:
    st.subheader("4️⃣ الـ Zones النهائية — proximal / distal / departure")
    st.markdown("اللي عدّى **departure gates** (dep_ratio ≥ 2.0 و dep_atr ≥ 0.5) بقى zone. خط متصل = **proximal** (دخول)، متقطّع = **distal** (ستوب). المرفوضة باهتة.")
    fig = _base_candles(view, highlight_base=False)
    vstart = len(df) - len(view)
    for z in result["zones"]:
        if z["end"] < vstart:
            continue
        color = COLOR_BULL if z["zone_type"] == "demand" else COLOR_BEAR
        _add_zone_box(fig, df, z, color, f"{z['formation']} (dep {z['dep_ratio']:.1f})")
    for z in result["rejected"]:
        if z["end"] < vstart:
            continue
        sub = df.iloc[z["start"]: z["end"] + 1]
        fig.add_shape(type="rect", x0=df.index[z["start"]], x1=df.index[z["end"]],
                      y0=sub["low"].min(), y1=sub["high"].max(),
                      line=dict(color="#666", width=1, dash="dot"), fillcolor="#666", opacity=0.08)
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("✅ صالحة", len(result["zones"]))
    c2.metric("❌ مرفوضة", len(result["rejected"]))

elif step_idx == 4:
    st.subheader("5️⃣ التقييم — الـ 5 معايير (SETS)")
    st.markdown("نقاط: **strength · freshness · time · curve · trend**. المجموع → ★★★ (≥7) / ★★ (≥5) / ★.")
    zones = result["zones"]
    if not zones:
        st.warning("مفيش zones.")
    else:
        rows = []
        for z in zones:
            total = z.get("sets_total", 0)
            stars = "★★★" if total >= 7 else ("★★" if total >= 5 else "★")
            rows.append({
                "formation": z["formation"], "type": z["zone_type"],
                "strength": z.get("strength_score", 0), "freshness": z.get("freshness_score", 0),
                "time": z.get("time_score", 0), "curve": z.get("curve_score", 0),
                "trend": z.get("trend_score", 0), "SETS total": total, "rating": stars,
                "formation_time": df.index[z["end"]].strftime("%Y-%m-%d %H:%M")})
        sdf = pd.DataFrame(rows).sort_values("SETS total", ascending=False)
        st.dataframe(sdf, use_container_width=True, height=420)
        c1, c2, c3 = st.columns(3)
        c1.metric("★★★", int((sdf["SETS total"] >= 7).sum()))
        c2.metric("★★", int(((sdf["SETS total"] >= 5) & (sdf["SETS total"] < 7)).sum()))
        c3.metric("★", int((sdf["SETS total"] < 5).sum()))

elif step_idx == 5:
    st.subheader("6️⃣ الـ Labeling — الصفقة من الدخول للنتيجة")
    st.markdown("الدخول عند **proximal**، الستوب عند **distal** (± buffer)، الهدف **3R**. المحاكاة لحد TP (✅) / SL (❌) / timeout.")
    zones = result["zones"]
    labeled = [z for z in zones if z.get("label") is not None]
    if not labeled:
        st.warning("مفيش صفقات اتعملت.")
    else:
        opts = {
            f"{z['formation']} @ {df.index[z['end']].strftime('%Y-%m-%d')} "
            f"→ {z['exit_reason']} ({'WIN' if z['label'] == 1 else 'LOSS'})": i
            for i, z in enumerate(labeled)}
        choice = st.selectbox("اختار صفقة:", list(opts.keys()))
        z = labeled[opts[choice]]
        start_pos = max(0, z["start"] - 10)
        end_pos = min(len(df) - 1, (z.get("exit_bar") or z["end"]) + 10)
        tv = df.iloc[start_pos: end_pos + 1]
        fig = _base_candles(tv, highlight_base=False)
        color = COLOR_BULL if z["zone_type"] == "demand" else COLOR_BEAR
        _add_zone_box(fig, df, z, color, z["formation"], extend_bars=(end_pos - z["end"]))
        x0, x1 = df.index[z["start"]], df.index[end_pos]
        for level, name, dash, lc in [
            (z["entry"], "Entry", "solid", "#ffffff"),
            (z["stop"], "Stop", "dash", COLOR_BEAR),
            (z["tp"], "TP (3R)", "dash", COLOR_BULL)]:
            fig.add_shape(type="line", x0=x0, x1=x1, y0=level, y1=level,
                          line=dict(color=lc, width=1.5, dash=dash))
            fig.add_annotation(x=x1, y=level, text=name, showarrow=False,
                               font=dict(color=lc, size=11), xanchor="right", yanchor="bottom")
        if z.get("entry_bar") is not None:
            fig.add_trace(go.Scatter(x=[df.index[z["entry_bar"]]], y=[z["entry"]], mode="markers",
                                     marker=dict(symbol="circle", size=12, color="#ffffff"), name="ENTRY"))
        if z.get("exit_bar") is not None:
            ec = COLOR_BULL if z["exit_reason"] == "tp" else COLOR_BEAR
            fig.add_trace(go.Scatter(x=[df.index[z["exit_bar"]]],
                                     y=[z["tp"] if z["exit_reason"] == "tp" else z["stop"]],
                                     mode="markers", marker=dict(symbol="x", size=14, color=ec),
                                     name=f"EXIT ({z['exit_reason']})"))
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("الاتجاه", "🟢 LONG" if z["direction"] == "long" else "🔴 SHORT")
        c2.metric("النتيجة", z["exit_reason"].upper(), "WIN" if z["label"] == 1 else "LOSS")
        c3.metric("pnl (R)", f"{z.get('pnl_r', 0):.2f}")
        c4.metric("bars held", z.get("bars_held", 0))
        with st.expander("📊 الـ features للمنطقة دي"):
            st.json({
                "dep_ratio": z.get("dep_ratio"), "dep_atr": z.get("dep_atr"),
                "compactness_ratio": z.get("compactness_ratio"), "leg_strength": z.get("leg_strength"),
                "curve_third": z.get("curve_third"), "curve_pos": z.get("curve_pos"),
                "trend_aligned": z.get("trend_aligned"), "itf_trend": z.get("itf_trend_at_formation"),
                "htf_trend": z.get("htf_trend_at_formation"), "sets_total": z.get("sets_total")})

st.markdown("---")
st.caption("💡 ده الـ Scenario Player — بيشغّل نفس دوال build_dataset.py على داتا حقيقية. غيّر الـ step من الشريط الجانبي.")
