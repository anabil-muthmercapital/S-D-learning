#!/usr/bin/env python
# =============================================================================
# dashboard_pipeline.py — Interactive "watch the pipeline run" dashboard
# =============================================================================
# بيستورد نفس دوال build_dataset — اللي بتشوفه هنا = نفس اللي بيتحط في الـ dataset.
# التشغيل:  cd /Users/an/Desktop/S-D-learning && streamlit run dashboard_pipeline.py
# =============================================================================

from __future__ import annotations

import subprocess
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
from utils.feature_engine import FEATURE_COLS, build_features
from utils.freshness import add_freshness
from utils.htf_range import add_curve_score
from utils.labeler import DEFAULT_HTF_LTF_MAP, label_zones
from utils.legs_formation import detect_formations
from utils.regime import compute_regime_series
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
    "7. ML Decision",
]

MODEL_PATH = PROJECT_ROOT / "data" / "xgb_model.json"
ML_THRESHOLD = 0.52

# ---- Forward-tracking integration -------------------------------------
SIGNALS_CSV = PROJECT_ROOT / "data" / "forward_signals.csv"
FORWARD_START_FILE = PROJECT_ROOT / "data" / "forward_test_start.json"
FORWARD_TEST_SCRIPT = PROJECT_ROOT / "forward_test.py"
UPDATE_SIGNALS_SCRIPT = PROJECT_ROOT / "update_signals.py"
# Backtest baseline expectancy in R — every closed forward-trade is compared
# against this number to show whether live behaviour matches the historical edge.
BACKTEST_BASELINE_R: float = 0.42
SCRIPT_TIMEOUT_SEC = 1800


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
    feats = pd.DataFrame()
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

        regime_series = None
        if "1d" in data:
            try:
                regime_series = compute_regime_series(data["1d"])
            except Exception:  # noqa: BLE001
                regime_series = None
        try:
            feats = build_features(
                zones,
                ltf_df,
                symbol=symbol,
                asset_class=_asset_class(symbol),
                timeframe=ltf,
                regime_series=regime_series,
            )
        except Exception:  # noqa: BLE001
            feats = pd.DataFrame()
    return {
        "ltf_df": ltf_df,
        "all_clusters": all_clusters,
        "passed_clusters": passed_clusters,
        "failed_clusters": failed_clusters,
        "formations": formations,
        "zones": zones,
        "rejected": rejected,
        "features": feats,
    }


@st.cache_resource(show_spinner=False)
def load_xgb_model(path: str):
    import xgboost as xgb

    model = xgb.XGBClassifier()
    model.load_model(path)
    return model


def _base_candles(df: pd.DataFrame, highlight_base: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=COLOR_BULL,
            decreasing_line_color=COLOR_BEAR,
            name="Price",
        )
    )
    if highlight_base and "is_base" in df.columns:
        bb = df[df["is_base"]]
        if not bb.empty:
            # Anchor the triangle exactly at the candle high (no % offset).
            # The previous `high * 1.001` looked fine on Forex but on
            # high-priced or zoomed-in views the arrow floated far above the
            # wick. Marker size is in pixels → consistent across scales.
            fig.add_trace(
                go.Scatter(
                    x=bb.index,
                    y=bb["high"],
                    mode="markers",
                    marker=dict(
                        symbol="triangle-down",
                        size=9,
                        color=COLOR_DOJI_SAFE,
                        line=dict(width=0),
                    ),
                    name="base candle",
                    hoverinfo="skip",
                    cliponaxis=False,
                )
            )
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color="#d1d4dc"),
        xaxis=dict(gridcolor=CHART_GRID, rangeslider=dict(visible=False)),
        yaxis=dict(gridcolor=CHART_GRID),
        height=620,
        margin=dict(l=10, r=10, t=30, b=10),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    return fig


def _add_zone_box(fig, df, zone, color, label, extend_bars=30):
    bs, be = zone["start"], zone["end"]
    x0 = df.index[bs]
    end_pos = min(be + extend_bars, len(df) - 1)
    x1 = df.index[end_pos]
    prox, dist = zone["proximal"], zone["distal"]
    top, bot = max(prox, dist), min(prox, dist)
    fig.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=bot,
        y1=top,
        line=dict(color=color, width=1),
        fillcolor=color,
        opacity=0.15,
        layer="below",
    )
    fig.add_shape(
        type="line", x0=x0, x1=x1, y0=prox, y1=prox, line=dict(color=color, width=2)
    )
    fig.add_shape(
        type="line",
        x0=x0,
        x1=x1,
        y0=dist,
        y1=dist,
        line=dict(color=color, width=1, dash="dash"),
    )
    fig.add_annotation(
        x=x0,
        y=top,
        text=label,
        showarrow=False,
        font=dict(color=color, size=11),
        xanchor="left",
        yanchor="bottom",
    )


# =============================================================================
# Forward Live tracking — integration with forward_test.py / update_signals.py
# =============================================================================


def _run_script(script_path: Path, args: list[str], label: str) -> None:
    """Invoke a sibling script as a subprocess and surface stdout/stderr.

    The dashboard never imports the script as a module because the scripts
    do their own argparse / printing and are designed to be standalone.
    We capture output so the user sees exactly what they would in a shell.
    """
    if not script_path.exists():
        st.error(f"الـ script مش موجود: {script_path}")
        return
    cmd = [sys.executable, str(script_path), *args]
    with st.spinner(f"بشغّل {label} ..."):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(PROJECT_ROOT),
                timeout=SCRIPT_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            st.error(f"⏱️ {label} تخطّى الـ timeout ({SCRIPT_TIMEOUT_SEC}s).")
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"فشل تشغيل {label}: {type(exc).__name__}: {exc}")
            return
    # Bust the @st.cache_data on _load_forward_signals so the table refreshes.
    _load_forward_signals.clear()
    with st.expander(f"📋 خرج {label}", expanded=(result.returncode != 0)):
        if result.returncode == 0:
            st.success(f"✓ تمّت بنجاح (exit {result.returncode})")
        else:
            st.error(f"✗ فشل (exit {result.returncode})")
        if result.stdout:
            st.code(result.stdout, language="text")
        if result.stderr:
            st.code(result.stderr, language="text")


@st.cache_data(show_spinner=False, ttl=15)
def _load_forward_signals() -> pd.DataFrame:
    """Read forward_signals.csv with a short TTL so it auto-refreshes.

    Returns an empty DataFrame when the file is missing — the caller
    decides how to render that state.
    """
    if not SIGNALS_CSV.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(SIGNALS_CSV)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    # Parse timestamps once (kept as object for empty strings).
    for col in ("formation_time", "entry_time", "exit_time", "timestamp_detected"):
        if col in df.columns:
            df[col + "_dt"] = pd.to_datetime(df[col], utc=True, errors="coerce")
    for col in ("pnl_r", "model_prob", "timeout_pnl_r"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _render_signal_chart(row: pd.Series) -> None:
    """Plot the signal with the SAME visual contract as Step 6 (Labeling).

    Visual elements (in order):
      * candlesticks around the base
      * base candles flagged (down-triangle markers) so the user sees
        exactly which bars formed the zone
      * coloured rectangle from base_start → end_of_view, between
        proximal (entry) and distal — proximal is the SOLID line,
        distal is the DASHED line, matching _add_zone_box used by
        the Labeling chart
      * formation label (RBD / DBR / RBR / DBD) anchored on the box
      * Entry / Stop / TP horizontal lines across the full view
      * Entry / Exit markers if the trade has progressed
    """
    try:
        data = load_enriched_timeframes(row["symbol"], timeframes=[row["timeframe"]])
        ltf_df = data[row["timeframe"]]
    except Exception as exc:  # noqa: BLE001
        st.error(f"فشل تحميل الـ data: {type(exc).__name__}: {exc}")
        return

    formation_ts = pd.to_datetime(row["formation_time"], utc=True, errors="coerce")
    if pd.isna(formation_ts):
        st.warning("formation_time غير صالح.")
        return
    end_pos_base = int(ltf_df.index.searchsorted(formation_ts, side="left"))
    if end_pos_base >= len(ltf_df) or ltf_df.index[end_pos_base] != formation_ts:
        st.warning("الـ formation bar مش موجود في الـ data المحمّلة.")
        return

    # ---- locate base START -------------------------------------------------
    # New schema has base_start_time. Fallback for legacy rows: assume base
    # was a single candle (start == end). Either way we get a valid iloc.
    base_start_ts_raw = row.get("base_start_time", "")
    if pd.notna(base_start_ts_raw) and str(base_start_ts_raw):
        base_start_ts = pd.to_datetime(base_start_ts_raw, utc=True, errors="coerce")
        start_pos_base = int(ltf_df.index.searchsorted(base_start_ts, side="left"))
        if (
            start_pos_base >= len(ltf_df)
            or ltf_df.index[start_pos_base] != base_start_ts
        ):
            start_pos_base = end_pos_base  # safety fallback
    else:
        start_pos_base = end_pos_base

    # ---- viewport: 10 bars before base, extend past exit (if any) ----------
    view_end = end_pos_base + 50
    exit_ts_raw = row.get("exit_time", "")
    if pd.notna(exit_ts_raw) and str(exit_ts_raw):
        exit_ts = pd.to_datetime(exit_ts_raw, utc=True, errors="coerce")
        if pd.notna(exit_ts):
            exit_pos = int(ltf_df.index.searchsorted(exit_ts, side="right"))
            view_end = max(view_end, exit_pos + 10)
    view_start = max(0, start_pos_base - 10)
    view_end = min(len(ltf_df) - 1, view_end)

    # ---- build a small zone dict so we can reuse _add_zone_box -------------
    # _add_zone_box wants {start, end, proximal, distal} in iloc terms,
    # which is exactly what the Labeling page passes too.
    distal_raw = row.get("distal", None)
    distal_val = (
        float(distal_raw)
        if distal_raw not in (None, "") and pd.notna(distal_raw)
        else float(row["stop"])
    )
    zone_for_box = {
        "start": start_pos_base,
        "end": end_pos_base,
        "proximal": float(row["entry"]),  # entry == proximal by construction
        "distal": distal_val,
        "zone_type": str(row["zone_type"]),
    }

    # Slice candles and mark base bars so highlight_base lights them up.
    tv = ltf_df.iloc[view_start : view_end + 1].copy()
    tv["is_base"] = False
    base_mask_global = slice(start_pos_base, end_pos_base + 1)
    tv.loc[ltf_df.index[base_mask_global], "is_base"] = True

    fig = _base_candles(tv, highlight_base=True)
    color = COLOR_BULL if zone_for_box["zone_type"] == "demand" else COLOR_BEAR
    formation_label = str(row.get("formation", "") or zone_for_box["zone_type"])
    _add_zone_box(
        fig,
        ltf_df,
        zone_for_box,
        color,
        formation_label,
        extend_bars=(view_end - end_pos_base),
    )

    # ---- Entry / Stop / TP horizontal lines across the full view ----------
    entry = float(row["entry"])
    stop = float(row["stop"])
    tp = float(row["tp"])
    x0, x1 = tv.index[0], tv.index[-1]
    for level, name, dash, lc in [
        (entry, "Entry", "solid", "#ffffff"),
        (stop, "Stop", "dash", COLOR_BEAR),
        (tp, "TP (3R)", "dash", COLOR_BULL),
    ]:
        fig.add_shape(
            type="line",
            x0=x0,
            x1=x1,
            y0=level,
            y1=level,
            line=dict(color=lc, width=1.5, dash=dash),
        )
        fig.add_annotation(
            x=x1,
            y=level,
            text=name,
            showarrow=False,
            font=dict(color=lc, size=11),
            xanchor="right",
            yanchor="bottom",
        )

    # ---- Entry marker (white dot) -----------------------------------------
    entry_ts_raw = row.get("entry_time", "")
    if pd.notna(entry_ts_raw) and str(entry_ts_raw):
        et = pd.to_datetime(entry_ts_raw, utc=True, errors="coerce")
        if pd.notna(et):
            fig.add_trace(
                go.Scatter(
                    x=[et],
                    y=[entry],
                    mode="markers",
                    marker=dict(symbol="circle", size=12, color="#ffffff"),
                    name="ENTRY",
                )
            )

    # ---- Exit marker (X, colour by reason) --------------------------------
    if pd.notna(exit_ts_raw) and str(exit_ts_raw):
        xt = pd.to_datetime(exit_ts_raw, utc=True, errors="coerce")
        if pd.notna(xt):
            reason = str(row.get("exit_reason", "") or "")
            exit_color = (
                COLOR_BULL
                if reason == "tp"
                else (COLOR_BEAR if reason in ("sl", "death") else "#aaaaaa")
            )
            exit_price = tp if reason == "tp" else (stop if reason == "sl" else entry)
            fig.add_trace(
                go.Scatter(
                    x=[xt],
                    y=[exit_price],
                    mode="markers",
                    marker=dict(symbol="x", size=14, color=exit_color),
                    name=f"EXIT ({reason or '?'})",
                )
            )

    st.plotly_chart(fig, use_container_width=True)


def render_forward_live() -> None:
    """Self-contained Forward Live page — runs scripts, shows live record."""
    st.title("📡 OTA S&D — Forward Live Tracking")
    st.caption(
        "سجل الـ signals الحية اللي بيطلعها forward_test.py وبيتحدّث بـ update_signals.py."
    )

    # ---- Run buttons --------------------------------------------------------
    bc1, bc2, bc3, bc4 = st.columns([1.1, 1.1, 1, 1])
    no_refresh = bc4.checkbox(
        "no-refresh",
        value=False,
        help="تخطّى الـ yfinance re-pull. مفيد لو الـ data محدّثة من شويّة.",
    )
    extra = ["--no-refresh"] if no_refresh else []
    if bc1.button("▶️ تشغيل forward_test", use_container_width=True):
        _run_script(FORWARD_TEST_SCRIPT, extra, "forward_test.py")
    if bc2.button("🔄 تحديث الـ signals", use_container_width=True):
        _run_script(UPDATE_SIGNALS_SCRIPT, extra, "update_signals.py")
    if bc3.button("🧹 إعادة تحميل الـ CSV", use_container_width=True):
        _load_forward_signals.clear()
        st.rerun()

    # ---- Load signals -------------------------------------------------------
    df = _load_forward_signals()
    if df.empty:
        st.info(
            "مفيش signals بعد. اضغط **▶️ تشغيل forward_test** عشان تطلّع أول دفعة، "
            "أو لو لسه ما اشتغلش حد، شغّل من الـ terminal:\n\n"
            "```bash\npython forward_test.py\n```"
        )
        return

    # ---- Headline metrics ---------------------------------------------------
    status = df["status"].astype(str)
    closed = df[status == "closed"].copy()
    closed_valid = closed.dropna(subset=["pnl_r"])
    n_closed = len(closed_valid)
    n_pending = int((status == "pending").sum())
    n_open = int((status == "open").sum())
    n_expired = int((status == "expired_no_entry").sum())

    if n_closed > 0:
        win_rate = float((closed_valid["pnl_r"] > 0).mean() * 100)
        total_r = float(closed_valid["pnl_r"].sum())
        avg_r = float(closed_valid["pnl_r"].mean())
    else:
        win_rate = total_r = avg_r = 0.0

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Closed", n_closed)
    mc2.metric("Win Rate", f"{win_rate:.1f}%")
    mc3.metric("Total R", f"{total_r:+.2f}")
    mc4.metric(
        "Avg R / trade",
        f"{avg_r:+.3f}",
        delta=(
            f"{avg_r - BACKTEST_BASELINE_R:+.3f} vs baseline (+{BACKTEST_BASELINE_R:.2f})"
            if n_closed > 0
            else None
        ),
    )
    mc5.metric("Pending / Open / Expired", f"{n_pending} / {n_open} / {n_expired}")

    if FORWARD_START_FILE.exists():
        try:
            import json

            payload = json.loads(FORWARD_START_FILE.read_text())
            start_date = pd.to_datetime(payload["forward_test_start"]).date()
            st.caption(
                f"⏱ forward-test start: **{start_date}** "
                f"(محفوظة في {FORWARD_START_FILE.name})"
            )
        except Exception:  # noqa: BLE001
            pass

    # ---- Throughput pacing --------------------------------------------------
    # The user looks at "1 signal in 24h" and worries the system is broken.
    # It usually isn't — it's just early. This widget projects the current
    # rate forward so they can see how long it takes to gather a meaningful
    # sample (rule of thumb: ~30 closed trades for the avg-R metric to
    # stabilise, ~100 for confidence the live edge matches +0.42R baseline).
    if FORWARD_START_FILE.exists():
        try:
            import json

            payload = json.loads(FORWARD_START_FILE.read_text())
            start_ts = pd.to_datetime(payload["forward_test_start"], utc=True)
            elapsed_h = max(
                1.0,
                (pd.Timestamp.now(tz="UTC") - start_ts).total_seconds() / 3600.0,
            )
            elapsed_days = elapsed_h / 24.0
            n_total = len(df)
            signals_per_day = n_total / elapsed_days if elapsed_days > 0 else 0.0
            closed_per_day = (
                n_closed / elapsed_days if (elapsed_days > 0 and n_closed > 0) else 0.0
            )
            target_closed = 30  # heuristic sample-size threshold
            if closed_per_day > 0:
                days_to_target = max(0.0, (target_closed - n_closed) / closed_per_day)
            else:
                days_to_target = float("inf")

            with st.expander("📐 Throughput pacing — هل المعدل طبيعي؟", expanded=True):
                p1, p2, p3, p4 = st.columns(4)
                p1.metric(
                    "Elapsed",
                    f"{elapsed_h:.0f}h",
                    delta=f"~{elapsed_days:.1f} يوم",
                    delta_color="off",
                )
                p2.metric(
                    "Signals/day",
                    f"{signals_per_day:.2f}",
                    delta=f"{n_total} إجمالي",
                    delta_color="off",
                )
                p3.metric(
                    "Closed/day",
                    f"{closed_per_day:.2f}",
                    delta=f"{n_closed} closed",
                    delta_color="off",
                )
                if days_to_target == float("inf"):
                    p4.metric("⏳ to 30 closed", "—", delta="استنى أول صفقة")
                else:
                    p4.metric(
                        f"⏳ to {target_closed} closed",
                        f"~{days_to_target:.0f} يوم",
                    )
                st.caption(
                    "💡 الـ pipeline بيـ scan 50 رمز × 3 timeframes كل ساعة. "
                    "الـ ML threshold = 0.52 بيرفض 70-85% من الـ candidates "
                    "(ده الـ feature، مش bug). صبر أسبوع للوصول لـ ~15-30 signal "
                    "وشهر للوصول لـ sample إحصائي يقاس بـ +0.42R baseline."
                )
        except Exception:  # noqa: BLE001
            pass

    st.markdown("---")

    # ---- Equity curve -------------------------------------------------------
    st.markdown("### 📈 Equity Curve — Cumulative R")
    if n_closed == 0:
        st.info("لسه مفيش صفقات مقفولة (closed) عشان نرسم equity curve.")
    else:
        eq = closed_valid.copy()
        eq["exit_time_dt"] = pd.to_datetime(eq["exit_time"], utc=True, errors="coerce")
        eq = eq.dropna(subset=["exit_time_dt"]).sort_values("exit_time_dt")
        eq["cumulative_r"] = eq["pnl_r"].cumsum()
        fig_eq = go.Figure()
        fig_eq.add_trace(
            go.Scatter(
                x=eq["exit_time_dt"],
                y=eq["cumulative_r"],
                mode="lines+markers",
                line=dict(color=COLOR_BULL, width=2),
                marker=dict(size=6),
                name="cumulative R",
            )
        )
        # Backtest-baseline reference line (linear baseline of n * 0.42R)
        baseline_y = [BACKTEST_BASELINE_R * (i + 1) for i in range(len(eq))]
        fig_eq.add_trace(
            go.Scatter(
                x=eq["exit_time_dt"],
                y=baseline_y,
                mode="lines",
                line=dict(color="#888888", width=1, dash="dash"),
                name=f"backtest baseline ({BACKTEST_BASELINE_R:.2f}R / trade)",
            )
        )
        fig_eq.add_hline(y=0, line=dict(color="#444444", width=1))
        fig_eq.update_layout(
            paper_bgcolor=CHART_BG,
            plot_bgcolor=CHART_BG,
            font=dict(color="#d1d4dc"),
            xaxis=dict(gridcolor=CHART_GRID),
            yaxis=dict(gridcolor=CHART_GRID, title="Cumulative R"),
            height=380,
            margin=dict(l=10, r=10, t=30, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_eq, use_container_width=True)

    st.markdown("---")

    # ---- Filters + table ----------------------------------------------------
    st.markdown("### 🔍 الـ Signals")
    fc1, fc2, fc3 = st.columns(3)
    sym_opts = ["(all)"] + sorted(df["symbol"].astype(str).unique().tolist())
    tf_opts = ["(all)"] + sorted(df["timeframe"].astype(str).unique().tolist())
    status_opts = ["(all)"] + sorted(df["status"].astype(str).unique().tolist())
    sel_sym = fc1.selectbox("Symbol", sym_opts)
    sel_tf = fc2.selectbox("Timeframe", tf_opts)
    sel_status = fc3.selectbox("Status", status_opts)

    filt = df.copy()
    if sel_sym != "(all)":
        filt = filt[filt["symbol"].astype(str) == sel_sym]
    if sel_tf != "(all)":
        filt = filt[filt["timeframe"].astype(str) == sel_tf]
    if sel_status != "(all)":
        filt = filt[filt["status"].astype(str) == sel_status]

    show_cols = [
        c
        for c in [
            "symbol",
            "timeframe",
            "zone_type",
            "direction",
            "formation",
            "formation_time",
            "entry",
            "stop",
            "tp",
            "model_prob",
            "status",
            "entry_time",
            "exit_time",
            "exit_reason",
            "pnl_r",
            "bars_held",
        ]
        if c in filt.columns
    ]
    st.dataframe(
        filt[show_cols].reset_index(drop=True),
        use_container_width=True,
        height=380,
    )

    # ---- Per-signal chart ---------------------------------------------------
    st.markdown("### 🔎 رسم صفقة بعينها")
    if filt.empty:
        st.info("مفيش صفقات بعد الفلاتر دي.")
        return

    def _label(r: pd.Series) -> str:
        st_str = str(r.get("status", ""))
        outcome = ""
        if st_str == "closed" and pd.notna(r.get("pnl_r")):
            outcome = f" → {r.get('exit_reason', '')} ({float(r['pnl_r']):+.2f}R)"
        elif st_str == "expired_no_entry":
            outcome = " → expired"
        elif st_str == "open":
            outcome = " → OPEN"
        return (
            f"{r['symbol']} {r['timeframe']} · {r['formation']} · "
            f"{r['formation_time']}{outcome}"
        )

    labels = [_label(r) for _, r in filt.iterrows()]
    sel = st.selectbox("اختار صفقة:", labels, index=0)
    row = filt.iloc[labels.index(sel)]
    _render_signal_chart(row)


# =============================================================================
# Mode selector (top of sidebar) — switch between Scenario Player & Forward Live
# =============================================================================

st.sidebar.title("🧭 Mode")
MODES = ["🎬 Scenario Player", "📡 Forward Live"]
mode = st.sidebar.radio("Choose view:", MODES, index=0, label_visibility="collapsed")
st.sidebar.markdown("---")

if mode == "📡 Forward Live":
    render_forward_live()
    st.stop()


st.sidebar.title("⚙️ Controls")
all_symbols = [s for syms in WATCHLIST.values() for s in syms]
default_sym = all_symbols.index("USDJPY=X") if "USDJPY=X" in all_symbols else 0
symbol = st.sidebar.selectbox("Symbol", all_symbols, index=default_sym)
ltf = st.sidebar.selectbox("Timeframe", LTFS, index=2)
st.sidebar.markdown("---")
st.sidebar.markdown("### Pipeline step")
step = st.sidebar.radio(
    "Show up to:", PIPELINE_STEPS, index=0, label_visibility="collapsed"
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "كل خطوة بتبني على اللي قبلها — نفس ترتيب build_dataset.py. اللي بتشوفه = نفس اللي بيتحط في الـ dataset."
)

st.title("📈 OTA S&D — الـ Pipeline بيشتغل قدامك")
st.caption(f"**{symbol}** · **{ltf}** · asset class: **{_asset_class(symbol)}**")

try:
    with st.spinner(f"بشغّل الـ pipeline على {symbol} {ltf} ..."):
        result = run_pipeline(symbol, ltf)
except Exception as exc:  # noqa: BLE001
    st.error(f"فشل تشغيل الـ pipeline: {type(exc).__name__}: {exc}")
    st.info(
        f"اتأكد إن الرمز عنده CSVs في data/raw/ والفريمات المطلوبة موجودة ({sorted(REQUIRED_TFS[ltf])})."
    )
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
    st.markdown(
        "كل شمعة جسمها صغير (body_ratio ≤ 0.5) بتتعلّم كـ **base candle** (المثلثات الصفرا)."
    )
    st.plotly_chart(_base_candles(view, highlight_base=True), use_container_width=True)
    n_base = int(view["is_base"].sum()) if "is_base" in view.columns else 0
    st.metric("شموع base في آخر 250 شمعة", n_base)

elif step_idx == 1:
    st.subheader("2️⃣ كشف الـ Base Clusters")
    st.markdown(
        "الـ base candles المتراصة بتتجمّع في **clusters** بتعدّي **min_count** و **compactness** (≤ 2.5 ATR). 🟢 عدّى · 🔴 اترفض."
    )
    fig = _base_candles(view, highlight_base=False)
    vstart = len(df) - len(view)
    p_n = f_n = 0
    for c in result["passed_clusters"]:
        if c["end"] >= vstart:
            sub = df.iloc[c["start"] : c["end"] + 1]
            fig.add_shape(
                type="rect",
                x0=df.index[c["start"]],
                x1=df.index[c["end"]],
                y0=sub["low"].min(),
                y1=sub["high"].max(),
                line=dict(color=COLOR_BULL, width=1.5),
                fillcolor=COLOR_BULL,
                opacity=0.2,
            )
            p_n += 1
    for c in result["failed_clusters"]:
        if c["end"] >= vstart:
            sub = df.iloc[c["start"] : c["end"] + 1]
            fig.add_shape(
                type="rect",
                x0=df.index[c["start"]],
                x1=df.index[c["end"]],
                y0=sub["low"].min(),
                y1=sub["high"].max(),
                line=dict(color=COLOR_BEAR, width=1, dash="dot"),
                fillcolor=COLOR_BEAR,
                opacity=0.1,
            )
            f_n += 1
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 عدّت (عرض)", p_n)
    c2.metric("🔴 اترفضت (عرض)", f_n)
    c3.metric("إجمالي عدّت", len(result["passed_clusters"]))

elif step_idx == 2:
    st.subheader("3️⃣ الـ Formations — اتجاه المنطقة")
    st.markdown(
        "بنقيس الـ **leg-out**: لأعلى → 🟢 **Demand**، لأسفل → 🔴 **Supply**. اللي فشل في leg_strength أو clean_departure بيترفض."
    )
    fig = _base_candles(view, highlight_base=False)
    vstart = len(df) - len(view)
    d_n = s_n = 0
    for f in result["formations"]:
        if f["end"] < vstart:
            continue
        color = COLOR_BULL if f["zone_type"] == "demand" else COLOR_BEAR
        sub = df.iloc[f["start"] : f["end"] + 1]
        fig.add_shape(
            type="rect",
            x0=df.index[f["start"]],
            x1=df.index[f["end"]],
            y0=sub["low"].min(),
            y1=sub["high"].max(),
            line=dict(color=color, width=1.5),
            fillcolor=color,
            opacity=0.2,
        )
        fig.add_annotation(
            x=df.index[f["start"]],
            y=sub["high"].max(),
            text=f["formation"],
            showarrow=False,
            font=dict(color=color, size=10),
            xanchor="left",
            yanchor="bottom",
        )
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
    st.markdown(
        "اللي عدّى **departure gates** (dep_ratio ≥ 2.0 و dep_atr ≥ 0.5) بقى zone. خط متصل = **proximal** (دخول)، متقطّع = **distal** (ستوب). المرفوضة باهتة."
    )
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
        sub = df.iloc[z["start"] : z["end"] + 1]
        fig.add_shape(
            type="rect",
            x0=df.index[z["start"]],
            x1=df.index[z["end"]],
            y0=sub["low"].min(),
            y1=sub["high"].max(),
            line=dict(color="#666", width=1, dash="dot"),
            fillcolor="#666",
            opacity=0.08,
        )
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("✅ صالحة", len(result["zones"]))
    c2.metric("❌ مرفوضة", len(result["rejected"]))

elif step_idx == 4:
    st.subheader("5️⃣ التقييم — الـ 5 معايير (SETS)")
    st.markdown(
        "نقاط: **strength · freshness · time · curve · trend**. المجموع → ★★★ (≥7) / ★★ (≥5) / ★."
    )
    zones = result["zones"]
    if not zones:
        st.warning("مفيش zones.")
    else:
        rows = []
        for z in zones:
            total = z.get("sets_total", 0)
            stars = "★★★" if total >= 7 else ("★★" if total >= 5 else "★")
            rows.append(
                {
                    "formation": z["formation"],
                    "type": z["zone_type"],
                    "strength": z.get("strength_score", 0),
                    "freshness": z.get("freshness_score", 0),
                    "time": z.get("time_score", 0),
                    "curve": z.get("curve_score", 0),
                    "trend": z.get("trend_score", 0),
                    "SETS total": total,
                    "rating": stars,
                    "formation_time": df.index[z["end"]].strftime("%Y-%m-%d %H:%M"),
                }
            )
        sdf = pd.DataFrame(rows).sort_values("SETS total", ascending=False)
        st.dataframe(sdf, use_container_width=True, height=420)
        c1, c2, c3 = st.columns(3)
        c1.metric("★★★", int((sdf["SETS total"] >= 7).sum()))
        c2.metric("★★", int(((sdf["SETS total"] >= 5) & (sdf["SETS total"] < 7)).sum()))
        c3.metric("★", int((sdf["SETS total"] < 5).sum()))

elif step_idx == 5:
    st.subheader("6️⃣ الـ Labeling — الصفقة من الدخول للنتيجة")
    st.markdown(
        "الدخول عند **proximal**، الستوب عند **distal** (± buffer)، الهدف **3R**. المحاكاة لحد TP (✅) / SL (❌) / timeout."
    )
    zones = result["zones"]
    labeled = [z for z in zones if z.get("label") is not None]
    if not labeled:
        st.warning("مفيش صفقات اتعملت.")
    else:
        opts = {
            f"{z['formation']} @ {df.index[z['end']].strftime('%Y-%m-%d')} "
            f"→ {z['exit_reason']} ({'WIN' if z['label'] == 1 else 'LOSS'})": i
            for i, z in enumerate(labeled)
        }
        choice = st.selectbox("اختار صفقة:", list(opts.keys()))
        z = labeled[opts[choice]]
        start_pos = max(0, z["start"] - 10)
        end_pos = min(len(df) - 1, (z.get("exit_bar") or z["end"]) + 10)
        tv = df.iloc[start_pos : end_pos + 1]
        fig = _base_candles(tv, highlight_base=False)
        color = COLOR_BULL if z["zone_type"] == "demand" else COLOR_BEAR
        _add_zone_box(
            fig, df, z, color, z["formation"], extend_bars=(end_pos - z["end"])
        )
        x0, x1 = df.index[z["start"]], df.index[end_pos]
        for level, name, dash, lc in [
            (z["entry"], "Entry", "solid", "#ffffff"),
            (z["stop"], "Stop", "dash", COLOR_BEAR),
            (z["tp"], "TP (3R)", "dash", COLOR_BULL),
        ]:
            fig.add_shape(
                type="line",
                x0=x0,
                x1=x1,
                y0=level,
                y1=level,
                line=dict(color=lc, width=1.5, dash=dash),
            )
            fig.add_annotation(
                x=x1,
                y=level,
                text=name,
                showarrow=False,
                font=dict(color=lc, size=11),
                xanchor="right",
                yanchor="bottom",
            )
        if z.get("entry_bar") is not None:
            fig.add_trace(
                go.Scatter(
                    x=[df.index[z["entry_bar"]]],
                    y=[z["entry"]],
                    mode="markers",
                    marker=dict(symbol="circle", size=12, color="#ffffff"),
                    name="ENTRY",
                )
            )
        if z.get("exit_bar") is not None:
            ec = COLOR_BULL if z["exit_reason"] == "tp" else COLOR_BEAR
            fig.add_trace(
                go.Scatter(
                    x=[df.index[z["exit_bar"]]],
                    y=[z["tp"] if z["exit_reason"] == "tp" else z["stop"]],
                    mode="markers",
                    marker=dict(symbol="x", size=14, color=ec),
                    name=f"EXIT ({z['exit_reason']})",
                )
            )
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("الاتجاه", "🟢 LONG" if z["direction"] == "long" else "🔴 SHORT")
        c2.metric(
            "النتيجة", z["exit_reason"].upper(), "WIN" if z["label"] == 1 else "LOSS"
        )
        c3.metric("pnl (R)", f"{z.get('pnl_r', 0):.2f}")
        c4.metric("bars held", z.get("bars_held", 0))
        with st.expander("📊 الـ features للمنطقة دي"):
            st.json(
                {
                    "dep_ratio": z.get("dep_ratio"),
                    "dep_atr": z.get("dep_atr"),
                    "compactness_ratio": z.get("compactness_ratio"),
                    "leg_strength": z.get("leg_strength"),
                    "curve_third": z.get("curve_third"),
                    "curve_pos": z.get("curve_pos"),
                    "trend_aligned": z.get("trend_aligned"),
                    "itf_trend": z.get("itf_trend_at_formation"),
                    "htf_trend": z.get("htf_trend_at_formation"),
                    "sets_total": z.get("sets_total"),
                }
            )

elif step_idx == 6:
    st.subheader("7️⃣ قرار الـ ML — XGBoost بيقول TAKE ولا SKIP")
    st.markdown(
        f"الموديل المُجمَّد بيشوف الـ 25 feature بتاعت كل zone وبيطلع احتمالية ربح. "
        f"لو **prob ≥ {ML_THRESHOLD:.2f}** → **TAKE** · غير كده → **SKIP**. "
        f"بنقارن قراره بالنتيجة الفعلية من خطوة 6 عشان نشوف بيفلتر صح ولا لأ."
    )
    feats = result.get("features", pd.DataFrame())
    if feats is None or feats.empty:
        st.warning("مفيش features (لازم تبقى فيه labeled zones من خطوة 6).")
    elif not MODEL_PATH.exists():
        st.error(f"الموديل مش موجود في {MODEL_PATH}. شغّل train_model.py الأول.")
    else:
        try:
            model = load_xgb_model(str(MODEL_PATH))
        except Exception as exc:  # noqa: BLE001
            st.error(f"فشل تحميل الموديل: {type(exc).__name__}: {exc}")
        else:
            X = feats[FEATURE_COLS]
            probs = model.predict_proba(X)[:, 1]
            decisions = ["TAKE" if p >= ML_THRESHOLD else "SKIP" for p in probs]
            labels = feats["label"].astype(int).tolist()
            outcomes = ["WIN" if y == 1 else "LOSS" for y in labels]

            correct = []
            for d, y in zip(decisions, labels):
                if d == "TAKE":
                    correct.append("✅" if y == 1 else "❌")
                else:
                    correct.append("✅" if y == 0 else "❌")

            ft = feats["formation_time"]
            if not pd.api.types.is_datetime64_any_dtype(ft):
                ft = pd.to_datetime(ft, errors="coerce")
            zone_types = (
                feats["zone_type"].tolist()
                if "zone_type" in feats.columns
                else [""] * len(feats)
            )
            formation_names = [
                z["formation"] for z in result["zones"] if z.get("label") is not None
            ]
            if len(formation_names) != len(feats):
                formation_names = ["-"] * len(feats)

            table = pd.DataFrame(
                {
                    "formation": formation_names,
                    "type": zone_types,
                    "formation_time": ft.dt.strftime("%Y-%m-%d %H:%M"),
                    "model_prob": [round(float(p), 4) for p in probs],
                    "decision": decisions,
                    "actual": outcomes,
                    "correct?": correct,
                }
            ).sort_values("model_prob", ascending=False)
            st.dataframe(table, use_container_width=True, height=420)

            take_mask = pd.Series([d == "TAKE" for d in decisions])
            skip_mask = ~take_mask
            n_take = int(take_mask.sum())
            n_skip = int(skip_mask.sum())
            y_arr = pd.Series(labels)
            take_win_rate = float(y_arr[take_mask].mean() * 100) if n_take else 0.0
            skip_loss_rate = (
                float((1 - y_arr[skip_mask]).mean() * 100) if n_skip else 0.0
            )
            baseline = float(y_arr.mean() * 100) if len(y_arr) else 0.0

            st.markdown("#### 📊 قيمة الفلترة — confusion summary")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("إجمالي صفقات", len(feats))
            c2.metric("baseline win-rate", f"{baseline:.1f}%")
            c3.metric(
                f"TAKE → فعلاً WIN ({n_take})",
                f"{take_win_rate:.1f}%",
                delta=f"{take_win_rate - baseline:+.1f}% vs baseline",
            )
            c4.metric(
                f"SKIP → فعلاً LOSS ({n_skip})",
                f"{skip_loss_rate:.1f}%",
            )

            tp = int(((take_mask) & (y_arr == 1)).sum())
            fp = int(((take_mask) & (y_arr == 0)).sum())
            tn = int(((skip_mask) & (y_arr == 0)).sum())
            fn = int(((skip_mask) & (y_arr == 1)).sum())
            cm = pd.DataFrame(
                [[tp, fp], [fn, tn]],
                index=["actual WIN", "actual LOSS"],
                columns=["model TAKE", "model SKIP"],
            )
            with st.expander("🔢 confusion matrix (raw counts)"):
                st.dataframe(cm, use_container_width=True)
            st.caption(
                "الموديل التزم بنفس threshold = 0.52 المُستخدم في الـ backtest. "
                "لو TAKE win-rate أعلى من baseline → الفلترة بتضيف قيمة."
            )

st.markdown("---")
st.caption(
    "💡 ده الـ Scenario Player — بيشغّل نفس دوال build_dataset.py على داتا حقيقية. غيّر الـ step من الشريط الجانبي."
)
