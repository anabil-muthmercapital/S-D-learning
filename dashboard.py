"""
dashboard.py — Pre-labeling visual validation tool for the S&D pipeline.

This Streamlit app runs the full Origin-To-Algo zone detection chain end-to-end
for one (symbol, timeframe) pair and exposes EVERY intermediate result so the
detection logic can be inspected visually and numerically before any ML
labeling work begins.

Run:
    streamlit run dashboard.py

No labels are generated here — this is purely an inspection / validation
dashboard for detection + scoring output.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.config import (
    ALL_SYMBOLS,
    CHART_BG,
    CHART_GRID,
    COLOR_BEAR,
    COLOR_BULL,
    DEFAULT_TIMEFRAMES,
    DEPARTURE_ATR_MIN,
    DEPARTURE_RATIO_MIN,
    HTF_REF,
)
from utils.base_detector import detect_bases
from utils.data_loader import load_enriched_timeframes
from utils.freshness import add_freshness, find_death_bar
from utils.htf_range import add_curve_score
from utils.legs_formation import FORMATION_MAP, detect_formations, measure_legs
from utils.nested_zones import merge_zones
from utils.sets_scoring import add_sets_score
from utils.time_scoring import add_time_score
from utils.trend_alignment import add_trend_score
from utils.zone_detector import detect_zones

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="S&D Pipeline — Validation Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark TradingView palette (mirrors notebooks/07_verify_all_scenarios.ipynb)
# ---------------------------------------------------------------------------

# Zone fills: notebook uses ~0.15 for passed, ~0.06 for rejected. Faded
# variants used when the zone has already died (matches lifespan logic).
DEMAND_FILL_PASSED_ALIVE = "rgba(38, 166, 154, 0.18)"
DEMAND_FILL_PASSED_DEAD = "rgba(38, 166, 154, 0.07)"
DEMAND_FILL_REJ_ALIVE = "rgba(38, 166, 154, 0.07)"
DEMAND_FILL_REJ_DEAD = "rgba(38, 166, 154, 0.03)"
DEMAND_EDGE = "#26a69a"

SUPPLY_FILL_PASSED_ALIVE = "rgba(239, 83, 80, 0.18)"
SUPPLY_FILL_PASSED_DEAD = "rgba(239, 83, 80, 0.07)"
SUPPLY_FILL_REJ_ALIVE = "rgba(239, 83, 80, 0.07)"
SUPPLY_FILL_REJ_DEAD = "rgba(239, 83, 80, 0.03)"
SUPPLY_EDGE = "#ef5350"

FAILED_BASE_FILL = "rgba(176, 190, 197, 0.10)"
FAILED_BASE_LINE = "rgba(176, 190, 197, 0.55)"
DEATH_LINE = "rgba(255, 255, 255, 0.55)"

RATING_RANK = {"★": 1, "★★": 2, "★★★": 3}
RATING_FILTERS = {"All": 1, "★★ or better": 2, "★★★ only": 3}


# ---------------------------------------------------------------------------
# Pipeline (cached)
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner="Loading + enriching data…")
def load_data(symbol: str) -> dict[str, pd.DataFrame]:
    """Load all timeframes for *symbol* with ATR warmed on full history."""
    return load_enriched_timeframes(symbol)


def _formation_diagnostic(df: pd.DataFrame, passed_bases: list[dict]) -> dict:
    """Replicate detect_formations() bookkeeping so we can show drop-off counts.

    Returns counts for: insufficient_history (legs is None), flat (no
    formation mapping), weak_leg, and confirmed formations.
    """
    counts = {
        "passed_bases": len(passed_bases),
        "insufficient_history": 0,
        "flat": 0,
        "weak_leg": 0,
        "dirty_departure": 0,
        "confirmed": 0,
    }
    for cluster in passed_bases:
        legs = measure_legs(df, cluster["start"], cluster["end"])
        if legs is None:
            counts["insufficient_history"] += 1
            continue
        if (legs["leg_in_dir"], legs["leg_out_dir"]) not in FORMATION_MAP:
            counts["flat"] += 1
            continue
        if not legs["leg_strength_ok"]:
            counts["weak_leg"] += 1
            continue
        if not legs["clean_departure"]:
            counts["dirty_departure"] += 1
            continue
        counts["confirmed"] += 1
    return counts


def _zone_rejection_reason(z: dict) -> str:
    """Why did a zone fail the departure gates?"""
    if z.get("passed"):
        return ""
    dr_fail = z["dep_ratio"] < DEPARTURE_RATIO_MIN
    da_fail = z["dep_atr"] < DEPARTURE_ATR_MIN
    if dr_fail and da_fail:
        return f"dep_ratio<{DEPARTURE_RATIO_MIN} + dep_atr<{DEPARTURE_ATR_MIN}"
    if dr_fail:
        return f"dep_ratio<{DEPARTURE_RATIO_MIN}"
    if da_fail:
        return f"dep_atr<{DEPARTURE_ATR_MIN}"
    return "unknown"


def annotate_lifespan(df: pd.DataFrame, zones: list[dict]) -> list[dict]:
    """Set ``death_bar``, ``death_time``, ``is_alive`` on every zone in-place.

    Uses ``utils.freshness.find_death_bar`` so the dashboard's lifespan view
    and the freshness score share one source of truth.

    * ``death_bar``  : iloc of the breaking bar, or ``None`` if still alive.
    * ``death_time`` : timestamp of the breaking bar, or ``None`` if alive.
    * ``is_alive``   : bool — True when the zone has not been broken yet.
    """
    n = len(df)
    for z in zones:
        death = find_death_bar(df, z)
        if death is None:
            z["death_bar"] = None
            z["death_time"] = None
            z["is_alive"] = True
        else:
            safe = min(death, n - 1)
            z["death_bar"] = safe
            z["death_time"] = df.index[safe]
            z["is_alive"] = False
    return zones


def _base_failure_reason(b: dict) -> str:
    if not b["min_count_passed"] and not b["compactness_passed"]:
        return "min_count + compactness"
    if not b["min_count_passed"]:
        return "min_count"
    if not b["compactness_passed"]:
        return "compactness"
    return ""


@st.cache_data(show_spinner="Running detection pipeline…")
def run_pipeline(symbol: str, timeframe: str, apply_nested: bool) -> dict:
    """Run the full pipeline for one (symbol, tf) and return all artefacts.

    All zones (passed + rejected) are scored so the table can show every zone
    with a consistent column set. Merging is applied only to passed zones.
    """
    data = load_data(symbol)
    if timeframe not in data:
        raise ValueError(f"Timeframe {timeframe!r} not available for {symbol!r}.")
    df = data[timeframe]

    # Phase 4 — bases
    passed_bases, failed_bases = detect_bases(df)

    # Phase 5 — formations (with diagnostic breakdown of the drop-off)
    formations = detect_formations(df, passed_bases)
    formation_diag = _formation_diagnostic(df, passed_bases)

    # Phase 6 — zones
    passed_zones, rejected_zones = detect_zones(df, formations)

    # Phases 8–12 — score every zone (passed + rejected) so the table is uniform.
    all_zones = passed_zones + rejected_zones
    if all_zones:
        add_freshness(df, all_zones)
        add_time_score(all_zones)

        htf_key = HTF_REF.get(timeframe, "1d")
        if htf_key not in data:
            htf_key = "1d" if "1d" in data else timeframe
        add_curve_score(all_zones, data[htf_key], df.index)

        add_trend_score(all_zones, df)
        add_sets_score(all_zones)
        annotate_lifespan(df, all_zones)

    # Phase 13 — merge transitively-overlapping passed zones (optional).
    merged_zones = merge_zones(passed_zones) if apply_nested else passed_zones
    if merged_zones and merged_zones is not passed_zones:
        # Merged zones inherit fields from their leader; recompute lifespan on
        # the merged geometry so the death bar reflects the merged distal.
        annotate_lifespan(df, merged_zones)

    return {
        "df": df,
        "passed_bases": passed_bases,
        "failed_bases": failed_bases,
        "formations": formations,
        "formation_diag": formation_diag,
        "passed_zones": passed_zones,
        "rejected_zones": rejected_zones,
        "merged_zones": merged_zones,
    }


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Chart — dark TradingView style with filled-rectangle zones.
# Mirrors plot_zones() in notebooks/07_verify_all_scenarios.ipynb.
# ---------------------------------------------------------------------------


def _x_labels(df: pd.DataFrame) -> list[str]:
    """Stringified timestamps for a category x-axis (no weekend / holiday gaps)."""
    return df.index.strftime("%Y-%m-%d %H:%M").tolist()


def _zone_hover_text(z: dict, *, passed: bool) -> str:
    """Rich HTML tooltip with every annotated field on the zone."""
    keys = [
        "zone_type",
        "formation",
        "proximal",
        "distal",
        "zone_width",
        "dep_ratio",
        "dep_atr",
        "departure",
        "base_count",
        "touches",
        "leg_strength",
        "clean_departure",
        "is_alive",
        "death_time",
        "freshness_score",
        "time_score",
        "curve_third",
        "curve_pos",
        "curve_score",
        "trend",
        "trend_score",
        "strength_score",
        "sets_total",
        "sets_rating",
    ]
    rows = [f"<b>status</b>: {'passed' if passed else 'rejected'}"]
    for k in keys:
        if k in z and z[k] is not None:
            v = z[k]
            if isinstance(v, float):
                v = f"{v:.4f}"
            elif hasattr(v, "strftime"):
                v = v.strftime("%Y-%m-%d %H:%M")
            rows.append(f"<b>{k}</b>: {v}")
    return "<br>".join(rows)


def _add_zone_to_fig(
    fig: go.Figure,
    z: dict,
    x_labels: list[str],
    *,
    passed: bool,
) -> None:
    """Draw a zone as a filled rectangle spanning its living lifespan.

    Mirrors the notebook's add_shape(type="rect", ...) approach: a semi-
    transparent fill between proximal and distal, plus a solid proximal line
    and a dashed distal line clipped to the zone's lifespan (formation bar
    → death bar | right edge if still alive).
    """
    n = len(x_labels)
    end_idx = z["end"]
    if end_idx >= n:
        return

    death_idx = z.get("death_bar")
    is_alive = z.get("is_alive", death_idx is None)
    if death_idx is None:
        right_idx = n - 1
    else:
        right_idx = max(end_idx, min(death_idx, n - 1))

    x0 = x_labels[end_idx]
    x1 = x_labels[right_idx]

    if z["zone_type"] == "demand":
        edge = DEMAND_EDGE
        if passed:
            fill = DEMAND_FILL_PASSED_ALIVE if is_alive else DEMAND_FILL_PASSED_DEAD
        else:
            fill = DEMAND_FILL_REJ_ALIVE if is_alive else DEMAND_FILL_REJ_DEAD
    else:
        edge = SUPPLY_EDGE
        if passed:
            fill = SUPPLY_FILL_PASSED_ALIVE if is_alive else SUPPLY_FILL_PASSED_DEAD
        else:
            fill = SUPPLY_FILL_REJ_ALIVE if is_alive else SUPPLY_FILL_REJ_DEAD

    # Single source of truth for both the rectangle and the lines:
    # the zone's own stored proximal/distal. By construction
    #   y_lo = min(prox, dist)   y_hi = max(prox, dist)
    # so the proximal and distal lines below sit EXACTLY on the
    # rectangle's two edges (proximal on top for demand / on bottom
    # for supply, distal on the opposite side).
    prox = z["proximal"]
    dist = z["distal"]
    y_lo = min(prox, dist)
    y_hi = max(prox, dist)

    # Filled rectangle — the zone body. Border is transparent so that
    # the explicit proximal/distal lines below ARE the visible edges
    # (otherwise two strokes stack at each edge and Plotly's anti-
    # aliasing makes them read as two parallel lines floating off
    # the rectangle).
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=x0,
        x1=x1,
        y0=y_lo,
        y1=y_hi,
        fillcolor=fill,
        line=dict(color="rgba(0,0,0,0)", width=0),
        layer="below",
    )

    # Proximal line — solid, drawn at exactly y = zone["proximal"].
    # This IS the top edge of the rectangle for demand zones and the
    # bottom edge for supply zones (because of the min/max above).
    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=x0,
        x1=x1,
        y0=prox,
        y1=prox,
        line=dict(
            color=edge,
            width=1.4 if passed else 1.0,
            dash="solid" if passed else "dot",
        ),
        layer="below",
    )

    # Distal line — dashed, drawn at exactly y = zone["distal"].
    # This is the opposite edge of the rectangle by construction.
    fig.add_shape(
        type="line",
        xref="x",
        yref="y",
        x0=x0,
        x1=x1,
        y0=dist,
        y1=dist,
        line=dict(
            color=edge,
            width=1.0,
            dash="dash" if passed else "dot",
        ),
        layer="below",
    )

    # Death marker — thin vertical line at the breaking bar.
    if not is_alive and death_idx is not None and 0 <= death_idx < n:
        x_death = x_labels[min(death_idx, n - 1)]
        fig.add_shape(
            type="line",
            xref="x",
            yref="y",
            x0=x_death,
            x1=x_death,
            y0=y_lo,
            y1=y_hi,
            line=dict(color=DEATH_LINE, width=1.2, dash="dot"),
            layer="above",
        )

    # Invisible scatter spanning the rectangle to capture hover with full info.
    mid_y = (y_lo + y_hi) / 2
    hover = _zone_hover_text(z, passed=passed)
    fig.add_trace(
        go.Scatter(
            x=[x0, x1],
            y=[mid_y, mid_y],
            mode="lines",
            line=dict(color="rgba(0,0,0,0)", width=14),
            hoverinfo="text",
            hovertext=[hover, hover],
            showlegend=False,
        )
    )

    # On-chart label — formation + star rating at the formation bar (notebook style).
    rating = z.get("sets_rating", "")
    formation = z.get("formation", "")
    label = f"{formation} {rating}".strip()
    if label:
        fig.add_annotation(
            x=x0,
            y=prox,
            text=f"<b>{label}</b>",
            showarrow=False,
            xanchor="left",
            yanchor="bottom" if z["zone_type"] == "demand" else "top",
            font=dict(size=10, color=edge),
            bgcolor="rgba(19,23,34,0.6)",
        )


def _add_failed_base_to_fig(
    fig: go.Figure,
    b: dict,
    x_labels: list[str],
) -> None:
    """Draw a small faint rectangle over the failed cluster's bars only."""
    bs, be = b["start"], b["end"]
    if be >= len(x_labels):
        return
    x0 = x_labels[bs]
    x1 = x_labels[be]
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=x0,
        x1=x1,
        y0=b["base_low"],
        y1=b["base_high"],
        fillcolor=FAILED_BASE_FILL,
        line=dict(color=FAILED_BASE_LINE, width=0.8, dash="dot"),
        layer="below",
    )


def build_chart(
    df: pd.DataFrame,
    passed_zones: list[dict],
    rejected_zones: list[dict],
    failed_bases: list[dict],
    *,
    show_passed: bool,
    show_rejected: bool,
    show_failed_bases: bool,
    title: str,
) -> go.Figure:
    """TradingView-style dark candlestick chart with filled-rectangle zones."""
    x_labels = _x_labels(df)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x_labels,
                open=df["open"],
                high=df["high"],
                low=df["low"],
                close=df["close"],
                increasing=dict(line=dict(color=COLOR_BULL), fillcolor=COLOR_BULL),
                decreasing=dict(line=dict(color=COLOR_BEAR), fillcolor=COLOR_BEAR),
                name="OHLC",
                showlegend=False,
            )
        ]
    )

    if show_failed_bases:
        for b in failed_bases:
            _add_failed_base_to_fig(fig, b, x_labels)

    if show_rejected:
        for z in rejected_zones:
            _add_zone_to_fig(fig, z, x_labels, passed=False)

    if show_passed:
        for z in passed_zones:
            _add_zone_to_fig(fig, z, x_labels, passed=True)

    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        height=720,
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="closest",
        xaxis=dict(
            type="category",
            rangeslider=dict(visible=False),
            gridcolor=CHART_GRID,
            showgrid=True,
            tickangle=-45,
            nticks=20,
        ),
        yaxis=dict(
            gridcolor=CHART_GRID,
            showgrid=True,
            side="right",
            title="Price",
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------

TABLE_COLUMNS = [
    "status",
    "rejection_reason",
    "is_alive",
    "death_time",
    "sets_rating",
    "sets_total",
    "zone_type",
    "formation",
    "proximal",
    "distal",
    "zone_width",
    "dep_ratio",
    "dep_atr",
    "departure",
    "strength_score",
    "time_score",
    "freshness_score",
    "curve_score",
    "curve_third",
    "trend_score",
    "trend",
    "base_count",
    "touches",
    "leg_strength",
    "clean_departure",
    "avg_atr",
    "start",
    "end",
    "death_bar",
]


def zones_to_dataframe(
    passed: list[dict],
    rejected: list[dict],
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Build the unified zones table with passed + rejected rows."""
    rows = []
    for z in passed:
        row = dict(z)
        row["status"] = "passed"
        row["rejection_reason"] = ""
        rows.append(row)
    for z in rejected:
        row = dict(z)
        row["status"] = "rejected"
        row["rejection_reason"] = _zone_rejection_reason(z)
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=TABLE_COLUMNS)

    out = pd.DataFrame(rows)

    # Add formation timestamp for readability
    if "end" in out.columns and len(df):
        out["formation_time"] = (
            out["end"].clip(upper=len(df) - 1).map(lambda i: df.index[i])
        )

    cols = ["formation_time"] + [c for c in TABLE_COLUMNS if c in out.columns]
    cols += [c for c in out.columns if c not in cols]
    out = out[cols]

    if "sets_total" in out.columns:
        out = out.sort_values(
            by=["status", "sets_total"], ascending=[True, False]
        ).reset_index(drop=True)

    return out


def _style_table(df: pd.DataFrame):
    """Color-code the sets_rating column."""
    if df.empty:
        return df

    def color_rating(val):
        return {
            "★★★": "background-color: #1b5e20; color: white;",
            "★★": "background-color: #f9a825; color: black;",
            "★": "background-color: #424242; color: white;",
        }.get(val, "")

    styler = df.style
    if "sets_rating" in df.columns:
        styler = styler.applymap(color_rating, subset=["sets_rating"])
    return styler


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def sidebar_controls():
    st.sidebar.title("S&D Validation Controls")

    st.sidebar.markdown("### Symbol")
    options = sorted(set(ALL_SYMBOLS))
    default_idx = options.index("USDJPY=X") if "USDJPY=X" in options else 0
    picked = st.sidebar.selectbox("Watchlist", options, index=default_idx)
    free_text = st.sidebar.text_input(
        "Or enter a ticker", value="", placeholder="e.g. EURUSD=X"
    ).strip()
    symbol = free_text or picked

    st.sidebar.markdown("### Timeframe")
    tf = st.sidebar.selectbox(
        "Timeframe",
        DEFAULT_TIMEFRAMES,
        index=DEFAULT_TIMEFRAMES.index("1d"),
    )

    st.sidebar.markdown("### Display")
    show_passed = st.sidebar.toggle("Show passed zones", value=True)
    show_rejected = st.sidebar.toggle("Show rejected zones", value=False)
    show_failed_bases = st.sidebar.toggle("Show failed bases", value=False)
    apply_nested = st.sidebar.toggle("Apply nested-merge", value=False)

    rating_choice = st.sidebar.radio(
        "Minimum rating",
        list(RATING_FILTERS.keys()),
        index=0,
        horizontal=True,
    )
    min_rating = RATING_FILTERS[rating_choice]

    run_clicked = st.sidebar.button("▶ Run pipeline", type="primary")

    return {
        "symbol": symbol,
        "timeframe": tf,
        "show_passed": show_passed,
        "show_rejected": show_rejected,
        "show_failed_bases": show_failed_bases,
        "apply_nested": apply_nested,
        "min_rating": min_rating,
        "run_clicked": run_clicked,
    }


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------


def summary_metrics(df: pd.DataFrame, passed: list[dict]) -> None:
    if not passed:
        st.info("No passed zones in this run.")
        return

    total = len(passed)
    demand = sum(1 for z in passed if z["zone_type"] == "demand")
    supply = total - demand
    n3 = sum(1 for z in passed if z.get("sets_rating") == "★★★")
    n2 = sum(1 for z in passed if z.get("sets_rating") == "★★")
    n1 = sum(1 for z in passed if z.get("sets_rating") == "★")
    fresh = sum(1 for z in passed if z.get("touches", 0) == 0)
    pct_fresh = (fresh / total * 100) if total else 0.0

    last_price = float(df["close"].iloc[-1])

    # Nearest zone by proximal distance
    def dist(z):
        return abs(z["proximal"] - last_price)

    nearest = min(passed, key=dist)
    nearest_dist = dist(nearest)
    nearest_pct = (nearest_dist / last_price * 100) if last_price else 0.0

    cols = st.columns(7)
    cols[0].metric("Total zones", total)
    cols[1].metric("Demand / Supply", f"{demand} / {supply}")
    cols[2].metric("★★★", n3)
    cols[3].metric("★★", n2)
    cols[4].metric("★", n1)
    cols[5].metric("% fresh (0 touches)", f"{pct_fresh:.0f}%")
    cols[6].metric(
        f"Nearest {nearest['zone_type']}",
        f"{nearest['proximal']:.4f}",
        delta=f"{nearest_pct:+.2f}% vs {last_price:.4f}",
        delta_color="off",
    )


# ---------------------------------------------------------------------------
# Validation panel
# ---------------------------------------------------------------------------


def validation_panel(result: dict) -> None:
    passed_bases = result["passed_bases"]
    failed_bases = result["failed_bases"]
    formations = result["formations"]
    formation_diag = result["formation_diag"]
    passed_zones = result["passed_zones"]
    rejected_zones = result["rejected_zones"]

    with st.expander("Stage 1 — Bases", expanded=False):
        total = len(passed_bases) + len(failed_bases)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total clusters", total)
        c2.metric("Passed", len(passed_bases))
        c3.metric("Failed", len(failed_bases))
        if failed_bases:
            reasons = pd.Series(
                [_base_failure_reason(b) for b in failed_bases]
            ).value_counts()
            st.write("**Failure reasons**")
            st.dataframe(reasons.rename("count"), use_container_width=True)

    with st.expander("Stage 2 — Formations", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Passed bases", formation_diag["passed_bases"])
        c2.metric("Insufficient hist.", formation_diag["insufficient_history"])
        c3.metric("Flat leg", formation_diag["flat"])
        c4.metric("Weak leg", formation_diag["weak_leg"])
        c5.metric("Confirmed", formation_diag["confirmed"])
        if formations:
            ft = pd.Series([f["formation"] for f in formations]).value_counts()
            st.write("**Formation type breakdown**")
            st.dataframe(ft.rename("count"), use_container_width=True)

    with st.expander("Stage 3 — Zones (departure gate)", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Formations", len(formations))
        c2.metric("Passed departure", len(passed_zones))
        c3.metric("Rejected departure", len(rejected_zones))
        if rejected_zones:
            reasons = pd.Series(
                [_zone_rejection_reason(z) for z in rejected_zones]
            ).value_counts()
            st.write("**Rejection reasons**")
            st.dataframe(reasons.rename("count"), use_container_width=True)

    with st.expander("Stage 4 — Scoring distributions", expanded=False):
        all_zones = passed_zones + rejected_zones
        if not all_zones:
            st.info("No zones to score.")
        else:
            df_scores = pd.DataFrame(all_zones)
            score_cols = [
                "strength_score",
                "time_score",
                "freshness_score",
                "curve_score",
                "trend_score",
                "sets_total",
            ]
            present = [c for c in score_cols if c in df_scores.columns]
            if not present:
                st.info("No score columns found.")
            else:
                cols = st.columns(min(3, len(present)))
                for i, col_name in enumerate(present):
                    fig = go.Figure(
                        data=[
                            go.Histogram(
                                x=df_scores[col_name],
                                marker=dict(color=COLOR_BULL),
                                nbinsx=10,
                            )
                        ]
                    )
                    fig.update_layout(
                        title=col_name,
                        template="plotly_dark",
                        paper_bgcolor=CHART_BG,
                        plot_bgcolor=CHART_BG,
                        height=260,
                        margin=dict(l=10, r=10, t=40, b=10),
                        xaxis=dict(gridcolor=CHART_GRID),
                        yaxis=dict(gridcolor=CHART_GRID),
                    )
                    cols[i % len(cols)].plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ctrl = sidebar_controls()

    st.title("Supply & Demand — Pre-labeling Validation Dashboard")
    st.caption(
        f"Symbol: **{ctrl['symbol']}** · Timeframe: **{ctrl['timeframe']}** · "
        "Click *Run pipeline* in the sidebar to (re)compute. "
        "Display toggles don't trigger recomputation — cached per (symbol, tf, nested-merge)."
    )

    # Clear cache when explicitly asked
    if ctrl["run_clicked"]:
        run_pipeline.clear()
        load_data.clear()

    try:
        result = run_pipeline(ctrl["symbol"], ctrl["timeframe"], ctrl["apply_nested"])
    except FileNotFoundError as e:
        st.error(f"Data not found: {e}")
        return
    except Exception as e:  # noqa: BLE001
        st.error(f"Pipeline failed: {type(e).__name__}: {e}")
        return

    df = result["df"]
    passed_zones_all = (
        result["merged_zones"] if ctrl["apply_nested"] else result["passed_zones"]
    )
    rejected_zones_all = result["rejected_zones"]

    # Filter by minimum rating
    min_rank = ctrl["min_rating"]
    passed_zones = [
        z
        for z in passed_zones_all
        if RATING_RANK.get(z.get("sets_rating", "★"), 1) >= min_rank
    ]
    rejected_zones = [
        z
        for z in rejected_zones_all
        if RATING_RANK.get(z.get("sets_rating", "★"), 1) >= min_rank
    ]

    # ---- Summary metrics
    summary_metrics(df, passed_zones)

    st.divider()

    # ---- Chart
    title = (
        f"{ctrl['symbol']} {ctrl['timeframe']} — "
        f"{len(passed_zones)} passed · {len(rejected_zones)} rejected"
        + (" · merged" if ctrl["apply_nested"] else "")
    )
    fig = build_chart(
        df,
        passed_zones,
        rejected_zones,
        result["failed_bases"],
        show_passed=ctrl["show_passed"],
        show_rejected=ctrl["show_rejected"],
        show_failed_bases=ctrl["show_failed_bases"],
        title=title,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- Validation panel (expanders)
    st.subheader("Stage-by-stage validation")
    validation_panel(result)

    # ---- Zone table
    st.subheader("Zone table")
    table_df = zones_to_dataframe(passed_zones, rejected_zones, df)
    if table_df.empty:
        st.info("No zones to display.")
    else:
        st.dataframe(
            _style_table(table_df),
            use_container_width=True,
            height=420,
        )


if __name__ == "__main__":
    main()
