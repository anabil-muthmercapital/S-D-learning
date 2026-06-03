# Missing Concepts — Roadmap for NB08 → NB18

> NB01–NB07 covered concepts **1–9** from [`ordered-concepts.md`](ordered-concepts.md): Candle, ATR, Body Ratio, Base, Compactness, Leg, Formations (RBR/DBD/DBR/RBD), Departure, Proximal/Distal.
>
> What's still missing — the concepts that turn a "detected zone" into a **tradeable zone with a score and an entry plan** — is listed below in the exact order they should be built.

---

## Quick map

| #  | Concept            | Notebook                         | Status | Blocks |
|----|--------------------|----------------------------------|--------|--------|
| 10 | Freshness          | `08_freshness.ipynb`             | ⬜ todo | — |
| 11 | Time (base count)  | `09_time_score.ipynb`            | ⬜ todo | — |
| 12 | The Curve          | `10_curve_position.ipynb`        | ⬜ todo | needs HTF range |
| 13 | Trend Alignment    | `11_trend_alignment.ipynb`       | ⬜ todo | swing structure |
| 14 | S.E.T.S Score      | `12_sets_scoring.ipynb`          | ⬜ todo | needs 10–13 |
| 15 | Nested Zones       | `13_nested_zones.ipynb`          | ⬜ todo | — |
| 16 | Entry / SL / TP    | `14_trade_execution.ipynb`       | ⬜ todo | — |
| 17 | Timeframes         | `15_timeframes.ipynb`            | ⬜ todo | multi-tf data |
| 18 | Liquidity / SH     | `16_liquidity_stop_hunts.ipynb`  | ⬜ todo | — |
| 19 | FVG                | `17_fvg.ipynb`                   | ⬜ todo | — |
| 20 | Risk Management    | `18_risk_management.ipynb`       | ⬜ todo | needs 16 |

---

## 10. Freshness — `08_freshness.ipynb`
**Question:** has this zone been touched before? Untouched zones are stronger.

- For every confirmed zone, walk forward and count how many times price re-entered the zone box.
- A zone **dies** the moment a candle closes beyond the **distal** line (the structure is broken).

| Touches | Score |
|---------|-------|
| 0       | 2     |
| 1       | 1     |
| ≥ 2     | 0     |

**Inputs:** zones from NB07. **Output:** `freshness_score` column.

---

## 11. Time — `09_time_score.ipynb`
**Question:** how many candles is the base? Fewer = stronger.

| Base candles | Score |
|--------------|-------|
| ≤ 2          | 2     |
| 3            | 1     |
| ≥ 4          | 0     |

A single explosive base candle is the strongest. Long bases = indecision = weaker reaction.

---

## 12. The Curve — `10_curve_position.ipynb`
**Question:** where does the zone sit inside the higher-timeframe range?

- Take the HTF range: `htf_high − htf_low`.
- Split into thirds: **Low / Mid / High**.
- A **demand** zone in the Low third is best (price is at value, lots of room up).
- A **supply** zone in the High third is best.
- Anything in the Mid third is mediocre.
- Demand in High / Supply in Low → counter-curve, weakest.

| Position (vs. zone type) | Score |
|--------------------------|-------|
| With the curve (demand-low, supply-high) | 2 |
| Mid third                                | 1 |
| Against the curve                        | 0 |

---

## 13. Trend Alignment — `11_trend_alignment.ipynb`
**Question:** does the zone trade *with* the current trend?

- Build M1 swing structure: higher-highs + higher-lows → **uptrend**; lower-highs + lower-lows → **downtrend**.
- A break of a recent **supply** zone confirms uptrend; break of **demand** confirms downtrend.

| Alignment            | Score |
|----------------------|-------|
| Zone aligned with trend  | 2 |
| Sideways / unclear       | 1 |
| Counter-trend            | 0 |

---

## 14. S.E.T.S Score — `12_sets_scoring.ipynb`
**Question:** combine everything into one number.

$$\text{total} = \text{Strength} + \text{Time} + \text{Freshness} + \text{Trend} + \text{Curve}$$

- **Strength** comes from NB06's departure ratio (map to 0/1/2).
- **Time, Freshness, Trend, Curve** from NB08-11.

| Total | Rating |
|-------|--------|
| ≥ 7   | ★★★ A-setup |
| 5–6   | ★★ B-setup  |
| ≤ 4   | ★ skip      |

This is the **decision gate** — only ★★★ zones make it to execution.

---

## 15. Nested Zones — `13_nested_zones.ipynb`
**Question:** when a smaller zone sits inside a bigger one, do we trade them as one or two?

$$\text{overlap\_ratio} = \frac{\text{overlap}}{\min(\text{width}_1,\, \text{width}_2)} \geq 0.5 \;\Rightarrow\; \text{merge}$$

- Inner zone gives a sharper entry; outer zone gives a safer stop.
- Solves NB07's **G_nested** false negative.

---

## 16. Trade Execution — `14_trade_execution.ipynb`
**Question:** translate a confirmed zone into entry, stop, target.

For a **demand** zone (buy):

```
entry  = proximal                          (the near edge)
sl     = distal − 0.1 × ATR                (buffer below the far edge)
risk   = entry − sl
tp     = entry + 3 × risk                  (R:R = 1:3)
```

Supply mirrors the geometry (sell at proximal, stop above distal).

The 0.1 × ATR buffer absorbs typical stop-hunt wicks without giving up too much R.

---

## 17. Timeframes — `15_timeframes.ipynb`
- **HTF** (4H/D) → bias and curve
- **MTF** (1H) → zone detection
- **LTF** (5–15m) → entry trigger

Rule: zones from HTF take priority; trade them only when the LTF agrees.

---

## 18. Liquidity & Stop Hunts — `16_liquidity_stop_hunts.ipynb`
- Identify obvious swing-high/swing-low clusters where retail stops sit.
- A wick that pokes the cluster and reverses is a **stop hunt** — often the trigger candle for a zone reaction.

---

## 19. FVG — `17_fvg.ipynb`
**Fair Value Gap** = a 3-candle pattern where candle-1's wick and candle-3's wick don't overlap, leaving a price gap on candle-2.

```
bullish FVG:  low[i+1]  > high[i-1]
bearish FVG:  high[i+1] < low[i-1]
```

Used as a refined entry inside a wider zone.

---

## 20. Risk Management — `18_risk_management.ipynb`
**The most important notebook of the series.**

$$\text{position\_size} = \frac{\text{account} \times \text{risk\_percent}}{\text{entry} - \text{sl}}$$

Defaults: `risk_percent = 0.5%` per trade, hard cap 1%. No exceptions.

A perfect signal with bad sizing is still a losing strategy. A mediocre signal with disciplined sizing survives.

---

## Build order
Strict left-to-right: **08 → 09 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18**.

Each notebook follows the same template established in NB01–07:
- English markdown, KaTeX for formulas (`$...$`, `$$...$$`)
- Section headers `## N. Title`
- Pipe tables for thresholds, no emojis
- Dark TradingView theme (BG `#131722`, GRID `#1e222d`, BULL `#26a69a`, BEAR `#ef5350`)
- Re-declare constants at the top so each notebook runs standalone
- End with a Plotly visualization
