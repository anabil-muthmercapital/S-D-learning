"""
build_fixtures.py — Synthetic GOLDEN test dataset for zone detection.
Each scenario is hand-crafted so the expected outcome is known in advance.
This is a learning/verification tool: run detect_zones() on it and check
that what comes out matches the 'expected' column documented below.

Scenarios:
  A_demand_RBR   -> 1 demand zone (RBR)        textbook demand
  B_supply_DBD   -> 1 supply zone (DBD)        textbook supply
  C_weak_dep     -> NO zone                    departure too small
  D_wide_base    -> NO zone                    base not tight (compactness fail)
  E_doji_base    -> 1 demand zone              doji base, div-by-zero guard
  F_fresh_vs_tested -> 1 demand zone, price returns to it (freshness test)
  G_nested       -> overlapping demand zones   (nesting/confluence test)
  H_drop_base_rally (DBR) -> 1 demand zone     drop-base-rally formation
  I_rally_base_drop (RBD) -> 1 supply zone     rally-base-drop formation
  J_long_base    -> NO zone                    base longer than BASE_MAX_CANDLES
"""
import pandas as pd

rows = []

def add(o, h, l, c, scenario, note):
    rows.append({"open": o, "high": h, "low": l, "close": c,
                 "scenario": scenario, "note": note})

# ---------------------------------------------------------------------
# WARM-UP: enough bars for ATR(14) to stabilise (~1.5 range)
# ---------------------------------------------------------------------
p = 100.0
for k in range(16):
    o = p
    c = p + (0.6 if k % 2 == 0 else -0.5)
    h = max(o, c) + 0.5
    l = min(o, c) - 0.5
    add(o, h, l, c, "warmup", "ATR warm-up")
    p = c

# ---------------------------------------------------------------------
# A — TEXTBOOK DEMAND (RBR). Strong rally IN (3 bars), tight base, strong rally OUT.
#     leg-in must clear ~1.5*ATR, so make it big and clean.
# ---------------------------------------------------------------------
add(100.0, 100.6, 99.6, 100.4, "A_demand_RBR", "pre")
add(100.4, 103.2, 100.3, 103.0, "A_demand_RBR", "LEG-IN rally bar 1 (big up)")
add(103.0, 105.6, 102.9, 105.4, "A_demand_RBR", "LEG-IN rally bar 2 (big up)")
add(105.4, 105.8, 105.0, 105.3, "A_demand_RBR", "BASE 1 (small body)")
add(105.3, 105.7, 104.9, 105.5, "A_demand_RBR", "BASE 2 (small body)")
add(105.5, 105.9, 105.1, 105.2, "A_demand_RBR", "BASE 3 (small body)")
add(105.2, 110.0, 105.1, 109.8, "A_demand_RBR", "LEG-OUT rally (departure)")
add(109.8, 110.4, 109.0, 110.1, "A_demand_RBR", "after")
add(110.1, 111.5, 109.8, 111.2, "A_demand_RBR", "continuation up")

# ---------------------------------------------------------------------
# B — TEXTBOOK SUPPLY (DBD). Strong drop in, tight base, strong drop out.
# ---------------------------------------------------------------------
add(111.2, 111.5, 108.5, 108.8, "B_supply_DBD", "LEG-IN drop bar 1 (big down)")
add(108.8, 109.0, 106.0, 106.3, "B_supply_DBD", "LEG-IN drop bar 2 (big down)")
add(106.3, 106.7, 105.9, 106.1, "B_supply_DBD", "BASE 1 (small body)")
add(106.1, 106.5, 105.8, 106.2, "B_supply_DBD", "BASE 2 (small body)")
add(106.2, 106.6, 105.9, 106.0, "B_supply_DBD", "BASE 3 (small body)")
add(106.0, 106.2, 101.5, 101.8, "B_supply_DBD", "LEG-OUT drop (departure)")
add(101.8, 102.3, 101.0, 101.3, "B_supply_DBD", "after")
add(101.3, 101.8, 100.5, 100.8, "B_supply_DBD", "continuation down")

# ---------------------------------------------------------------------
# C — WEAK DEPARTURE. Tight base, valid legs, but leg-out is small.
#     Expect NO zone (fails DEPARTURE_RATIO_MIN / DEPARTURE_ATR_MIN).
# ---------------------------------------------------------------------
add(100.8, 101.0, 100.2, 100.9, "C_weak_dep", "pre")
add(100.9, 103.6, 100.8, 103.4, "C_weak_dep", "LEG-IN rally (big up)")
add(103.4, 103.8, 103.0, 103.5, "C_weak_dep", "BASE 1")
add(103.5, 103.9, 103.1, 103.6, "C_weak_dep", "BASE 2")
add(103.6, 104.0, 103.2, 103.7, "C_weak_dep", "BASE 3")
add(103.7, 104.3, 103.6, 104.1, "C_weak_dep", "LEG-OUT TOO SMALL (weak)")
add(104.1, 104.4, 103.7, 104.0, "C_weak_dep", "no follow-through")
add(104.0, 104.3, 103.6, 103.9, "C_weak_dep", "flat")

# ---------------------------------------------------------------------
# D — WIDE BASE. Big candles where the base should be -> not tight.
#     Expect NO zone (fails BASE_MAX_ATR_WIDTH).
# ---------------------------------------------------------------------
add(103.9, 104.2, 103.2, 104.0, "D_wide_base", "pre")
add(104.0, 107.0, 103.9, 106.8, "D_wide_base", "LEG-IN rally (big up)")
add(106.8, 110.5, 105.0, 106.0, "D_wide_base", "WIDE bar (huge range)")
add(106.0, 111.0, 104.5, 110.5, "D_wide_base", "WIDE bar (huge range)")
add(110.5, 115.5, 110.3, 115.2, "D_wide_base", "LEG-OUT up")
add(115.2, 115.8, 114.5, 115.0, "D_wide_base", "after")

# ---------------------------------------------------------------------
# E — DOJI BASE. Near-zero body base candles. Tests div-by-zero guard.
#     Expect 1 demand zone (or graceful handling, never a crash).
# ---------------------------------------------------------------------
add(115.0, 115.3, 114.4, 115.1, "E_doji_base", "pre")
add(115.1, 118.0, 115.0, 117.8, "E_doji_base", "LEG-IN rally (big up)")
add(117.8, 120.6, 117.7, 120.4, "E_doji_base", "LEG-IN rally bar 2")
add(120.4, 120.9, 120.0, 120.41, "E_doji_base", "DOJI base 1 (body~0.01)")
add(120.41, 120.85, 120.05, 120.40, "E_doji_base", "DOJI base 2 (body~0.01)")
add(120.40, 125.0, 120.3, 124.8, "E_doji_base", "LEG-OUT big rally")
add(124.8, 125.4, 124.0, 125.0, "E_doji_base", "after")

# ---------------------------------------------------------------------
# F — FRESH vs TESTED. A clean demand zone, then price RETURNS and touches
#     the proximal, then bounces. Tests freshness/touch counting (future build).
# ---------------------------------------------------------------------
add(125.0, 125.3, 124.4, 125.1, "F_fresh_vs_tested", "pre")
add(125.1, 128.0, 125.0, 127.8, "F_fresh_vs_tested", "LEG-IN rally (big up)")
add(127.8, 130.6, 127.7, 130.4, "F_fresh_vs_tested", "LEG-IN rally bar 2")
add(130.4, 130.8, 130.0, 130.3, "F_fresh_vs_tested", "BASE 1")
add(130.3, 130.7, 129.9, 130.5, "F_fresh_vs_tested", "BASE 2")
add(130.5, 135.0, 130.4, 134.8, "F_fresh_vs_tested", "LEG-OUT rally (departure)")
add(134.8, 135.4, 133.5, 134.0, "F_fresh_vs_tested", "peak")
add(134.0, 134.5, 131.0, 131.3, "F_fresh_vs_tested", "pulling back toward zone")
add(131.3, 131.6, 130.2, 130.6, "F_fresh_vs_tested", "TOUCHES proximal (~130.8) -> tested")
add(130.6, 133.0, 130.4, 132.8, "F_fresh_vs_tested", "bounce off zone")

# ---------------------------------------------------------------------
# G — NESTED. A small demand zone whose price range sits INSIDE a larger
#     demand zone formed earlier nearby. Tests overlap/nesting (future build).
# ---------------------------------------------------------------------
add(132.8, 133.1, 132.0, 132.9, "G_nested", "pre")
add(132.9, 135.8, 132.8, 135.6, "G_nested", "LEG-IN rally big (outer)")
add(135.6, 136.2, 135.0, 135.4, "G_nested", "OUTER base 1")
add(135.4, 136.0, 134.8, 135.7, "G_nested", "OUTER base 2")
add(135.7, 139.5, 135.6, 139.3, "G_nested", "OUTER leg-out")
add(139.3, 139.8, 137.0, 137.4, "G_nested", "pullback into outer base region")
add(137.4, 137.8, 135.3, 135.6, "G_nested", "INNER base forms inside outer 1")
add(135.6, 136.0, 135.1, 135.8, "G_nested", "INNER base 2")
add(135.8, 140.0, 135.7, 139.8, "G_nested", "INNER leg-out (departure)")
add(139.8, 140.3, 139.0, 140.0, "G_nested", "after")

# ---------------------------------------------------------------------
# H — DROP-BASE-RALLY (DBR) = demand. Drop IN, base, rally OUT.
# ---------------------------------------------------------------------
add(140.0, 140.3, 139.5, 140.1, "H_DBR", "pre")
add(140.1, 140.4, 137.2, 137.5, "H_DBR", "LEG-IN drop (big down)")
add(137.5, 137.8, 134.8, 135.1, "H_DBR", "LEG-IN drop bar 2")
add(135.1, 135.5, 134.7, 135.3, "H_DBR", "BASE 1")
add(135.3, 135.7, 134.9, 135.0, "H_DBR", "BASE 2")
add(135.0, 139.8, 134.9, 139.6, "H_DBR", "LEG-OUT rally (departure up)")
add(139.6, 140.1, 139.0, 139.8, "H_DBR", "after")

# ---------------------------------------------------------------------
# I — RALLY-BASE-DROP (RBD) = supply. Rally IN, base, drop OUT.
# ---------------------------------------------------------------------
add(139.8, 142.8, 139.7, 142.6, "I_RBD", "LEG-IN rally (big up)")
add(142.6, 145.4, 142.5, 145.2, "I_RBD", "LEG-IN rally bar 2")
add(145.2, 145.6, 144.8, 145.0, "I_RBD", "BASE 1")
add(145.0, 145.4, 144.6, 145.1, "I_RBD", "BASE 2")
add(145.1, 145.3, 140.5, 140.8, "I_RBD", "LEG-OUT drop (departure down)")
add(140.8, 141.3, 140.0, 140.4, "I_RBD", "after")

# ---------------------------------------------------------------------
# J — LONG BASE. 7 base candles (> BASE_MAX_CANDLES=5). Expect NO zone.
# ---------------------------------------------------------------------
add(140.4, 140.7, 139.8, 140.5, "J_long_base", "pre")
add(140.5, 143.5, 140.4, 143.3, "J_long_base", "LEG-IN rally (big up)")
add(143.3, 143.7, 142.9, 143.4, "J_long_base", "BASE 1")
add(143.4, 143.8, 143.0, 143.5, "J_long_base", "BASE 2")
add(143.5, 143.9, 143.1, 143.6, "J_long_base", "BASE 3")
add(143.6, 144.0, 143.2, 143.5, "J_long_base", "BASE 4")
add(143.5, 143.9, 143.1, 143.4, "J_long_base", "BASE 5")
add(143.4, 143.8, 143.0, 143.5, "J_long_base", "BASE 6 (too long)")
add(143.5, 143.9, 143.1, 143.6, "J_long_base", "BASE 7 (too long)")
add(143.6, 148.0, 143.5, 147.8, "J_long_base", "LEG-OUT rally")
add(147.8, 148.3, 147.0, 148.0, "J_long_base", "after")

df = pd.DataFrame(rows)
df.index = pd.date_range("2024-01-01", periods=len(df), freq="W-MON", tz="UTC")
df.index.name = "Date"
df["volume"] = 1_000_000

df.to_csv("/home/claude/fixtures_labeled.csv")
df[["open", "high", "low", "close", "volume"]].to_csv("/home/claude/fixtures.csv")

print(f"Built {len(df)} candles across {df['scenario'].nunique()} scenarios:")
print(df.groupby("scenario", sort=False).size().to_string())
