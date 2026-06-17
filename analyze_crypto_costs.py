#!/usr/bin/env python
# =============================================================================
# analyze_crypto_costs.py — هل الكريبتو رابح بعد تكلفة binance؟ (قياس فقط)
# =============================================================================
#
# سكريبت قياس مستقل — مش بيعدّل costs.py ولا الـ pipeline ولا الموديل.
# بياخد صفقات الكريبتو الفعلية من data/dataset.csv ويعيد حساب الـ net
# بثلاث سيناريوهات تكلفة:
#   1. OLD      — أرقام costs.py الحالية (commission 0.1%/side).
#   2. BINANCE  — commission 0.06%/side (BNB+referral)، نفس spread/slip.
#   3. BINANCE+ — أفضل حالة: commission 0.06% + spread/slip أضيق (عملات كبيرة).
#
# وبيقارن الكريبتو بالأصول النظيفة (us_stocks, etfs, fx, commodities, indices)
# عشان نشوف: هل الكريبتو بقى يقارَن بيهم، ولا لسه ورا؟
#
# الـ net per trade = gross_r − cost_r، حيث:
#   gross_r = pnl_r (tp→ كسب فعلي، sl→ -1، timeout→ pnl_r المسجّل)
#   cost_r  = round-trip cost بالـ R = (2 × per_side_frac × entry) / risk
#
# التشغيل:  cd /Users/an/Desktop/S-D-learning && python analyze_crypto_costs.py
# =============================================================================

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET = PROJECT_ROOT / "data" / "dataset.csv"

# ---------------------------------------------------------------------------
# سيناريوهات التكلفة (per side، fraction من سعر الدخول)
# ---------------------------------------------------------------------------
# الكريبتو في costs.py الحالي: spread 5bps + slip 5bps + commission 10bps.
SCENARIOS = {
    "OLD (comm 0.10%)": {"spread": 5e-4, "slip": 5e-4, "comm": 10e-4},
    "BINANCE (comm 0.06%)": {"spread": 5e-4, "slip": 5e-4, "comm": 6e-4},
    "BINANCE+ (tight, big coins)": {"spread": 3e-4, "slip": 3e-4, "comm": 6e-4},
    "BINANCE++ (zero slip ideal)": {"spread": 2e-4, "slip": 1e-4, "comm": 6e-4},
}


def cost_r(row, spread, slip, comm) -> float:
    """round-trip cost بالـ R لصف صفقة واحد، بأرقام تكلفة معيّنة."""
    entry = float(row["entry"])
    risk = float(row["risk"])
    if entry <= 0 or risk <= 0:
        return 0.0
    per_side = spread + slip + comm
    return (2.0 * per_side * entry) / risk


def gross_r(row) -> float:
    """الـ gross R للصفقة (قبل التكلفة)."""
    reason = str(row.get("exit_reason", ""))
    pnl = row.get("pnl_r", np.nan)
    if reason == "tp":
        return float(pnl) if pd.notna(pnl) else 3.0
    if reason == "sl":
        return -1.0
    if reason == "timeout":
        return float(pnl) if pd.notna(pnl) else 0.0
    return float(pnl) if pd.notna(pnl) else 0.0


def main() -> int:
    if not DATASET.exists():
        print(f"[fatal] مش لاقي {DATASET}")
        return 1

    df = pd.read_csv(DATASET)
    print(f"إجمالي الصفقات في الـ dataset: {len(df)}")
    print(f"الفئات الموجودة: {sorted(df['asset_class'].unique())}\n")

    crypto = df[df["asset_class"] == "crypto"].copy()
    if crypto.empty:
        print("⚠️  مفيش صفقات crypto في الـ dataset!")
        print("    الكريبتو متشال من البناء. محتاج تبنيه بـ crypto الأول:")
        print("    (تأكد إن WATCHLIST فيه crypto وأعد build_dataset.py)")
        # نكمّل نحلّل الأصول النظيفة على الأقل
    else:
        print(f"صفقات الكريبتو: {len(crypto)}  ({crypto['symbol'].nunique()} عملة)")
        crypto["gross_r"] = crypto.apply(gross_r, axis=1)

        print("\n" + "=" * 70)
        print("الكريبتو — net expectancy بكل سيناريو تكلفة")
        print("=" * 70)
        print(f"  gross (قبل التكلفة) : {crypto['gross_r'].mean():+.4f} R")
        print(f"  win rate            : {(crypto['label']==1).mean()*100:.1f}%\n")
        for name, c in SCENARIOS.items():
            crypto["cost_r"] = crypto.apply(
                lambda r: cost_r(r, c["spread"], c["slip"], c["comm"]), axis=1
            )
            net = (crypto["gross_r"] - crypto["cost_r"]).mean()
            avg_cost = crypto["cost_r"].mean()
            flag = "✅ موجب" if net > 0 else "❌ سالب"
            print(f"  {name:<32} cost={avg_cost:.4f}R  net={net:+.4f}R  {flag}")

        # توزيع التكلفة (عشان نشوف لو فيه صفقات ستوبها ضيق بتتكلّف كتير)
        crypto["cost_r"] = crypto.apply(
            lambda r: cost_r(r, **{"spread": 5e-4, "slip": 5e-4, "comm": 6e-4}), axis=1
        )
        print(f"\n  توزيع cost_r (binance): median={crypto['cost_r'].median():.3f}R  "
              f"p90={crypto['cost_r'].quantile(0.9):.3f}R  max={crypto['cost_r'].max():.3f}R")
        # نسبة الصفقات اللي التكلفة بتاكل أغلب الـ gross
        heavy = (crypto["cost_r"] > 0.3).mean() * 100
        print(f"  صفقات تكلفتها > 0.3R: {heavy:.1f}%  (ستوب ضيق = تكلفة قاتلة)")

    # ---- مقارنة بالأصول النظيفة (بأرقام costs.py الحالية) ----
    print("\n" + "=" * 70)
    print("مقارنة: net per trade بالفئة (gross − cost، أرقام costs.py الحالية)")
    print("=" * 70)
    # نعيد حساب الـ cost للأصول النظيفة بأرقامها الأصلية (تقريب بسيط)
    clean_costs = {
        "us_stocks": {"spread": 1.5e-4, "slip": 2e-4, "comm": 0.0},
        "etfs": {"spread": 1e-4, "slip": 1.5e-4, "comm": 0.0},
        "indices": {"spread": 1.5e-4, "slip": 2e-4, "comm": 0.0},
        "commodities": {"spread": 2e-4, "slip": 3e-4, "comm": 0.0},
        "fx": {"spread": 0.5e-4 / 1.10, "slip": 0.5e-4 / 1.10, "comm": 0.0},
    }
    for ac in ["us_stocks", "etfs", "fx", "commodities", "indices"]:
        sub = df[df["asset_class"] == ac].copy()
        if sub.empty:
            continue
        sub["gross_r"] = sub.apply(gross_r, axis=1)
        c = clean_costs[ac]
        sub["cost_r"] = sub.apply(
            lambda r: cost_r(r, c["spread"], c["slip"], c["comm"]), axis=1
        )
        net = (sub["gross_r"] - sub["cost_r"]).mean()
        print(f"  {ac:<14} n={len(sub):5}  gross={sub['gross_r'].mean():+.3f}R  "
              f"cost={sub['cost_r'].mean():.3f}R  net={net:+.3f}R")

    print("\n" + "=" * 70)
    print("الحُكم")
    print("=" * 70)
    if not crypto.empty:
        crypto["cost_r"] = crypto.apply(
            lambda r: cost_r(r, 5e-4, 5e-4, 6e-4), axis=1
        )
        net_binance = (crypto["gross_r"] - crypto["cost_r"]).mean()
        if net_binance > 0.1:
            print(f"  ✅ الكريبتو بقى رابح بوضوح بأرقام binance (+{net_binance:.3f}R).")
            print("     يستاهل نعيد تدريب موديل يشمله ونتحقق OOS.")
        elif net_binance > 0:
            print(f"  🟡 الكريبتو موجب هامشياً ({net_binance:+.3f}R) — هشّ، الـ slippage الحقيقي ممكن ياكله.")
            print("     مخاطرة عالية. لازم forward test منفصل قبل أي ثقة.")
        else:
            print(f"  ❌ الكريبتو لسه سالب بأرقام binance ({net_binance:+.3f}R).")
            print("     حتى مع تخفيض العمولة، الـ spread+slip بياكلوه. مرفوض.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
