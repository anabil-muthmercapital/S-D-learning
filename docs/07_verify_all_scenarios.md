# 07 — التحقق من كل السيناريوهات (Verify All)

## الهدف
تشغّل `detect_zones()` على الـ 11 سيناريو كلهم وتقارن النتيجة بالمتوقّع. ده الـ notebook اللي بيجمّع كل اللي فهمته ويمسك الـ bugs.

## ليه آخر واحد؟
لأنه بيعتمد على إنك فاهم كل الأجزاء قبله. هنا مش بتتعلم مفهوم جديد — بتتأكد إن النظام ككل بيتصرف صح.

## المتوقّع (ورقة الإجابات)
| سيناريو | المتوقّع | الحالة الحالية |
|---|---|---|
| A_demand_RBR | 1 demand | ✅ شغّال |
| B_supply_DBD | 1 supply | ✅ شغّال |
| C_weak_dep | لا شيء | ✅ شغّال |
| D_wide_base | لا شيء | ❌ لقى منطقة (bug) |
| E_doji_base | 1 demand | ✅ شغّال |
| F_fresh_vs_tested | 1 demand | ❌ ملقاش (bug) |
| G_nested | مناطق متداخلة | ❌ ملقاش (bug) |
| H_DBR | 1 demand | ✅ شغّال |
| I_RBD | 1 supply | ✅ شغّال |
| J_long_base | لا شيء | ✅ شغّال |

## هتعمل إيه
1. تشغّل الكاشف على كل البيانات.
2. تربط كل منطقة مكتشفة بالسيناريو بتاعها.
3. تعمل جدول مقارنة (متوقّع مقابل فعلي).
4. تركّز على الـ 3 اللي فيهم مشكلة (D, F, G) وتحقق في كل واحد.

## الكود اللي تكتبه بإيدك
```python
import sys; sys.path.insert(0, "..")
import pandas as pd
from utils.models import CandlePrimitives
from utils.zone_detector import detect_zones

df = pd.read_csv("../fixtures.csv", index_col=0, parse_dates=True)
labeled = pd.read_csv("../fixtures_labeled.csv", index_col=0, parse_dates=True)
dfe = CandlePrimitives.enrich_dataframe(df)
zones = detect_zones(dfe)

for z in zones:
    scen = labeled["scenario"].iloc[z.base_start]
    print(f"{z.zone_type:6s} {z.formation} | {scen} | "
          f"prox={z.proximal:.2f} dist={z.distal:.2f}")
```

## السؤال اللي بيقفله
- D لقى منطقة المفروض يرفضها — ليه؟ (فلتر الـ compactness مش بيمسك الحالة دي).
- F و G ملقوش حاجة — ليه؟ (الـ leg detection بيلخبط مع الحركات المركّبة).
- أنهي bug من دول خطير في الـ production وأنهي مقبول مؤقتاً؟

## علامة إنك خلّصت
عندك جدول واضح بيقول كودك قوي فين وضعيف فين، وتقدر تشرح كل سطر فيه. دي نقطة الانطلاق لإصلاح الـ bugs وبناء الـ freshness/SETS بعد كده.
