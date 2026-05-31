# 05 — الـ Legs و الـ Formation (الجزء الأصعب)

## الهدف
تفهم إزاي الكود بيحدد الـ leg-in (الحركة الداخلة للـ base) و الـ leg-out (الحركة الخارجة)، وإزاي بيجمعهم في formation (RBR, DBD, DBR, RBD).

## ليه ده أصعب notebook؟
ده الجزء اللي كشفنا إنه بيلخبط في سيناريوهات F و G. الـ leg detection حساس جداً للبيانات اللي قبل وبعد الـ base. لو فهمت ده، فهمت قلب الكود.

## الأجزاء اللي تركّز عليها في الكود
في `zone_detector.py`، خطوات 4, 5, 6:
- **leg-in:** `leg_in_net = c[base_start - 1] - o[leg_in_start]` ← انتبه: لو `base_start = 0` ده بيرجع آخر شمعة (المستقبل) = الـ lookahead bug.
- **leg-out:** `leg_out_net = c[leg_out_end] - c[base_end]`.
- **التصنيف:** `_classify_move()` — up لو الحركة ≥ threshold، down لو ≤ −threshold، flat غير كده.
- **الـ formation map:** `_FORMATION_MAP` — بيربط (اتجاه leg-in, اتجاه leg-out) بـ formation.

## هتعمل إيه
1. تتبع الـ leg-in و leg-out لـ scenario A خطوة خطوة.
2. تشوف ليه scenario F (الـ pullback) و G (الـ nested) بيفشلوا — الـ legs بتتقاس غلط بسبب الحركات المركّبة.
3. تفهم جدول الـ formation: ازاي (up, up) = RBR = demand، و (down, down) = DBD = supply.

## الكود اللي تكتبه بإيدك
```python
import sys; sys.path.insert(0, "..")
import pandas as pd
from utils.models import CandlePrimitives
from utils.zone_detector import _atr, _classify_move, _FORMATION_MAP
from utils.config import LEG_ATR_MIN_MULT, DEPARTURE_CANDLES

df = pd.read_csv("../fixtures.csv", index_col=0, parse_dates=True)
dfe = CandlePrimitives.enrich_dataframe(df)
o, h, l, c = (dfe[x].values for x in ["open","high","low","close"])
atr = _atr(dfe)

print("جدول الـ formation:")
for (lin, lout), form in _FORMATION_MAP.items():
    print(f"  leg-in={lin:5s} leg-out={lout:5s} -> {form}")
```

## السؤال اللي بيقفله
- ليه (up, up) = demand مش supply؟ (رالي → base → رالي = المشترين أقوى).
- إيه اللي بيحصل لو الـ leg-in طلع flat؟ (الكود بيرفض المنطقة).
- في الـ lookahead: لو `base_start = 0`، إيه قيمة `c[base_start - 1]`؟ ليه دي خطيرة؟

## علامة إنك خلّصت
تقدر تشرح ليه F و G فشلوا، وتقول هل ده عيب في الكود ولا في طريقة قياس الـ leg.
