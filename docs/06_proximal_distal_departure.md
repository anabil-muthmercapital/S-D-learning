# 06 — Proximal / Distal و الـ Departure

## الهدف
تتحقق إزاي الكود بيحدد حدود المنطقة (proximal/distal) وبيحسب قوة الخروج (departure). دول مفهومين إنت فاهمهم — هنا تشوفهم في الكود.

## الأجزاء اللي تركّز عليها في الكود
في `zone_detector.py`، خطوات 7, 8:
- **proximal/distal:** في الـ demand، `proximal = أعلى الـ base` (أقرب نقطة للسعر الراجع)، `distal = أسفل الـ base` (مكان الستوب). في الـ supply ينعكسوا.
- **departure:** بيتقاس بالـ PEAK (أعلى/أقل نقطة وصلها السعر في نافذة الخروج)، مش بالإغلاق. ده fix #3 في الكود (round-trip trap).
- **القياسين:** `departure_ratio = departure / zone_width` و `departure_atr = departure / avg_atr`. لازم الاتنين يعدّوا.

## نقطة مهمة تكتشفها
في الملف التأسيسي، الـ departure_ratio بيتقاس **بالنسبة لعرض الـ base**. بس انتبه: ده بيدّي أرقام مختلفة عن القياس بالـ ATR. الكود بيستخدم الاتنين كـ AND-gate. افهم الفرق بينهم.

## هتعمل إيه
1. تاخد scenario A وتحسب proximal/distal بإيدك (أعلى وأسفل الـ base).
2. تحسب departure (المسافة اللي السعر خرجها).
3. تحسب القياسين وتتأكد الاتنين فوق الحد.
4. تجرّب scenario C (الـ departure الضعيف) وتشوف أنهي قياس رفضه.

## الكود اللي تكتبه بإيدك
```python
import sys; sys.path.insert(0, "..")
import pandas as pd
from utils.models import CandlePrimitives
from utils.zone_detector import detect_zones

df = pd.read_csv("../fixtures.csv", index_col=0, parse_dates=True)
dfe = CandlePrimitives.enrich_dataframe(df)
zones = detect_zones(dfe)

for z in zones:
    print(f"{z.zone_type} {z.formation}: prox={z.proximal:.2f} dist={z.distal:.2f} "
          f"width={z.zone_width:.2f} dep={z.departure:.2f} "
          f"dep_ratio={z.departure_ratio:.2f} dep_atr={z.departure_atr:.2f}")
```

## السؤال اللي بيقفله
- في demand، ليه الـ proximal = أعلى الـ base مش أسفله؟ (السعر بينزل من فوق، فأول حاجة بيلمسها هي القمة).
- ليه بنقيس الـ departure بالـ PEAK مش بالإغلاق؟ (عشان حركة خرجت ورجعت تفضل دليل قوة).
- ليه محتاجين القياسين (ratio + atr) مش واحد؟ (الـ base الضيق بيكبّر الـ ratio وهمياً).

## علامة إنك خلّصت
تقدر تاخد منطقة وتقول entry/stop بتوعها فين قبل ما تشغّل الكود.
