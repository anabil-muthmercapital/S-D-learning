# 02 — تشريح الشمعة (Candle Primitives)

## الهدف
تتحقق إن كلاس `CandlePrimitives` في `models.py` بيحسب صح. ده أصغر وحدة في النظام كله — كل حاجة فوقها بتعتمد عليها.

## ليه دلوقتي؟
إنت قلت إنك فاهم candle breakdown و base vs leg. الـ notebook ده بيثبّت الفهم ده على الكود الفعلي بأرقام حقيقية — مش نظري.

## هتعمل إيه
1. تاخد شمعة واحدة من البيانات.
2. تحسب بإيدك (ورقة وقلم): `range`, `body`, `body_ratio`.
3. تعمل نفس الشمعة بـ `CandlePrimitives` وتقارن — لازم نفس الأرقام.
4. تفهم `is_base` (الـ body_ratio ≤ 0.5) و `is_bullish`.

## الكود اللي تكتبه بإيدك
```python
import sys; sys.path.insert(0, "..")
from utils.models import CandlePrimitives

# شمعة من scenario A — شوف note بتاعها في fixtures_labeled
c = CandlePrimitives(open=105.4, high=105.8, low=105.0, close=105.3)

print("range  =", round(c.candle_range, 3))          # high - low
print("body   =", round(c.body_size, 3))             # |close - open|
print("ratio  =", round(c.body_to_range_ratio, 3))   # body / range
print("is_base   =", c.is_base)                       # ratio <= 0.5 ?
print("is_bullish=", c.is_bullish)                    # close > open ?
```

## التحقق المطلوب
قبل ما تشغّل الكود، احسب إنت الأول على ورقة:
- `range = 105.8 − 105.0 = ?`
- `body  = |105.3 − 105.4| = ?`
- `ratio = body / range = ?`
- دي base candle ولا leg candle؟

بعدين شغّل وقارن. لو أرقامك زي الكود → فهمت.

## السؤال اللي بيقفله
- ليه بنستخدم `body_ratio` مش `body` لوحده؟ (تلميح: شمعة جسمها 1 دولار على سهم بيتحرك 2 دولار غير شمعة جسمها 1 دولار على سهم بيتحرك 50 دولار).
- إمتى `body_ratio` بيطلع صفر؟ (الـ doji).

## علامة إنك خلّصت
تقدر تاخد أي شمعة وتقول "دي base ولا leg" قبل ما تشغّل الكود، وتطلع صح.
