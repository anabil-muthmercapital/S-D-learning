# 04 — كشف الـ Base (Base Detection)

## الهدف
تتبع إزاي `detect_zones()` بيلاقي مجموعة شموع الـ base (الـ base cluster). ده الجزء الأول من الدالة — الـ `while` loop اللي بيمشي على الشموع.

## ليه دلوقتي؟
ده بيقفل مفهوم الـ compactness عملياً. إنت فاهمه نظرياً؛ هنا تشوفه بيتطبّق في الكود سطر بسطر.

## الأجزاء اللي تركّز عليها في الكود
في `zone_detector.py`، خطوات 1, 2, 3:
1. **لازم تبدأ على base candle** (`b_ratio[i] > BASE_BODY_RATIO_MAX → عدّي`).
2. **مدّ الـ cluster** (طول ما الشموع base ومش زيادة عن `BASE_MAX_CANDLES`).
3. **فلتر الـ tightness** (`base_width / avg_atr > BASE_MAX_ATR_WIDTH → ارفض`).

## هتعمل إيه
1. تاخد scenario A (الـ base عند index معيّن — شوف fixtures_labeled).
2. تتأكد إزاي الكود بيكتشف إن الشموع دي base.
3. تحسب `base_width` و `compactness_ratio` بإيدك وتقارن بالكود.
4. **مهم:** تروح لـ scenario D (الـ base العريض) وتشوف ليه المفروض يترفض — وده اللي هنحقق فيه إنه bug.

## الكود اللي تكتبه بإيدك
```python
import sys; sys.path.insert(0, "..")
import pandas as pd
from utils.models import CandlePrimitives
from utils.config import BASE_BODY_RATIO_MAX, BASE_MAX_ATR_WIDTH

df = pd.read_csv("../fixtures.csv", index_col=0, parse_dates=True)
labeled = pd.read_csv("../fixtures_labeled.csv", index_col=0, parse_dates=True)
dfe = CandlePrimitives.enrich_dataframe(df)

# شوف أي شموع في scenario D
mask = labeled["scenario"] == "D_wide_base"
print(labeled[mask][["note"]])
print()
print("body_to_range_ratio لكل شمعة في D:")
print(dfe[mask][["body_to_range_ratio"]])
```

## السؤال اللي بيقفله
- ليه الكود بيبدأ بس على base candle؟
- في scenario D، الـ base المفروض يكون عريض — احسب `base_width / ATR`. هل فعلاً أكبر من 2.5؟ لو لأ، يبقى البيانات هي المشكلة. لو أيوة، يبقى الكود مش بيرفض صح → bug.

## علامة إنك خلّصت
تقدر تقول بالظبط ليه scenario D لقى منطقة رغم إن المفروض يرفضها (ده أول bug هتمسكه بنفسك).
