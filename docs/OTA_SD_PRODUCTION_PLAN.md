# OTA S&D AI Trader — خطة الإنتاج التفصيلية (Production Architecture Plan)

> **الغرض:** خطة هندسية كاملة لتحويل الـ research code الحالي لنظام إنتاجي حقيقي: automation للـ forward paper trading عبر Dagster + dashboard شامل للمراقبة والحُكم على الصفقات. مكتوبة لـ tech lead / AI engineer.
>
> **مبدأ حاكم:** نبني على اللي موجود (الـ edge متحقّق منه)، مش من الصفر. ونبني **بسيط أولاً** (paper trading) قبل أي تعقيد (HRP/portfolio). متبنيش طبقات معقّدة قبل ما الـ edge يثبت لايف.

---

## 0) الوضع الحالي (نقطة البداية)

عندنا research codebase شغّال ومتحقّق منه:
- pipeline كامل: كشف مناطق S&D → تقييم → labeling (Triple Barrier) → 25 features → XGBoost model (محفوظ في xgb_model.json).
- edge متحقّق منه: +0.42R/trade out-of-sample على الأصول النظيفة (us_stocks, commodities, etfs, fx, indices)، threshold 0.52.
- crypto/macro مستبعدين (خاسرين هيكلياً بالتكاليف).
- **الناقص:** كل ده كان batch research على داتا تاريخية. مفيش (1) تشغيل آلي مجدول، (2) توليد إشارات حية، (3) تنفيذ/تتبّع صفقات، (4) dashboard للمراقبة.

الهدف من الخطة: نبني الـ 4 حاجات دول كنظام إنتاجي مراقَب.

---

## 1) الـ Stack المختار (بأحدث الإصدارات — يونيو 2026)

### Orchestration: **Dagster 1.13.x** (آخر إصدار، Production/Stable)
- **ليه Dagster:** asset-based orchestration (مش job-based زي Airflow). كل مرحلة في الـ pipeline = "asset" ليه lineage واضح، observability، و freshness tracking. مثالي لـ data pipelines اللي بتعتمد على بعض.
- **الميزات اللي هنستخدمها:**
  - **Software-Defined Assets (SDA):** كل خطوة (download → detect → signal → execute) asset.
  - **FreshnessPolicy (بقت GA في 1.12):** نتأكد إن الداتا مش بايتة (مثلاً: إشارات الـ 1h لازم تتحدّث كل ساعة).
  - **Schedules + Sensors:** تشغيل مجدول (كل ساعة/يوم) + sensors تتفاعل مع أحداث.
  - **Schedule exclusions:** نتجاهل عطلات السوق/الويكند (ميزة في 1.12).
  - **Dagster UI:** مراقبة الـ runs، الـ lineage، الأخطاء (built-in).
- Python 3.10-3.14 (متوافق).
- `pip install dagster dagster-webserver`

### Dashboard: **Streamlit (للبداية) → Reflex (لو احتجنا real-time)**
- **القرار:** ابدأ بـ **Streamlit** للـ MVP — أسرع في البناء، عندنا dashboard.py بالفعل بـ Streamlit. كفاية لـ 80% من الاحتياجات (عرض صفقات، إحصائيات، شارت).
- **ترقية لاحقة لـ Reflex** *لو* احتجنا: real-time WebSocket updates (الصفقات بتتحدّث لحظياً)، routing متعدد الصفحات، state management حقيقي، auth. Reflex = Python كامل بيتحوّل لـ React. الـ migration path معروف.
- **متجنّبش:** Dash (callback fragmentation معقّد)، أو بناء frontend منفصل (مضيعة وقت في مرحلة الـ MVP).
- `pip install streamlit plotly` (الموجود) — والترقية لاحقاً `pip install reflex`.

### Storage: **DuckDB + Parquet** (للبداية) → **PostgreSQL** (للإنتاج الحقيقي)
- **DuckDB:** قاعدة بيانات تحليلية embedded، سريعة جداً مع Parquet، صفر إعداد. مثالية لتخزين الـ ledger والإشارات والإحصائيات في مرحلة الـ paper trading.
- **ترقية لـ PostgreSQL** لما نروح live (concurrent writes، durability، multi-process).
- `pip install duckdb`

### Data feed: **yfinance (الحالي) → broker API (للايف)**
- yfinance كفاية للـ paper trading (داتا مؤجّلة 15 دقيقة مقبولة للفريمات 1h+).
- للايف لاحقاً: broker API (Interactive Brokers / Alpaca للأسهم، OANDA للفوركس).

### اللغة/الأدوات المساعدة:
- `pydantic` v2 (الموجود في models.py) — validation للإشارات والصفقات.
- `pandas` + `numpy` (الموجود).
- `xgboost` + `scikit-learn` (الموجود).
- `apscheduler` كـ fallback بسيط لو Dagster overkill في البداية (بس Dagster أفضل).

---

## 2) المعمارية الكاملة (Architecture)

```
┌─────────────────────────────────────────────────────────────────┐
│                      DAGSTER ORCHESTRATION                        │
│                                                                   │
│  [Schedule: كل ساعة عند إغلاق الشمعة، باستثناء عطلات السوق]      │
│                                                                   │
│  asset_1: ingest_market_data                                      │
│    └─ يحمّل آخر شموع (1h/4h/1d) للأصول النظيفة من yfinance        │
│    └─ FreshnessPolicy: لازم تتحدّث كل ساعة                        │
│           ↓                                                       │
│  asset_2: detect_zones                                            │
│    └─ يشغّل pipeline الكشف+التقييم على الداتا الجديدة             │
│    └─ (detect_bases → formations → zones → scores → regime)       │
│           ↓                                                       │
│  asset_3: generate_signals                                        │
│    └─ يحسب features، يطبّق xgb_model.json، يفلتر threshold 0.52   │
│    └─ يطلّع: إشارات "خُد صفقة عند X، stop Y، tp Z"               │
│    └─ يفلتر الأصول النظيفة بس (crypto/macro مستبعدين)            │
│           ↓                                                       │
│  asset_4: paper_execute                                           │
│    └─ يفتح صفقات ورقية للإشارات الجديدة                          │
│    └─ يتابع الصفقات المفتوحة (هل ضربت SL/TP/timeout؟)            │
│    └─ يطبّق التكاليف (من costs.py) + concurrency cap (5)         │
│           ↓                                                       │
│  asset_5: update_ledger                                           │
│    └─ يحدّث DuckDB: الصفقات، النتايج، الإحصائيات الحية           │
│    └─ يحسب: win rate, net expectancy, drawdown مقابل المتوقّع    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              STREAMLIT DASHBOARD (يقرأ من DuckDB)                  │
│                                                                   │
│  صفحة 1: Live Trades — الصفقات المفتوحة دلوقتي + الـ R الحالي     │
│  صفحة 2: History — الصفقات المقفولة، فلترة بالأصل/الفريم/النتيجة  │
│  صفحة 3: Performance — win rate, expectancy, equity curve, DD     │
│           مقابل المتوقّع من الباك تيست (هل الأداء الحي مطابق؟)    │
│  صفحة 4: Charts — المناطق النشطة على الشارت (زي dashboard الحالي) │
│  صفحة 5: Signals Log — كل الإشارات اللي اتولّدت + ليه اتقبلت/رُفضت│
└─────────────────────────────────────────────────────────────────┘
```

---

## 3) خطة التنفيذ — مراحل (Phased Roadmap)

### المرحلة 0: الفهم (قبل أي بناء) ⚠️ أولوية مطلقة
**الهدف:** المستخدم (tech lead) يفهم كل سطر في الـ research code قبل ما يبني عليه نظام إنتاجي.
- جلسة مراجعة منظّمة، ملف ملف، بالترتيب: كشف → تقييم → labeling → features → model → backtest.
- لكل ملف: "إيه بيعمل" + "ليه القرار ده" + "إزاي بيتجنّب lookahead".
- **مخرَج:** المستخدم قادر يشرح أي جزء بثقة.
- **مدة تقديرية:** عدة جلسات.

### المرحلة 1: Refactor الـ research code لـ production-ready modules
**الهدف:** نحوّل السكريبتات (build_dataset, train_model) لـ functions قابلة لإعادة الاستخدام في Dagster assets.
- استخراج منطق "توليد إشارة لمنطقة واحدة" في function نظيفة `generate_signal(zone, model) → Signal | None`.
- استخراج منطق "متابعة صفقة" `update_trade(trade, latest_bars) → TradeUpdate`.
- تعريف Pydantic models: `Signal`, `PaperTrade`, `TradeOutcome`.
- **مهم:** الحفاظ على كل ضمانات الـ lookahead — الإشارة الحية تتولّد بنفس منطق الباك تيست بالظبط (الفرق الوحيد: مفيش مستقبل، بنشتغل على آخر شمعة مقفولة).
- **مخرَج:** library functions نظيفة + Pydantic schemas.

### المرحلة 2: DuckDB schema + الـ ledger
**الهدف:** قاعدة بيانات تخزّن كل حاجة.
- جداول: `signals` (كل إشارة اتولّدت)، `paper_trades` (الصفقات)، `equity_curve` (snapshots)، `performance_stats`.
- functions: `record_signal`, `open_trade`, `update_open_trades`, `close_trade`, `compute_stats`.
- **مخرَج:** DuckDB DB + data access layer.

### المرحلة 3: Dagster assets + scheduling
**الهدف:** الـ pipeline الآلي.
- الـ 5 assets الموضّحين فوق.
- Schedule: كل ساعة عند إغلاق الشمعة (مع schedule exclusions للويكند/العطلات).
- FreshnessPolicy على asset البيانات.
- error handling + retries (Dagster بيوفّرهم).
- **مخرَج:** Dagster project شغّال، بيتشغّل آلياً، مراقَب من Dagster UI.

### المرحلة 4: Streamlit dashboard
**الهدف:** المراقبة والحُكم.
- الـ 5 صفحات الموضّحين فوق.
- يقرأ من DuckDB (read-only).
- أهم صفحة: **Performance** — الأداء الحي مقابل المتوقّع (هل +0.42R بيتحقّق لايف؟).
- **مخرَج:** dashboard شغّال.

### المرحلة 5: Forward paper trading (التشغيل الفعلي)
**الهدف:** الاختبار الحقيقي على بيانات لايف.
- تشغيل النظام 2-3 شهور، paper فقط.
- مراقبة يومية من الـ dashboard.
- تسجيل: الأداء الحي، الانحراف عن المتوقّع، أي سلوك غريب.
- **معيار النجاح:** net expectancy لايف ≥ +0.20R (بعد slippage حقيقي) → الـ edge صحيح.
- **مخرَج:** قرار: نروح live ولا نعيد التفكير.

### المرحلة 6 (لو نجحت 5): طبقات التحسين + Live
- HRP / clustering لإدارة المخاطرة (لما الصفقات المتزامنة تكتر).
- regime-based position sizing.
- broker API integration (تنفيذ حقيقي).
- ترقية Streamlit → Reflex لو احتجنا real-time.
- ترقية DuckDB → PostgreSQL.

---

## 4) خوارزميات الـ Risk & Portfolio (من الـ shortlist — للمرحلة 6 مش قبلها)

من ملف الـ 35 algorithm، دول اللي ينفعوا للـ risk/portfolio (مش للـ edge):

1. **HRP (Hierarchical Risk Parity):** توزيع المخاطرة عبر صفقات متزامنة بحيث الأصول المترابطة (مثلاً عدة صفقات "دولار") ماتاخدش مخاطرة مجمّعة. **متى:** لما يكون فيه 5+ صفقات متزامنة بانتظام.

2. **Hierarchical/Spectral Clustering:** تجميع الأصول في بلوكات مخاطرة حسب الارتباط الفعلي (مش القطاع الاسمي). توزيع المخاطرة على بلوكات مستقلة. **متى:** لما يكبر universe الأصول.

3. **GMM/HMM regime (عملناه كـ feature):** ممكن يُستخدم كمان في **position sizing** — تصغّر الحجم في regime متقلّب، تكبّره في هدوء. **متى:** بعد ما الـ paper trading يأكّد الـ edge.

**مبدأ:** دول تحسينات للتنفيذ، مش للـ edge. **متبنيهمش قبل ما الـ edge يثبت لايف** (المرحلة 5). إضافتهم بدري = تعقيد بلا فايدة مثبتة.

---

## 5) قرارات معمارية محسومة

- **Orchestration:** Dagster (asset-based، مش Airflow job-based).
- **Dashboard:** Streamlit أولاً، Reflex لو احتجنا real-time.
- **Storage:** DuckDB أولاً، PostgreSQL للايف.
- **ابدأ بسيط:** paper trading بسيط قبل أي HRP/portfolio/regime-sizing.
- **حافظ على ضمانات الـ lookahead:** الإشارة الحية = نفس منطق الباك تيست على آخر شمعة مقفولة.
- **متروحش live قبل ما الـ forward paper test ينجح (≥+0.20R بعد slippage).**
- **الأصول النظيفة بس** (crypto/macro مستبعدين)، threshold ML 0.52.

---

## 6) الترتيب الموصى به للسيشن الجديدة

1. **ابدأ بالمرحلة 0 (الفهم):** راجع الكود ملف ملف. ده الأساس.
2. وازي بدء المرحلة 1 (refactor) بعد ما تفهم كل جزء.
3. بعدها 2 → 3 → 4 → 5 بالترتيب.
4. المرحلة 6 (التحسينات + live) بس بعد نجاح الـ paper trading.

**أول خطوة عملية في السيشن الجديدة:** قرّر — تبدأ بالفهم (مراجعة الكود)، ولا تبدأ بناء المرحلة 1 (الـ refactor) على طول؟ (الموصى به: الفهم أولاً).


---

## 7) تحديث: مرحلة features متقدمة قبل الإنتاج

قبل ما ندخل في الـ refactor الإنتاجي، فيه دورة تحسين أخيرة على الـ features (لأن الـ learning curve أكّد إن ده الطريق الوحيد المتبقّي للتحسين):

**المرحلة 0.5 (بين الفهم والإنتاج): إضافة 6 features متقدمة من المنهج**
- nesting (cross-TF) + FVG + liquidity sweep (تفاصيلها في OTA_SD_PROJECT_STATE.md قسم 17).
- إعادة بناء dataset + تدريب + قياس.
- لو حسّنت فوق +0.42R → نعتمدها في الموديل الإنتاجي.
- لو لأ → نمشي بالموديل الحالي (+0.42R) — مش بلوكر للإنتاج.

**مهم:** دي آخر دورة ML قبل ما نجمّد الموديل وننتقل للبناء الإنتاجي. متفضلش تلف في تحسين الموديل للأبد — بعد الدورة دي، الموديل **مجمّد** وبننتقل للـ forward test. الـ edge الحالي كفاية للبدء.

**الترتيب المحدّث:**
1. المرحلة 0: الفهم (مراجعة الكود ملف ملف). ← استخدم OTA_SD_LEARNING_GUIDE.md
2. المرحلة 0.5: الـ 6 features الجديدة + قياس. ← دورة ML أخيرة
3. المرحلة 1-6: الإنتاج (refactor → DuckDB → Dagster → dashboard → forward test → live).
