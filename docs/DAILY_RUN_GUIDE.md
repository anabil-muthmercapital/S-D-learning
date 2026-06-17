# 🚀 FORWARD TEST — دليل التشغيل اليومي

> الملف ده هو الـ **playbook اليومي** بتاعك. كل ما تفتح اللاپتوب اعمل الخطوات بالترتيب.

---

## 📋 جدول المحتويات

1. [الـ Setup مرة واحدة فقط](#1-الـ-setup-مرة-واحدة-فقط)
2. [التشغيل اليومي — 3 خطوات](#2-التشغيل-اليومي--3-خطوات)
3. [الـ Dashboard — لو عايز تتفرّج](#3-الـ-dashboard--لو-عايز-تتفرّج)
4. [استكشاف الأخطاء (Troubleshooting)](#4-استكشاف-الأخطاء-troubleshooting)
5. [الـ Cron — لو غيّرت رأيك ومش هتعمل shutdown](#5-الـ-cron--لو-غيّرت-رأيك-ومش-هتعمل-shutdown)
6. [أوامر مهمة لازم تعرفها](#6-أوامر-مهمة-لازم-تعرفها)

---

## 1. الـ Setup مرة واحدة فقط

### A. تأكد من الـ Python

```bash
/opt/anaconda3/bin/python --version
```

لازم يطلع: `Python 3.13.x`. لو طلع غير كده اتكلم معايا.

### B. تأكد من الـ pin file

```bash
cat /Users/an/Desktop/S-D-learning/data/forward_test_start.json
```

**لازم تشوف:**

```json
{
  "forward_test_start": "2026-06-16T00:00:00+00:00",
  "written_at": "2026-06-16T11:46:44+00:00",
  "reason": "first-run init"
}
```

⚠️ **لو الـ date تغيّر أو الملف مش موجود → لا تكمل، اتصل بـ Copilot.**

### C. اعمل alias مرة واحدة (يوفّر عليك كتابة كل يوم)

افتح ملف الـ zshrc:

```bash
open -a TextEdit ~/.zshrc
```

ضيف السطور دي في آخر الملف:

```bash
# OTA S&D Forward Test shortcuts
alias sd-go='cd ~/Desktop/S-D-learning && scripts/cron_forward.sh forward && scripts/cron_forward.sh update'
alias sd-status='cd ~/Desktop/S-D-learning && scripts/cron_status.sh'
alias sd-dash='cd ~/Desktop/S-D-learning && streamlit run dashboard_pipeline.py'
alias sd-log='tail -f ~/Desktop/S-D-learning/data/logs/forward_test.log'
```

احفظ واقفل. بعدين شغّل:

```bash
source ~/.zshrc
```

✅ **خلصت الـ setup. كده عندك 4 أوامر سحرية:**

- `sd-go` — يشغّل forward_test + update_signals
- `sd-status` — يوريك الحالة بسرعة
- `sd-dash` — يفتح الـ dashboard
- `sd-log` — يتابع الـ log live

---

## 2. التشغيل اليومي — 3 خطوات

### 🌅 أول ما تفتح اللاپتوب الصبح

```bash
sd-go
```

ده هيشغّل اتنين بالترتيب:

1. `forward_test.py` → يلقّط أي signals جديدة اتكوّنت من آخر مرة شغّلت
2. `update_signals.py` → يحدّث الـ pending/open signals (لو السعر لمس entry/tp/sl)

⏱ **هياخد ~5-10 دقايق** (لإن بيـ download fresh data من yfinance لـ 50 رمز).

### 🕐 خلال اليوم (كل 1-2 ساعة لو فاضي)

```bash
sd-go
```

كل ما تشغّله أكتر → تفوّت trades أقل. لو شغّلته مرة واحدة في اليوم هتفوّت ~70% من الـ signals.

| تشغّل كل     | بتلقط من الـ trades |
| ------------ | ------------------- |
| كل ساعة      | ~85%                |
| كل ساعتين    | ~65%                |
| كل 4 ساعات   | ~40%                |
| مرة في اليوم | ~20%                |

### 🌙 قبل ما تقفل اللاپتوب بالليل

```bash
sd-go
```

عشان تحدّث آخر signals قبل النوم. الـ Mac هيقفل، يصحى تاني بكرة، تشغّل تاني.

### 🔍 لو عايز تتأكد إن كل حاجة تمام

```bash
sd-status
```

هيوريك:

- الـ pin date (لازم 2026-06-16)
- عدد الـ signals (pending / open / closed)
- آخر 5 سطور من الـ log
- الـ processes الشغّالة دلوقتي

---

## 3. الـ Dashboard — لو عايز تتفرّج

```bash
sd-dash
```

هيفتح في الـ browser على [http://localhost:8501](http://localhost:8501)

من الـ sidebar اختار:

**🎬 Scenario Player** → عشان تشوف الـ pipeline بيشتغل على أي symbol step-by-step.

**📡 Forward Live** → عشان تشوف الـ forward test:

- 📊 الـ metrics (Closed · Win Rate · Total R · Avg R/trade vs baseline)
- 📈 الـ Equity Curve (مقارنة بالـ backtest baseline 0.42R)
- 🔍 الـ filters (Symbol / TF / Status)
- 📋 الـ table بكل الـ signals
- 🎯 رسم أي signal بعينه مع entry/stop/tp markers

**أزرار مهمة في الـ Forward Live page:**

- `▶️ تشغيل forward_test` — نفس `sd-go` بس من غير الـ update
- `🔄 تحديث الـ signals` — يشغّل update_signals.py
- `🧹 إعادة تحميل الـ CSV` — لو عملت تغيير من الـ terminal والـ dashboard مش بيظهره

---

## 4. استكشاف الأخطاء (Troubleshooting)

### ❌ `command not found: sd-go`

```bash
source ~/.zshrc
```

أو اقفل الـ terminal وافتحه تاني.

### ❌ `scripts/cron_forward.sh: Permission denied`

```bash
chmod +x ~/Desktop/S-D-learning/scripts/*.sh
```

### ❌ الـ output بيقول `[skip] previous run still active`

في run قديم لسه شغّال. شوف بـ:

```bash
ps aux | grep forward_test | grep -v grep
```

لو في process قديم عمره أكتر من 15 دقيقة (stuck) اقتله بـ:

```bash
pkill -f "forward_test.py"
pkill -f "update_signals.py"
rm -rf ~/Desktop/S-D-learning/data/locks/
```

### ❌ مفيش signals بعد ساعات من التشغيل

ده **مش error**. الـ S&D system بيطلع ~1 signal كل 3 أيام لكل 5 رموز. شوف الـ log:

```bash
sd-status
```

لازم تشوف سطور زي:

```
historical zones skipped (formed before start) : XXX
```

ده معناه إن الـ pipeline شغّال وبيرفض الـ zones التاريخية صح.

### ❌ الـ dashboard فاضي (Forward Live)

اضغط `🔄 تحديث الـ signals` و `🧹 إعادة تحميل الـ CSV`.

### ❌ خطأ `libxgboost.dylib` أو `libomp.dylib`

لا تستخدم `.venv` — استخدم anaconda python:

```bash
which python   # لازم يطلع /opt/anaconda3/bin/python
```

كل الـ scripts بتستخدم `/opt/anaconda3/bin/python` مباشرة، فمفيش مشكلة.

### ⚠️ الـ pin file اتغيّر لوحده

ده مش مفروض يحصل. لو لقيت `forward_test_start` غير `2026-06-16`، الـ system هيرفض يشتغل بفلتر `--since` أصغر منه. **اتصل بـ Copilot قبل ما تعمل أي حاجة.**

---

## 5. الـ Cron — لو غيّرت رأيك ومش هتعمل shutdown

لو يوم قررت تخلي الـ Mac شغّال 24/7 وعايز كل حاجة تتعمل automatic كل ساعة:

### تركيب الـ cron

```bash
( crontab -l 2>/dev/null; cat ~/Desktop/S-D-learning/scripts/crontab.txt ) | crontab -
```

تأكد:

```bash
crontab -l
```

لازم تشوف:

```
6 * * * * /Users/an/Desktop/S-D-learning/scripts/cron_forward.sh forward
16 * * * * /Users/an/Desktop/S-D-learning/scripts/cron_forward.sh update
```

### شيل الـ cron

```bash
crontab -l | grep -v cron_forward.sh | crontab -
```

⚠️ **مهم:** الـ cron مفيد بس لو الـ Mac شغّال 24/7. لو بتعمل shutdown → مش هيفيد.

### بديل أحسن لو الـ Mac هيبقى مقفول معظم اليوم

**GitHub Actions** — بيشتغل على servers تانية، مجاني، 24/7 فعليًا. اتكلم مع Copilot لو عايز تعمله.

---

## 6. أوامر مهمة لازم تعرفها

### شغّل dataset جديد (لو عملت تغيير في الـ pipeline)

```bash
cd ~/Desktop/S-D-learning
/opt/anaconda3/bin/python build_dataset.py
```

### إعادة تدريب الموديل (لو عندك dataset جديد)

```bash
/opt/anaconda3/bin/python train_model.py
```

### اعمل backtest

```bash
/opt/anaconda3/bin/python backtest.py
```

### شوف الـ logs live

```bash
sd-log
```

اضغط `Ctrl+C` للخروج.

### مسح الـ logs لو كبرت

```bash
: > ~/Desktop/S-D-learning/data/logs/forward_test.log
: > ~/Desktop/S-D-learning/data/logs/update_signals.log
```

### ⚠️ إعادة تعيين الـ forward test (DESTRUCTIVE)

**لا تعمل ده إلا لو متأكد جدًا.** بيمسح الـ pin date ويعيد تعيينها للنهارده:

```bash
cd ~/Desktop/S-D-learning
/opt/anaconda3/bin/python forward_test.py --reset-forward-test --yes-i-really-want-to-reset
```

---

## 📌 خلاصة سريعة (TL;DR)

```bash
# لما تفتح اللاپتوب الصبح:
sd-go

# خلال اليوم لما تفتكر:
sd-go

# عايز تتفرج:
sd-dash

# عايز تتأكد:
sd-status

# قبل ما تقفل اللاپتوب:
sd-go
```

ده كل اللي محتاجه. سيب الباقي للـ system.

---

## 📂 خرايط الملفات المهمة

| الملف                          | الوصف                                 |
| ------------------------------ | ------------------------------------- |
| `data/forward_test_start.json` | الـ pin date (immutable)              |
| `data/forward_signals.csv`     | الـ signals log (forward test data)   |
| `data/logs/forward_test.log`   | log كل runs الـ forward_test          |
| `data/logs/update_signals.log` | log كل runs الـ update_signals        |
| `data/locks/`                  | الـ locks اللي بتمنع التشغيل المتزامن |
| `scripts/cron_forward.sh`      | الـ wrapper الرئيسي                   |
| `scripts/cron_status.sh`       | الـ health check                      |
| `scripts/crontab.txt`          | الـ cron entries (للنسخ والتركيب)     |

---

**آخر تحديث:** 2026-06-17
**الـ Pin Date:** 2026-06-16 (immutable)
