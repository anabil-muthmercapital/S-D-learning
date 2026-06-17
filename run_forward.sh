#!/bin/bash
# =============================================================================
# run_forward.sh — تشغيل دورة الـ forward test كاملة (محرّك + متابع)
# =============================================================================
# بيشغّل forward_test.py (يكتشف إشارات جديدة) ثم update_signals.py (يتابع
# الـ pending)، ويسجّل كل تشغيلة في log بالتاريخ. مصمّم عشان يتنادى من cron
# تلقائياً — فمتعتمدش على إنك تفتكر تشغّله بنفسك.
#
# الاستخدام اليدوي:
#   cd /Users/an/Desktop/S-D-learning && ./run_forward.sh
#
# أو من cron (شوف التعليمات في رد Claude).
# =============================================================================

set -uo pipefail

# جذر المشروع
PROJECT_DIR="/Users/an/Desktop/S-D-learning"
cd "$PROJECT_DIR"

# الـ Python بتاع الـ venv (uv-managed). الـ symlink ده بيوصل للـ interpreter
# الصح ومعاه كل المكتبات المثبّتة للمشروع.
PYTHON="$PROJECT_DIR/.venv/bin/python"

# مجلد الـ logs (بيتعمل لو مش موجود)
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# ملف log واحد لكل يوم
TODAY=$(date +%Y-%m-%d)
LOG_FILE="$LOG_DIR/forward_$TODAY.log"

# طابع زمني لكل تشغيلة
{
  echo ""
  echo "=================================================================="
  echo "RUN @ $(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "=================================================================="
} >> "$LOG_FILE"

# 1) المحرّك — يكتشف إشارات جديدة (من نقطة البداية المثبّتة)
echo "--- forward_test.py ---" >> "$LOG_FILE"
"$PYTHON" forward_test.py >> "$LOG_FILE" 2>&1 || echo "[warn] forward_test.py exited non-zero" >> "$LOG_FILE"

# 2) المتابع — يحدّث حالة الإشارات الـ pending/open
echo "--- update_signals.py ---" >> "$LOG_FILE"
"$PYTHON" update_signals.py >> "$LOG_FILE" 2>&1 || echo "[warn] update_signals.py exited non-zero" >> "$LOG_FILE"

echo "--- done @ $(date '+%H:%M:%S') ---" >> "$LOG_FILE"
