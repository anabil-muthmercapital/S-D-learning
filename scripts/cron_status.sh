#!/bin/bash
# =============================================================================
# cron_status.sh — quick health check for the scheduled forward test.
# Shows: cron entries, last run times, signal counts, recent log tails.
# Usage: scripts/cron_status.sh
# =============================================================================
set -u

PROJECT_ROOT="/Users/an/Desktop/S-D-learning"
LOG_DIR="${PROJECT_ROOT}/data/logs"
SIGNALS="${PROJECT_ROOT}/data/forward_signals.csv"
PIN="${PROJECT_ROOT}/data/forward_test_start.json"

bar() { printf '%.0s=' {1..72}; echo; }

bar
echo "OTA S&D — Forward Test Status   ($(date '+%Y-%m-%d %H:%M:%S %Z'))"
bar

echo ""
echo "[ Pin file ]"
if [[ -f "${PIN}" ]]; then
  cat "${PIN}"
else
  echo "  (no pin file — run forward_test.py once to create it)"
fi

echo ""
echo "[ Cron entries ]"
if crontab -l 2>/dev/null | grep -q cron_forward.sh; then
  crontab -l | grep cron_forward.sh | sed 's/^/  /'
else
  echo "  (no cron entries installed — see scripts/crontab.txt)"
fi

echo ""
echo "[ Signal log ]"
if [[ -f "${SIGNALS}" ]]; then
  TOTAL=$(($(wc -l < "${SIGNALS}") - 1))
  echo "  rows: ${TOTAL}"
  if [[ "${TOTAL}" -gt 0 ]]; then
    # tally by status (column 14 of the schema)
    awk -F, 'NR>1 {print $14}' "${SIGNALS}" | sort | uniq -c | sed 's/^/    /'
  fi
else
  echo "  (no signal log yet)"
fi

echo ""
echo "[ Last 5 lines — forward_test.log ]"
if [[ -f "${LOG_DIR}/forward_test.log" ]]; then
  tail -5 "${LOG_DIR}/forward_test.log" | sed 's/^/  /'
else
  echo "  (no log yet)"
fi

echo ""
echo "[ Last 5 lines — update_signals.log ]"
if [[ -f "${LOG_DIR}/update_signals.log" ]]; then
  tail -5 "${LOG_DIR}/update_signals.log" | sed 's/^/  /'
else
  echo "  (no log yet)"
fi

echo ""
echo "[ Running processes ]"
PS_OUT=$(ps aux | grep -E "forward_test\.py|update_signals\.py" | grep -v grep)
if [[ -z "${PS_OUT}" ]]; then
  echo "  (none running right now)"
else
  echo "${PS_OUT}" | awk '{printf "  pid=%-7s started=%-9s cpu=%s%%  cmd=%s %s\n", $2, $9, $3, $11, $12}'
fi

echo ""
bar
