#!/bin/bash
# =============================================================================
# cron_forward.sh — single entry-point invoked by launchd / cron.
#
# Why this wrapper instead of calling python directly from crontab:
#
#   1. Cron has an almost-empty PATH and no shell rc files. We must use an
#      absolute path to python AND cd into the project so the in-repo
#      `utils/` package imports resolve.
#
#   2. We must serialise runs with flock — if a slow yfinance pull is still
#      running when the next cron minute fires, two writers would race on
#      data/forward_signals.csv and corrupt it. flock makes the second run
#      exit immediately (non-blocking).
#
#   3. We append to a single log file per script with timestamps so it is
#      readable and survives across runs. Rotate manually when it gets big
#      (or `: > data/logs/forward_test.log` to truncate).
#
# Usage (from crontab):
#   /Users/an/Desktop/S-D-learning/scripts/cron_forward.sh forward
#   /Users/an/Desktop/S-D-learning/scripts/cron_forward.sh update
# =============================================================================
set -uo pipefail

PROJECT_ROOT="/Users/an/Desktop/S-D-learning"
PYTHON="/opt/anaconda3/bin/python"
LOCK_DIR="${PROJECT_ROOT}/data/locks"
LOG_DIR="${PROJECT_ROOT}/data/logs"

mkdir -p "${LOCK_DIR}" "${LOG_DIR}"

MODE="${1:-}"
case "${MODE}" in
  forward)
    SCRIPT="forward_test.py"
    LOCK="${LOCK_DIR}/forward_test.lock"
    LOG="${LOG_DIR}/forward_test.log"
    ;;
  update)
    SCRIPT="update_signals.py"
    LOCK="${LOCK_DIR}/update_signals.lock"
    LOG="${LOG_DIR}/update_signals.log"
    ;;
  *)
    echo "usage: $0 {forward|update}" >&2
    exit 2
    ;;
esac

cd "${PROJECT_ROOT}" || exit 3

# Atomic, portable lock (macOS has no flock by default).
# mkdir succeeds only if the directory does not exist, and is atomic on the
# kernel level — so two simultaneous runs can never both succeed.
# Stale locks (process killed) are cleaned up by checking the PID inside.
LOCK_DIR_PATH="${LOCK}.d"
if ! mkdir "${LOCK_DIR_PATH}" 2>/dev/null; then
  # Lock exists — check if owner is still alive.
  if [[ -f "${LOCK_DIR_PATH}/pid" ]]; then
    OLD_PID="$(cat "${LOCK_DIR_PATH}/pid" 2>/dev/null || echo '')"
    if [[ -n "${OLD_PID}" ]] && kill -0 "${OLD_PID}" 2>/dev/null; then
      printf '[%s] [skip] previous %s run still active (pid=%s) — skipping\n' \
        "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${MODE}" "${OLD_PID}" >> "${LOG}"
      exit 0
    fi
    # Stale lock — owner gone. Take it over.
    printf '[%s] [warn] stale lock from pid=%s — taking over\n' \
      "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${OLD_PID}" >> "${LOG}"
  fi
  # Force-acquire by replacing the pid file.
fi
echo "$$" > "${LOCK_DIR_PATH}/pid"
trap 'rm -rf "${LOCK_DIR_PATH}"' EXIT

{
  printf '\n========================================================================\n'
  printf '[%s] starting %s\n' "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${SCRIPT}"
  printf '========================================================================\n'
  "${PYTHON}" "${SCRIPT}"
  EXIT_CODE=$?
  printf '[%s] %s finished (exit=%d)\n' \
    "$(date -u +'%Y-%m-%dT%H:%M:%SZ')" "${SCRIPT}" "${EXIT_CODE}"
} >> "${LOG}" 2>&1
