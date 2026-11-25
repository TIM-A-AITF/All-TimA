#!/bin/sh
set -e
mkdir -p "$TOR_RAB_WORKSPACE/runs" "$CHECKLIST_KATALOG_DIR"
# salin katalog SBM bawaan bila volume masih kosong (dibangun ulang otomatis bila SBM berubah)
if [ -d /app/Checklist-Rule-TOR-RAB/katalog_sbm ]; then
  cp -n /app/Checklist-Rule-TOR-RAB/katalog_sbm/*.json "$CHECKLIST_KATALOG_DIR"/ 2>/dev/null || true
fi
exec "$@"
