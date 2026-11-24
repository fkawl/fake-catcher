#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MODE="${1:-cpu}"
case "$MODE" in
  cpu) INDEX=https://download.pytorch.org/whl/cpu ;;
  cu124) INDEX=https://download.pytorch.org/whl/cu124 ;;
  *) echo 'Usage: bash scripts/setup_linux.sh cpu|cu124'; exit 2 ;;
esac
python3 -m venv .venv
.venv/bin/python -m pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0 --index-url "$INDEX"
.venv/bin/python -m pip install --no-cache-dir -r requirements.txt
.venv/bin/python -m fakecatcher.doctor
