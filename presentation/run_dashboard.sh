#!/bin/sh
# Live demo dashboard. Run from anywhere: sh presentation/run_dashboard.sh   (opens http://localhost:8501)
cd "$(dirname "$0")/.." && exec .venv/bin/streamlit run presentation/dashboard_app.py \
  --theme.base light --theme.primaryColor "#2C5AA0" --theme.backgroundColor "#FFFFFF" --theme.secondaryBackgroundColor "#F5F5F7" \
  --theme.textColor "#1D1D1F" --browser.gatherUsageStats false --server.headless false
