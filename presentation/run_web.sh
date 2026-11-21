#!/bin/sh
# React demo: builds presentation/web once (offline after npm install), then serves it with the model API on http://127.0.0.1:8600
cd "$(dirname "$0")/web" || exit 1
zsh -ic 'npm run build --silent' >/dev/null || exit 1
cd ../.. || exit 1
(sleep 4 && open http://127.0.0.1:8600) &
exec .venv/bin/uvicorn --app-dir presentation/web server:app --host 127.0.0.1 --port 8600
