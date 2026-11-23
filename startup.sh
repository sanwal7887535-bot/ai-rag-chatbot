#!/usr/bin/env bash
set -euo pipefail

APP_PORT="${PORT:-${WEBSITES_PORT:-8501}}"

if [[ -n "${CHROMA_DB_DIR:-}" ]]; then
    mkdir -p "${CHROMA_DB_DIR}"
fi

if [[ -n "${LLM_LOG_PATH:-}" ]]; then
    mkdir -p "$(dirname "${LLM_LOG_PATH}")"
fi

if [[ -n "${HOME:-}" && ! -f "${HOME}/.streamlit/credentials.toml" ]]; then
    mkdir -p "${HOME}/.streamlit"
    printf '[general]\nemail = ""\n' > "${HOME}/.streamlit/credentials.toml"
fi

exec python -m streamlit run app.py \
    --server.port="${APP_PORT}" \
    --server.address="0.0.0.0" \
    --server.headless=true
