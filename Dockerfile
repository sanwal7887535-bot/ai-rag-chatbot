FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface \
    CHROMA_DB_DIR=/app/data/chroma_db \
    LLM_LOG_PATH=/app/data/llm_calls.jsonl \
    LOG_REDACT_CONTENT=true \
    PORT=8501

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --system appuser && \
    useradd --system --gid appuser --create-home --home-dir /home/appuser appuser

COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# Pre-download the embedding model into the image cache for reliable Azure cold starts.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY app.py config.py rag_pipeline.py tools.py llm_client.py startup.sh ./
COPY .streamlit/config.toml ./.streamlit/config.toml

RUN sed -i 's/\r$//' /app/startup.sh && \
    chmod +x /app/startup.sh && \
    mkdir -p /app/data/chroma_db /home/appuser/.streamlit && \
    printf '[general]\nemail = ""\n' > /home/appuser/.streamlit/credentials.toml && \
    chown -R appuser:appuser /app /home/appuser/.streamlit

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\", \"8501\")}/_stcore/health', timeout=5)"

ENTRYPOINT ["./startup.sh"]
