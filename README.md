# AI RAG Chatbot (Azure-Ready)

Unified Streamlit chatbot supporting multi-turn conversation, hybrid PDF RAG (BM25 + ChromaDB cosine similarity via Reciprocal Rank Fusion), and tool calling (`get_current_datetime`, `search_web`).

## Architecture

- [app.py](app.py) — Unified Streamlit UI (`Local / Cloud AI RAG Chatbot`), PDF upload/management, optional pasted reference text, source citations, and session turn logs.
- [config.py](config.py) — Centralized environment-first configuration (`GROQ` default, optional `OLLAMA` / `GEMINI`), validation, and secret sanitization.
- [rag_pipeline.py](rag_pipeline.py) — PDF text extraction (`pymupdf`), chunking (`langchain-text-splitters`), embeddings (`all-MiniLM-L6-v2`), persistent ChromaDB (`RAGStore`), legacy collection migration, and hybrid RRF retrieval.
- [tools.py](tools.py) — Validated `ToolRegistry`, timezone-aware `get_current_datetime`, and lazily initialized Tavily `search_web`.
- [llm_client.py](llm_client.py) — Shared OpenAI-compatible client (`groq` / `ollama`) and `google-genai` client (`gemini`), prompt-injection isolation (untrusted reference context kept out of `system` role), and redacted JSONL audit logging.
- [test_rag.py](test_rag.py) — Isolated `unittest` test suite covering config validation, security sanitization, RAG indexing/deletion/retrieval, tool execution, and mocked LLM orchestration.

## Local Development

1. Activate the virtual environment and install dependencies:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Configure secrets via environment variables (see [.env.example](.env.example)) or `.streamlit/secrets.toml` (never committed):
   ```toml
   GROQ_API_KEY = "..."
   TAVILY_API_KEY = "..."
   ```
3. Run the application:
   ```powershell
   streamlit run app.py
   ```
4. Run the test suite:
   ```powershell
   python -m unittest test_rag.py -v
   ```

## Security Design

- **No committed secrets**: `.gitignore` and `.dockerignore` exclude `.streamlit/secrets.toml`, `.env`, `chroma_db/`, and `llm_calls.jsonl`.
- **Environment-first secrets + Key Vault ready**: `config.py` reads environment variables before any optional local `secrets.toml`.
- **Error & log sanitization**: Configured secrets are scrubbed from UI errors and logs (`[REDACTED]`), `client.showErrorDetails = "none"` hides tracebacks from browsers, and `LOG_REDACT_CONTENT=true` redacts prompt/document contents in production JSONL logs.
- **Upload & input hardening**: Enforces PDF magic-header checks, password-protected PDF rejection, `MAX_PDF_MB` size limits, filename sanitization, `MAX_PROMPT_CHARS`, `MAX_CONTEXT_CHARS`, and `MAX_CHAT_HISTORY`.
- **Prompt-injection isolation**: Retrieved PDF chunks and pasted text are placed in user-role reference sections rather than elevated `system` messages, and system instructions treat reference data as untrusted text.
- **Streamlit protections**: `server.enableCORS = true`, `server.enableXsrfProtection = true`, and `browser.gatherUsageStats = false` in [.streamlit/config.toml](.streamlit/config.toml).
- **Non-root container**: `Dockerfile` creates and runs as `appuser`.

## Azure Deployment

> [!IMPORTANT]
> ChromaDB uses embedded SQLite/HNSW files on disk. Keep `minReplicas: 1` / `maxReplicas: 1` (single instance) and point `CHROMA_DB_DIR` to persistent storage (Azure Files mount on Container Apps or `/home/data/chroma_db` on App Service) so uploaded PDFs survive restarts without SQLite locking conflicts.

### Option A: Azure Container Apps (Recommended)

1. Build and push the container image to Azure Container Registry (ACR):
   ```bash
   az acr build --registry <ACR_NAME> --image ai-rag-chatbot:latest .
   ```
2. Provision an Azure Container Apps environment with an Azure Storage (`AzureFile`) share mounted at `/app/data`, and grant the Container App's Managed Identity **Key Vault Secrets User** access to your Azure Key Vault (`GROQ-API-KEY`, `TAVILY-API-KEY`).
3. Update placeholders in [azure-container-app.yaml](azure-container-app.yaml) and deploy:
   ```bash
   az containerapp create --resource-group <RG> --name ai-rag-chatbot --yaml azure-container-app.yaml
   ```
4. Enable **Microsoft Entra ID Authentication (Easy Auth)** in the Container App (`Authentication` blade) so only authorized tenant users can access the UI.

### Option B: Azure App Service for Linux

1. Create a Python 3.13 Linux Web App and enable WebSockets + persistent App Service storage:
   ```bash
   az webapp config set --resource-group <RG> --name <APP_NAME> --startup-file "bash startup.sh" --web-sockets-enabled true
   az webapp config appsettings set --resource-group <RG> --name <APP_NAME> --settings \
     WEBSITES_PORT=8501 \
     WEBSITES_ENABLE_APP_SERVICE_STORAGE=true \
     SCM_DO_BUILD_DURING_DEPLOYMENT=true \
     LLM_PROVIDER=groq \
     LLM_MODEL=openai/gpt-oss-120b \
     ALLOW_PROVIDER_SWITCH=false \
     CHROMA_DB_DIR=/home/data/chroma_db \
     LLM_LOG_PATH=/home/data/llm_calls.jsonl \
     LOG_REDACT_CONTENT=true \
     GROQ_API_KEY="@Microsoft.KeyVault(VaultName=<KV_NAME>;SecretName=GROQ-API-KEY)" \
     TAVILY_API_KEY="@Microsoft.KeyVault(VaultName=<KV_NAME>;SecretName=TAVILY-API-KEY)"
   ```
2. Enable **System-assigned Managed Identity** on the Web App, grant it **Key Vault Secrets User** on `<KV_NAME>`, and enable **Microsoft Entra ID Authentication (Easy Auth)** in the App Service `Authentication` settings.
