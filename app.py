"""Unified Streamlit UI for multi-turn AI chat, tool calling, and hybrid PDF RAG."""

from __future__ import annotations

from typing import Any

import streamlit as st

from config import DEFAULT_MODELS, Settings, load_settings
from llm_client import format_reference_context, generate_reply
from rag_pipeline import (
    RAGStore,
    extract_pdf_bytes,
    sanitize_filename,
    split_text_into_chunks,
)
from tools import create_tool_registry

NO_PDF_OPTION = "None (General Chat / Web)"


@st.cache_resource
def get_rag_store(db_dir: str, embedding_model_name: str) -> RAGStore:
    """Cache the ChromaDB client and embedding model across Streamlit reruns."""
    return RAGStore(db_dir=db_dir, embedding_model_name=embedding_model_name)


def _get_easy_auth_user() -> str | None:
    """Read optional Azure Easy Auth user identity header if present."""
    context = getattr(st, "context", None)
    headers = getattr(context, "headers", None) if context is not None else None
    if not headers:
        return None
    return headers.get("X-MS-CLIENT-PRINCIPAL-NAME") or headers.get(
        "x-ms-client-principal-name"
    )


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm_logs" not in st.session_state:
        st.session_state.llm_logs = []
    if "selected_pdf" not in st.session_state:
        st.session_state.selected_pdf = NO_PDF_OPTION


def _render_sidebar(
    settings: Settings, store: RAGStore
) -> tuple[str, str, list[dict[str, Any]]]:
    with st.sidebar:
        st.header("Configuration & Documents")

        auth_user = _get_easy_auth_user()
        if auth_user:
            st.info(f"Signed in as: `{auth_user}`")

        available_providers = settings.available_providers()
        if settings.allow_provider_switch and len(available_providers) > 1:
            active_provider = st.selectbox(
                "LLM Provider",
                options=available_providers,
                index=0,
                key="active_provider",
                format_func=lambda item: item.upper(),
            )
        else:
            active_provider = settings.llm_provider

        active_model = (
            settings.llm_model
            if active_provider == settings.llm_provider
            else DEFAULT_MODELS[active_provider]
        )

        st.caption(f"Provider: **{active_provider.upper()}** (`{active_model}`)")
        web_status = "Enabled (Tavily)" if settings.tavily_api_key else "Disabled"
        st.caption(f"Web Search: {web_status} · TZ: `{settings.default_timezone}`")

        st.divider()
        st.subheader("PDF Knowledge Base")
        uploaded_files = st.file_uploader(
            "Upload PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            max_upload_size=settings.max_pdf_mb,
        )

        if st.button("Index uploaded PDFs", disabled=not uploaded_files):
            for uploaded_file in uploaded_files:
                if uploaded_file.size > settings.max_pdf_bytes:
                    st.error(
                        f"{uploaded_file.name} exceeds the {settings.max_pdf_mb} MB limit."
                    )
                    continue

                try:
                    safe_name = sanitize_filename(uploaded_file.name)
                    with st.spinner(f"Indexing {safe_name}..."):
                        pages = extract_pdf_bytes(
                            uploaded_file.getvalue(),
                            max_bytes=settings.max_pdf_bytes,
                        )
                        chunks = split_text_into_chunks(
                            pages,
                            chunk_size=settings.chunk_size,
                            chunk_overlap=settings.chunk_overlap,
                        )
                        if not chunks:
                            st.warning(
                                f"{safe_name} contains no extractable text. "
                                "Scanned PDFs require OCR before indexing."
                            )
                            continue
                        embeddings = store.generate_embeddings(chunks)
                        stored_count = store.store_chunks(chunks, embeddings, safe_name)
                        st.session_state.selected_pdf = safe_name
                    st.success(
                        f"Indexed {safe_name}: {stored_count} chunks from {len(pages)} pages."
                    )
                except Exception as error:
                    safe_error = settings.sanitize_text(str(error))
                    st.error(f"Could not index {uploaded_file.name}: {safe_error}")

        indexed_docs = store.get_indexed_documents()
        if indexed_docs:
            with st.expander(f"Manage indexed PDFs ({len(indexed_docs)})"):
                for doc in indexed_docs:
                    st.write(
                        f"**{doc['filename']}** — {doc['chunks']} chunks, "
                        f"{len(doc['pages'])} pages"
                    )

                doc_to_remove = st.selectbox(
                    "Remove an indexed PDF",
                    options=[doc["filename"] for doc in indexed_docs],
                    key="document_to_remove",
                )
                if st.button("Remove selected PDF"):
                    removed = store.remove_document(doc_to_remove)
                    if st.session_state.get("selected_pdf") == doc_to_remove:
                        st.session_state.selected_pdf = NO_PDF_OPTION
                    st.success(f"Removed {doc_to_remove} ({removed} chunks).")
                    st.rerun()

        st.divider()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.llm_logs = []
            st.rerun()

    return active_provider, active_model, indexed_docs


def main() -> None:
    st.set_page_config(
        page_title="AI RAG Chatbot",
        page_icon="💬",
        layout="centered",
    )

    try:
        settings = load_settings()
        store = get_rag_store(
            str(settings.chroma_db_dir),
            settings.embedding_model_name,
        )
    except Exception as error:
        st.error(f"Application startup error: {error}")
        st.stop()

    _init_session_state()
    active_provider, active_model, indexed_docs = _render_sidebar(settings, store)
    tool_registry = create_tool_registry(settings)

    st.title("AI RAG Chatbot")
    st.caption(f"Powered by {active_provider.upper()} ({active_model})")

    pdf_options = [NO_PDF_OPTION, *[doc["filename"] for doc in indexed_docs]]
    if st.session_state.get("selected_pdf") not in pdf_options:
        st.session_state.selected_pdf = NO_PDF_OPTION

    selected_pdf = st.selectbox(
        "Document context",
        options=pdf_options,
        key="selected_pdf",
    )

    with st.expander("Optional pasted reference text"):
        pasted_text = st.text_area(
            "Paste text to query alongside or instead of a PDF:",
            height=150,
            max_chars=settings.max_context_chars,
            placeholder="Paste reference text here (leave empty for normal chat/web/PDF)...",
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                st.caption("Retrieved sources: " + "; ".join(message["sources"]))

    if raw_prompt := st.chat_input("Type your message here..."):
        prompt = raw_prompt.strip()
        if not prompt:
            st.warning("Please enter a non-empty message.")
            st.stop()
        if len(prompt) > settings.max_prompt_chars:
            st.warning(
                f"Message exceeds the {settings.max_prompt_chars}-character limit."
            )
            st.stop()

        turn_log: dict[str, Any] = {
            "prompt": "[REDACTED]" if settings.log_redact_content else prompt,
            "events": [],
        }
        st.session_state.llm_logs.append(turn_log)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            retrieved_chunks: list[dict[str, Any]] = []
            if selected_pdf != NO_PDF_OPTION:
                retrieval_results = store.retrieve_relevant_chunks(
                    prompt,
                    filename=selected_pdf,
                    k=settings.rag_top_k,
                )
                documents = retrieval_results["documents"][0]
                metadatas = retrieval_results["metadatas"][0]
                retrieved_chunks = [
                    {"text": text, **metadata}
                    for text, metadata in zip(documents, metadatas)
                ]

            reference_context, sources = format_reference_context(
                retrieved_chunks,
                pasted_text,
                max_chars=settings.max_context_chars,
            )

            chat_messages = [
                {"role": msg["role"], "content": msg["content"]}
                for msg in st.session_state.messages
            ]

            with st.chat_message("assistant"):
                with st.spinner("Generating response..."):
                    answer = generate_reply(
                        messages=chat_messages,
                        settings=settings,
                        tool_registry=tool_registry,
                        turn_log=turn_log,
                        reference_context=reference_context,
                        provider=active_provider,
                        model=active_model,
                    )
                    st.markdown(answer)
                    if sources:
                        st.caption("Retrieved sources: " + "; ".join(sources))

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )

        except Exception as error:
            error_message = settings.sanitize_text(
                str(error) or type(error).__name__
            )
            turn_log["events"].append(
                {
                    "type": "error",
                    "error_type": type(error).__name__,
                    "message": error_message,
                }
            )
            st.error(f"Request failed ({type(error).__name__}): {error_message}")
            st.session_state.messages.pop()

    with st.expander("LLM call logs"):
        if st.session_state.llm_logs:
            for index, log in enumerate(st.session_state.llm_logs, start=1):
                st.markdown(f"**Turn {index}: {log['prompt']}**")
                for event in log["events"]:
                    st.caption(event["type"].replace("_", " ").title())
                    st.json(event)
        else:
            st.caption("LLM requests and responses will appear here.")


if __name__ == "__main__":
    main()
