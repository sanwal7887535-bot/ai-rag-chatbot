"""Isolated unit and integration tests for config, RAG, tools, and LLM client."""

from __future__ import annotations

import gc
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import chromadb

from config import load_settings, sanitize_error_message
from llm_client import (
    REFUSAL_TEXT,
    build_system_prompt,
    format_reference_context,
    generate_reply,
)
from rag_pipeline import (
    COLLECTION_NAME,
    LEGACY_COLLECTION_NAME,
    RAGStore,
    compute_document_id,
    extract_pdf_bytes,
    sanitize_filename,
    split_text_into_chunks,
)
from tools import ToolRegistry, create_tool_registry, get_current_datetime, search_web


def _fake_embedder(texts: list[str]) -> list[list[float]]:
    """Deterministic 4-D normalized vectors keyed by simple keyword presence."""
    vectors: list[list[float]] = []
    for text in texts:
        lower = text.casefold()
        raw = [
            1.0 if "laminar" in lower else 0.1,
            1.0 if "autoclave" in lower else 0.1,
            1.0 if "centrifuge" in lower else 0.1,
            float(len(lower) % 10 + 1) / 10.0,
        ]
        norm = sum(x * x for x in raw) ** 0.5
        vectors.append([x / norm for x in raw])
    return vectors


class ConfigTests(unittest.TestCase):
    def test_env_overrides_secrets_and_handles_missing_secrets_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_settings(
                base_dir=Path(tmp),
                env={
                    "LLM_PROVIDER": "groq",
                    "GROQ_API_KEY": "env-groq-secret-1234",
                    "LOG_REDACT_CONTENT": "true",
                    "MAX_PDF_MB": "10",
                },
            )
            self.assertEqual(settings.llm_provider, "groq")
            self.assertEqual(settings.llm_model, "openai/gpt-oss-120b")
            self.assertEqual(settings.groq_api_key, "env-groq-secret-1234")
            self.assertTrue(settings.log_redact_content)
            self.assertEqual(settings.max_pdf_bytes, 10 * 1024 * 1024)

    def test_invalid_configs_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                load_settings(
                    base_dir=Path(tmp),
                    env={"LLM_PROVIDER": "unknown"},
                    local_secrets={},
                )
            with self.assertRaises(ValueError):
                load_settings(
                    base_dir=Path(tmp),
                    env={"GROQ_API_URL": "http://insecure.example.com/v1/chat"},
                    local_secrets={},
                )
            with self.assertRaises(ValueError):
                load_settings(
                    base_dir=Path(tmp),
                    env={"CHUNK_SIZE": "200", "CHUNK_OVERLAP": "200"},
                    local_secrets={},
                )
            with self.assertRaises(ValueError):
                load_settings(
                    base_dir=Path(tmp),
                    env={"DEFAULT_TIMEZONE": "Invalid/Zone"},
                    local_secrets={},
                )

    def test_sanitize_error_message(self) -> None:
        cleaned = sanitize_error_message(
            "Request failed with Bearer secret_token_xyz!",
            ("secret_token_xyz", None),
        )
        self.assertEqual(cleaned, "Request failed with Bearer [REDACTED]!")


class RagPipelineTests(unittest.TestCase):
    def test_filename_sanitization_and_pdf_validation(self) -> None:
        self.assertEqual(
            sanitize_filename("../../evil[page 1]\n.pdf"),
            "evilpage 1.pdf",
        )
        with self.assertRaises(ValueError):
            extract_pdf_bytes(b"")
        with self.assertRaises(ValueError):
            extract_pdf_bytes(b"Not a PDF file")
        with self.assertRaises(ValueError):
            extract_pdf_bytes(b"%PDF-1.4\n" + b"0" * 100, max_bytes=50)

    def test_case_insensitive_upsert_cleans_stale_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RAGStore(db_dir=tmp, embed_fn=_fake_embedder)
            try:
                initial_chunks = [
                    {"text": f"Laminar flow cabinet detail {i}", "page": 1}
                    for i in range(4)
                ]
                store.store_chunks(
                    initial_chunks, store.generate_embeddings(initial_chunks), "Report.pdf"
                )
                self.assertEqual(store.collection.count(), 4)

                updated_chunks = [{"text": "Autoclave overview", "page": 1}]
                store.store_chunks(
                    updated_chunks, store.generate_embeddings(updated_chunks), "report.pdf"
                )
                self.assertEqual(store.collection.count(), 1)
                docs = store.get_indexed_documents()
                self.assertEqual(len(docs), 1)
                self.assertEqual(docs[0]["filename"], "report.pdf")
                self.assertEqual(docs[0]["chunks"], 1)
            finally:
                store.close()
                del store
                gc.collect()

    def test_legacy_migration_and_permanent_deletion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            client = chromadb.PersistentClient(path=tmp)
            legacy = client.get_or_create_collection(name=LEGACY_COLLECTION_NAME)
            doc_id = compute_document_id("old.pdf")
            legacy.upsert(
                ids=[f"{doc_id}_chunk_0"],
                documents=["Legacy laminar flow note"],
                embeddings=_fake_embedder(["Legacy laminar flow note"]),
                metadatas=[{"filename": "old.pdf", "document_id": doc_id, "page": 1}],
            )
            if hasattr(client, "close"):
                client.close()
            del legacy, client
            gc.collect()

            store = RAGStore(db_dir=tmp, embed_fn=_fake_embedder)
            try:
                collection_names = {
                    c.name if hasattr(c, "name") else c
                    for c in store.client.list_collections()
                }
                self.assertIn(COLLECTION_NAME, collection_names)
                self.assertNotIn(LEGACY_COLLECTION_NAME, collection_names)
                self.assertEqual(store.collection.count(), 1)

                removed = store.remove_document("old.pdf")
                self.assertEqual(removed, 1)
                self.assertEqual(store.collection.count(), 0)
            finally:
                store.close()
                del store
                gc.collect()

            # Re-opening must NOT resurrect the deleted document.
            store_reloaded = RAGStore(db_dir=tmp, embed_fn=_fake_embedder)
            try:
                self.assertEqual(store_reloaded.collection.count(), 0)
            finally:
                store_reloaded.close()
                del store_reloaded
                gc.collect()

    def test_hybrid_retrieval_and_sample_pdf_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = RAGStore(db_dir=tmp, embed_fn=_fake_embedder)
            try:
                chunks = [
                    {"text": "A centrifuge spins samples at high RPM.", "page": 1},
                    {"text": "A laminar flow cabinet provides sterile airflow.", "page": 2},
                    {"text": "An autoclave sterilizes media with pressurized steam.", "page": 3},
                ]
                store.store_chunks(
                    chunks, store.generate_embeddings(chunks), "lab.pdf"
                )
                results = store.retrieve_relevant_chunks(
                    "What is a laminar flow cabinet?", "lab.pdf", k=2
                )
                self.assertEqual(len(results["documents"][0]), 2)
                self.assertIn("laminar flow cabinet", results["documents"][0][0].lower())

                with self.assertRaises(ValueError):
                    store.retrieve_relevant_chunks("   ", "lab.pdf", k=2)
                with self.assertRaises(ValueError):
                    store.retrieve_relevant_chunks("question", "lab.pdf", k=0)
            finally:
                store.close()
                del store
                gc.collect()

        sample_pdf = Path(__file__).with_name("sample.pdf")
        if sample_pdf.is_file():
            pages = extract_pdf_bytes(sample_pdf.read_bytes())
            self.assertGreater(len(pages), 0)
            pdf_chunks = split_text_into_chunks(pages)
            self.assertGreater(len(pdf_chunks), 0)


class ToolsAndLlmTests(unittest.TestCase):
    def test_tool_registry_and_datetime(self) -> None:
        dt = get_current_datetime("Asia/Karachi")
        self.assertEqual(set(dt.keys()), {"date", "day", "time", "timezone"})

        with tempfile.TemporaryDirectory() as tmp:
            no_tavily = load_settings(
                base_dir=Path(tmp), env={"GROQ_API_KEY": "k"}, local_secrets={}
            )
            reg_no_web = create_tool_registry(no_tavily)
            self.assertTrue(reg_no_web.has_tool("get_current_datetime"))
            self.assertFalse(reg_no_web.has_tool("search_web"))

            with_tavily = load_settings(
                base_dir=Path(tmp),
                env={"GROQ_API_KEY": "k", "TAVILY_API_KEY": "tvly-key"},
                local_secrets={},
            )
            reg_with_web = create_tool_registry(with_tavily)
            self.assertTrue(reg_with_web.has_tool("search_web"))

        registry = ToolRegistry()
        registry.register(
            name="echo",
            function=lambda text: {"echo": text},
            description="Echo input",
            parameters={
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
                "additionalProperties": False,
            },
        )
        turn_log: dict = {"events": []}
        self.assertEqual(
            registry.execute("echo", '{"text": "hi"}', turn_log), {"echo": "hi"}
        )
        with self.assertRaises(ValueError):
            registry.execute("echo", "{}", turn_log)
        with self.assertRaises(ValueError):
            registry.execute("echo", '{"text": "hi", "extra": 1}', turn_log)
        with self.assertRaises(ValueError):
            search_web("   ", api_key="tvly-123")

    def test_generate_reply_tool_loop_immutability_and_redaction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            log_path = tmp_path / "calls.jsonl"
            settings = load_settings(
                base_dir=tmp_path,
                env={
                    "LLM_PROVIDER": "groq",
                    "GROQ_API_KEY": "groq-secret-9999",
                    "LLM_LOG_PATH": str(log_path),
                    "ENABLE_FILE_LOGGING": "true",
                    "LOG_REDACT_CONTENT": "true",
                    "MAX_TOOL_ROUNDS": "2",
                },
                local_secrets={},
            )
            registry = create_tool_registry(settings)

            first_resp = MagicMock()
            first_resp.status_code = 200
            first_resp.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_current_datetime",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
            second_resp = MagicMock()
            second_resp.status_code = 200
            second_resp.json.return_value = {
                "choices": [
                    {"message": {"role": "assistant", "content": "Answer [doc.pdf, page 2]."}}
                ]
            }

            ref_text, sources = format_reference_context(
                [{"text": "Sample text", "filename": "doc.pdf", "page": 2}],
                "Extra note",
            )
            self.assertEqual(sources, ["doc.pdf, page 2"])
            self.assertIn(REFUSAL_TEXT, build_system_prompt(registry, has_reference_context=True))
            self.assertNotIn(REFUSAL_TEXT, build_system_prompt(registry, has_reference_context=False))

            turn_log: dict = {"prompt": "[REDACTED]", "events": []}
            with patch("llm_client.requests.post", side_effect=[first_resp, second_resp]) as post_mock:
                reply = generate_reply(
                    messages=[{"role": "user", "content": "What date is it?"}],
                    settings=settings,
                    tool_registry=registry,
                    turn_log=turn_log,
                    reference_context=ref_text,
                )

            self.assertEqual(reply, "Answer [doc.pdf, page 2].")
            sent_first_payload = post_mock.call_args_list[0].kwargs["json"]
            # Ensure only 1 system message exists and untrusted RAG text stays in user role.
            system_msgs = [m for m in sent_first_payload["messages"] if m["role"] == "system"]
            self.assertEqual(len(system_msgs), 1)
            self.assertIn("Reference Information", sent_first_payload["messages"][-1]["content"])

            # Verify earlier logged request was NOT mutated by tool call append.
            req_events = [e for e in turn_log["events"] if e["type"] == "model_request"]
            self.assertEqual(len(req_events), 2)
            self.assertEqual(len(req_events[0]["messages"]), 2)
            self.assertEqual(len(req_events[1]["messages"]), 4)

            # Verify JSONL redaction works cleanly.
            lines = log_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["provider"], "groq")
            self.assertIn("[REDACTED", record["model_response"]["text"])
            self.assertEqual(record["tool_events"][1]["result"], "[REDACTED]")

    def test_secret_sanitization_on_network_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_settings(
                base_dir=Path(tmp),
                env={
                    "LLM_PROVIDER": "groq",
                    "GROQ_API_KEY": "super-secret-abc",
                    "ENABLE_FILE_LOGGING": "false",
                },
                local_secrets={},
            )
            registry = create_tool_registry(settings)
            turn_log: dict = {"events": []}
            import requests

            with patch(
                "llm_client.requests.post",
                side_effect=requests.exceptions.ConnectionError(
                    "Failed with key super-secret-abc"
                ),
            ):
                with self.assertRaises(RuntimeError) as ctx:
                    generate_reply(
                        messages=[{"role": "user", "content": "hi"}],
                        settings=settings,
                        tool_registry=registry,
                        turn_log=turn_log,
                    )
            self.assertNotIn("super-secret-abc", str(ctx.exception))
            self.assertIn("[REDACTED]", str(ctx.exception))

    def test_max_tool_rounds_exceeded_omits_tools_on_final_round(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = load_settings(
                base_dir=Path(tmp),
                env={
                    "LLM_PROVIDER": "ollama",
                    "ENABLE_FILE_LOGGING": "false",
                    "MAX_TOOL_ROUNDS": "1",
                },
                local_secrets={},
            )
            registry = create_tool_registry(settings)
            tool_resp = MagicMock()
            tool_resp.status_code = 200
            tool_resp.json.return_value = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "get_current_datetime",
                                        "arguments": "{}",
                                    },
                                }
                            ],
                        }
                    }
                ]
            }
            turn_log: dict = {"events": []}
            with patch("llm_client.requests.post", side_effect=[tool_resp, tool_resp]) as post_mock:
                with self.assertRaises(RuntimeError) as ctx:
                    generate_reply(
                        messages=[{"role": "user", "content": "time?"}],
                        settings=settings,
                        tool_registry=registry,
                        turn_log=turn_log,
                    )
            self.assertIn("Maximum tool-call rounds exceeded", str(ctx.exception))
            self.assertIn("tools", post_mock.call_args_list[0].kwargs["json"])
            self.assertNotIn("tools", post_mock.call_args_list[1].kwargs["json"])

    def test_app_module_imports_and_helpers(self) -> None:
        import app as chatbot_app

        class AttrDict(dict):
            __getattr__ = dict.__getitem__
            __setattr__ = dict.__setitem__

        self.assertEqual(chatbot_app.NO_PDF_OPTION, "None (General Chat / Web)")
        with patch.object(chatbot_app.st, "session_state", AttrDict()):
            chatbot_app._init_session_state()
            self.assertEqual(chatbot_app.st.session_state["messages"], [])
            self.assertEqual(chatbot_app.st.session_state["llm_logs"], [])
            self.assertEqual(
                chatbot_app.st.session_state["selected_pdf"],
                chatbot_app.NO_PDF_OPTION,
            )

        fake_context = MagicMock()
        fake_context.headers = {"X-MS-CLIENT-PRINCIPAL-NAME": "alice@example.com"}
        with patch.object(chatbot_app.st, "context", fake_context):
            self.assertEqual(chatbot_app._get_easy_auth_user(), "alice@example.com")


if __name__ == "__main__":
    unittest.main()
