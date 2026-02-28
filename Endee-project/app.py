#!/usr/bin/env python3
"""
Gradio-based RAG Chatbot UI powered by Endee vector database.

Launch:
    python app.py
"""

from __future__ import annotations

import gradio as gr
from rich.console import Console

from rag_pipeline import RAGPipeline, warm_up

console = Console()

# ── Initialise RAG pipeline ─────────────────────────────────────────────────

rag = RAGPipeline()

# Pre-load models at startup so first browser query doesn't timeout
warm_up()


# ── Chat function ────────────────────────────────────────────────────────────

def chat(message: str, history: list) -> str:
    """Process a user message through the RAG pipeline and return the answer."""
    if not message.strip():
        return "Please enter a question."

    try:
        result = rag.ask(message, return_context=False)
        answer = result["answer"]

        # Append sources
        sources = result.get("sources", [])
        if sources:
            source_names = list({s["source"] for s in sources})
            answer += f"\n\n---\n📄 **Sources:** {', '.join(source_names)}"

        return answer

    except Exception as e:
        console.print_exception()
        return f"⚠️ An error occurred: {str(e)}"


# ── File upload + ingestion ──────────────────────────────────────────────────

def ingest_uploaded_files(files) -> str:
    """Ingest uploaded files into the vector store."""
    if not files:
        return "No files uploaded."

    from document_loader import chunk_text, load_document
    from embeddings import get_embedder
    from vector_store import VectorStore

    store = VectorStore()
    embedder = get_embedder()
    store.ensure_index()

    total_chunks = 0
    processed_files = []

    for file in files:
        try:
            text, source = load_document(file.name)
            chunks = chunk_text(text)
            if not chunks:
                processed_files.append(f"⚠️ {source}: no content extracted")
                continue

            vectors = embedder.embed_texts(chunks)
            count = store.upsert_chunks(chunks, vectors, source=source)
            total_chunks += count
            processed_files.append(f"✅ {source}: {count} chunks")
        except Exception as e:
            processed_files.append(f"❌ {file.name}: {str(e)}")

    summary = "\n".join(processed_files)
    return f"**Ingestion complete — {total_chunks} total chunks indexed.**\n\n{summary}"


# ── Gradio UI ────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Endee RAG Chatbot",
    ) as demo:
        gr.Markdown(
            """
            # 🔍 Endee RAG Chatbot
            **Ask questions about your documents** — powered by
            [Endee](https://endee.io) vector database and HuggingFace models.

            1. **Upload** documents below (PDF, TXT, MD, DOCX, HTML)
            2. **Ask** questions in the chat
            """
        )

        with gr.Tabs():
            # ── Chat Tab ────────────────────────────────────────────────
            with gr.Tab("💬 Chat"):
                gr.ChatInterface(
                    fn=chat,
                    description="Ask anything about your ingested documents.",
                    examples=[
                        "What are the main topics in the documents?",
                        "Summarize the key points.",
                        "What does the document say about…?",
                    ],
                    fill_height=True,
                )

            # ── Upload Tab ──────────────────────────────────────────────
            with gr.Tab("📁 Upload Documents"):
                gr.Markdown("Upload documents to index them in the Endee vector store.")
                file_upload = gr.File(
                    label="Drop files here",
                    file_count="multiple",
                    file_types=[".txt", ".md", ".pdf", ".docx", ".html", ".htm"],
                )
                ingest_btn = gr.Button("🚀 Ingest Documents", variant="primary")
                ingest_output = gr.Markdown(label="Results")

                ingest_btn.click(
                    fn=ingest_uploaded_files,
                    inputs=[file_upload],
                    outputs=[ingest_output],
                )

            # ── Info Tab ────────────────────────────────────────────────
            with gr.Tab("ℹ️ Info"):
                gr.Markdown(
                    f"""
                    ### Configuration
                    | Setting | Value |
                    |---|---|
                    | **Embedding Model** | `{rag.embedder.model_name}` |
                    | **LLM Model** | `{__import__('config').HF_MODEL_ID}` |
                    | **Vector DB** | Endee @ `{__import__('config').ENDEE_HOST}` |
                    | **Index Name** | `{rag.store.index_name}` |
                    | **Vector Dimension** | `{rag.store.dimension}` |
                    | **Chunk Size** | `{__import__('config').CHUNK_SIZE}` chars |
                    | **Top-K Retrieval** | `{rag.top_k}` |

                    ### How it works
                    1. **Ingest** — Documents are chunked, embedded with
                       `{rag.embedder.model_name}`, and stored in Endee.
                    2. **Retrieve** — Your question is embedded and the top-K
                       most similar chunks are fetched from Endee via cosine similarity.
                    3. **Generate** — The retrieved context and your question are
                       sent to the LLM, which produces a grounded answer.
                    """
                )

    return demo


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    console.print("[bold green]Starting Endee RAG Chatbot…[/bold green]")
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
