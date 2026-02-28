"""
RAG pipeline — ties together embedding, retrieval, and generation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from rich.console import Console

import config
from embeddings import get_embedder
from vector_store import VectorStore

console = Console()

# ── System prompt template ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a helpful assistant. Answer the user's question using ONLY the provided context. Be concise and accurate. Do NOT repeat the context or the question. If the context does not contain enough information, say "I don't have enough information to answer that."
"""

USER_TEMPLATE = """\
Context:
{context}

Question: {question}

Provide a direct, concise answer:"""


# ── LLM generation backends ─────────────────────────────────────────────────

def _generate_with_api(prompt: str, system: str) -> str:
    """Use the HuggingFace Inference API (serverless)."""
    from huggingface_hub import InferenceClient

    client = InferenceClient(
        model=config.HF_MODEL_ID,
        token=config.HF_API_TOKEN,
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    response = client.chat_completion(
        messages=messages,
        max_tokens=1024,
        temperature=0.3,
    )
    return response.choices[0].message.content.strip()


def _generate_with_local(prompt: str, system: str) -> str:
    """Use a local HuggingFace model via transformers pipeline."""
    import torch
    from transformers import pipeline as hf_pipeline

    # Cache the pipeline in a module-level variable
    if not hasattr(_generate_with_local, "_pipe"):
        console.print(f"[yellow]Loading local model '{config.HF_MODEL_ID}'…[/yellow]")
        # Detect best available device
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        _generate_with_local._pipe = hf_pipeline(
            "text-generation",
            model=config.HF_MODEL_ID,
            device=device,
            dtype=torch.float16 if device != "cpu" else torch.float32,
        )
        console.print(f"[green]Model loaded on {device}[/green]")
    pipe = _generate_with_local._pipe

    full_prompt = f"<|system|>\n{system}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n"
    from transformers import GenerationConfig
    gen_cfg = GenerationConfig(
        max_new_tokens=512,
        temperature=0.3,
        do_sample=True,
    )
    output = pipe(
        full_prompt,
        generation_config=gen_cfg,
        return_full_text=False,
    )
    return output[0]["generated_text"].strip()


def generate(prompt: str, system: str = SYSTEM_PROMPT) -> str:
    """Route to the configured LLM backend."""
    if config.USE_LOCAL_LLM:
        return _generate_with_local(prompt, system)
    return _generate_with_api(prompt, system)


def warm_up() -> None:
    """Pre-load embedding and LLM models so the first query is fast."""
    console.print("[yellow]Warming up models…[/yellow]")
    from embeddings import get_embedder
    get_embedder().embed_text("warmup")
    if config.USE_LOCAL_LLM:
        _generate_with_local("Say hi", "You are helpful.")
    console.print("[green]Models ready![/green]")


# ── RAG Pipeline ─────────────────────────────────────────────────────────────

class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        top_k: int | None = None,
    ) -> None:
        self.embedder = get_embedder()
        self.store = vector_store or VectorStore()
        self.top_k = top_k or config.TOP_K

    def retrieve(self, query: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        """Embed the query and fetch the most relevant chunks from Endee."""
        query_vec = self.embedder.embed_text(query)
        results = self.store.search(query_vec, top_k=top_k or self.top_k)
        return results

    def build_context(self, results: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a context string for the LLM."""
        parts: List[str] = []
        for i, r in enumerate(results, 1):
            meta = r.get("meta") or r.get("metadata") or {}
            text = meta.get("text", "")
            source = meta.get("source", "unknown")
            similarity = r.get("similarity", 0)
            parts.append(
                f"[{i}] (source: {source}, score: {similarity:.3f})\n{text}"
            )
        return "\n\n---\n\n".join(parts)

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        return_context: bool = False,
    ) -> Dict[str, Any]:
        """Full RAG: retrieve → build context → generate answer.

        Returns dict with keys: answer, sources, context (optional).
        """
        # 1. Retrieve
        results = self.retrieve(question, top_k=top_k)

        if not results:
            answer = "I couldn't find any relevant documents to answer your question. Please ingest some documents first."
            return {
                "answer": answer,
                "sources": [],
                "context": "" if return_context else None,
            }

        # 2. Build context
        context_str = self.build_context(results)

        # 3. Generate
        user_prompt = USER_TEMPLATE.format(context=context_str, question=question)
        answer = generate(user_prompt)

        # 4. Collect sources
        sources = []
        for r in results:
            meta = r.get("meta") or r.get("metadata") or {}
            sources.append(
                {
                    "source": meta.get("source", "unknown"),
                    "similarity": r.get("similarity", 0),
                    "chunk_preview": (meta.get("text", ""))[:150] + "…",
                }
            )

        output: Dict[str, Any] = {"answer": answer, "sources": sources}
        if return_context:
            output["context"] = context_str
        return output
