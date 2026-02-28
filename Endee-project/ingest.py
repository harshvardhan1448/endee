#!/usr/bin/env python3
"""
CLI tool to ingest documents into the Endee vector database.

Usage:
    python ingest.py                        # Ingest all docs from ./documents/
    python ingest.py /path/to/file.pdf      # Ingest a single file
    python ingest.py /path/to/folder        # Ingest all files in a folder
    python ingest.py --reset                # Delete index and re-create it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import config
from document_loader import chunk_text, load_directory, load_document
from embeddings import get_embedder
from vector_store import VectorStore

console = Console()


def ingest_file(path: str | Path, store: VectorStore, embedder) -> int:
    """Ingest a single file. Returns number of chunks upserted."""
    text, source = load_document(path)
    chunks = chunk_text(text)
    if not chunks:
        console.print(f"  [yellow]No chunks produced for {source}[/yellow]")
        return 0

    console.print(f"  Embedding {len(chunks)} chunks from [bold]{source}[/bold]…")
    vectors = embedder.embed_texts(chunks)
    count = store.upsert_chunks(chunks, vectors, source=source)
    return count


def ingest_directory(directory: str | Path, store: VectorStore, embedder) -> int:
    """Ingest all supported files in a directory. Returns total chunks upserted."""
    docs = load_directory(directory)
    if not docs:
        console.print("[yellow]No supported documents found.[/yellow]")
        return 0

    total = 0
    for text, source in docs:
        chunks = chunk_text(text)
        if not chunks:
            continue
        console.print(f"  Embedding {len(chunks)} chunks from [bold]{source}[/bold]…")
        vectors = embedder.embed_texts(chunks)
        count = store.upsert_chunks(chunks, vectors, source=source)
        total += count

    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest documents into Endee RAG")
    parser.add_argument(
        "path",
        nargs="?",
        default=None,
        help="Path to a file or directory (default: ./documents/)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete and re-create the index before ingesting",
    )
    args = parser.parse_args()

    console.print(Panel("[bold blue]Endee RAG — Document Ingestion[/bold blue]"))

    store = VectorStore()
    embedder = get_embedder()

    # Reset index if requested
    if args.reset:
        try:
            store.delete_index()
        except Exception:
            pass

    store.ensure_index()

    console.print(f"[dim]Embedding model: {embedder.model_name} (dim={embedder.dimension})[/dim]")
    console.print(f"[dim]Index: {store.index_name}[/dim]")
    console.print()

    # Determine target
    target = Path(args.path) if args.path else config.DOCUMENTS_DIR

    if target.is_file():
        total = ingest_file(target, store, embedder)
    elif target.is_dir():
        total = ingest_directory(target, store, embedder)
    else:
        console.print(f"[red]Path not found: {target}[/red]")
        sys.exit(1)

    # Summary
    console.print()
    table = Table(title="Ingestion Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Target", str(target))
    table.add_row("Chunks upserted", str(total))
    table.add_row("Index", store.index_name)
    console.print(table)


if __name__ == "__main__":
    main()
