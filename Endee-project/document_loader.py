"""
Document loader and text chunker.

Supports: .txt, .md, .pdf, .docx, .html
Splits documents into overlapping chunks for embedding.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

from rich.console import Console

import config

console = Console()


# ── Text extraction per file type ────────────────────────────────────────────

def _load_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_markdown(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n\n".join(pages)


def _load_docx(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _load_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")
    return soup.get_text(separator="\n", strip=True)


_LOADERS = {
    ".txt": _load_txt,
    ".md": _load_markdown,
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".html": _load_html,
    ".htm": _load_html,
}


# ── Public helpers ───────────────────────────────────────────────────────────

def load_document(path: str | Path) -> Tuple[str, str]:
    """Load a single file and return (text, source_name).

    Raises ValueError for unsupported file types.
    """
    path = Path(path)
    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {list(_LOADERS.keys())}"
        )
    text = loader(path)
    return text, path.name


def load_directory(
    directory: str | Path | None = None,
    recursive: bool = True,
) -> List[Tuple[str, str]]:
    """Load all supported files from a directory.

    Returns list of (text, source_name) tuples.
    """
    directory = Path(directory or config.DOCUMENTS_DIR)
    results: List[Tuple[str, str]] = []

    pattern = "**/*" if recursive else "*"
    for path in sorted(directory.glob(pattern)):
        if path.is_file() and path.suffix.lower() in _LOADERS:
            try:
                text, name = load_document(path)
                if text.strip():
                    results.append((text, name))
                    console.print(f"  [green]✓[/green] Loaded {name} ({len(text):,} chars)")
                else:
                    console.print(f"  [yellow]⚠[/yellow] Skipped {name} (empty)")
            except Exception as exc:
                console.print(f"  [red]✗[/red] Failed to load {path.name}: {exc}")

    return results


# ── Chunking ─────────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[str]:
    """Split text into overlapping chunks by character count.

    Uses paragraph/sentence boundaries when possible for cleaner splits.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP

    # Normalise whitespace
    text = text.strip()
    if not text:
        return []

    # Split into paragraphs first, then recombine into chunks
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: List[str] = []
    current_chunk = ""

    for para in paragraphs:
        # If adding this paragraph would exceed chunk_size, flush
        if current_chunk and len(current_chunk) + len(para) + 2 > chunk_size:
            chunks.append(current_chunk)
            # Keep overlap from the end of the previous chunk
            if chunk_overlap > 0:
                current_chunk = current_chunk[-chunk_overlap:] + "\n\n" + para
            else:
                current_chunk = para
        else:
            current_chunk = (current_chunk + "\n\n" + para).strip() if current_chunk else para

    if current_chunk.strip():
        chunks.append(current_chunk)

    # If any single chunk is still too large, force-split by characters
    final_chunks: List[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final_chunks.append(chunk)
        else:
            for i in range(0, len(chunk), chunk_size - chunk_overlap):
                sub = chunk[i : i + chunk_size]
                if sub.strip():
                    final_chunks.append(sub.strip())

    return final_chunks
