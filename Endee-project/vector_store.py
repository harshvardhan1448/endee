"""
Endee vector-store wrapper.

Handles index creation, upserting embeddings, and similarity search
through the official Endee Python SDK.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

import requests as _requests
from endee import Endee, Precision
from rich.console import Console

import config

console = Console()

# The Docker image may use precision names like "int8d" / "int16d" while
# the SDK enum uses "int8" / "int16".  We try SDK values first; if the
# server rejects them we fall back to the "d"-suffixed variants.
_PRECISION_FALLBACKS = {
    "int8": "int8d",
    "int16": "int16d",
}


class VectorStore:
    """High-level interface to the Endee vector database."""

    def __init__(
        self,
        index_name: str | None = None,
        dimension: int | None = None,
        host: str | None = None,
        auth_token: str | None = None,
    ) -> None:
        self.index_name = index_name or config.ENDEE_INDEX_NAME
        self.dimension = dimension or config.ENDEE_DIMENSION
        self.host = host or config.ENDEE_HOST
        self.auth_token = auth_token or config.ENDEE_AUTH_TOKEN

        # Initialise client
        if self.auth_token:
            self.client = Endee(self.auth_token)
        else:
            self.client = Endee()

        # Set custom base URL if not default
        base_api = f"{self.host}/api/v1"
        self.client.set_base_url(base_api)

        self._index = None

    # ── index management ────────────────────────────────────────────────
    def _create_index_raw(self, precision: str) -> None:
        """Create an index via a direct REST call to bypass SDK validation
        issues when the server expects precision names like 'int8d'."""
        url = f"{self.host}/api/v1/index/create"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = self.auth_token
        payload = {
            "index_name": self.index_name,
            "dim": self.dimension,
            "space_type": "cosine",
            "M": 16,
            "ef_con": 128,
            "precision": precision,
            "checksum": -1,
        }
        resp = _requests.post(url, json=payload, headers=headers, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to create index: {resp.text}")

    def ensure_index(self) -> None:
        """Create the index if it doesn't already exist."""
        try:
            existing = self.client.list_indexes()
            # list_indexes() may return a dict like {"indexes": [...]} or a list
            if isinstance(existing, dict):
                index_list = existing.get("indexes", [])
            else:
                index_list = existing or []
            names = [
                idx.get("name", idx) if isinstance(idx, dict) else str(idx)
                for idx in index_list
            ]
            if self.index_name in names:
                console.print(f"[green]Index '{self.index_name}' already exists.[/green]")
                self._index = self.client.get_index(name=self.index_name)
                return
        except Exception:
            pass  # list may fail if index doesn't exist yet

        console.print(f"[yellow]Creating index '{self.index_name}' (dim={self.dimension})…[/yellow]")

        # Try SDK first; fall back to raw REST if precision naming differs
        created = False
        for precision in (Precision.INT8, _PRECISION_FALLBACKS.get("int8", "int8d")):
            try:
                if isinstance(precision, str) and precision not in Precision.__members__.values():
                    # SDK Pydantic would reject this; use raw HTTP
                    self._create_index_raw(precision)
                else:
                    self.client.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        space_type="cosine",
                        precision=precision,
                    )
                created = True
                break
            except Exception as exc:
                console.print(f"  [dim]Precision '{precision}' failed: {exc}[/dim]")

        if not created:
            raise RuntimeError(
                "Could not create Endee index — all precision variants failed."
            )

        self._index = self.client.get_index(name=self.index_name)
        console.print(f"[green]Index '{self.index_name}' created successfully.[/green]")

    @property
    def index(self):
        if self._index is None:
            self.ensure_index()
        return self._index

    def delete_index(self) -> None:
        """Delete the current index."""
        self.client.delete_index(name=self.index_name)
        self._index = None
        console.print(f"[red]Index '{self.index_name}' deleted.[/red]")

    def describe(self) -> Dict[str, Any]:
        """Return index metadata."""
        return self.index.describe()

    # ── upsert ──────────────────────────────────────────────────────────
    @staticmethod
    def _make_id(text: str, source: str) -> str:
        """Deterministic ID from content + source to allow idempotent upserts."""
        return hashlib.sha256(f"{source}::{text[:200]}".encode()).hexdigest()[:24]

    def upsert_chunks(
        self,
        texts: List[str],
        vectors: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        source: str = "unknown",
        batch_size: int = 500,
    ) -> int:
        """Upsert text chunks with their embeddings into Endee.

        Returns the number of vectors upserted.
        """
        self.ensure_index()
        metadatas = metadatas or [{}] * len(texts)
        total = 0

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch = []
            for i in range(start, end):
                doc_id = self._make_id(texts[i], source)
                meta = {
                    **metadatas[i],
                    "text": texts[i],
                    "source": source,
                    "chunk_index": i,
                }
                batch.append(
                    {
                        "id": doc_id,
                        "vector": vectors[i],
                        "meta": meta,
                    }
                )
            self.index.upsert(batch)
            total += len(batch)
            console.print(f"  ↳ upserted {total}/{len(texts)} chunks")

        return total

    # ── search ──────────────────────────────────────────────────────────
    def search(
        self,
        query_vector: List[float],
        top_k: int | None = None,
        ef: int = 128,
    ) -> List[Dict[str, Any]]:
        """Perform similarity search and return results with metadata."""
        self.ensure_index()
        top_k = top_k or config.TOP_K
        results = self.index.query(
            vector=query_vector,
            top_k=top_k,
            ef=ef,
            include_vectors=False,
        )
        return results
