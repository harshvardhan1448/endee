# Endee Vector Database — Overview

Endee (nD) is a high-performance open source vector database designed for speed and efficiency. It can handle up to 1 billion vectors on a single node, delivering significant performance gains through optimized indexing and execution.

## Key Features

- **HNSW Algorithm**: Uses Hierarchical Navigable Small World graphs for fast approximate nearest neighbor (ANN) search.
- **Multiple Precision Options**: Supports Binary, INT8, INT16, FLOAT16, and FLOAT32 quantization for controlling the trade-off between speed, memory, and accuracy.
- **Hybrid Search**: Combines dense vector search with sparse vector search for improved retrieval — ideal for RAG pipelines requiring both semantic and keyword matching.
- **Filtering**: Supports $eq (exact match), $in (match any in list), and $range (numeric range) operators for filtering query results.
- **Multiple Distance Metrics**: Cosine similarity, L2 (Euclidean), and Inner Product.
- **SIMD Optimized**: Supports AVX2, AVX512, NEON (Apple Silicon), and SVE2 for hardware-accelerated vector operations.

## Index Types

### Dense Indexes
Dense indexes enable semantic search — finding items based on meaning rather than exact keyword matches. Use dense indexes for:
- Finding semantically similar documents
- Powering recommendation systems
- Enabling image/video similarity search

### Hybrid Indexes
Hybrid indexes combine dense vector search with sparse vector search. To create a hybrid index, specify the sparse dimension parameter along with the dense dimension. This is ideal for document retrieval in RAG pipelines where you need both semantic and keyword matching.

## Getting Started

Endee can be run via Docker:
```
docker compose up -d
```

The Python SDK is available via pip:
```
pip install endee
```

Initialize the client and create an index:
```python
from endee import Endee, Precision
client = Endee()
client.create_index(name="my_index", dimension=384, space_type="cosine", precision=Precision.INT8)
```

## Architecture

Endee is built in C++ (96.3% of codebase) with CMake build system. It exposes a REST API on port 8080 with a built-in dashboard for monitoring. SDKs are available for Python, TypeScript, Java, and Go.
