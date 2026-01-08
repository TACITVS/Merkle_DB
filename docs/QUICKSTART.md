# MerkleDb Quick Start Tutorial

This guide walks you through setting up MerkleDb from scratch and performing your first vector operations.

---

## Prerequisites

Before you begin, ensure you have the following installed:

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Elixir | ~> 1.14 | Runtime platform |
| Erlang/OTP | 25+ | Virtual machine |
| GCC | Any recent | C compilation |
| NASM | Any recent | Assembly compilation |
| Make | Any | Build automation |

### Windows-Specific Setup

1. **Install Erlang/OTP**: Download from [erlang.org](https://www.erlang.org/downloads)
2. **Install Elixir**: Download from [elixir-lang.org](https://elixir-lang.org/install.html)
3. **Install MinGW64**: For GCC on Windows
4. **Install NASM**: Download from [nasm.us](https://www.nasm.us/)
5. **Add to PATH**: Ensure all tools are in your system PATH

### Verify Installation

```bash
# Check Elixir version
elixir --version
# Should show: Elixir 1.14.x or higher

# Check Erlang version
erl -eval "erlang:display(erlang:system_info(otp_release)), halt()." -noshell
# Should show: "25" or higher

# Check GCC
gcc --version

# Check NASM
nasm --version
```

---

## Step 1: Clone and Install

```bash
# Clone the repository
git clone https://github.com/TACITVS/Merkle_DB.git
cd Merkle_DB

# Install Elixir dependencies
mix deps.get

# Generate NIF bridge (compiles native code)
mix run gen_bridge.exs

# Compile everything
mix compile
```

If you see any errors during `gen_bridge.exs`, ensure GCC and NASM are properly installed and in your PATH.

---

## Step 2: Start the Server

### Development Mode (No Authentication)

```bash
# Start in development mode
mix run --no-halt
```

You should see output like:
```
[info] Starting MerkleDb...
[info] HTTP server listening on port 4000
[info] Raft cluster initialized
```

### Production Mode (With Authentication)

```bash
# Set required environment variables
# Windows:
set MERKLE_DB_API_KEY=your_secure_random_key_here
set MIX_ENV=prod

# Linux/Mac:
export MERKLE_DB_API_KEY=your_secure_random_key_here
export MIX_ENV=prod

# Start the server
mix run --no-halt
```

---

## Step 3: Verify Server Health

Open a new terminal and check the server status:

```bash
# Check if server is running
curl http://localhost:4000/health/live

# Expected response:
# {"status":"ok","timestamp":"2024-01-08T12:00:00Z"}

# Check if server is ready to accept traffic
curl http://localhost:4000/health/ready

# Expected response:
# {"status":"ready","checks":{"kv_store":"ok","raft":"ok","cache":"ok"},"timestamp":"..."}
```

---

## Step 4: Create Your First Collection

A collection is a container for vectors with a specific dimensionality.

```bash
# Create a collection named "my_vectors" with 128-dimensional f32 vectors
curl -X POST http://localhost:4000/v1/collections/my_vectors \
  -H "Content-Type: application/json" \
  -d '{"dim": 128, "precision": "f32"}'

# Expected response:
# {"status":"created","collection":"my_vectors","dim":128,"precision":"f32"}
```

### Collection Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `dim` | Yes | Vector dimensionality (1-10,000) |
| `precision` | No | `f64`, `f32` (default), or `int8` |

---

## Step 5: Insert Vectors

### Single Vector

```bash
curl -X POST http://localhost:4000/v1/my_vectors/vectors \
  -H "Content-Type: application/json" \
  -d '[{
    "id": "doc_001",
    "vector": [0.1, 0.2, 0.3, 0.4, 0.5, ...],
    "metadata": {"title": "My First Document", "category": "tutorial"}
  }]'
```

### Batch Insert (Recommended)

```bash
curl -X POST http://localhost:4000/v1/my_vectors/vectors \
  -H "Content-Type: application/json" \
  -d '[
    {"id": "doc_001", "vector": [0.1, 0.2, ...], "metadata": {"title": "Doc 1"}},
    {"id": "doc_002", "vector": [0.3, 0.4, ...], "metadata": {"title": "Doc 2"}},
    {"id": "doc_003", "vector": [0.5, 0.6, ...], "metadata": {"title": "Doc 3"}}
  ]'

# Expected response:
# {"status":"ok","inserted":3}
```

### Vector with Text (Auto-Embedding)

If you have text embedding configured:

```bash
curl -X POST http://localhost:4000/v1/my_vectors/vectors \
  -H "Content-Type: application/json" \
  -d '[{
    "id": "article_001",
    "text": "Machine learning is a subset of artificial intelligence...",
    "metadata": {"source": "wikipedia", "topic": "ML"}
  }]'
```

---

## Step 6: Search for Similar Vectors

### K-Nearest Neighbors Search

```bash
curl -X POST http://localhost:4000/v1/my_vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.15, 0.25, 0.35, ...],
    "k": 10
  }'

# Expected response:
# {
#   "results": [
#     {"id": "doc_001", "score": 0.95, "metadata": {"title": "Doc 1"}},
#     {"id": "doc_002", "score": 0.87, "metadata": {"title": "Doc 2"}},
#     ...
#   ],
#   "count": 10,
#   "latency_ms": 2.5
# }
```

### Search with Threshold

Only return results above a similarity threshold:

```bash
curl -X POST http://localhost:4000/v1/my_vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.15, 0.25, 0.35, ...],
    "k": 10,
    "threshold": 0.8
  }'
```

### Semantic Search (Text Query)

```bash
curl -X POST http://localhost:4000/v1/my_vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "text": "artificial intelligence applications",
    "k": 5
  }'
```

---

## Step 7: Using Metadata Filters

Filter search results based on metadata:

```bash
curl -X POST http://localhost:4000/v1/my_vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.15, 0.25, ...],
    "k": 10,
    "filter": {
      "category": {"$eq": "tutorial"}
    }
  }'
```

### Supported Filter Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equals | `{"field": {"$eq": "value"}}` |
| `$ne` | Not equals | `{"field": {"$ne": "value"}}` |
| `$gt` | Greater than | `{"field": {"$gt": 10}}` |
| `$gte` | Greater or equal | `{"field": {"$gte": 10}}` |
| `$lt` | Less than | `{"field": {"$lt": 10}}` |
| `$lte` | Less or equal | `{"field": {"$lte": 10}}` |
| `$in` | In array | `{"field": {"$in": ["a", "b"]}}` |
| `$contains` | Contains substring | `{"field": {"$contains": "text"}}` |

### Combining Filters

```bash
curl -X POST http://localhost:4000/v1/my_vectors/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.15, 0.25, ...],
    "k": 10,
    "filter": {
      "$and": [
        {"category": {"$eq": "tutorial"}},
        {"year": {"$gte": 2023}}
      ]
    }
  }'
```

---

## Step 8: Build an Index (For Large Collections)

For collections with 10,000+ vectors, build an IVF index for faster search:

```bash
curl -X POST http://localhost:4000/v1/my_vectors/build_index \
  -H "Content-Type: application/json" \
  -d '{"type": "ivf", "clusters": 100}'
```

Index building runs in the background. Check progress:

```bash
curl http://localhost:4000/health/detailed

# Look for "index_build" in the response:
# "index_build": {"status": "building", "progress": 0.75, ...}
```

---

## Step 9: Create a Checkpoint

Persist your data to disk:

```bash
curl -X POST http://localhost:4000/v1/my_vectors/checkpoint

# Response:
# {"status":"ok","path":"data/checkpoint_1704700000.bin"}
```

---

## Step 10: Clean Up

### Delete a Collection

```bash
curl -X DELETE http://localhost:4000/v1/collections/my_vectors

# Response:
# {"status":"deleted","collection":"my_vectors"}
```

### Stop the Server

Press `Ctrl+C` in the terminal running MerkleDb. The server will:
1. Stop accepting new connections
2. Drain existing requests
3. Flush the Write-Ahead Log
4. Save a snapshot
5. Gracefully exit

---

## Complete Example: Building a Semantic Search App

Here's a complete workflow for building a document search application:

```bash
# 1. Start the server
mix run --no-halt

# 2. Create a collection for 768-dimensional embeddings (common for BERT models)
curl -X POST http://localhost:4000/v1/collections/documents \
  -H "Content-Type: application/json" \
  -d '{"dim": 768, "precision": "f32"}'

# 3. Insert some documents (vectors would come from your embedding model)
curl -X POST http://localhost:4000/v1/documents/vectors \
  -H "Content-Type: application/json" \
  -d '[
    {"id": "doc1", "vector": [...768 floats...], "metadata": {"title": "Introduction to AI", "author": "Smith"}},
    {"id": "doc2", "vector": [...768 floats...], "metadata": {"title": "Machine Learning Basics", "author": "Jones"}},
    {"id": "doc3", "vector": [...768 floats...], "metadata": {"title": "Deep Learning Guide", "author": "Smith"}}
  ]'

# 4. Search for similar documents
curl -X POST http://localhost:4000/v1/documents/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [...query embedding...],
    "k": 5,
    "filter": {"author": {"$eq": "Smith"}}
  }'

# 5. Build index once you have 10k+ documents
curl -X POST http://localhost:4000/v1/documents/build_index \
  -H "Content-Type: application/json" \
  -d '{"type": "ivf", "clusters": 256}'

# 6. Create checkpoint for persistence
curl -X POST http://localhost:4000/v1/documents/checkpoint
```

---

## Using the Python SDK

For a more convenient interface, use the Python SDK:

```python
from merkledb import MerkleDb
import numpy as np

# Connect to MerkleDb
db = MerkleDb("http://localhost:4000")

# Create collection
db.create_collection("embeddings", dim=768)

# Insert vectors
vectors = [
    {"id": f"doc_{i}", "vector": np.random.rand(768).tolist(), "metadata": {"index": i}}
    for i in range(1000)
]
db.insert("embeddings", vectors)

# Search
query = np.random.rand(768).tolist()
results = db.search("embeddings", vector=query, k=10)

for hit in results:
    print(f"ID: {hit['id']}, Score: {hit['score']:.4f}")
```

See [Python SDK Documentation](../sdk/python/README.md) for more details.

---

## Troubleshooting

### Server Won't Start

1. **Port in use**: Change port with `MERKLE_DB_HTTP_PORT=4001`
2. **Missing dependencies**: Run `mix deps.get` again
3. **NIF compilation failed**: Ensure GCC and NASM are in PATH

### Slow Queries

1. **Build an index**: Use IVF indexing for 10k+ vectors
2. **Reduce k**: Smaller k values are faster
3. **Use Int8 quantization**: 4x smaller memory footprint

### Out of Memory

1. **Reduce cache size**: Set `MERKLE_DB_CACHE_SIZE=10000`
2. **Use Int8 precision**: `{"precision": "int8"}` when creating collection
3. **Build IVF index**: Reduces memory during search

---

## Next Steps

- [Configuration Guide](CONFIGURATION.md) - Fine-tune MerkleDb settings
- [API Reference](API.md) - Complete HTTP API documentation
- [Deployment Guide](DEPLOYMENT.md) - Production deployment instructions
- [Module 2: Core Operations](tutorial/module2_basics.md) - Deep dive into vector operations
