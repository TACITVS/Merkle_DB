# MerkleDb: High-Performance AVX2-Accelerated Vector Database

[![Performance: AVX2](https://img.shields.io/badge/Performance-AVX2%20%2F%20SIMD-orange)](https://github.com/TACITVS/Merkle_DB)
[![Bridge: Zero--Copy NIF](https://img.shields.io/badge/Bridge-Zero--Copy%20NIF-blue)](https://github.com/TACITVS/Merkle_DB)
[![Platform: Elixir](https://img.shields.io/badge/Platform-Elixir%20%2F%20Erlang-purple)](https://elixir-lang.org/)
[![Consensus: Raft](https://img.shields.io/badge/Consensus-Raft-red)](https://github.com/rabbitmq/ra)
[![Production Ready](https://img.shields.io/badge/Production-Ready-green)](https://github.com/TACITVS/Merkle_DB)

**MerkleDb** is a cutting-edge, fault-tolerant vector database engineered for extreme performance and absolute data consistency. It seamlessly bridges the high-level concurrency and distribution of **Elixir/OTP** with the raw power of **x86-64 Assembly (AVX2)**.

Designed for production AI workloads, MerkleDb is a distributed analytics platform capable of processing millions of vectors with microsecond latency, guaranteed by the **Raft consensus algorithm**.

---

## Table of Contents

- [Key Features](#-key-features)
- [Quick Start](#-quick-start)
- [Production Configuration](#-production-configuration)
- [Authentication & Security](#-authentication--security)
- [HTTP API Reference](#-http-api-reference)
- [Python SDK](#-python-sdk)
- [Deployment Guide](#-deployment-guide)
- [Architecture](#-architecture)
- [Documentation](#-documentation)

---

## Key Features

### 1. Hardcore SIMD Performance
At the core of MerkleDb is a library written in **64-bit Assembly**, utilizing **AVX2 (Advanced Vector Extensions)** to perform 256-bit SIMD operations.
- **Microsecond Latency**: Brute-force and indexed searches optimized at the transistor level.
- **24x Speedup**: Massive performance gains over native implementations for vector reductions.

### 2. Fault-Tolerant & Distributed (Raft)
MerkleDb is built for high availability. By integrating the **Raft consensus algorithm**, we ensure that your data is replicated and consistent across a cluster of nodes.
- **Strong Consistency**: Every write is committed to a quorum before being acknowledged.
- **Linearizable Reads**: Guaranteed to see the latest writes.
- **Automatic Leader Election**: If a node fails, the cluster automatically elects a new leader in milliseconds.
- **Self-Healing**: Nodes automatically catch up with the log after downtime.

### 3. Zero-Copy Architecture
Our **V7 Bridge Generator** eliminates NIF performance pitfalls:
- **Direct Binary Access**: Elixir binaries are mapped directly to C/Assembly pointers without copying.
- **Zero Memory Bloat**: Output buffers are pre-allocated, allowing the hardware to write results directly into BEAM-managed memory.
- **Dirty Scheduler Support**: Heavy jobs (HNSW build, training) are offloaded to dirty schedulers to keep the VM responsive.

### 4. Production Ready Security
- **API Key Authentication**: Multi-key support with scopes (read, write, admin)
- **Rate Limiting**: Token bucket algorithm with per-IP and per-key limits
- **Input Validation**: Comprehensive validation for all API inputs
- **HTTPS/TLS**: Built-in TLS support via Cowboy
- **Health Checks**: Kubernetes-compatible liveness and readiness probes

### 5. Advanced Semantic Intelligence
- **Sliding Window Ingestion**: Context-aware chunking for long-form text.
- **Hybrid Search**: Combine vector similarity with metadata filtering.
- **Hierarchical Summarization**: Aggregate passage vectors into chapter and book-level "Topic Vectors."
- **IVF Indexing**: Inverted File index for fast approximate nearest neighbor search.
- **Int8 Quantization**: 4x memory reduction with minimal accuracy loss.

---

## Quick Start

### Prerequisites
- **Elixir ~> 1.14**
- **Erlang/OTP 25+**
- **GCC (MinGW64 on Windows or standard GCC on Linux/Mac)**
- **NASM (Netwide Assembler)**
- **Make**

### Installation

```bash
# Clone the repository
git clone https://github.com/TACITVS/Merkle_DB.git
cd Merkle_DB

# Install dependencies
mix deps.get

# Generate NIF bridge (if needed)
mix run gen_bridge.exs

# Compile
mix compile

# Start the server
mix run --no-halt
```

The server will start on `http://localhost:4000`.

### Verify Installation

```bash
# Check health
curl http://localhost:4000/health/ready

# Response:
# {"status":"ready","checks":{"kv_store":"ok","raft":"ok",...},"timestamp":"..."}
```

### Your First Vectors

```bash
# 1. Create a collection
curl -X POST http://localhost:4000/v1/collections/my_embeddings \
  -H "Content-Type: application/json" \
  -d '{"dim": 128, "precision": "f32"}'

# 2. Insert vectors
curl -X POST http://localhost:4000/v1/my_embeddings/vectors \
  -H "Content-Type: application/json" \
  -d '[
    {"id": "doc1", "vector": [0.1, 0.2, 0.3, ...], "metadata": {"title": "Hello"}},
    {"id": "doc2", "vector": [0.4, 0.5, 0.6, ...], "metadata": {"title": "World"}}
  ]'

# 3. Search
curl -X POST http://localhost:4000/v1/my_embeddings/search \
  -H "Content-Type: application/json" \
  -d '{"vector": [0.1, 0.2, 0.3, ...], "k": 10, "threshold": 0.5}'
```

---

## Production Configuration

MerkleDb uses environment variables for production configuration:

### Required Variables (Production)

| Variable | Description |
|----------|-------------|
| `MERKLE_DB_API_KEY` | Primary API key (required in production) |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MERKLE_DB_HTTP_PORT` | 4000 | HTTP server port |
| `MERKLE_DB_HTTPS_PORT` | 4443 | HTTPS server port |
| `MERKLE_DB_ENABLE_HTTPS` | true | Enable HTTPS (requires SSL certs) |
| `MERKLE_DB_SSL_CERT` | priv/ssl/cert.pem | SSL certificate path |
| `MERKLE_DB_SSL_KEY` | priv/ssl/key.pem | SSL private key path |
| `MERKLE_DB_DATA_DIR` | data | Data directory |
| `MERKLE_DB_RATE_LIMIT` | 100 | Requests per minute per client |
| `MERKLE_DB_RATE_BURST` | 20 | Burst capacity |
| `MERKLE_DB_CACHE_SIZE` | 50000 | Vector cache max entries |
| `MERKLE_DB_QUERY_TIMEOUT_MS` | 30000 | Query timeout in milliseconds |

### Example Production Setup (Windows)

```batch
set MERKLE_DB_API_KEY=your_secure_random_key_here
set MERKLE_DB_DATA_DIR=C:\MerkleDb\data
set MERKLE_DB_ENABLE_HTTPS=false
set MIX_ENV=prod

mix run --no-halt
```

### Generate a Secure API Key

```bash
mix run -e "IO.puts(:crypto.strong_rand_bytes(32) |> Base.url_encode64())"
```

See [docs/CONFIGURATION.md](docs/CONFIGURATION.md) for complete reference.

---

## Authentication & Security

### API Key Authentication

In production (`MIX_ENV=prod`), all `/v1/*` endpoints require authentication:

```bash
# Using Authorization header (recommended)
curl -H "Authorization: Bearer YOUR_API_KEY" \
  http://localhost:4000/v1/collections

# Using X-API-Key header
curl -H "X-API-Key: YOUR_API_KEY" \
  http://localhost:4000/v1/collections
```

### API Key Scopes

| Scope | Permissions |
|-------|-------------|
| `read` | Query collections and vectors |
| `write` | Insert, update, delete vectors |
| `admin` | Manage collections, view metrics, admin dashboard |

### Multiple API Keys

Configure multiple keys with different scopes:

```bash
# Format: key:name:scope1|scope2,key2:name2:scope
set MERKLE_DB_API_KEYS=abc123:primary:read|write|admin,xyz789:readonly:read
```

### Rate Limiting

Requests are rate-limited per IP and per API key:

- Default: 100 requests/minute with burst of 20
- Rate limit headers are included in responses:
  - `X-RateLimit-Limit`: Maximum requests per window
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Milliseconds until reset

When rate limited, you'll receive a `429 Too Many Requests` response with a `Retry-After` header.

### Health Checks (No Auth Required)

These endpoints are always accessible without authentication:

| Endpoint | Purpose |
|----------|---------|
| `GET /health/live` | Liveness probe - is the process running? |
| `GET /health/ready` | Readiness probe - can it serve traffic? |
| `GET /health/detailed` | Detailed metrics and status |

---

## HTTP API Reference

### Health Endpoints

```bash
# Liveness probe
GET /health/live
# Response: {"status":"ok","timestamp":"2024-01-08T12:00:00Z"}

# Readiness probe
GET /health/ready
# Response: {"status":"ready","checks":{...},"timestamp":"..."}

# Detailed health
GET /health/detailed
# Response: {"status":"ok","version":"0.2.0","uptime_seconds":3600,...}
```

### Collection Management

```bash
# List collections
GET /v1/collections
# Response: {"collections":["my_vectors","embeddings"]}

# Create collection
POST /v1/collections/:name
Content-Type: application/json
{"dim": 128, "precision": "f32"}
# precision: "f64" | "f32" | "int8"

# Delete collection
DELETE /v1/collections/:name
```

### Vector Operations

```bash
# Insert/Update vectors (batch)
POST /v1/:collection/vectors
Content-Type: application/json
[
  {"id": "vec1", "vector": [0.1, 0.2, ...], "metadata": {"key": "value"}},
  {"id": "vec2", "text": "Semantic text to embed", "metadata": {}}
]

# Search by vector
POST /v1/:collection/search
Content-Type: application/json
{"vector": [0.1, 0.2, ...], "k": 10, "threshold": 0.5}

# Search by text (semantic)
POST /v1/:collection/search
Content-Type: application/json
{"text": "search query", "k": 10}

# Create checkpoint
POST /v1/:collection/checkpoint
```

### Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request (validation error) |
| 401 | Unauthorized (missing/invalid API key) |
| 403 | Forbidden (insufficient scope) |
| 404 | Not Found |
| 409 | Conflict (already exists) |
| 429 | Rate Limited |
| 500 | Internal Server Error |

See [docs/API.md](docs/API.md) for complete API reference.

---

## Python SDK

MerkleDb includes a comprehensive Python SDK for easy integration with your AI pipelines.

### Installation

```bash
pip install merkledb
# or
pip install -e sdk/python
```

### Usage

```python
from merkledb import MerkleDb
import numpy as np

# Connect (with optional API key)
db = MerkleDb("http://localhost:4000", api_key="your_key")

# Create collection
db.create_collection("embeddings", dim=768, precision="f32")

# Insert vectors
vectors = [
    {"id": "doc1", "vector": np.random.rand(768).tolist(), "metadata": {"title": "Doc 1"}},
    {"id": "doc2", "vector": np.random.rand(768).tolist(), "metadata": {"title": "Doc 2"}}
]
db.insert("embeddings", vectors)

# Search
results = db.search("embeddings", vector=query_vec.tolist(), k=10)
for hit in results:
    print(f"{hit['id']}: {hit['score']:.4f}")

# Semantic search (auto-embeds text)
results = db.search("embeddings", text="machine learning", k=5)
```

See [sdk/python/README.md](sdk/python/README.md) for complete SDK documentation.

---

## Deployment Guide

### Windows Native Deployment

1. **Build a release**:
```batch
set MIX_ENV=prod
mix deps.get
mix release
```

2. **Configure environment**:
```batch
set MERKLE_DB_API_KEY=your_secure_key_here
set MERKLE_DB_DATA_DIR=C:\MerkleDb\data
set MERKLE_DB_ENABLE_HTTPS=false
```

3. **Run the release**:
```batch
_build\prod\rel\merkle_db\bin\merkle_db.bat start
```

4. **Run as Windows Service** (optional):
Use NSSM (Non-Sucking Service Manager) to install as a service.

### HTTPS Setup

1. **Generate self-signed certificates** (for testing):
```bash
openssl req -x509 -newkey rsa:4096 \
  -keyout priv/ssl/key.pem \
  -out priv/ssl/cert.pem \
  -days 365 -nodes \
  -subj "/CN=localhost"
```

2. **Enable HTTPS**:
```batch
set MERKLE_DB_ENABLE_HTTPS=true
set MERKLE_DB_SSL_CERT=priv/ssl/cert.pem
set MERKLE_DB_SSL_KEY=priv/ssl/key.pem
```

### Graceful Shutdown

MerkleDb handles shutdown gracefully:
1. Stops accepting new connections
2. Drains existing requests (5 second timeout)
3. Flushes Write-Ahead Log
4. Saves snapshot to disk
5. Leaves Raft cluster

Use `Ctrl+C` or send `SIGTERM` to initiate graceful shutdown.

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for complete deployment guide.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    HTTP API (Cowboy)                         │
│              /v1/* (Authenticated) │ /health/*              │
├─────────────────────────────────────────────────────────────┤
│    Rate Limiter    │   API Key Store   │  Input Validator   │
├─────────────────────────────────────────────────────────────┤
│                      Query Engine                            │
│          KNN │ Range │ Semantic │ Hybrid │ Filtered         │
├─────────────────────────────────────────────────────────────┤
│                      Storage Layer                           │
│     KV Store │ Vector Cache │ Text Store │ Payload Store    │
├─────────────────────────────────────────────────────────────┤
│                      Persistence                             │
│               WAL │ Snapshots │ Checkpoints                  │
├─────────────────────────────────────────────────────────────┤
│                   Raft Consensus (Ra)                        │
│            Leader Election │ Log Replication                 │
├─────────────────────────────────────────────────────────────┤
│                   Native Acceleration                        │
│          AVX2 SIMD │ Dot Product │ Distance │ K-Means       │
└─────────────────────────────────────────────────────────────┘
```

### Design Philosophy

MerkleDb adopts a **Hybrid Strategy**:
- **Elixir Control Plane**: Manages distribution, consensus (Raft), and fault recovery.
- **C/ASM Data Plane**: Handles the "hot" loops—the expensive vector math—using hardware-level optimizations.

---

## Documentation

- [Quick Start Tutorial](docs/QUICKSTART.md) - Step-by-step getting started guide
- [Configuration Guide](docs/CONFIGURATION.md) - Complete configuration reference
- [API Reference](docs/API.md) - Full HTTP API documentation
- [Deployment Guide](docs/DEPLOYMENT.md) - Production deployment instructions
- [Changelog](CHANGELOG.md) - Version history and release notes

### Developer Course

- [Module 1: Getting Started](docs/tutorial/module1_setup.md) - Installation and first instance
- [Module 2: Core Vector Operations](docs/tutorial/module2_basics.md) - KV API basics
- [Module 3: Going Distributed with Raft](docs/tutorial/module3_clustering.md) - Multi-node clusters
- [Module 4: Advanced Search & Analytics](docs/tutorial/module4_advanced.md) - IVF, quantization, semantic search

---

## Performance

- **Query Latency**: <1ms cached, <10ms indexed search
- **Throughput**: 10,000+ QPS on modern hardware
- **Index Build**: ~1M vectors/minute for IVF indexing
- **Memory**: Int8 quantization provides 4x compression

---

## Development

### Running Tests
```bash
mix test
```

### Code Quality
```bash
mix credo --strict
mix dialyzer
```

### Format Code
```bash
mix format
```

---

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and code quality checks
5. Submit a pull request

## Support

- GitHub Issues: https://github.com/TACITVS/Merkle_DB/issues

---

Developed by **TACITVS**
