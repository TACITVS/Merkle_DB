# MerkleDb HTTP API Reference

Complete reference for the MerkleDb REST API.

---

## Base URL

- **Development**: `http://localhost:4000`
- **Production HTTP**: `http://localhost:4000` (or configured port)
- **Production HTTPS**: `https://localhost:4443` (or configured port)

---

## Authentication

In production mode (`MIX_ENV=prod`), all `/v1/*` endpoints require authentication.

### Authentication Headers

```http
# Option 1: Bearer Token (recommended)
Authorization: Bearer YOUR_API_KEY

# Option 2: X-API-Key Header
X-API-Key: YOUR_API_KEY
```

### API Key Scopes

| Scope | Allowed Operations |
|-------|-------------------|
| `read` | GET requests, search queries |
| `write` | POST/PUT/DELETE for vectors |
| `admin` | Collection management, metrics, admin endpoints |

### Authentication Errors

| Code | Error | Description |
|------|-------|-------------|
| 401 | `unauthorized` | Missing or invalid API key |
| 403 | `forbidden` | Valid key but insufficient scope |

```json
{
  "error": "unauthorized",
  "message": "Invalid or missing API key"
}
```

---

## Rate Limiting

Requests are rate-limited per IP address and per API key.

### Rate Limit Headers

Every response includes rate limit information:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 45000
```

| Header | Description |
|--------|-------------|
| `X-RateLimit-Limit` | Maximum requests per minute |
| `X-RateLimit-Remaining` | Remaining requests in current window |
| `X-RateLimit-Reset` | Milliseconds until limit resets |

### Rate Limit Exceeded

When rate limited, you receive:

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 30

{
  "error": "rate_limited",
  "message": "Rate limit exceeded",
  "retry_after_seconds": 30
}
```

---

## Health Endpoints

Health check endpoints do not require authentication.

### GET /health/live

Liveness probe - checks if the process is running.

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2024-01-08T12:00:00.000Z"
}
```

### GET /health/ready

Readiness probe - checks if the server can serve traffic.

**Response:**
```json
{
  "status": "ready",
  "checks": {
    "kv_store": "ok",
    "raft": "ok",
    "cache": "ok",
    "wal": "ok"
  },
  "timestamp": "2024-01-08T12:00:00.000Z"
}
```

**Failure Response (503):**
```json
{
  "status": "not_ready",
  "checks": {
    "kv_store": "ok",
    "raft": "initializing",
    "cache": "ok"
  },
  "timestamp": "2024-01-08T12:00:00.000Z"
}
```

### GET /health/detailed

Detailed health and metrics information.

**Response:**
```json
{
  "status": "ok",
  "version": "0.2.0",
  "uptime_seconds": 3600,
  "query_metrics": {
    "total_queries": 10000,
    "qps_current": 150.5,
    "avg_latency_ms": 2.3,
    "p95_latency_ms": 5.1,
    "p99_latency_ms": 12.4
  },
  "cache_metrics": {
    "hit_rate": 0.85,
    "size_entries": 45000,
    "size_mb": 128.5
  },
  "system_metrics": {
    "memory_mb": 512.3,
    "vector_count": 100000,
    "indexed": true,
    "index_type": "ivf",
    "cluster_count": 256
  },
  "raft": {
    "state": "leader",
    "term": 5,
    "commit_index": 12345
  }
}
```

---

## Collection Management

### GET /v1/collections

List all collections.

**Required Scope:** `read`

**Response:**
```json
{
  "collections": [
    {
      "name": "embeddings",
      "dim": 768,
      "precision": "f32",
      "count": 50000,
      "indexed": true
    },
    {
      "name": "images",
      "dim": 512,
      "precision": "f32",
      "count": 10000,
      "indexed": false
    }
  ]
}
```

### POST /v1/collections/:name

Create a new collection.

**Required Scope:** `admin`

**URL Parameters:**
- `name` - Collection name (alphanumeric, underscores, hyphens; max 128 chars)

**Request Body:**
```json
{
  "dim": 768,
  "precision": "f32"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `dim` | integer | Yes | Vector dimensionality (1-10,000) |
| `precision` | string | No | `f64`, `f32` (default), or `int8` |

**Response (201):**
```json
{
  "status": "created",
  "collection": "embeddings",
  "dim": 768,
  "precision": "f32"
}
```

**Errors:**
- `400` - Invalid parameters
- `409` - Collection already exists

### GET /v1/collections/:name

Get collection details.

**Required Scope:** `read`

**Response:**
```json
{
  "name": "embeddings",
  "dim": 768,
  "precision": "f32",
  "count": 50000,
  "indexed": true,
  "index_type": "ivf",
  "cluster_count": 256,
  "created_at": "2024-01-01T00:00:00Z"
}
```

### DELETE /v1/collections/:name

Delete a collection and all its data.

**Required Scope:** `admin`

**Response:**
```json
{
  "status": "deleted",
  "collection": "embeddings"
}
```

**Errors:**
- `404` - Collection not found

---

## Vector Operations

### POST /v1/:collection/vectors

Insert or update vectors (batch operation).

**Required Scope:** `write`

**Request Body:**
```json
[
  {
    "id": "doc_001",
    "vector": [0.1, 0.2, 0.3, ...],
    "metadata": {
      "title": "Document Title",
      "category": "tutorial",
      "score": 95
    }
  },
  {
    "id": "doc_002",
    "text": "Text to be embedded automatically",
    "metadata": {
      "title": "Another Document"
    }
  }
]
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | string | Yes | Unique vector identifier (max 256 chars) |
| `vector` | array | Conditional | Float array matching collection dimension |
| `text` | string | Conditional | Text to auto-embed (requires embedding config) |
| `metadata` | object | No | Arbitrary JSON metadata |

**Note:** Either `vector` or `text` must be provided, not both.

**Response:**
```json
{
  "status": "ok",
  "inserted": 2,
  "updated": 0,
  "errors": []
}
```

**Partial Failure Response:**
```json
{
  "status": "partial",
  "inserted": 1,
  "updated": 0,
  "errors": [
    {
      "id": "doc_002",
      "error": "vector dimension mismatch: expected 768, got 512"
    }
  ]
}
```

### GET /v1/:collection/vectors/:id

Get a specific vector by ID.

**Required Scope:** `read`

**Response:**
```json
{
  "id": "doc_001",
  "vector": [0.1, 0.2, 0.3, ...],
  "metadata": {
    "title": "Document Title",
    "category": "tutorial"
  }
}
```

**Errors:**
- `404` - Vector not found

### DELETE /v1/:collection/vectors/:id

Delete a specific vector.

**Required Scope:** `write`

**Response:**
```json
{
  "status": "deleted",
  "id": "doc_001"
}
```

---

## Search Operations

### POST /v1/:collection/search

Search for similar vectors.

**Required Scope:** `read`

**Request Body (Vector Search):**
```json
{
  "vector": [0.1, 0.2, 0.3, ...],
  "k": 10,
  "threshold": 0.5,
  "filter": {
    "category": {"$eq": "tutorial"}
  },
  "include_vectors": false,
  "include_metadata": true
}
```

**Request Body (Text Search):**
```json
{
  "text": "machine learning tutorial",
  "k": 10,
  "filter": {
    "category": {"$eq": "tutorial"}
  }
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `vector` | array | Conditional | Query vector |
| `text` | string | Conditional | Text query (auto-embedded) |
| `k` | integer | No | Number of results (default: 10, max: 1000) |
| `threshold` | float | No | Minimum similarity score (0.0-1.0) |
| `filter` | object | No | Metadata filter expression |
| `include_vectors` | boolean | No | Include vectors in results (default: false) |
| `include_metadata` | boolean | No | Include metadata in results (default: true) |

**Response:**
```json
{
  "results": [
    {
      "id": "doc_001",
      "score": 0.95,
      "metadata": {
        "title": "Document Title",
        "category": "tutorial"
      }
    },
    {
      "id": "doc_002",
      "score": 0.87,
      "metadata": {
        "title": "Another Document",
        "category": "tutorial"
      }
    }
  ],
  "count": 2,
  "latency_ms": 2.5,
  "search_type": "indexed"
}
```

### Filter Operators

#### Comparison Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$eq` | Equals | `{"field": {"$eq": "value"}}` |
| `$ne` | Not equals | `{"field": {"$ne": "value"}}` |
| `$gt` | Greater than | `{"field": {"$gt": 10}}` |
| `$gte` | Greater or equal | `{"field": {"$gte": 10}}` |
| `$lt` | Less than | `{"field": {"$lt": 10}}` |
| `$lte` | Less or equal | `{"field": {"$lte": 10}}` |

#### Array/String Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$in` | Value in array | `{"field": {"$in": ["a", "b", "c"]}}` |
| `$nin` | Value not in array | `{"field": {"$nin": ["x", "y"]}}` |
| `$contains` | String contains | `{"field": {"$contains": "substring"}}` |
| `$starts_with` | String starts with | `{"field": {"$starts_with": "prefix"}}` |
| `$ends_with` | String ends with | `{"field": {"$ends_with": "suffix"}}` |

#### Logical Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `$and` | All conditions must match | `{"$and": [{...}, {...}]}` |
| `$or` | Any condition must match | `{"$or": [{...}, {...}]}` |
| `$not` | Negate condition | `{"$not": {"field": {"$eq": "value"}}}` |

#### Complex Filter Example

```json
{
  "vector": [0.1, 0.2, ...],
  "k": 10,
  "filter": {
    "$and": [
      {"category": {"$in": ["tutorial", "guide"]}},
      {"score": {"$gte": 80}},
      {
        "$or": [
          {"author": {"$eq": "John"}},
          {"verified": {"$eq": true}}
        ]
      }
    ]
  }
}
```

---

## Index Operations

### POST /v1/:collection/build_index

Build an IVF index for faster approximate search.

**Required Scope:** `admin`

**Request Body:**
```json
{
  "type": "ivf",
  "clusters": 256,
  "sample_size": 10000
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | No | Index type: `ivf` (default) |
| `clusters` | integer | No | Number of clusters (default: sqrt(n)) |
| `sample_size` | integer | No | Training sample size |

**Response:**
```json
{
  "status": "building",
  "job_id": "idx_12345",
  "estimated_time_seconds": 120
}
```

Index building runs in the background. Check progress via `/health/detailed`.

### DELETE /v1/:collection/index

Remove the index (revert to brute-force search).

**Required Scope:** `admin`

**Response:**
```json
{
  "status": "ok",
  "message": "Index removed, using brute-force search"
}
```

---

## Persistence Operations

### POST /v1/:collection/checkpoint

Create a checkpoint (snapshot) of the collection.

**Required Scope:** `admin`

**Response:**
```json
{
  "status": "ok",
  "checkpoint_path": "data/snapshots/checkpoint_1704700000.bin",
  "vector_count": 50000,
  "size_bytes": 153600000
}
```

### GET /v1/:collection/checkpoints

List available checkpoints.

**Required Scope:** `read`

**Response:**
```json
{
  "checkpoints": [
    {
      "path": "data/snapshots/checkpoint_1704700000.bin",
      "created_at": "2024-01-08T10:00:00Z",
      "vector_count": 50000,
      "size_bytes": 153600000
    },
    {
      "path": "data/snapshots/checkpoint_1704696400.bin",
      "created_at": "2024-01-08T09:00:00Z",
      "vector_count": 49000,
      "size_bytes": 150528000
    }
  ]
}
```

---

## Admin Endpoints

### GET /v1/admin/metrics

Get detailed server metrics.

**Required Scope:** `admin`

**Response:**
```json
{
  "query_metrics": {
    "total_queries": 100000,
    "qps_current": 250.5,
    "avg_latency_ms": 1.8,
    "median_latency_ms": 1.2,
    "p95_latency_ms": 4.5,
    "p99_latency_ms": 10.2,
    "slowest_query_ms": 45.3
  },
  "cache_metrics": {
    "hit_rate": 0.892,
    "total_hits": 89200,
    "total_misses": 10800,
    "size_entries": 50000,
    "size_mb": 156.8
  },
  "system_metrics": {
    "memory_mb": 1024.5,
    "memory_gb": 1.0,
    "uptime_seconds": 86400
  },
  "collections": {
    "embeddings": {
      "count": 50000,
      "indexed": true,
      "index_type": "ivf"
    }
  }
}
```

### POST /v1/admin/cache/clear

Clear the vector cache.

**Required Scope:** `admin`

**Response:**
```json
{
  "status": "ok",
  "cleared_entries": 50000
}
```

---

## Response Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Missing/invalid API key |
| 403 | Forbidden - Insufficient scope |
| 404 | Not Found |
| 409 | Conflict - Resource already exists |
| 429 | Too Many Requests - Rate limited |
| 500 | Internal Server Error |
| 503 | Service Unavailable - Server not ready |

---

## Error Response Format

All errors follow this format:

```json
{
  "error": "error_code",
  "message": "Human-readable error description",
  "details": {
    "field": "Additional context if available"
  }
}
```

### Common Error Codes

| Code | Description |
|------|-------------|
| `invalid_request` | Malformed JSON or missing fields |
| `validation_error` | Field validation failed |
| `unauthorized` | Authentication required |
| `forbidden` | Insufficient permissions |
| `not_found` | Resource doesn't exist |
| `already_exists` | Resource already exists |
| `rate_limited` | Too many requests |
| `internal_error` | Server error |
| `timeout` | Operation timed out |

---

## See Also

- [Quick Start Guide](QUICKSTART.md)
- [Configuration Guide](CONFIGURATION.md)
- [Deployment Guide](DEPLOYMENT.md)
