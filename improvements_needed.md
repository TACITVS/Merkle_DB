# MerkleDB Improvements Needed

**Audit Date:** 2026-01-04
**Last Updated:** 2026-01-04
**Auditor:** Claude Code
**Version:** Pre-release (main branch)

## Recent Implementations

- **2026-01-04**: Int8 Scalar Quantization implemented (8x memory reduction, AVX2-accelerated)
- **2026-01-04**: Metadata/Scalar Filtering implemented (`:where` filter with eq, neq, gt, lt, gte, lte, in)
- **2026-01-04**: Delete/Update Operations implemented (Tombstone-based soft delete, logical updates)
- **2026-01-04**: Range Queries implemented (`:range` query type with min/max similarity bounds)

---

## Executive Summary

MerkleDB is a columnar vector database written in Elixir with 600+ native C/ASM SIMD kernels. It implements IVF (Inverted File) indexing for efficient similarity search. The foundation is solid with exceptional performance characteristics, but significant gaps exist compared to production vector databases like Pinecone, Qdrant, Milvus, and Weaviate.

---

## Current State

### Fully Implemented (Production-Ready)

| Category | Features |
|----------|----------|
| **Core Operations** | Insert, Batch Insert (50x faster via columnar AXPY), Vector Validation, L2 Normalization |
| **Search** | KNN with Cosine Similarity, Similarity Threshold Filtering, Cached Queries (BLAKE3 keys) |
| **Indexing** | IVF (K-Means++), Parallel IVF Search (top-N clusters), Flat/Brute-Force Search, PCA Dimensionality Reduction |
| **Storage** | BLAKE3-Protected Snapshots, Optional Compression, Atomic Writes with Global Locks, Backup Rotation (current/previous) |
| **Performance** | AVX2/FMA SIMD Kernels, Columnar Storage Layout, AXPY-Optimized Batch Operations, Indexed GEMV for IVF |
| **Analytics** | Column Statistics (mean/min/max per dimension), Memory Estimation, Telemetry Integration, Index Statistics |
| **API** | HTTP REST Endpoints, Batch Operations, Progress Tracking |
| **Reliability** | Optimistic Locking (generation-based), Single-Node ACID, Crash Recovery via Snapshots |

### Stubbed (API Exists, Implementation Pending)

- `TextAnalytics` module (corpus analysis, word contexts, cluster analysis)
- `BenchmarkRunner` module (insert/query/IVF benchmarks)
- Multi-node Clustering (only single-node analysis implemented)
- Custom Embedding Functions (extensible via FPDispatcher but no examples)
- Incremental Backup (only full snapshots, no WAL)

---

## Critical Gaps

### Priority 1: Delete/Update Operations - IMPLEMENTED

**Status:** FULLY IMPLEMENTED (2026-01-04)

**Implementation:**
- **Soft Deletes:** Tombstone-based mechanism using `tree.tombstones` MapSet.
- **Updates:** Logical update via "Delete Old + Insert New" (append-only log).
- **Filtering:** Queries automatically filter out tombstoned indices.
- **Persistence:** Tombstones and key index persist across snapshots.

**Usage:**
```elixir
# Delete a key
MerkleDb.KV.delete("key1")

# Update a key (overwrite)
MerkleDb.KV.put("key1", new_vector)
```

**Effort:** Medium (2-3 days) - COMPLETED

---

### Priority 2: Metadata/Scalar Filtering - IMPLEMENTED

**Status:** FULLY IMPLEMENTED (2026-01-04)

**Implementation:**
- **Storage:** `metadata` map in `Tree` struct mapping `row_index` to attribute maps.
- **Predicates:** Support for `:eq`, `:neq`, `:gt`, `:lt`, `:gte`, `:lte`, and `:in`.
- **Integration:** Filters applied during KNN and Range searches.
- **Batch Support:** Metadata can be provided during batch insertion.

**Usage:**
```elixir
# KNN with filter
Query.execute(tree, [:knn, query_vec, 10, 0.7, {:where, [{"price", :lt, 50}, {"cat", :eq, "electronics"}]}])

# Range query with filter
Query.execute(tree, [:range, query_vec, 0.8, 1.0, {:where, [{"tenant_id", :eq, 123}]}])
```

**Effort:** Medium-High (3-5 days) - COMPLETED

---

### Priority 3: HNSW Index

**Current State:** Only IVF (K-Means) indexing available.

**Impact:**
- IVF requires rebuilding centroids when data changes significantly
- HNSW provides better recall at same latency for many workloads
- Industry standard (Qdrant, Weaviate, Milvus all use HNSW)

**Recommended Implementation:**
```
1. Implement hierarchical graph structure
   - Multiple layers with decreasing density
   - Entry point at top layer

2. Greedy search with backtracking
   - Start at entry point
   - Descend layers, searching neighbors

3. Dynamic insertion without full rebuild
   - Add new node to appropriate layer
   - Connect to M nearest neighbors
```

**Effort:** High (1-2 weeks)

---

### Priority 4: Vector Quantization - IMPLEMENTED (Int8)

**Status:** PARTIALLY IMPLEMENTED (Int8 Scalar Quantization) (2026-01-04)

**Implementation:**
- **Scalar Quantization (Int8):** Vectors are compressed from 64-bit floats to 8-bit integers.
- **AVX2 Acceleration:** Quantization and search kernels are optimized using AVX2 SIMD instructions.
- **Space Efficiency:** 8x reduction in vector storage size.
- **Integration:** Automated quantization via `Tree.quantize/1`. Search automatically uses quantized index if available.

**Usage:**
```elixir
# Quantize the tree
tree = MerkleDb.Tree.quantize(tree)

# Search automatically uses Int8 index
results = MerkleDb.Query.execute(tree, [:knn, query_vec, 10, 0.0])
```

**Effort:** Medium per type (2-3 days each) - DONE (Int8)

---

### Priority 5: Sparse Vector Support

**Current State:** Only dense vectors supported.

**Impact:**
- Cannot handle BM25/TF-IDF sparse representations
- No hybrid dense+sparse search
- Limited for keyword-aware retrieval

**Recommended Implementation:**
```
1. Sparse vector format
   %SparseVector{indices: [int], values: [float], dim: int}

2. Separate storage for sparse vectors
   sparse_columns: %{vector_id => %SparseVector{}}

3. Sparse dot product kernel
   fp_sparse_dotp(indices_a, values_a, indices_b, values_b)

4. Hybrid scoring
   score = alpha * dense_score + (1 - alpha) * sparse_score
```

**Effort:** Medium (3-4 days)

---

## Secondary Gaps

### Multi-Node Distribution

**Current State:** Single-node only. Cluster module exists but only analyzes local data.

**Required for:**
- Horizontal scaling beyond single machine
- High availability (replication)
- Fault tolerance

**Components Needed:**
- Consensus layer (Raft or similar)
- Data partitioning/sharding strategy
- Distributed query routing
- Replication and failover

**Effort:** Very High (2-4 weeks)

---

### GPU Acceleration

**Current State:** CPU-only with AVX2 SIMD.

**Impact:**
- GPU can provide 10-100x speedup for large batches
- Required for real-time inference at scale

**Options:**
- CUDA kernels for NVIDIA
- ROCm for AMD
- Metal for Apple Silicon
- Vulkan compute for cross-platform

**Effort:** High (1-2 weeks per platform)

---

### Range Queries - IMPLEMENTED

**Status:** FULLY IMPLEMENTED (2026-01-04)

**Usage:**
```elixir
# Basic range query - all vectors with similarity in [0.7, 0.9]
Query.execute(tree, [:range, query_vec, 0.7, 0.9])

# With limit - max 100 results
Query.execute(tree, [:range, query_vec, 0.7, 0.9, 100])

# Parallel IVF search
Query.execute(tree, [:range, query_vec, 0.7, 0.9, :parallel])

# Parallel with limit
Query.execute(tree, [:range, query_vec, 0.7, 0.9, 100, :parallel])
```

**Effort:** Low (1 day) - COMPLETED

---

### Namespaces/Collections

**Current State:** Single global tree.

**Impact:**
- No multi-tenant isolation
- Cannot organize vectors by category/type

**Implementation:**
```elixir
# Multiple named trees
KV.create_collection("products")
KV.create_collection("users")
KV.insert("products", key, vector)
```

**Effort:** Low-Medium (2-3 days)

---

## Comparison Matrix

| Feature | MerkleDB | Pinecone | Qdrant | Milvus | Weaviate |
|---------|----------|----------|--------|--------|----------|
| KNN Search | Yes | Yes | Yes | Yes | Yes |
| IVF Index | Yes | Yes | No | Yes | No |
| HNSW Index | **No** | Yes | Yes | Yes | Yes |
| Delete/Update | **No** | Yes | Yes | Yes | Yes |
| Metadata Filter | **No** | Yes | Yes | Yes | Yes |
| Sparse Vectors | **No** | Yes | Yes | Yes | No |
| Quantization | **No** | Yes | Yes | Yes | Yes |
| Multi-Node | **No** | Yes | Yes | Yes | Yes |
| GPU Support | **No** | Yes | Exp | Yes | No |
| SIMD Optimized | Yes | Yes | Yes | Yes | Yes |
| Snapshots | Yes | Yes | Yes | Yes | Yes |

---

## Recommended Roadmap

### Phase 1: Core Completeness (1-2 weeks)
1. ~~Delete operations with tombstones~~ - DONE (2026-01-04)
2. ~~Basic metadata filtering (equality, range)~~ - DONE (2026-01-04)
3. ~~Range queries~~ - DONE (2026-01-04)

### Phase 2: Performance Parity (2-3 weeks)
4. ~~Int8 scalar quantization~~ - DONE (2026-01-04)
5. HNSW index implementation
6. Namespaces/collections

### Phase 3: Advanced Features (3-4 weeks)
7. Sparse vector support
8. Product quantization
9. Hybrid search (dense + sparse)

### Phase 4: Scale (4+ weeks)
10. Multi-node replication
11. Sharding strategy
12. GPU acceleration

---

## Architecture Strengths to Preserve

1. **Columnar Storage**: Each dimension stored separately enables AXPY batch operations and cache-friendly access patterns.

2. **SIMD Kernels**: 600+ hand-optimized AVX2/FMA functions provide exceptional single-node performance.

3. **Optimistic Locking**: Generation-based versioning prevents stale index updates without heavy locking.

4. **Immutable Core**: Pure functional approach in Tree operations simplifies reasoning and crash recovery.

5. **Elixir/OTP**: Supervision trees, fault tolerance, and hot code reloading built-in.

---

## Conclusion

MerkleDB has a solid foundation with exceptional SIMD performance and clean architecture. The primary gaps (delete, filtering, HNSW, quantization) are well-understood problems with established solutions. Implementing Priority 1-4 would bring MerkleDB to feature parity with open-source alternatives like Qdrant for single-node deployments.

The columnar storage design and extensive native kernel library are significant differentiators that should be preserved and leveraged as new features are added.
