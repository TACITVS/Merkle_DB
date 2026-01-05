# MerkleDB Improvements Needed

**Audit Date:** 2026-01-04
**Last Updated:** 2026-01-04
**Auditor:** Claude Code
**Version:** Pre-release (main branch)

## Recent Implementations

- **2026-01-04**: Sparse Vector Support implemented (Hybrid dense+sparse search)
- **2026-01-04**: Namespaces/Collections implemented (Multi-tenant support via `KV` module)
- **2026-01-04**: HNSW Indexing implemented (Fast approximate nearest neighbor search)
- **2026-01-04**: HNSW top-k search + ef_search/ef_construction neighbor selection implemented
- **2026-01-04**: Int8 Scalar Quantization implemented (8x memory reduction, AVX2-accelerated)
- **2026-01-04**: Metadata/Scalar Filtering implemented (`:where` filter with eq, neq, gt, lt, gte, lte, in)
- **2026-01-04**: Payload/metadata filter DSL expanded (string ops, nested fields, PayloadStore merge)
- **2026-01-04**: Delete/Update Operations implemented (Tombstone-based soft delete, logical updates)
- **2026-01-04**: Range Queries implemented (`:range` query type with min/max similarity bounds)
- **2026-01-04**: PCA transform + variance metrics implemented
- **2026-01-04**: Benchmarks use real inserts (no simulation)

---

## Executive Summary

MerkleDB is a columnar vector database written in Elixir with 600+ native C/ASM SIMD kernels. It implements IVF indexing plus HNSW, sparse/hybrid search, and Int8 quantization for efficient similarity search. The foundation is solid with strong single-node parity; remaining gaps are product quantization, multi-node distribution/sharding, GPU acceleration, and tree-level incremental backup and embedding extension hooks.

---

## Current State

### Fully Implemented (Production-Ready)

| Category | Features |
|----------|----------|
| **Core Operations** | Insert, Batch Insert (columnar AXPY), Delete/Update (tombstones), Vector Validation, L2 Normalization |
| **Search** | KNN, Range, Sparse, Hybrid, Similarity Threshold Filtering, Cached Queries (BLAKE3 keys) |
| **Filtering** | Metadata/payload filters (atom/string ops: eq/neq/gt/lt/gte/lte/in/not_in/contains/starts_with/exists; nested fields) |
| **Indexing** | IVF (K-Means++), Parallel IVF Search (top-N clusters), HNSW (top-k, ef_search), Flat/Brute-Force Search, PCA (fit + transform + variance) |
| **Compression** | Int8 scalar quantization (AVX2 accelerated) |
| **Storage** | BLAKE3-protected snapshots per collection, optional compression, atomic writes with global locks, backup rotation (current/previous) |
| **Replication/Sync** | Oplog (ETS/DETS), delta fetch/apply, snapshot export/import |
| **Performance** | AVX2/FMA SIMD kernels, columnar storage, AXPY batch ops, indexed GEMV for IVF |
| **Analytics** | Column statistics, memory estimation, telemetry aggregation, index statistics, text analytics |
| **API** | HTTP REST endpoints, batch operations, progress tracking |
| **Reliability** | Optimistic locking, crash recovery via snapshots; WAL/segment StorageEngine (experimental) |

### Remaining Gaps (Not Implemented or Partial)

- Product quantization (PQ) for higher compression/recall tradeoffs.
- Multi-node distribution (consensus/sharding/query routing); replication oplog exists but no cluster orchestration.
- Tree-level incremental backup/restore (WAL exists for `StorageEngine`, not wired to `Persistence` snapshots).
- Custom embedding hooks (only `TextEmbedding` built-in today).
- GPU acceleration (CUDA/ROCm/Metal/Vulkan).

---

## Completed Priorities (2026-01-04)

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
- **Predicates:** Support for atom/string ops: `eq/neq/gt/lt/gte/lte/in/not_in/contains/starts_with/exists`, plus dot-notation nested fields.
- **Integration:** Filters applied during KNN and Range searches against merged metadata + payloads (PayloadStore + Tree metadata).
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

### Priority 3: HNSW Index - IMPLEMENTED

**Status:** FULLY IMPLEMENTED (2026-01-04)

**Implementation:**
- **Hierarchical Graph:** Multi-layer navigable small world graph implemented in C.
- **Native Resource:** Index managed as an Erlang NIF resource for efficiency and safety.
- **Search:** Greedy top-layer traversal + best-first search at layer 0 with `ef_search` and top-k results.
- **Build:** `ef_construction` search used during insert to select neighbors per layer.
- **Integration:** Automated building via `Tree.build_hnsw/2`. Seamlessly integrated into `Query.execute/2`.

**Usage:**
```elixir
# Build HNSW index
tree = MerkleDb.Tree.build_hnsw(tree, m: 16, ef_construction: 64)

# Search automatically uses HNSW
results = MerkleDb.Query.execute(tree, [:knn, query_vec, 10, 0.0])
```

**Effort:** High (1-2 weeks) - COMPLETED

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

### Priority 5: Sparse Vector Support - IMPLEMENTED

**Status:** FULLY IMPLEMENTED (2026-01-04)

**Implementation:**
- **SparseVector Format:** Dedicated struct storing indices (int32) and values (float64) as binaries.
- **Dedicated Storage:** `sparse_vectors` map in `Tree` struct for secondary sparse representations.
- **Native Kernel:** Highly optimized `fp_sparse_dotp` NIF for fast index-intersection based dot products.
- **Hybrid Scoring:** Combined dense + sparse ranking with configurable `alpha` weight.

**Usage:**
```elixir
# Add sparse representation to existing vector
tree = MerkleDb.Tree.insert_sparse(tree, "V1", [{0, 1.0}, {15, 0.5}], 1000)

# Hybrid search
results = MerkleDb.Query.execute(tree, [:hybrid, dense_vec, sparse_query, 10, 0.5, [alpha: 0.7]])
```

**Effort:** Medium (3-4 days) - COMPLETED

---

## Secondary Gaps

### Multi-Node Distribution

**Current State:** Single-node only. Replication provides oplog + snapshot/delta sync, but no consensus, sharding, or distributed routing.

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

### Namespaces/Collections - IMPLEMENTED

**Status:** FULLY IMPLEMENTED (2026-01-04)

**Implementation:**
- **Architecture:** `KV` GenServer manages a map of trees `%{"collection_name" => %Tree{}}`.
- **API:** All KV operations accept an optional `collection` argument (defaulting to "default").
- **Persistence:** Each collection is persisted to a separate snapshot file (`snapshot-<name>-current.bin`).
- **Discovery:** Automatic discovery and loading of existing collections on startup.

**Usage:**
```elixir
# Create separate collections
MerkleDb.KV.create_collection("users")
MerkleDb.KV.create_collection("products")

# Insert into specific collection
MerkleDb.KV.put("users", "alice", user_vec)
MerkleDb.KV.put("products", "laptop", prod_vec)

# Search specific collection
tree = MerkleDb.KV.snapshot("users")
results = MerkleDb.Query.execute(tree, [:knn, query_vec, 10, 0.0])
```

**Effort:** Low-Medium (2-3 days) - COMPLETED

---

## Secondary Gaps

| Feature | MerkleDB | Pinecone | Qdrant | Milvus | Weaviate |
|---------|----------|----------|--------|--------|----------|
| KNN Search | Yes | Yes | Yes | Yes | Yes |
| IVF Index | Yes | Yes | No | Yes | No |
| HNSW Index | Yes | Yes | Yes | Yes | Yes |
| Delete/Update | Yes | Yes | Yes | Yes | Yes |
| Metadata Filter | Yes | Yes | Yes | Yes | Yes |
| Sparse Vectors | Yes | Yes | Yes | Yes | No |
| Quantization | Yes (Int8) | Yes | Yes | Yes | Yes |
| Multi-Node | **Partial** | Yes | Yes | Yes | Yes |
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
5. ~~HNSW index implementation~~ - DONE (2026-01-04)
6. ~~Namespaces/collections~~ - DONE (2026-01-04)

### Phase 3: Advanced Features (3-4 weeks)
7. ~~Sparse vector support~~ - DONE (2026-01-04)
8. Product quantization (PQ) for higher compression
9. Tree-level incremental backups (WAL integration with Persistence snapshots)
10. Embedding extension hooks + examples (beyond TextEmbedding)
11. Hybrid search (dense + sparse) - DONE (2026-01-04)

### Phase 4: Scale (4+ weeks)
12. Multi-node replication (consensus + membership, build on oplog)
13. Sharding strategy + distributed query routing
14. GPU acceleration (CUDA/ROCm/Metal/Vulkan)

---

## Architecture Strengths to Preserve

1. **Columnar Storage**: Each dimension stored separately enables AXPY batch operations and cache-friendly access patterns.

2. **SIMD Kernels**: 600+ hand-optimized AVX2/FMA functions provide exceptional single-node performance.

3. **Optimistic Locking**: Generation-based versioning prevents stale index updates without heavy locking.

4. **Immutable Core**: Pure functional approach in Tree operations simplifies reasoning and crash recovery.

5. **Elixir/OTP**: Supervision trees, fault tolerance, and hot code reloading built-in.

---

## Conclusion

MerkleDB has a solid foundation with exceptional SIMD performance and clean architecture. The former critical gaps (delete, filtering, HNSW, quantization) are now implemented, giving strong single-node parity. Remaining work is product quantization, multi-node distribution/sharding, GPU acceleration, and tree-level incremental backup and embedding extensibility.

The columnar storage design and extensive native kernel library are significant differentiators that should be preserved and leveraged as new features are added.
