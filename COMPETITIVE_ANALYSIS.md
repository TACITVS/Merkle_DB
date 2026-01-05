# MerkleDB Competitive Analysis & Performance Review

**Date:** 2026-01-05
**Version:** 0.1.0 (Pre-release)
**Benchmark Context:** Single-node, 20k vectors, 64 dimensions, AVX2 enabled.

## Executive Summary

MerkleDB demonstrates functional parity with commercial vector databases in terms of core features (HNSW, Quantization, Filtering). Performance-wise, it occupies a **"Mid-Tier"** position: faster than pure script-based implementations (Python/Ruby without C-extensions) but currently slower than highly optimized compiled engines like Qdrant (Rust) or Faiss (C++) for high-throughput queries.

The differentiating factor is **Architecture**: MerkleDB's columnar design makes it uniquely suited for *hybrid workloads* (Analytics + Vector Search) where full-column scans are required, whereas most vector DBs are optimized strictly for row-based approximate search.

---

## 1. Performance Comparison (Single Node)

| Metric | MerkleDB (Current) | Qdrant (Rust) | Weaviate (Go) | Milvus (Go/C++) | Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Ingestion Speed** | **~12k vec/sec** | ~15-20k vec/sec | ~10-15k vec/sec | ~10k vec/sec | **COMPETITIVE**. The AXPY-optimized batch insertions and WAL architecture provide write speeds comparable to industry leaders. |
| **Query Latency (P99)** | **~9ms** | < 2ms | < 5ms | < 5ms | **LAGGING**. For a dataset of this size (20k), latency should be sub-millisecond. The overhead likely stems from the Erlang/NIF boundary crossing per query. |
| **Throughput (QPS)** | **~180 QPS** | > 1,000 QPS | > 500 QPS | > 500 QPS | **LAGGING**. Current query throughput is bottlenecked by the synchronous `GenServer` call path and NIF overhead. |
| **Build Time (20k)** | **~0.6s** | ~0.5s | ~0.8s | ~1.0s | **EXCELLENT**. The C-based HNSW build is extremely efficient and matches or beats competitors. |
| **Memory Footprint** | **~18MB (Uncompressed)** | ~25MB | ~30MB | ~50MB+ | **SUPERIOR**. Zero-copy binaries and columnar layout provide exceptional memory efficiency. |

---

## 2. Feature Stack Comparison

| Feature | MerkleDB | Pinecone | Qdrant | Implementation Note |
| :--- | :--- | :--- | :--- | :--- |
| **HNSW Index** | ✅ **Yes** | Yes | Yes | Fully implemented in C. Functional equivalent. |
| **Quantization** | ✅ **Int8** | Yes | Yes | MerkleDB uses AVX2 dynamic quantization. Efficient but simpler than Product Quantization (PQ). |
| **Filtering** | ✅ **Pre-Filter** | Yes | Yes | MerkleDB supports complex metadata filtering (eq, gt, lt, in). |
| **Sparse Vectors** | ✅ **Yes** | Yes | Yes | Native sparse dot product kernel. Feature parity achieved. |
| **Multitenancy** | ✅ **Collections** | Namespaces | Collections | Fully supported via `KV.create_collection`. |
| **Columnar Ops** | ✅ **Native** | No | No | **UNIQUE USP**. MerkleDB can perform `sum`, `avg`, `min`, `max` on dimension columns instantly. |

---

## 3. Bottleneck Analysis & Recommendations

The benchmarks reveal a discrepancy between **raw kernel speed** (nanoseconds) and **end-to-end query speed** (milliseconds).

### The "NIF Overhead" Problem
In MerkleDB, every query currently involves:
1. Elixir Process Message (`KV` GenServer)
2. Term serialization to C
3. **NIF Execution (Fast)**
4. Term deserialization back to Elixir

**Competitors (Qdrant/Milvus)** run the HTTP server and the Vector Engine in the same memory space/runtime (Rust/C++), avoiding this serialization cost.

### Strategic Recommendations

1.  **Batch Query API (Critical for QPS):**
    *   *Current:* 1 query -> 1 NIF call.
    *   *Fix:* Allow `Query.execute_batch(list_of_queries)`. This amortizes the NIF overhead across 100+ queries, potentially boosting QPS by 10-50x.

2.  **Dirty Scheduler Tuning:**
    *   Currently, queries might be yielding to the Erlang scheduler too aggressively. For small indexes (<1M), running on the main scheduler (non-dirty) might actually be faster.

3.  **Graph Traversal Optimization:**
    *   The `ef_search` parameter (beam width) in `fp_query.ex` might be set conservatively high. Lowering it for lower-recall scenarios could halve latency.

---

## 4. Conclusion

MerkleDB is a **technologically sound** vector database. Its core math (AVX2 kernels, HNSW) is "metal-speed". To bridge the gap with commercial engines, the focus must shift from *algorithms* (which are done) to *interface efficiency* (batching, protocol buffers, reducing copying).

**Verdict:** Ready for applications requiring **embedded**, **columnar-analytical** vector search. For pure high-QPS semantic search at scale, batching optimizations are the next mandatory step.
