# MerkleDB Performance Benchmark

## Configuration
- **Vectors**: 20000
- **Dimensions**: 64
- **Collection**: `stress_test`
- **Environment**: `win32`
- **Date**: 2026-01-05 20:17:35.359000Z

## Results

### 1. Ingestion
- **Batch Insert**: 1143.91ms
- **Throughput**: **17483.89 vectors/sec**

### 2. Indexing (HNSW)
- **Build Time**: 666.21ms
- **Parameters**: M=16, ef_construction=64

### 3. Query Latency (Dense)
- **Total Time**: 7063.65ms (1000 queries)
- **QPS**: **141.57**
- **Avg Latency**: 7.064ms

### 4. Metadata Filtering
- **Scenario**: KNN k=10 + `category == 'electronics'`
- **Time**: 116111.97ms (500 queries)

### 5. Concurrency Stress
- **Workload**: 1000 writes + 1000 reads in parallel
- **Total Time**: 10455.65ms
- **Stability**: ✅ Passed (No crashes/deadlocks)

### 6. Int8 Quantization
- **Quantization Time**: 100.97ms
- **Quantized QPS**: **98.78**
- **Speedup**: 0.7x vs Float64

### 7. Hybrid Search (Dense + Sparse)
- **Time**: 2865.36ms (100 queries)
- **Status**: ✅ Verified

## Final Database Statistics
```elixir
%{
  count: 21000,
  memory_mb: 18.15,
  cluster_count: 0,
  dimensions: 64,
  has_ivf_index: false,
  tombstones: 1000,
  quantized: true,
  active_count: 20000,
  has_hnsw_index: true,
  metadata_entries: 21000,
  sparse_vector_count: 100
}
```
