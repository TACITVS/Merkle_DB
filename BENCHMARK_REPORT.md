# MerkleDB Performance Benchmark

## Configuration
- **Vectors**: 20000
- **Dimensions**: 64
- **Collection**: `stress_test`
- **Environment**: `win32`
- **Date**: 2026-01-05 18:32:38.068000Z

## Results

### 1. Ingestion
- **Batch Insert**: 1678.23ms
- **Throughput**: **11917.3 vectors/sec**

### 2. Indexing (HNSW)
- **Build Time**: 1340.42ms
- **Parameters**: M=16, ef_construction=64

### 3. Query Latency (Dense)
- **Total Time**: 9690.52ms (1000 queries)
- **QPS**: **103.19**
- **Avg Latency**: 9.691ms

### 4. Metadata Filtering
- **Scenario**: KNN k=10 + `category == 'electronics'`
- **Time**: 106059.26ms (500 queries)

### 5. Concurrency Stress
- **Workload**: 1000 writes + 1000 reads in parallel
- **Total Time**: 11856.08ms
- **Stability**: ✅ Passed (No crashes/deadlocks)

### 6. Int8 Quantization
- **Quantization Time**: 166.4ms
- **Quantized QPS**: **89.73**
- **Speedup**: 0.87x vs Float64

### 7. Hybrid Search (Dense + Sparse)
- **Time**: 2698.65ms (100 queries)
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
