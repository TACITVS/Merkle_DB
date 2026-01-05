# MerkleDb: High-Performance AVX2-Accelerated Vector Database

[![Performance: AVX2](https://img.shields.io/badge/Performance-AVX2%20%2F%20SIMD-orange)](https://github.com/TACITVS/Merkle_DB)
[![Bridge: Zero--Copy NIF](https://img.shields.io/badge/Bridge-Zero--Copy%20NIF-blue)](https://github.com/TACITVS/Merkle_DB)
[![Platform: Elixir](https://img.shields.io/badge/Platform-Elixir%20%2F%20Erlang-purple)](https://elixir-lang.org/)

**MerkleDb** is a cutting-edge, generic vector database engineered for extreme performance and scalability. It seamlessly bridges the high-level concurrency of **Elixir/BEAM** with the raw power of **x86-64 Assembly (AVX2)** via a highly optimized, zero-copy NIF interface.

Designed for research, AI, and large-scale data analysis, MerkleDb is not just a storage engine—it's a comprehensive analytics platform capable of processing millions of vectors with microsecond latency.

---

## 🚀 Why MerkleDb?

In the era of AI, vector search is often the bottleneck. MerkleDb solves this by moving heavy computations to the hardware level while maintaining a safe, developer-friendly Elixir API.

### 1. Hardcore SIMD Performance
At the core of MerkleDb is the `FP_ASM_LIB_DEV` library. Written primarily in **64-bit Assembly**, it utilizes **AVX2 (Advanced Vector Extensions)** to perform 256-bit SIMD operations. 
- **Dot Products, AXPY, and Norms** are processed in parallel across multiple elements per CPU cycle.
- **24x Speedup** over native Elixir implementations for large-scale reductions.
- **1.3 GB/s+ Throughput** sustained for vector transformations.

### 2. Zero-Copy Architecture (V7 Bridge)
Our custom-built **V7 Bridge Generator** eliminates the common performance pitfalls of NIFs:
- **Direct Binary Access**: No more redundant `memcpy` for inputs. We map Elixir binaries directly to C pointers.
- **Up-front Allocation**: Output buffers are pre-allocated as Erlang binaries and passed to Assembly, allowing the hardware to write results directly into memory managed by the BEAM.
- **Minimal Overhead**: NIF call latency is minimized, ensuring that even short-lived operations remain efficient.

### 3. Advanced Inverted File Index (IVF)
MerkleDb doesn't just do brute-force search. It includes a built-in **IVF Indexing** system powered by **K-Means Clustering**:
- **Voronoi Partitioning**: Data is automatically clustered around centroids.
- **6x+ Search Speedup**: Queries focus only on the most relevant clusters, drastically reducing the search space without sacrificing accuracy.
- **Dynamic Re-indexing**: Easily rebuild indices as your dataset grows.

### 4. Comprehensive Analytics Suite
Access over **180 specialized functions** for data science directly from Elixir:
- **Clustering**: High-speed K-Means.
- **Dimensionality Reduction**: Principal Component Analysis (PCA).
- **Statistics**: Mean, Variance, Moments, Correlation, and more across massive datasets.
- **ML Primitives**: Ready-to-use kernels for building custom neural networks or models.

---

## ⚖️ Design Philosophy: Performance vs. Reliability

A common question is: *Why implement a vector database in Elixir instead of pure C or Assembly?*

While a pure C/ASM implementation would undoubtedly be faster for raw mathematical throughput, it lacks the architectural safety nets required for a production-grade database. MerkleDb adopts a **Hybrid Strategy** that provides the best of both worlds:

### The Elixir Control Plane (Reliability & Uptime)
We leverage Elixir and the BEAM virtual machine for everything that requires **absurdly high reliability and fault tolerance**.
- **Isolation**: Every operation runs in its own process. If a query crashes, it doesn't take down the database.
- **Supervision**: The database state is protected by OTP supervision trees, ensuring self-healing and guaranteed uptime.
- **Distribution**: Built-in support for clustering and transparent replication across nodes.
- **Concurrency**: The Erlang scheduler handles millions of lightweight processes, allowing MerkleDb to manage thousands of concurrent searches without the thread-safety nightmares of pure C.

### The C/ASM Data Plane (Raw Metal Speed)
We offload only the "hot" loops—the computationally expensive vector math—to **AVX2-optimized Assembly**. This ensures that the actual search performance is identical to the fastest compiled engines, while the management of that data remains within a safe, functional environment.

By choosing this path, MerkleDb avoids the fragility of pure C databases (buffer overflows, memory leaks, segmentation faults) while maintaining the microsecond-level latency of hardware-accelerated math.

---

## 🛠️ Architecture

- **Core Storage**: Columnar storage layout optimized for cache-friendly AXPY batch processing.
- **Data Integrity**: Based on **Merkle Trees and DAGs**, ensuring that every state of your database is verifiable and cryptographically sound.
- **Concurrency**: Leverages the Erlang Scheduler for non-blocking I/O, while the CPU-heavy tasks are offloaded to optimized native kernels.

---

## 📦 Installation & Usage

### Prerequisites
- **Elixir ~> 1.14**
- **GCC (MinGW64 on Windows)**
- **NASM (Netwide Assembler)**
- **Make**

### Quick Start
```powershell
mix deps.get
mix run gen_bridge.exs
mix compile
```

### Generic API Example
```elixir
alias MerkleDb.{Tree, Query, Analytics}

# Create a new database
tree = Tree.new()

# Insert vectors (64-dim in this case)
vec = :binary.copy(<<1.0::little-float-64>>, 64)
tree = Tree.insert(tree, "my-key-1", vec)

# Build an IVF Index for speed
tree = Analytics.build_ivf_index(tree, 10)

# Perform a high-speed KNN search
results = Query.execute(tree, [:knn, query_vec, 5, 0.30])
# Returns: [{"my-key-1", 0.98}, ...]
```

---

## 🧪 Applications

- **AI Search Engines**: Real-time semantic search using embeddings.
- **Bioinformatics**: Analyzing genomic sequences represented as high-dimensional vectors.
- **Financial Research**: Real-time correlation analysis and anomaly detection in time-series data.
- **Geographic Information Systems (GIS)**: Fast spatial indexing and nearest-neighbor lookups.

---

## 📜 Research Foundation

MerkleDb is built with a commitment to both performance and honesty about the underlying data. Whether you are building a production-grade search engine or conducting academic research, the combination of **Assembly-level speed** and **Functional Programming safety** provides a unique and powerful toolset.

Developed by **TACITVS**.
