# MerkleDb: High-Performance AVX2-Accelerated Vector Database

[![Performance: AVX2](https://img.shields.io/badge/Performance-AVX2%20%2F%20SIMD-orange)](https://github.com/TACITVS/Merkle_DB)
[![Bridge: Zero--Copy NIF](https://img.shields.io/badge/Bridge-Zero--Copy%20NIF-blue)](https://github.com/TACITVS/Merkle_DB)
[![Platform: Elixir](https://img.shields.io/badge/Platform-Elixir%20%2F%20Erlang-purple)](https://elixir-lang.org/)
[![Consensus: Raft](https://img.shields.io/badge/Consensus-Raft-red)](https://github.com/rabbitmq/ra)

**MerkleDb** is a cutting-edge, fault-tolerant vector database engineered for extreme performance and absolute data consistency. It seamlessly bridges the high-level concurrency and distribution of **Elixir/OTP** with the raw power of **x86-64 Assembly (AVX2)**.

Designed for production AI workloads, MerkleDb is a distributed analytics platform capable of processing millions of vectors with microsecond latency, guaranteed by the **Raft consensus algorithm**.

---

## 🚀 Key Features

### 1. Hardcore SIMD Performance
At the core of MerkleDb is a library written in **64-bit Assembly**, utilizing **AVX2 (Advanced Vector Extensions)** to perform 256-bit SIMD operations. 
- **Microsecond Latency**: Brute-force and indexed searches optimized at the transistor level.
- **24x Speedup**: Massive performance gains over native implementations for vector reductions.

### 2. Fault-Tolerant & Distributed (Raft)
MerkleDb is built for high availability. By integrating the **Raft consensus algorithm**, we ensure that your data is replicated and consistent across a cluster of nodes.
- **Strong Consistency**: Every write is committed to a quorum before being acknowledged.
- **Automatic Leader Election**: If a node fails, the cluster automatically elects a new leader in milliseconds.
- **Self-Healing**: Nodes automatically catch up with the log after downtime.

### 3. Zero-Copy Architecture
Our **V7 Bridge Generator** eliminates NIF performance pitfalls:
- **Direct Binary Access**: Elixir binaries are mapped directly to C/Assembly pointers.
- **Zero Memory Bloat**: Output buffers are pre-allocated, allowing the hardware to write results directly into BEAM-managed memory.

### 4. Advanced Semantic Intelligence
- **Sliding Window Ingestion**: Context-aware chunking for long-form text.
- **Hybrid Search**: Combine vector similarity with metadata filtering.
- **Hierarchical Summarization**: Aggregate passage vectors into chapter and book-level "Topic Vectors."

---

## 📚 Developer Course: From Zero to Distributed Cluster

We believe in making powerful tools accessible. Follow this course to master MerkleDb.

### [Module 1: Getting Started](docs/tutorial/module1_setup.md)
*Clone the repo, install dependencies, and run your first local instance.*

### [Module 2: Core Vector Operations](docs/tutorial/module2_basics.md)
*Learn the KV API: Creating collections, inserting vectors, and performing KNN searches.*

### [Module 3: Going Distributed with Raft](docs/tutorial/module3_clustering.md)
*Spin up a 3-node cluster and witness fault tolerance in action.*

### [Module 4: Advanced Search & Analytics](docs/tutorial/module4_advanced.md)
*IVF Indexing, Quantization, and Semantic Search over whole books.*

---

## 🛠️ Installation

### Prerequisites
- **Elixir ~> 1.14**
- **GCC (MinGW64 on Windows)**
- **NASM (Netwide Assembler)**
- **Make**

### Build
```powershell
git clone https://github.com/TACITVS/Merkle_DB.git
cd Merkle_DB
mix deps.get
mix run gen_bridge.exs
mix compile
```

---

## 📖 API at a Glance

```elixir
alias MerkleDb.KV

# Create a strongly consistent collection
KV.create_collection("products", dim: 128, precision: :f32)

# Insert data via Raft Quorum
KV.put("products", "id_1", vector_data, %{category: "electronics"})

# Perform AVX2-accelerated search
results = MerkleDb.Query.execute(KV.snapshot("products"), [:knn, query_vec, 10, 0.8])
```

---

## ⚖️ Design Philosophy

MerkleDb adopts a **Hybrid Strategy**:
- **Elixir Control Plane**: Manages distribution, consensus (Raft), and fault recovery.
- **C/ASM Data Plane**: Handles the "hot" loops—the expensive vector math—using hardware-level optimizations.

Developed by **TACITVS**.