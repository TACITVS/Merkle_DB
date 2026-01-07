# Module 4: Advanced Search & Analytics

In this final module, we will explore how MerkleDb handles large-scale datasets using IVF indexing and its unique semantic search capabilities.

## 1. IVF Indexing (Inverted File Index)

When you have millions of vectors, brute-force search becomes slow. MerkleDb uses IVF indexing to partition the vector space.

### Building an Index
```elixir
# Build an IVF index with 100 clusters
MerkleDb.IndexBuilder.start_build(100, collection: "my_large_collection")
```

### How it works:
1. MerkleDb runs a high-speed **AVX2-accelerated K-Means** clustering.
2. Vectors are assigned to the nearest centroid.
3. Search only checks the top-K clusters, providing a **6x+ speedup**.

---

## 2. Quantization

MerkleDb supports **Product Quantization (PQ)** and **Scalar Quantization** to reduce the memory footprint of vectors.

```elixir
# Create a quantized collection
# This converts f64 vectors to int8 internally for search
KV.create_collection("quantized_data", dim: 128, quantized: true)
```

---

## 3. Semantic Book Search

One of MerkleDb's most powerful features is searching over entire books.

### Sliding Window Ingestion
Instead of chunking by fixed size, MerkleDb supports sliding windows to preserve context.

```elixir
# Chunk a text file
chunks = MerkleDb.Ingestor.chunk_file("my_book.txt", 500, 100)

# Embed and insert each chunk
Enum.with_index(chunks) |> Enum.each(fn {chunk, idx} ->
  vec = MerkleDb.TextEmbedding.embed(chunk)
  MerkleDb.KV.put("my_book", "chunk_#{idx}", vec, %{"text" => chunk})
end)
```

### Hierarchical Search
You can search at different levels of granularity:
- **Passage Level**: Find specific quotes.
- **Topic Level**: Find chapters discussing a specific concept.

```elixir
# Search for concepts, not just keywords
Query.execute(KV.snapshot("my_book"), [:semantic, "The concept of suffering in stoicism", 5, 0.7])
```

---

## 4. Hardware-Accelerated Analytics

MerkleDb exposes its internal AVX2 kernels for general-purpose analytics.

```elixir
alias MerkleDb.Analytics

# Calculate column stats (mean, variance, etc.) across 1 million vectors
{:ok, stats} = Analytics.column_stats(KV.snapshot("large_data"), 0)
IO.inspect(stats.mean)
```

---

## 🎉 Completion

You have completed the MerkleDb Developer Course! You are now equipped to build high-performance, fault-tolerant AI applications.

For further reading, check out the [Full API Reference](../spec/api_spec.md).
