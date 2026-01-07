# MerkleDb API Reference

This document provides a comprehensive reference for the public Elixir API of MerkleDb.

## `MerkleDb.KV`
The primary interface for collection and data management. All state-changing operations are replicated via Raft.

### Collection Management
- `create_collection(name, opts \ [])`: Creates a new collection. 
  - `opts`: `[:dim, :precision (:f32 | :f64), :quantized (boolean)]`.
- `drop_collection(name)`: Deletes a collection and its data.
- `list_collections()`: Returns a list of all collection names.
- `reset(collection \ "default")`: Clears all vectors in a collection.

### Data Operations
- `put(collection \ "default", key, vector, metadata \ %{})`: Inserts or updates a vector.
- `put_batch(collection \ "default", key_vector_pairs)`: High-performance batch insertion.
  - `key_vector_pairs`: List of `{key, vector}` or `{key, vector, metadata}`.
- `delete(collection \ "default", key)`: Marks a key as deleted.

### State & Persistence
- `snapshot(collection \ "default")`: Returns the current immutable `%MerkleDb.Tree{}` for the collection.
- `generation(collection \ "default")`: Returns the current version number of the collection state.
- `checkpoint(collection \ "default")`: Forces a binary checkpoint to disk for faster recovery.

---

## `MerkleDb.Query`
Used to perform searches against a specific tree snapshot.

### Functions
- `execute(tree, query)`: Executes a search query.
  - **KNN**: `[:knn, query_vector, k, threshold]`
  - **Range**: `[:range, query_vector, min_sim, max_sim]`
  - **Semantic**: `[:semantic, "search text", k, threshold]`
  - **Hybrid**: Add `{:where, filters}` to any of the above.
- `execute_batch(tree, query_vectors, k, threshold)`: Performs multiple KNN searches in parallel.

---

## `MerkleDb.Analytics`
Hardware-accelerated data analysis functions.

### Functions
- `build_ivf_index(tree, k, max_iter \ 100)`: Builds an Inverted File Index using AVX2 K-Means.
- `reduce_dimensions(tree, n_components)`: Performs Principal Component Analysis (PCA) to reduce vector dimensionality.
- `column_stats(tree, dim_idx)`: Calculates `mean`, `min`, `max`, and `variance` for a specific dimension.

---

## `MerkleDb.Raft`
Controls the distributed consensus layer.

### Functions
- `join_cluster(peer_node)`: Connects the current node to an existing cluster.
- `get_state()`: Returns the full internal state of the Raft machine (Strongly Consistent).

---

## `MerkleDb.Tree`
The underlying data structure for collections.

### Fields
- `count`: Number of active vectors.
- `dim`: Dimensionality of the vectors.
- `precision`: `:f32` or `:f64`.
- `root_hash`: The Merkle root of the collection.
