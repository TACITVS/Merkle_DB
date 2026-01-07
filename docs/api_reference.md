# MerkleDb API Reference

This document provides a comprehensive reference for the MerkleDb interfaces.

## 1. Elixir API (`MerkleDb.KV`)
The primary interface for collection and data management within the BEAM VM. All state-changing operations are replicated via Raft.

### Collection Management
- `create_collection(name, opts \\ [])`: Creates a new collection. 
  - `opts`: `[:dim, :precision (:f32 | :f64)]`.
- `drop_collection(name)`: Deletes a collection and its data.
- `list_collections()`: Returns a list of all collection names.
- `reset(collection \\ "default")`: Clears all vectors in a collection.

### Data Operations
- `put(collection \\ "default", key, vector, metadata \\ %{})`: Inserts or updates a vector.
- `put_batch(collection \\ "default", key_vector_pairs)`: High-performance batch insertion.
  - `key_vector_pairs`: List of `{key, vector}` or `{key, vector, metadata}`.
- `delete(collection \\ "default", key)`: Marks a key as deleted.

### State & Persistence
- `snapshot(collection \\ "default")`: Returns the current immutable `%MerkleDb.Tree{}` for the collection (Linearizable Read).
- `generation(collection \\ "default")`: Returns the current version number of the collection state.
- `checkpoint(collection \\ "default")`: Forces a binary checkpoint to disk for faster recovery.

---

## 2. HTTP API (Port 4000)
Secure REST API for external clients. Requires `Authorization: Bearer <token>`.

### Collections
- **GET** `/v1/collections`: List all collections.
- **POST** `/v1/collections/:name`: Create a collection.
  - Body: `{"dim": 128, "precision": "f32"}`
- **DELETE** `/v1/collections/:name`: Drop a collection.

### Data
- **POST** `/v1/:collection/vectors`: Insert or update vectors.
  - Body: `[{"id": "k1", "vector": [...], "metadata": {...}}, ...]`
  - Note: Supports `text` field instead of `vector` for semantic embedding.

### Search
- **POST** `/v1/:collection/search`: Execute a search.
  - Body: `{"vector": [...], "k": 10, "threshold": 0.0}`
  - Or: `{"text": "query string", ...}` for semantic search.

---

## 3. Python SDK
A Python wrapper around the HTTP API.

```python
from sdk.python.merkledb import MerkleDb
db = MerkleDb("http://localhost:4000")
```

### Methods
- `db.create_collection(name, dim, precision="f32")`
- `db.list_collections()`
- `db.drop_collection(name)`
- `db.insert(collection, items)`:
  - `items`: List of dicts `{'id': str, 'vector': list/numpy, 'metadata': dict}`
- `db.search(collection, vector=None, text=None, k=10, threshold=0.0)`
- `db.checkpoint(collection)`

---

## 4. Query & Analytics (Elixir)

### `MerkleDb.Query`
- `execute(tree, query)`: Executes a search query.
  - **KNN**: `[:knn, query_vector, k, threshold]`
  - **Range**: `[:range, query_vector, min_sim, max_sim]`
  - **Semantic**: `[:semantic, "search text", k, threshold]`
  - **Hybrid**: Add `{:where, filters}` to any of the above.
- `execute_batch(tree, query_vectors, k, threshold)`: Performs multiple KNN searches in parallel.

### `MerkleDb.Analytics`
- `build_ivf_index(tree, k)`: Builds an Inverted File Index using AVX2 K-Means.
- `reduce_dimensions(tree, n_components)`: PCA for dimensionality reduction.
- `column_stats(tree, dim_idx)`: Calculates statistical properties of a dimension.
