# Module 2: Core Vector Operations

In this module, we will learn how to interact with MerkleDb using the `KV` and `Query` modules.

## 1. Creating a Collection

Collections are the primary way to organize data. They define the dimensionality and precision of your vectors.

```elixir
alias MerkleDb.KV

# Create a collection for 384-dimensional vectors (e.g., from a BERT model)
# We use f32 (32-bit floats) to save memory.
KV.create_collection("my_vectors", dim: 384, precision: :f32)
```

---

## 2. Inserting Data

MerkleDb supports single and batch inserts. In a local instance, these are immediately durable via the Write-Ahead Log (WAL).

### Single Insert
```elixir
vector = for _ <- 1..384, do: :rand.uniform()
KV.put("my_vectors", "doc_1", vector, %{"title" => "First Document"})
```

### Batch Insert (High Performance)
```elixir
batch = [
  {"doc_2", vector_a, %{"tags" => ["news"]}},
  {"doc_3", vector_b, %{"tags" => ["tech"]}}
]
KV.put_batch("my_vectors", batch)
```

---

## 3. Performing a Search

MerkleDb uses a snapshot-based query model. This means you query a consistent "view" of the data at a specific point in time.

### Brute-Force KNN
```elixir
alias MerkleDb.Query

# 1. Get a snapshot of the collection
tree = KV.snapshot("my_vectors")

# 2. Execute query: Find 10 nearest neighbors with similarity > 0.8
results = Query.execute(tree, [:knn, query_vector, 10, 0.8])

# Result format: [{"doc_1", 0.95}, {"doc_2", 0.88}, ...]
```

---

## 4. Understanding Snapshots

The `KV.snapshot/1` function returns a `%MerkleDb.Tree{}` structure. This structure is:
- **Immutable**: It won't change even if you insert more data.
- **Efficient**: It shares most of its data with other snapshots.
- **Verifiable**: It is a Merkle Tree, meaning you can prove a specific vector belongs to this snapshot.

In the next module, we will see how these operations change when running in a distributed cluster.
