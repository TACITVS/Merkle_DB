# MerkleDB API Specification

Version: 1.0.0
Date: 2026-01-01

## Overview

This document defines the public API contract for MerkleDB.

## Core API

### Collection Management

#### create_collection/2

Create a new vector collection. Replicated via Raft.

```elixir
@spec create_collection(name, opts) :: :ok | {:error, reason}

# Parameters:
#   name   - string, unique collection identifier
#   opts   - keyword list of options
#
# Options:
#   :dim         - integer, vector dimension (e.g., 384, 768)
#   :precision   - :f32 | :f64 (default: :f64) - f32 halves memory for embeddings
#   :quantized   - boolean, enable scalar quantization for search
#
# Returns:
#   :ok on success
#   {:error, :already_exists} if name taken
```

#### drop_collection/1

Delete a collection and all its data. Replicated via Raft.

```elixir
@spec drop_collection(name) :: :ok | {:error, reason}
```

#### list_collections/0

List all collection names.

```elixir
@spec list_collections() :: [String.t()]
```

### Data Operations

#### put/4

Insert or update a record. Replicated via Raft for Strong Consistency.

```elixir
@spec put(collection, id, vector, metadata) :: :ok | {:error, reason}

# Parameters:
#   collection - string, collection name
#   id         - string, record identifier
#   vector     - binary or list, must match collection dimension
#   metadata   - map, arbitrary metadata
#
# Returns:
#   :ok
#   {:error, :collection_not_found}
```

#### put_batch/2

Batch insert/update multiple records.

```elixir
@spec put_batch(collection, records) :: :ok | {:error, reason}
```

#### delete/2

Delete a record by ID.

```elixir
@spec delete(collection, id) :: :ok | {:error, reason}

# Deletes are soft until compaction.
# The record will be excluded from future snapshots.
```

#### get/3

Get a record by ID, optionally from a specific snapshot.

```elixir
@spec get(collection, id, opts) :: {:ok, record} | {:error, reason}

# Options:
#   :snapshot_root - binary (32 bytes), query specific snapshot
#
# Returns:
#   {:ok, %{id: id, vector: [...], payload: %{}, version: v}}
#   {:error, :not_found} if record doesn't exist
#   {:error, :snapshot_not_found} if snapshot_root invalid
```

### Query Operations

#### query/4

Search for similar vectors or concepts.

```elixir
@spec query(collection, query_input, top_k, opts) :: {:ok, results} | {:error, reason}

# Parameters:
#   collection  - string, collection name
#   query_input - list of floats (vector) OR string (semantic search text)
#   top_k       - integer, number of results (1-1000)
#   opts        - keyword list
```

### Snapshot Operations

#### commit/1

Create an immutable snapshot and return its root hash.

```elixir
@spec commit(collection) :: {:ok, snapshot_root} | {:error, reason}

# Returns:
#   {:ok, <<snapshot_root::binary-32>>}
#
# The snapshot_root is the cryptographic commitment to:
#   - All record data (vectors + payloads)
#   - Current index state
#   - Collection schema
#
# After commit, the snapshot is immutable and can be:
#   - Used for queries via :snapshot_root option
#   - Used to generate inclusion proofs
#   - Exported for replication
```

#### snapshot_list/1

List all snapshots for a collection.

```elixir
@spec snapshot_list(collection) :: {:ok, [snapshot_info]} | {:error, reason}

# Returns:
#   {:ok, [%{
#     root: <<binary-32>>,
#     timestamp: DateTime.t(),
#     record_count: integer,
#     index_type: atom
#   }]}
```

#### gc_snapshots/2

Garbage collect old snapshots.

```elixir
@spec gc_snapshots(collection, policy) :: {:ok, deleted_count} | {:error, reason}

# Policy options:
#   {:keep_last, n}          - keep last n snapshots
#   {:older_than, duration}  - delete snapshots older than duration
#   {:keep_roots, [roots]}   - delete all except specified roots
```

### Proof Operations

#### prove_inclusion/3

Generate an inclusion proof for a record in a snapshot.

```elixir
@spec prove_inclusion(collection, id, snapshot_root) :: {:ok, proof} | {:error, reason}

# Returns:
#   {:ok, %Proof{
#     snapshot_root: <<binary-32>>,
#     record_id: id,
#     record_hash: <<binary-32>>,
#     path: [{:left | :right, <<sibling_hash::binary-32>>}, ...]
#   }}
#
#   {:error, :not_found} if record not in snapshot
#   {:error, :snapshot_not_found} if snapshot doesn't exist
```

#### verify_inclusion/3

Verify an inclusion proof (client-side, no server needed).

```elixir
@spec verify_inclusion(snapshot_root, record, proof) :: boolean

# Parameters:
#   snapshot_root - binary (32 bytes), expected root
#   record        - %{id: id, vector: [...], payload: %{}, version: v}
#   proof         - %Proof{} from prove_inclusion
#
# Returns:
#   true if proof is valid
#   false if proof is invalid or record doesn't match
#
# This function is designed to run client-side with no
# database access. It only needs the BLAKE3 hash function.
```

### Export/Import Operations

#### export_snapshot/2

Export a snapshot as a portable bundle.

```elixir
@spec export_snapshot(collection, snapshot_root) :: {:ok, path} | {:error, reason}

# Creates a self-contained bundle containing:
#   - Manifest with snapshot_root
#   - All segments needed to reconstruct data
#   - Index files
#   - Checksums for integrity
#
# Returns path to the bundle file.
```

#### import_snapshot/2

Import a snapshot bundle into a collection.

```elixir
@spec import_snapshot(collection, bundle_path) :: {:ok, snapshot_root} | {:error, reason}

# Verifies integrity and imports the snapshot.
# Returns the snapshot_root of the imported data.
```

## Filtering

Filters are expressed as nested maps/tuples:

```elixir
# Equality
{:eq, "field", value}

# Comparison
{:gt, "field", value}
{:gte, "field", value}
{:lt, "field", value}
{:lte, "field", value}

# Set membership
{:in, "field", [value1, value2]}

# String contains
{:contains, "field", "substring"}

# Boolean combinations
{:and, [filter1, filter2, ...]}
{:or, [filter1, filter2, ...]}
{:not, filter}
```

Example:
```elixir
MerkleDB.query("docs", query_vec, 10,
  filter: {:and, [
    {:eq, "category", "news"},
    {:gte, "score", 0.8}
  ]}
)
```

## Error Codes

| Code | Description |
|------|-------------|
| :not_found | Collection or record not found |
| :already_exists | Collection name already taken |
| :dimension_mismatch | Vector dimension doesn't match schema |
| :invalid_vector | Vector contains NaN or Infinity |
| :payload_too_large | Payload exceeds size limit |
| :snapshot_not_found | Snapshot root doesn't exist |
| :proof_invalid | Proof verification failed |
| :index_building | Index is currently being built |
| :collection_locked | Collection is locked for maintenance |

## Telemetry Events

The following telemetry events are emitted:

```elixir
[:merkle_db, :query, :start]
[:merkle_db, :query, :stop]
[:merkle_db, :upsert, :start]
[:merkle_db, :upsert, :stop]
[:merkle_db, :commit, :start]
[:merkle_db, :commit, :stop]
[:merkle_db, :prove, :start]
[:merkle_db, :prove, :stop]
```
