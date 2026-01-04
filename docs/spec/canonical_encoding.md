# MerkleDB Canonical Encoding Specification

Version: 1.0.0
Date: 2026-01-01

## Overview

This document specifies the exact byte-level encoding for all hashable
structures in MerkleDB. Canonical encoding ensures deterministic hashes.

## General Rules

1. **No ambiguous concatenation**: Every field is either fixed-size or
   length-prefixed.

2. **Little-endian**: All multi-byte integers use little-endian byte order.

3. **Domain separation**: Every hash input starts with a 1-byte tag.

4. **No padding**: No alignment padding between fields.

5. **UTF-8 strings**: All strings are UTF-8 encoded, length-prefixed.

## Domain Separation Tags

```
TAG_LEAF       = 0x00  # Leaf node (record)
TAG_INTERNAL   = 0x01  # Internal Merkle node
TAG_INDEX      = 0x02  # Index state
TAG_MANIFEST   = 0x03  # Snapshot manifest
TAG_SEGMENT    = 0x04  # Segment data
TAG_CENTROID   = 0x05  # IVF centroid
TAG_POSTING    = 0x06  # Posting list
TAG_EMPTY      = 0xFF  # Empty/null marker
```

## Record (Leaf) Encoding

A record consists of: id, vector, payload, version.

### Leaf Hash Computation

```
leaf_hash = BLAKE3(
    TAG_LEAF           ||  # 1 byte: 0x00
    id                 ||  # 16 bytes: u128 little-endian
    dim                ||  # 4 bytes: u32 little-endian
    vector_bytes       ||  # dim * 4 bytes: float32[] little-endian
    payload_len        ||  # 4 bytes: u32 little-endian
    payload_bytes      ||  # payload_len bytes: canonical JSON/CBOR
    version                # 8 bytes: u64 little-endian
)
```

### Field Details

| Field | Type | Size | Description |
|-------|------|------|-------------|
| TAG_LEAF | u8 | 1 | Domain tag (0x00) |
| id | u128 | 16 | Record identifier, little-endian |
| dim | u32 | 4 | Vector dimension count |
| vector_bytes | [f32] | dim*4 | IEEE 754 float32, little-endian |
| payload_len | u32 | 4 | Payload byte length |
| payload_bytes | bytes | var | Canonical payload encoding |
| version | u64 | 8 | Record version/timestamp |

### Vector Encoding

```elixir
# Elixir encoding example
def encode_vector(vector) when is_list(vector) do
  vector
  |> Enum.map(fn f -> <<f::little-float-32>> end)
  |> IO.iodata_to_binary()
end
```

**Constraints**:
- No NaN values allowed (reject on insert)
- No Infinity values allowed (reject on insert)
- Dimension must match collection schema

### Payload Encoding

Payload is a key-value map encoded as canonical JSON:
1. Keys sorted lexicographically (UTF-8 byte order)
2. No whitespace
3. Numbers in shortest representation
4. No trailing commas
5. UTF-8 encoding

Example:
```json
{"category":"news","score":0.95,"tags":["ai","ml"]}
```

If payload is empty, payload_len = 0 and payload_bytes is empty.

## Internal Node Encoding

Internal nodes combine two child hashes.

```
internal_hash = BLAKE3(
    TAG_INTERNAL    ||  # 1 byte: 0x01
    left_hash       ||  # 32 bytes
    right_hash          # 32 bytes
)
```

Total input: 65 bytes.

## Tree Construction

### Leaf Ordering

1. Collect all records
2. Sort by id (ascending, lexicographic on u128 bytes)
3. Compute leaf_hash for each record in sorted order

### Building the Tree

```
Level 0 (leaves): [h0, h1, h2, h3, h4, h5, h6, h7]
Level 1:          [H(h0,h1), H(h2,h3), H(h4,h5), H(h6,h7)]
Level 2:          [H(H01,H23), H(H45,H67)]
Level 3 (root):   [H(H0123, H4567)]
```

### Odd Number of Nodes

If a level has an odd number of nodes, the last node is promoted
to the next level without hashing (NOT duplicated):

```
Level 0: [h0, h1, h2]
Level 1: [H(h0,h1), h2]  # h2 promoted
Level 2: [H(H01, h2)]    # root
```

**Rationale**: Duplicating would create ambiguity and waste computation.

## Index State Encoding

The index state captures the current state of all indexes.

```
index_state_hash = BLAKE3(
    TAG_INDEX           ||  # 1 byte: 0x02
    index_type          ||  # 1 byte: 0=none, 1=IVF, 2=HNSW
    num_centroids       ||  # 4 bytes: u32
    centroid_hashes     ||  # num_centroids * 32 bytes
    num_postings        ||  # 4 bytes: u32
    posting_hashes          # num_postings * 32 bytes
)
```

### Centroid Hash

```
centroid_hash = BLAKE3(
    TAG_CENTROID    ||  # 1 byte: 0x05
    centroid_id     ||  # 4 bytes: u32
    dim             ||  # 4 bytes: u32
    centroid_vector     # dim * 4 bytes: float32[]
)
```

### Posting List Hash

```
posting_hash = BLAKE3(
    TAG_POSTING     ||  # 1 byte: 0x06
    cluster_id      ||  # 4 bytes: u32
    num_ids         ||  # 4 bytes: u32
    record_ids          # num_ids * 16 bytes: sorted u128[]
)
```

## Snapshot Manifest Encoding

The manifest binds everything together.

```
manifest_hash = BLAKE3(
    TAG_MANIFEST        ||  # 1 byte: 0x03
    version             ||  # 1 byte: encoding version (0x01)
    tree_root           ||  # 32 bytes: Merkle tree root
    index_state_hash    ||  # 32 bytes: index commitment
    record_count        ||  # 8 bytes: u64
    timestamp           ||  # 8 bytes: u64 Unix micros
    schema_hash             # 32 bytes: collection schema hash
)
```

The manifest_hash IS the snapshot_root returned by commit().

## Schema Hash

```
schema_hash = BLAKE3(
    TAG_SEGMENT         ||  # 1 byte: 0x04 (reusing segment tag)
    name_len            ||  # 4 bytes
    collection_name     ||  # var bytes: UTF-8
    dim                 ||  # 4 bytes: u32
    metric_type             # 1 byte: 0=cosine, 1=dot, 2=L2
)
```

## Empty Tree

An empty collection (no records) has a special root:

```
empty_tree_root = BLAKE3(TAG_EMPTY)  # 32 bytes
```

## Proof Encoding

See [proof_format.md](proof_format.md) for inclusion proof encoding.

## Test Vectors

### Test Vector 1: Single Record

```
Input:
  id = 1 (u128)
  vector = [1.0, 2.0, 3.0] (3 floats)
  payload = {} (empty)
  version = 0

Encoded leaf input (hex):
  00                                    # TAG_LEAF
  01000000000000000000000000000000      # id = 1 (u128 LE)
  03000000                              # dim = 3 (u32 LE)
  0000803f 00000040 00004040            # [1.0, 2.0, 3.0] (f32 LE)
  00000000                              # payload_len = 0
  0000000000000000                      # version = 0

Expected leaf_hash (BLAKE3, hex):
  [computed at implementation time]
```

### Test Vector 2: Internal Node

```
Input:
  left_hash  = 0x0000...0001 (32 bytes, all zeros except last byte = 1)
  right_hash = 0x0000...0002 (32 bytes, all zeros except last byte = 2)

Encoded internal input (hex):
  01                                    # TAG_INTERNAL
  0000...0001                           # left_hash (32 bytes)
  0000...0002                           # right_hash (32 bytes)

Expected internal_hash (BLAKE3, hex):
  [computed at implementation time]
```

## Implementation Checklist

- [ ] Implement encode_leaf/4
- [ ] Implement encode_internal/2
- [ ] Implement encode_index_state/1
- [ ] Implement encode_manifest/1
- [ ] Implement canonical JSON encoder
- [ ] Add NaN/Infinity validation
- [ ] Add test vectors
- [ ] Verify cross-platform consistency
