# MerkleDB Inclusion Proof Format

Version: 1.0.0
Date: 2026-01-01

## Overview

This document specifies the format for inclusion proofs that allow
clients to verify record membership in a snapshot without trusting
the server.

## Proof Structure

An inclusion proof demonstrates that a specific record was part of
the dataset when a snapshot was committed.

### Components

1. **Snapshot Root**: The 32-byte root hash being verified against
2. **Record**: The full record data (id, vector, payload, version)
3. **Leaf Index**: Position of the record in the sorted leaf array
4. **Sibling Path**: Sequence of sibling hashes from leaf to root

## Binary Encoding

```
Proof = {
    version:        u8          # Proof format version (0x01)
    snapshot_root:  [u8; 32]    # Expected root hash
    record_id:      u128        # Record identifier
    leaf_index:     u64         # Position in tree
    path_length:    u8          # Number of path elements (max 64)
    path:           [PathElement; path_length]
}

PathElement = {
    direction:      u8          # 0x00 = left sibling, 0x01 = right sibling
    sibling_hash:   [u8; 32]    # Hash of sibling node
}
```

### Total Size

- Fixed header: 1 + 32 + 16 + 8 + 1 = 58 bytes
- Per path element: 1 + 32 = 33 bytes
- Typical proof (20 levels): 58 + 20*33 = 718 bytes
- Maximum proof (64 levels): 58 + 64*33 = 2,170 bytes

## Verification Algorithm

```elixir
def verify_inclusion(snapshot_root, record, proof) do
  # Step 1: Recompute leaf hash from record
  leaf_hash = compute_leaf_hash(
    record.id,
    record.vector,
    record.payload,
    record.version
  )

  # Step 2: Walk up the tree using sibling path
  current_hash = leaf_hash

  for {direction, sibling_hash} <- proof.path do
    current_hash = case direction do
      :left ->
        # Sibling is on the left
        compute_internal_hash(sibling_hash, current_hash)
      :right ->
        # Sibling is on the right
        compute_internal_hash(current_hash, sibling_hash)
    end
  end

  # Step 3: Compare with expected root
  current_hash == snapshot_root
end

defp compute_leaf_hash(id, vector, payload, version) do
  data = <<
    0x00,                           # TAG_LEAF
    id::little-128,
    length(vector)::little-32,
    encode_vector(vector)::binary,
    byte_size(payload)::little-32,
    payload::binary,
    version::little-64
  >>
  Blake3.hash(data)
end

defp compute_internal_hash(left, right) do
  Blake3.hash(<<0x01, left::binary-32, right::binary-32>>)
end
```

## Direction Convention

The `direction` field indicates where the **sibling** is relative to
the current node:

- `0x00` (`:left`): Sibling is the LEFT child, current is RIGHT child
  - Hash order: `H(sibling, current)`

- `0x01` (`:right`): Sibling is the RIGHT child, current is LEFT child
  - Hash order: `H(current, sibling)`

## JSON Representation

For API responses and debugging:

```json
{
  "version": 1,
  "snapshot_root": "a1b2c3...64hex",
  "record_id": "12345",
  "leaf_index": 42,
  "path": [
    {"direction": "right", "sibling": "d4e5f6...64hex"},
    {"direction": "left",  "sibling": "789abc...64hex"},
    {"direction": "right", "sibling": "def012...64hex"}
  ]
}
```

## Proof Generation

```elixir
def prove_inclusion(tree, record_id, snapshot_root) do
  # Find the snapshot
  {:ok, snapshot} = get_snapshot(snapshot_root)

  # Find record position in sorted leaves
  leaves = snapshot.leaves  # Already sorted by id
  leaf_index = binary_search(leaves, record_id)

  if leaf_index == :not_found do
    {:error, :not_found}
  else
    # Get record data
    record = Enum.at(leaves, leaf_index)

    # Build path from leaf to root
    path = build_merkle_path(snapshot.tree, leaf_index)

    {:ok, %Proof{
      version: 1,
      snapshot_root: snapshot_root,
      record_id: record_id,
      leaf_index: leaf_index,
      path: path
    }}
  end
end

defp build_merkle_path(tree, leaf_index) do
  # Tree is stored as array: [root, level1..., level2..., ..., leaves...]
  # Navigate from leaf up to root, collecting siblings

  path = []
  current_index = leaf_index
  level_size = tree.leaf_count
  level_offset = tree.total_nodes - level_size

  while level_size > 1 do
    sibling_index = if rem(current_index, 2) == 0 do
      current_index + 1  # Sibling is to the right
    else
      current_index - 1  # Sibling is to the left
    end

    direction = if rem(current_index, 2) == 0, do: :right, else: :left

    sibling_hash = if sibling_index < level_size do
      get_node(tree, level_offset + sibling_index)
    else
      # Odd node count: no sibling, promoted directly
      nil
    end

    if sibling_hash do
      path = [{direction, sibling_hash} | path]
    end

    # Move up one level
    current_index = div(current_index, 2)
    level_size = div(level_size + 1, 2)
    level_offset = level_offset - level_size
  end

  Enum.reverse(path)
end
```

## Edge Cases

### Empty Collection

Cannot generate proofs for empty collections. Return `{:error, :empty_collection}`.

### Single Record

Proof has empty path. Leaf hash equals tree root.

```json
{
  "snapshot_root": "abc...",
  "leaf_index": 0,
  "path": []
}
```

### Record Not Found

Return `{:error, :not_found}` if record_id is not in the snapshot.

### Deleted Records

Deleted records are not in the Merkle tree and cannot have proofs generated.

## Security Considerations

1. **Proof binding**: The proof is bound to a specific snapshot_root.
   A proof valid for one root is not valid for another.

2. **Record binding**: The proof verifies the exact record data.
   Any modification to id, vector, payload, or version invalidates the proof.

3. **Non-transferability**: Proofs cannot be reused across different records
   or different snapshots.

4. **Compactness**: Proof size is O(log n) where n is record count.
   Even for 10M records, proof is ~700 bytes.

## Test Vectors

### Test Case 1: Three Records

```
Records (sorted by id):
  id=1, vector=[1.0], payload={}, version=0
  id=2, vector=[2.0], payload={}, version=0
  id=3, vector=[3.0], payload={}, version=0

Tree structure:
       root
      /    \
    H01     h2
   /   \
  h0    h1

Proof for id=2:
  path = [
    {:left, h0_h1_hash},  # h2's sibling is H(h0,h1)
  ]

Verification:
  leaf_hash = compute_leaf_hash(2, [2.0], {}, 0)
  step1 = H(H01, leaf_hash)  # H01 is left sibling
  assert step1 == root
```

### Test Case 2: Power of Two Records

```
Records: 4 records with ids 1,2,3,4

Tree:
         root
        /    \
      H01    H23
     /  \   /  \
    h0  h1 h2  h3

Proof for id=3 (index 2):
  path = [
    {:right, h3},    # h2's sibling is h3 (right)
    {:left, H01}     # H23's sibling is H01 (left)
  ]
```
