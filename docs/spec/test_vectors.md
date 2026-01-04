# MerkleDB Test Vectors

Version: 1.0.0
Date: 2026-01-01

## Purpose

These test vectors allow verification that different implementations
produce identical hashes for the same input data.

## Hash Function

All tests use SHA-256 (via Erlang `:crypto.hash/2`).
BLAKE3 will have different values.

## Test Vector 1: Empty Tree

```
Input: No records

Expected tree_root (hex):
  [Empty hash = SHA256(0xFF)]
```

Generate with:
```elixir
MerkleDb.Crypto.hash_empty() |> MerkleDb.Crypto.to_hex()
```

## Test Vector 2: Single Record

```
Input:
  id = 1
  vector = [1.0, 2.0, 3.0]
  payload = {}
  version = 0

Leaf encoding (hex):
  00                              # TAG_LEAF
  01 00 00 00 00 00 00 00         # id (u128 LE, bytes 0-7)
  00 00 00 00 00 00 00 00         # id (u128 LE, bytes 8-15)
  03 00 00 00                     # dim = 3 (u32 LE)
  00 00 80 3f                     # 1.0 (float32 LE)
  00 00 00 40                     # 2.0 (float32 LE)
  00 00 40 40                     # 3.0 (float32 LE)
  02 00 00 00                     # payload_len = 2
  7b 7d                           # "{}" (empty JSON object)
  00 00 00 00 00 00 00 00         # version = 0 (u64 LE)
```

Generate with:
```elixir
{:ok, hash} = MerkleDb.Canonical.record_hash(1, [1.0, 2.0, 3.0], %{}, 0)
MerkleDb.Crypto.to_hex(hash)
```

## Test Vector 3: Three Records

```
Records:
  {1, [1.0, 2.0, 3.0], %{}, 0}
  {2, [4.0, 5.0, 6.0], %{}, 0}
  {3, [7.0, 8.0, 9.0], %{}, 0}

Tree structure:
       root
      /    \
    H01     h2
   /   \
  h0    h1

Where:
  h0 = leaf_hash(record_1)
  h1 = leaf_hash(record_2)
  h2 = leaf_hash(record_3)
  H01 = internal_hash(h0, h1)
  root = internal_hash(H01, h2)
```

Generate with:
```elixir
records = [
  {1, [1.0, 2.0, 3.0], %{}, 0},
  {2, [4.0, 5.0, 6.0], %{}, 0},
  {3, [7.0, 8.0, 9.0], %{}, 0}
]
{:ok, tree} = MerkleDb.Merkle.build_tree(records)
MerkleDb.Crypto.to_hex(tree.root)
```

## Test Vector 4: Proof Verification

```
For record {2, [4.0, 5.0, 6.0], %{}, 0} in the 3-record tree:

Proof structure:
  leaf_index = 1
  path = [
    {:left, h0},   # h1's sibling is h0 (left)
    {:right, h2}   # H01's sibling is h2 (right)
  ]

Verification:
  1. Compute h1 = leaf_hash(record_2)
  2. H01 = internal_hash(h0, h1)  # h0 is left sibling
  3. root = internal_hash(H01, h2)  # h2 is right sibling
  4. Compare with expected root
```

Generate with:
```elixir
records = [
  {1, [1.0, 2.0, 3.0], %{}, 0},
  {2, [4.0, 5.0, 6.0], %{}, 0},
  {3, [7.0, 8.0, 9.0], %{}, 0}
]
{:ok, tree} = MerkleDb.Merkle.build_tree(records)
{:ok, proof} = MerkleDb.Merkle.prove_inclusion(tree, 2)
record_2 = {2, [4.0, 5.0, 6.0], %{}, 0}
MerkleDb.Merkle.verify_inclusion(tree.root, record_2, proof)  # => true
```

## Test Vector 5: Proof Encoding

```elixir
# Generate encoded proof
{:ok, proof} = MerkleDb.Merkle.prove_inclusion(tree, 2)
encoded = MerkleDb.Merkle.encode_proof(proof)

# Structure:
#   version (1 byte)
#   snapshot_root (32 bytes)
#   record_id (16 bytes, u128 LE)
#   leaf_index (8 bytes, u64 LE)
#   path_length (1 byte)
#   path (path_length * 33 bytes each: 1 direction + 32 hash)

# Decode and verify roundtrip
{:ok, decoded} = MerkleDb.Merkle.decode_proof(encoded)
decoded.record_id == proof.record_id  # => true
decoded.path == proof.path  # => true
```

## Test Vector 6: Order Independence

```
# Different insertion order, same root

records_a = [{3, [7.0], %{}, 0}, {1, [1.0], %{}, 0}, {2, [2.0], %{}, 0}]
records_b = [{1, [1.0], %{}, 0}, {2, [2.0], %{}, 0}, {3, [7.0], %{}, 0}]

{:ok, tree_a} = MerkleDb.Merkle.build_tree(records_a)
{:ok, tree_b} = MerkleDb.Merkle.build_tree(records_b)

tree_a.root == tree_b.root  # => true (records are sorted by id)
```

## Generating Fresh Test Vectors

Run this to generate current test vector values:

```elixir
alias MerkleDb.{Crypto, Canonical, Merkle}

IO.puts("Empty hash: #{Crypto.to_hex(Crypto.hash_empty())}")

{:ok, h1} = Canonical.record_hash(1, [1.0, 2.0, 3.0], %{}, 0)
IO.puts("Single record hash: #{Crypto.to_hex(h1)}")

records = [
  {1, [1.0, 2.0, 3.0], %{}, 0},
  {2, [4.0, 5.0, 6.0], %{}, 0},
  {3, [7.0, 8.0, 9.0], %{}, 0}
]
{:ok, tree} = Merkle.build_tree(records)
IO.puts("Three record root: #{Crypto.to_hex(tree.root)}")
```

## Cross-Implementation Verification

To verify another implementation produces the same results:

1. Implement the encoding according to `canonical_encoding.md`
2. Use SHA-256 for hashing
3. Compare hash outputs with values generated above
4. All hashes must match exactly
