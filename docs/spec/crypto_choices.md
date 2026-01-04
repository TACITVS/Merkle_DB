# MerkleDB Cryptographic Choices

Version: 1.0.0
Date: 2026-01-01

## Hash Function Selection

### Choice: BLAKE3

We select BLAKE3 as the primary hash function for MerkleDB.

**Rationale**:
- Fast: ~3x faster than SHA-256 on modern CPUs
- Tree-friendly: Native support for parallel/incremental hashing
- Secure: 256-bit security level, based on BLAKE2
- Standardized: Published specification, multiple implementations
- No length extension attacks (unlike SHA-256)

**Parameters**:
- Output size: 32 bytes (256 bits)
- No custom key (using default keyed mode is optional for MAC)

### Alternative Considered: SHA-256

SHA-256 would also be acceptable but is slower. If BLAKE3 is unavailable,
SHA-256 is the fallback.

## Hash Output Format

All hash values are represented as:
- Binary: 32 bytes, big-endian
- Hex string: 64 lowercase hexadecimal characters
- Base64: 44 characters (with padding)

**Canonical representation**: Binary (32 bytes) is the canonical form.
Hex/Base64 are for display/transport only.

## Domain Separation Tags

To prevent cross-protocol attacks and ensure unambiguous hashing,
every hash input is prefixed with a domain separation tag.

### Tag Definitions

| Tag | Value (1 byte) | Context |
|-----|----------------|---------|
| TAG_LEAF | 0x00 | Leaf node (record hash) |
| TAG_INTERNAL | 0x01 | Internal Merkle tree node |
| TAG_INDEX_STATE | 0x02 | Index state commitment |
| TAG_MANIFEST | 0x03 | Snapshot manifest |
| TAG_SEGMENT | 0x04 | Segment checksum |
| TAG_CENTROID | 0x05 | IVF centroid hash |
| TAG_POSTING | 0x06 | Posting list hash |

### Tag Usage

Every hash computation MUST include the appropriate tag as the first byte:

```
hash = BLAKE3(tag || payload)
```

This ensures that a valid internal node hash can never collide with a
valid leaf hash, even if the payloads happen to match.

## Numeric Encoding

### Integers

- All integers are encoded as little-endian
- Sizes: u8, u16, u32, u64, u128 as needed
- Signed integers use two's complement

### Floating Point

- IEEE 754 binary32 (float32) for vectors
- Little-endian byte order
- **NaN handling**: NaN values are FORBIDDEN in vectors
  - Insertion must reject NaN values
  - If NaN is detected, return error

### Length Prefixes

- All variable-length fields are prefixed with their byte length
- Length is encoded as u32 little-endian (4 bytes)
- Maximum length: 2^32 - 1 bytes

## Canonical Ordering

### Record Ordering in Merkle Tree

Records are sorted by ID before tree construction:
- ID comparison: lexicographic on bytes (big-endian u128)
- This ensures deterministic tree structure

### Determinism Requirements

Given the same set of (id, vector, payload) records:
1. The leaf hashes are deterministic
2. The tree structure is deterministic
3. The root hash is deterministic

**Insertion order does NOT affect the final root hash.**

## Version Binding

The encoding version is bound into the manifest:

```
manifest_hash = BLAKE3(TAG_MANIFEST || version_u8 || tree_root || ...)
```

This allows detection of version mismatches during verification.

## Implementation Notes

### Elixir/Erlang

Use `:crypto.hash(:blake3, data)` if available, otherwise use
a NIF binding to the BLAKE3 C library.

### Verification Library

The client verifier should be a minimal, dependency-light module:
- Pure Elixir implementation preferred for auditability
- Optional NIF for performance

## Test Vectors

See [test_vectors.md](test_vectors.md) for canonical test cases.
