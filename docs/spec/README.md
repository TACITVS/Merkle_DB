# MerkleDB Specification Pack (D0)

Version: 1.0.0
Date: 2026-01-01
Status: DRAFT

## Overview

MerkleDB is a Verifiable Vector Database that combines:
- High-performance approximate nearest neighbor (ANN) search
- Cryptographic commitments via Merkle trees
- Immutable snapshots with verifiable inclusion proofs

## Documents

| Document | Description |
|----------|-------------|
| [threat_model.md](threat_model.md) | Security assumptions and threat model |
| [crypto_choices.md](crypto_choices.md) | Cryptographic primitives and parameters |
| [canonical_encoding.md](canonical_encoding.md) | Byte-level encoding specification |
| [api_spec.md](api_spec.md) | Public API contract |
| [proof_format.md](proof_format.md) | Inclusion proof format |

## Design Principles

P1. Correctness before speed; speed is worthless if semantics are unstable.
P2. Cryptographic commitments must be unambiguous: canonical encoding + domain separation.
P3. Snapshots are the product: root hashes are stable, versionable IDs.
P4. Separate concerns: Merkle for auditability/versioning, signatures for authenticity.
P5. Operational simplicity: crash-safe, recoverable, deterministic builds.

## Target Scale

- Vectors: up to 10M
- Dimensions: 384, 768, 1024 (typical embedding sizes)
- Top-K: 10, 50, 100
- Latency target: p95 < 50ms for query
- Recall target: >= 95% recall@10 vs brute-force

## Success Criteria

S1. Snapshot + proof verification works end-to-end
S2. Deterministic root for same logical dataset
S3. Crash recovery returns identical snapshot roots
S4. Client can verify inclusion proofs offline
