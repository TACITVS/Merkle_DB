# MerkleDB Threat Model

Version: 1.0.0
Date: 2026-01-01

## Scope

This document defines the security assumptions, adversary capabilities,
and security claims for MerkleDB's cryptographic verification layer.

## Adversary Model

### What the Adversary CAN Do

1. **Modify on-disk data**: Attacker has write access to storage files
   - Can corrupt vector data files
   - Can modify index files (centroids, postings)
   - Can alter manifest/metadata files

2. **Forge membership claims**: Attacker attempts to prove a record exists
   - Can construct fake inclusion proofs
   - Can claim a vector is in a snapshot when it is not

3. **Replay old snapshots**: Attacker presents stale data as current
   - Can serve old snapshot roots
   - Can return outdated query results

4. **Tamper with segment files**: Attacker modifies individual segments
   - Can flip bits in stored vectors
   - Can delete or truncate files

### What the Adversary CANNOT Do

1. **Break hash collision resistance**: We assume the chosen hash function
   (BLAKE3) is collision-resistant. Finding two different inputs that
   produce the same hash is computationally infeasible.

2. **Break preimage resistance**: Given a hash output, finding any input
   that produces that hash is computationally infeasible.

3. **Compromise the verifier**: The client-side verification code is
   assumed to be trusted and unmodified.

## Security Claims

### Claim 1: Tamper-Evident Snapshots

If any data in a committed snapshot is modified after commit(), the
snapshot root hash will change. A client holding the original root
can detect tampering.

**Verification**: Compare stored root with recomputed root.

### Claim 2: Inclusion Proof Correctness

Given:
- A snapshot root R
- A record (id, vector, payload)
- An inclusion proof P

The verify_inclusion(R, record, P) function returns true if and only if
the record was present in the snapshot that produced R.

**Soundness**: An adversary cannot forge a valid proof for a record
that was not in the original snapshot (assuming collision resistance).

**Completeness**: For any record in the snapshot, a valid proof exists.

### Claim 3: Binding to Index State

The snapshot root binds not just data but also the index state:
- IVF centroids
- Posting lists
- Index parameters

This ensures query results are reproducible for a given snapshot.

## Out of Scope (v1.0)

1. **Verifiable computation of ANN**: We do NOT prove that the top-K
   results are correct. The index could return suboptimal results.
   (Future: ZK proofs or TEE attestation)

2. **Authenticity/signatures**: Snapshot roots are not signed by default.
   (Future: optional signature layer)

3. **Confidentiality**: Data is not encrypted at rest.
   (Future: optional encryption layer)

4. **Availability**: No protection against denial of service.

5. **Key management**: No built-in key management for signatures.

## Trust Boundaries

```
+------------------+     +------------------+
|  Trusted Zone    |     |  Untrusted Zone  |
|                  |     |                  |
|  - Verifier code |     |  - Storage files |
|  - Snapshot root |     |  - Network data  |
|  - Client memory |     |  - Server process|
+------------------+     +------------------+
```

## Mitigation Strategies

| Threat | Mitigation |
|--------|------------|
| Data tampering | Merkle root verification on read |
| Forged proofs | Cryptographic proof verification |
| Replay attacks | Timestamp + sequence in manifest (optional) |
| Bit flips | Checksum per segment + Merkle verification |

## Assumptions

1. The hash function (BLAKE3) maintains its security properties
2. Random number generation for k-means seeding is adequate
3. Floating-point representation is consistent (IEEE 754)
4. System clock is reasonably accurate (for timestamps)
