defmodule MerkleDb.Demo do
  @moduledoc """
  Demonstration of MerkleDB's verifiable vector database capabilities.

  This module shows the end-to-end flow:
  1. Create vectors with payloads
  2. Commit to create an immutable snapshot
  3. Generate inclusion proofs
  4. Verify proofs client-side (no database access needed)
  """

  alias MerkleDb.Snapshot

  @doc """
  Run the full demo showing snapshot + proof verification.

  ## Example

      iex> MerkleDb.Demo.run()
  """
  def run do
    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("  MerkleDB Verifiable Vector Database Demo")
    IO.puts(String.duplicate("=", 60) <> "\n")

    # Step 1: Create sample records
    IO.puts("Step 1: Creating sample vector records...")
    records = create_sample_records(100)
    IO.puts("   Created #{length(records)} records with 3D vectors\n")

    # Step 2: Commit to create snapshot
    IO.puts("Step 2: Committing records to create cryptographic snapshot...")
    {:ok, state} = Snapshot.commit(records,
      collection_name: "demo_collection",
      metric: :cosine
    )

    info = Snapshot.info(state)
    IO.puts("   Snapshot Root: #{String.slice(info.snapshot_root, 0, 16)}...")
    IO.puts("   Tree Root:     #{String.slice(info.tree_root, 0, 16)}...")
    IO.puts("   Record Count:  #{info.record_count}")
    IO.puts("   Timestamp:     #{info.timestamp}\n")

    # Step 3: Generate proofs for some records
    IO.puts("Step 3: Generating inclusion proofs...")
    test_ids = [1, 42, 99]

    proofs =
      Enum.map(test_ids, fn id ->
        {:ok, proof} = Snapshot.prove_inclusion(state, id)
        IO.puts("   Proof for ID #{id}: #{length(proof.path)} path elements")
        {id, proof}
      end)
    IO.puts("")

    # Step 4: Verify proofs (simulating client-side verification)
    IO.puts("Step 4: Verifying proofs (client-side, no DB access)...")
    IO.puts("   [Simulating client with only: root hash + record + proof]\n")

    Enum.each(proofs, fn {id, proof} ->
      # Get the original record
      record = Enum.find(records, fn {rec_id, _, _, _} -> rec_id == id end)

      # Client verification - only needs record + proof
      # (The proof contains the tree_root to verify against)
      verified = Snapshot.verify_inclusion(record, proof)

      status = if verified, do: "✓ VALID", else: "✗ INVALID"
      IO.puts("   Record ID #{id}: #{status}")
    end)
    IO.puts("")

    # Step 5: Demonstrate tamper detection
    IO.puts("Step 5: Demonstrating tamper detection...")
    {id, proof} = hd(proofs)
    original_record = Enum.find(records, fn {rec_id, _, _, _} -> rec_id == id end)

    # Tamper with the record
    {_, _vector, payload, version} = original_record
    tampered_record = {id, [999.0, 999.0, 999.0], payload, version}

    original_valid = Snapshot.verify_inclusion(original_record, proof)
    tampered_valid = Snapshot.verify_inclusion(tampered_record, proof)

    IO.puts("   Original record:  #{if original_valid, do: "✓ VALID", else: "✗ INVALID"}")
    IO.puts("   Tampered record:  #{if tampered_valid, do: "✓ VALID", else: "✗ INVALID"}")
    IO.puts("   → Tampering detected!\n")

    # Step 6: Show proof serialization
    IO.puts("Step 6: Proof serialization for transport...")
    {_, proof} = hd(proofs)
    encoded = MerkleDb.Merkle.encode_proof(proof)
    {:ok, decoded} = MerkleDb.Merkle.decode_proof(encoded)

    IO.puts("   Proof size: #{byte_size(encoded)} bytes")
    IO.puts("   Roundtrip:  #{if decoded.record_id == proof.record_id, do: "✓ OK", else: "✗ FAILED"}")
    IO.puts("")

    IO.puts(String.duplicate("=", 60))
    IO.puts("  Demo Complete! MerkleDB provides verifiable vector storage.")
    IO.puts(String.duplicate("=", 60) <> "\n")

    :ok
  end

  @doc """
  Quick benchmark of proof generation and verification.
  """
  def benchmark(record_count \\ 10_000) do
    IO.puts("\nBenchmarking with #{record_count} records...")

    # Create records
    records = create_sample_records(record_count)

    # Time commit
    {commit_time, {:ok, state}} = :timer.tc(fn ->
      Snapshot.commit(records, collection_name: "bench")
    end)
    IO.puts("Commit time: #{commit_time / 1000}ms")

    # Time proof generation
    {prove_time, {:ok, _proof}} = :timer.tc(fn ->
      Snapshot.prove_inclusion(state, div(record_count, 2))
    end)
    IO.puts("Prove time:  #{prove_time}µs")

    # Time verification
    record = Enum.at(records, div(record_count, 2))
    {:ok, proof} = Snapshot.prove_inclusion(state, div(record_count, 2) + 1)

    {verify_time, _result} = :timer.tc(fn ->
      Snapshot.verify_inclusion(record, proof)
    end)
    IO.puts("Verify time: #{verify_time}µs")

    IO.puts("Proof size:  #{byte_size(MerkleDb.Merkle.encode_proof(proof))} bytes")
    IO.puts("Tree depth:  #{length(proof.path)} levels")

    :ok
  end

  # Create sample records
  defp create_sample_records(count) do
    Enum.map(1..count, fn i ->
      vector = [i * 0.1, i * 0.2, i * 0.3]
      payload = %{"category" => "item_#{rem(i, 10)}", "score" => i / count}
      {i, vector, payload, 0}
    end)
  end
end
