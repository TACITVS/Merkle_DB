defmodule MerkleDb.SnapshotIntegrationTest do
  use ExUnit.Case, async: false

  alias MerkleDb.{StorageEngine, SnapshotStore}

  @test_dir "test/tmp/snapshot_integration"

  setup do
    File.rm_rf!(@test_dir)
    File.mkdir_p!(@test_dir)

    on_exit(fn ->
      File.rm_rf!(@test_dir)
    end)

    :ok
  end

  defp engine_dir(name) do
    Path.join(@test_dir, name)
  end

  describe "commit and snapshot persistence" do
    test "commit saves snapshot to store" do
      dir = engine_dir("commit_saves")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      # Insert records
      for i <- 1..10 do
        :ok = StorageEngine.upsert(engine, i, [i * 1.0], %{"i" => i})
      end

      # Commit
      {:ok, root} = StorageEngine.commit(engine)

      # Verify snapshot was saved
      assert SnapshotStore.exists?(dir, root)

      StorageEngine.close(engine)
    end

    test "list_snapshots returns committed snapshots" do
      dir = engine_dir("list_snapshots")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      # Create multiple commits
      for batch <- 1..3 do
        :ok = StorageEngine.upsert(engine, batch, [batch * 1.0], %{})
        {:ok, _root} = StorageEngine.commit(engine)
      end

      {:ok, snapshots} = StorageEngine.list_snapshots(engine)

      assert length(snapshots) == 3

      StorageEngine.close(engine)
    end

    test "get_snapshot_info returns snapshot details" do
      dir = engine_dir("snapshot_info")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      for i <- 1..5 do
        :ok = StorageEngine.upsert(engine, i, [i * 1.0, i * 2.0], %{"idx" => i})
      end

      {:ok, root} = StorageEngine.commit(engine)
      {:ok, info} = StorageEngine.get_snapshot_info(engine, root)

      assert info.record_count == 5
      assert info.schema.name == "default"
      assert is_binary(info.snapshot_root)

      StorageEngine.close(engine)
    end
  end

  describe "prove_inclusion" do
    test "generates valid proof for committed record" do
      dir = engine_dir("prove_basic")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      # Insert records
      for i <- 1..10 do
        :ok = StorageEngine.upsert(engine, i, [i * 1.0], %{"i" => i}, 0)
      end

      # Commit
      {:ok, _root} = StorageEngine.commit(engine)

      # Generate proof for record 5
      {:ok, proof} = StorageEngine.prove_inclusion(engine, 5)

      assert proof.record_id == 5
      assert byte_size(proof.snapshot_root) == 32

      StorageEngine.close(engine)
    end

    test "proof can be verified client-side" do
      dir = engine_dir("prove_verify")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      # Insert records with known data
      record = {42, [1.5, 2.5, 3.5], %{"name" => "test"}, 0}
      {id, vector, payload, version} = record
      :ok = StorageEngine.upsert(engine, id, vector, payload, version)

      # Commit
      {:ok, _root} = StorageEngine.commit(engine)

      # Get proof
      {:ok, proof} = StorageEngine.prove_inclusion(engine, 42)

      # Verify client-side (no engine access needed)
      assert StorageEngine.verify_proof(record, proof)

      StorageEngine.close(engine)
    end

    test "get_with_proof returns record and proof together" do
      dir = engine_dir("get_with_proof")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      :ok = StorageEngine.upsert(engine, 1, [1.0, 2.0], %{"key" => "value"}, 100)
      {:ok, _root} = StorageEngine.commit(engine)

      {:ok, record, proof} = StorageEngine.get_with_proof(engine, 1)

      # Record is returned
      assert elem(record, 0) == 1
      assert elem(record, 1) == [1.0, 2.0]

      # Proof is valid
      assert proof.record_id == 1

      # Verification works
      record_tuple = {1, [1.0, 2.0], %{"key" => "value"}, 100}
      assert StorageEngine.verify_proof(record_tuple, proof)

      StorageEngine.close(engine)
    end

    test "prove_inclusion fails for non-existent record" do
      dir = engine_dir("prove_missing")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      :ok = StorageEngine.upsert(engine, 1, [1.0], %{})
      {:ok, _root} = StorageEngine.commit(engine)

      assert {:error, :not_found} = StorageEngine.prove_inclusion(engine, 999)

      StorageEngine.close(engine)
    end

    test "prove_inclusion fails without commit" do
      dir = engine_dir("prove_no_commit")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      :ok = StorageEngine.upsert(engine, 1, [1.0], %{})

      # No commit yet
      assert {:error, :no_snapshot} = StorageEngine.prove_inclusion(engine, 1)

      StorageEngine.close(engine)
    end
  end

  describe "snapshot-based queries" do
    test "can prove against specific snapshot" do
      dir = engine_dir("specific_snapshot")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      # First commit with record 1
      :ok = StorageEngine.upsert(engine, 1, [1.0], %{}, 0)
      {:ok, root1} = StorageEngine.commit(engine)

      # Second commit with record 2
      :ok = StorageEngine.upsert(engine, 2, [2.0], %{}, 0)
      {:ok, root2} = StorageEngine.commit(engine)

      # Prove record 1 against first snapshot
      {:ok, proof1} = StorageEngine.prove_inclusion(engine, 1, root1)
      assert proof1.record_id == 1

      # Prove record 2 against second snapshot
      {:ok, proof2} = StorageEngine.prove_inclusion(engine, 2, root2)
      assert proof2.record_id == 2

      # Record 2 should NOT be in first snapshot
      assert {:error, :not_found} = StorageEngine.prove_inclusion(engine, 2, root1)

      StorageEngine.close(engine)
    end
  end

  describe "tamper detection" do
    test "verification fails with tampered record" do
      dir = engine_dir("tamper_detect")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      :ok = StorageEngine.upsert(engine, 1, [1.0, 2.0, 3.0], %{"original" => true}, 0)
      {:ok, _root} = StorageEngine.commit(engine)

      {:ok, proof} = StorageEngine.prove_inclusion(engine, 1)

      # Original record verifies
      original = {1, [1.0, 2.0, 3.0], %{"original" => true}, 0}
      assert StorageEngine.verify_proof(original, proof)

      # Tampered vector fails
      tampered_vector = {1, [999.0, 999.0, 999.0], %{"original" => true}, 0}
      refute StorageEngine.verify_proof(tampered_vector, proof)

      # Tampered payload fails
      tampered_payload = {1, [1.0, 2.0, 3.0], %{"tampered" => true}, 0}
      refute StorageEngine.verify_proof(tampered_payload, proof)

      # Tampered version fails
      tampered_version = {1, [1.0, 2.0, 3.0], %{"original" => true}, 999}
      refute StorageEngine.verify_proof(tampered_version, proof)

      StorageEngine.close(engine)
    end
  end

  describe "snapshot garbage collection" do
    test "gc_snapshots removes old snapshots" do
      dir = engine_dir("gc_test")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      # Create 5 snapshots
      for i <- 1..5 do
        :ok = StorageEngine.upsert(engine, i, [i * 1.0], %{})
        {:ok, _} = StorageEngine.commit(engine)
        Process.sleep(10)  # Ensure different timestamps
      end

      {:ok, before} = StorageEngine.list_snapshots(engine)
      assert length(before) == 5

      # Keep only last 2
      {:ok, deleted} = StorageEngine.gc_snapshots(engine, keep_last: 2)
      assert deleted == 3

      {:ok, after_gc} = StorageEngine.list_snapshots(engine)
      assert length(after_gc) == 2

      StorageEngine.close(engine)
    end
  end

  describe "end-to-end verification flow" do
    test "full flow: insert, commit, prove, verify" do
      dir = engine_dir("e2e_flow")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      # Step 1: Insert records
      records = for i <- 1..100 do
        vector = [i * 0.1, i * 0.2, i * 0.3]
        payload = %{"category" => "item_#{rem(i, 5)}", "score" => i / 100}
        :ok = StorageEngine.upsert(engine, i, vector, payload, 0)
        {i, vector, payload, 0}
      end

      # Step 2: Commit
      {:ok, root} = StorageEngine.commit(engine)

      # Step 3: Generate proofs for random records
      test_ids = [1, 25, 50, 75, 100]

      proofs = for id <- test_ids do
        {:ok, proof} = StorageEngine.prove_inclusion(engine, id)
        {id, proof}
      end

      # Step 4: Verify all proofs (simulating client-side)
      for {id, proof} <- proofs do
        record = Enum.find(records, fn {rec_id, _, _, _} -> rec_id == id end)
        assert StorageEngine.verify_proof(record, proof),
               "Verification failed for record #{id}"
      end

      # Step 5: Verify snapshot info
      {:ok, info} = StorageEngine.get_snapshot_info(engine, root)
      assert info.record_count == 100

      StorageEngine.close(engine)
    end

    test "proof serialization for transport" do
      dir = engine_dir("proof_transport")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      :ok = StorageEngine.upsert(engine, 1, [1.0, 2.0], %{}, 0)
      {:ok, _} = StorageEngine.commit(engine)

      {:ok, proof} = StorageEngine.prove_inclusion(engine, 1)

      # Serialize proof
      encoded = MerkleDb.Merkle.encode_proof(proof)
      assert is_binary(encoded)

      # Deserialize
      {:ok, decoded} = MerkleDb.Merkle.decode_proof(encoded)

      # Verify with decoded proof
      record = {1, [1.0, 2.0], %{}, 0}
      assert StorageEngine.verify_proof(record, decoded)

      StorageEngine.close(engine)
    end
  end
end
