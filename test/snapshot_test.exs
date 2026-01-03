defmodule MerkleDb.SnapshotTest do
  use ExUnit.Case, async: true

  alias MerkleDb.Snapshot
  alias MerkleDb.Crypto

  # Helper to create test records
  defp make_record(id, value \\ nil) do
    value = value || id
    {id, [value * 1.0, value * 2.0, value * 3.0], %{"id" => id}, 0}
  end

  describe "commit/2" do
    test "empty collection" do
      {:ok, state} = Snapshot.commit([])
      assert state.manifest.record_count == 0
      assert byte_size(state.manifest.snapshot_root) == 32
    end

    test "single record" do
      {:ok, state} = Snapshot.commit([make_record(1)])
      assert state.manifest.record_count == 1
      assert byte_size(state.manifest.snapshot_root) == 32
    end

    test "multiple records" do
      records = Enum.map(1..100, &make_record/1)
      {:ok, state} = Snapshot.commit(records)
      assert state.manifest.record_count == 100
    end

    test "deterministic root for same data" do
      records = Enum.map(1..10, &make_record/1)

      # Commit twice - timestamps will differ, so tree_root should match
      {:ok, state1} = Snapshot.commit(records)
      {:ok, state2} = Snapshot.commit(Enum.shuffle(records))

      # Tree roots should be identical (data is sorted)
      assert state1.manifest.tree_root == state2.manifest.tree_root
    end

    test "different data produces different tree root" do
      {:ok, state1} = Snapshot.commit([make_record(1)])
      {:ok, state2} = Snapshot.commit([make_record(2)])
      refute state1.manifest.tree_root == state2.manifest.tree_root
    end

    test "infers dimension from first record" do
      records = [make_record(1)]  # 3-dimensional
      {:ok, state} = Snapshot.commit(records)
      assert state.schema.dim == 3
    end

    test "uses provided dimension" do
      {:ok, state} = Snapshot.commit([], dim: 384)
      assert state.schema.dim == 384
    end

    test "with collection name" do
      {:ok, state} = Snapshot.commit([], collection_name: "my_collection")
      assert state.schema.name == "my_collection"
    end

    test "with metric type" do
      {:ok, state1} = Snapshot.commit([], metric: :cosine)
      {:ok, state2} = Snapshot.commit([], metric: :dot)
      assert state1.schema.metric == :cosine
      assert state2.schema.metric == :dot
      # Different metrics produce different schema hashes
      refute state1.manifest.schema_hash == state2.manifest.schema_hash
    end

    test "with index state" do
      index_state = %{
        type: :ivf,
        centroids: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        postings: [[1, 2, 3], [4, 5, 6]]
      }

      {:ok, state} = Snapshot.commit([], index_state: index_state)
      # Index hash should not be empty hash
      refute state.manifest.index_state_hash == Crypto.hash_empty()
    end

    test "timestamp is recorded" do
      before = System.system_time(:microsecond)
      {:ok, state} = Snapshot.commit([])
      after_time = System.system_time(:microsecond)

      assert state.manifest.timestamp >= before
      assert state.manifest.timestamp <= after_time
    end
  end

  describe "prove_inclusion/2" do
    test "generates valid proof" do
      records = Enum.map(1..10, &make_record/1)
      {:ok, state} = Snapshot.commit(records)

      {:ok, proof} = Snapshot.prove_inclusion(state, 5)
      assert proof.record_id == 5
    end

    test "proof not found for missing record" do
      {:ok, state} = Snapshot.commit([make_record(1)])
      assert {:error, :not_found} = Snapshot.prove_inclusion(state, 999)
    end
  end

  describe "verify_inclusion/2 and verify_inclusion/3" do
    test "verifies valid proof (2-arity, uses embedded root)" do
      record = make_record(1)
      {:ok, state} = Snapshot.commit([record])
      {:ok, proof} = Snapshot.prove_inclusion(state, 1)

      # 2-arity version uses the tree_root embedded in the proof
      assert Snapshot.verify_inclusion(record, proof)
    end

    test "verifies valid proof (3-arity, explicit tree_root)" do
      record = make_record(1)
      {:ok, state} = Snapshot.commit([record])
      {:ok, proof} = Snapshot.prove_inclusion(state, 1)

      # 3-arity version verifies against explicit tree_root
      assert Snapshot.verify_inclusion(state.manifest.tree_root, record, proof)
    end

    test "end-to-end proof flow" do
      # This simulates the full client-server flow:
      # 1. Server commits records
      # 2. Server generates proof for a record
      # 3. Client verifies proof with only tree_root + record + proof

      records = Enum.map(1..100, &make_record/1)
      {:ok, state} = Snapshot.commit(records)

      # Pick a record to prove
      target_id = 42
      target_record = Enum.find(records, fn {id, _, _, _} -> id == target_id end)

      # Server generates proof
      {:ok, proof} = Snapshot.prove_inclusion(state, target_id)

      # Client has: tree_root (from manifest), the record, and the proof
      tree_root = state.manifest.tree_root

      # Client verification (this is the key security property)
      assert Snapshot.verify_inclusion(tree_root, target_record, proof)

      # Also verify the tree_root is bound in the snapshot
      assert proof.snapshot_root == tree_root
    end

    test "rejects tampered record" do
      record = make_record(1)
      {:ok, state} = Snapshot.commit([record])
      {:ok, proof} = Snapshot.prove_inclusion(state, 1)

      # Tamper with the record
      tampered = {1, [999.0, 999.0, 999.0], %{}, 0}
      refute Snapshot.verify_inclusion(tampered, proof)
    end

    test "rejects wrong root" do
      record = make_record(1)
      {:ok, state} = Snapshot.commit([record])
      {:ok, proof} = Snapshot.prove_inclusion(state, 1)

      wrong_root = Crypto.hash("wrong")
      refute Snapshot.verify_inclusion(wrong_root, record, proof)
    end
  end

  describe "root/1 and tree_root/1" do
    test "returns correct hashes" do
      {:ok, state} = Snapshot.commit([make_record(1)])

      assert byte_size(Snapshot.root(state)) == 32
      assert byte_size(Snapshot.tree_root(state)) == 32
      assert Snapshot.root(state) == state.manifest.snapshot_root
      assert Snapshot.tree_root(state) == state.manifest.tree_root
    end
  end

  describe "serialize/deserialize" do
    test "roundtrip" do
      records = Enum.map(1..10, &make_record/1)
      {:ok, state} = Snapshot.commit(records)

      serialized = Snapshot.serialize(state)
      assert is_binary(serialized)

      {:ok, deserialized} = Snapshot.deserialize(serialized)
      assert deserialized.manifest.snapshot_root == state.manifest.snapshot_root
      assert deserialized.manifest.tree_root == state.manifest.tree_root
      assert deserialized.manifest.record_count == state.manifest.record_count
    end

    test "invalid data returns error" do
      assert {:error, :invalid_data} = Snapshot.deserialize(<<1, 2, 3>>)
    end
  end

  describe "info/1" do
    test "returns readable info" do
      {:ok, state} = Snapshot.commit([make_record(1)],
        collection_name: "test_collection",
        dim: 3,
        metric: :cosine
      )

      info = Snapshot.info(state)

      assert is_binary(info.snapshot_root)
      assert String.length(info.snapshot_root) == 64  # hex string
      assert info.record_count == 1
      assert info.schema.name == "test_collection"
      assert info.schema.dim == 3
      assert info.schema.metric == :cosine
      assert %DateTime{} = info.timestamp
    end
  end

  describe "index state hashing" do
    test "same index state produces same hash" do
      index_state = %{
        type: :ivf,
        centroids: [[1.0, 2.0], [3.0, 4.0]],
        postings: [[1, 2], [3, 4]]
      }

      {:ok, state1} = Snapshot.commit([], index_state: index_state)
      {:ok, state2} = Snapshot.commit([], index_state: index_state)

      assert state1.manifest.index_state_hash == state2.manifest.index_state_hash
    end

    test "different index produces different hash" do
      index1 = %{type: :ivf, centroids: [[1.0]], postings: [[1]]}
      index2 = %{type: :ivf, centroids: [[2.0]], postings: [[1]]}

      {:ok, state1} = Snapshot.commit([], index_state: index1)
      {:ok, state2} = Snapshot.commit([], index_state: index2)

      refute state1.manifest.index_state_hash == state2.manifest.index_state_hash
    end
  end
end
