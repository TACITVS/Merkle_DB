defmodule MerkleDb.ReplicationTest do
  use ExUnit.Case, async: false

  alias MerkleDb.Replication
  alias MerkleDb.Replication.Operation

  setup do
    # Clear oplog before each test
    :ets.delete_all_objects(:replication_oplog)
    :ets.insert(:replication_meta, {:current_seq, 0})
    # Reset KV to empty tree so tests can use any dimension
    MerkleDb.KV.reset()
    :ok
  end

  describe "record_upsert/4" do
    test "records upsert operation and returns sequence number" do
      vector = <<1.0::little-float-64, 2.0::little-float-64>>
      payload = %{"category" => "test"}

      {:ok, seq1} = Replication.record_upsert("key1", vector, payload, 1)
      {:ok, seq2} = Replication.record_upsert("key2", vector, payload, 1)

      assert seq1 == 1
      assert seq2 == 2
    end

    test "current_seq returns latest sequence" do
      vector = <<1.0::little-float-64>>

      {:ok, _} = Replication.record_upsert("key1", vector, %{}, 1)
      {:ok, _} = Replication.record_upsert("key2", vector, %{}, 1)

      assert Replication.current_seq() == 2
    end
  end

  describe "record_delete/1" do
    test "records delete operation" do
      {:ok, seq} = Replication.record_delete("key1")

      assert seq == 1

      {:ok, ops} = Replication.get_deltas(since: 0)
      assert length(ops) == 1
      assert hd(ops).op == :delete
      assert hd(ops).key == "key1"
    end
  end

  describe "get_deltas/1" do
    test "returns operations since a sequence number" do
      vector = <<1.0::little-float-64>>

      {:ok, _} = Replication.record_upsert("key1", vector, %{}, 1)
      {:ok, _} = Replication.record_upsert("key2", vector, %{}, 1)
      {:ok, _} = Replication.record_upsert("key3", vector, %{}, 1)

      {:ok, ops} = Replication.get_deltas(since: 1)

      assert length(ops) == 2
      assert Enum.map(ops, & &1.key) == ["key2", "key3"]
    end

    test "returns empty list when no operations since" do
      {:ok, ops} = Replication.get_deltas(since: 100)

      assert ops == []
    end

    test "respects limit option" do
      vector = <<1.0::little-float-64>>

      for i <- 1..10 do
        {:ok, _} = Replication.record_upsert("key#{i}", vector, %{}, 1)
      end

      {:ok, ops} = Replication.get_deltas(since: 0, limit: 3)

      assert length(ops) == 3
    end

    test "returns operations in order" do
      vector = <<1.0::little-float-64>>

      {:ok, _} = Replication.record_upsert("a", vector, %{}, 1)
      {:ok, _} = Replication.record_upsert("b", vector, %{}, 1)
      {:ok, _} = Replication.record_upsert("c", vector, %{}, 1)

      {:ok, ops} = Replication.get_deltas(since: 0)

      seqs = Enum.map(ops, & &1.seq)
      assert seqs == Enum.sort(seqs)
    end
  end

  describe "apply_operations/1" do
    test "applies upsert operations" do
      operations = [
        %Operation{
          seq: 1,
          op: :upsert,
          key: "test_key",
          data: %{vector: <<1.0::little-float-64, 2.0::little-float-64>>, payload: %{"x" => 1}},
          timestamp: System.system_time(:millisecond)
        }
      ]

      {:ok, count} = Replication.apply_operations(operations)

      assert count == 1
    end

    test "applies delete operations" do
      # First insert
      MerkleDb.KV.put("to_delete", <<1.0::little-float-64>>)

      operations = [
        %Operation{
          seq: 1,
          op: :delete,
          key: "to_delete",
          data: nil,
          timestamp: System.system_time(:millisecond)
        }
      ]

      {:ok, count} = Replication.apply_operations(operations)

      assert count == 1
    end

    test "applies operations from JSON-like maps" do
      operations = [
        %{
          "op" => "upsert",
          "key" => "json_key",
          "data" => %{"vector" => <<1.0::little-float-64>>, "payload" => %{}}
        }
      ]

      {:ok, count} = Replication.apply_operations(operations)

      assert count == 1
    end
  end

  describe "status/0" do
    test "returns replication status" do
      vector = <<1.0::little-float-64>>
      {:ok, _} = Replication.record_upsert("key1", vector, %{}, 1)

      status = Replication.status()

      assert status.current_seq == 1
      assert status.oplog_size == 1
      assert is_integer(status.created_at)
    end
  end

  describe "compact/1" do
    test "removes old entries from oplog" do
      vector = <<1.0::little-float-64>>

      for i <- 1..100 do
        {:ok, _} = Replication.record_upsert("key#{i}", vector, %{}, 1)
      end

      assert Replication.status().oplog_size == 100

      {:ok, deleted} = Replication.compact(keep_last: 10)

      assert deleted == 90
      assert Replication.status().oplog_size == 10
    end
  end

  describe "export_snapshot/0 and import_snapshot/1" do
    test "export returns snapshot with vectors and metadata" do
      # Add some data
      vec1 = <<1.0::little-float-64, 2.0::little-float-64>>
      MerkleDb.KV.put("exp_key1", vec1)
      MerkleDb.PayloadStore.put("exp_key1", %{"test" => true})

      {:ok, snapshot} = Replication.export_snapshot()

      assert snapshot.type == :full_snapshot
      assert is_integer(snapshot.timestamp)
      assert is_map(snapshot.tree_stats)
    end
  end
end
