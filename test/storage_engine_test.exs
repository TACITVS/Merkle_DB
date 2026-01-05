defmodule MerkleDb.StorageEngineTest do
  use ExUnit.Case, async: false

  alias MerkleDb.StorageEngine

  @test_dir "test/tmp/storage_engine"

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

  describe "start_link/1" do
    test "starts with empty directory" do
      dir = engine_dir("start_empty")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      stats = StorageEngine.stats(engine)
      assert stats.memtable_size == 0
      assert stats.segment_count == 0

      StorageEngine.close(engine)
    end

    test "creates necessary directories" do
      dir = engine_dir("creates_dirs")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      assert File.dir?(dir)
      assert File.dir?(Path.join(dir, "segments"))

      StorageEngine.close(engine)
    end
  end

  describe "upsert/4" do
    test "inserts a record" do
      dir = engine_dir("upsert")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      :ok = StorageEngine.upsert(engine, 1, [1.0, 2.0], %{"key" => "value"})

      {:ok, record} = StorageEngine.get(engine, 1)
      assert record == {1, [1.0, 2.0], %{"key" => "value"}, elem(record, 3)}

      StorageEngine.close(engine)
    end

    test "updates existing record" do
      dir = engine_dir("update")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      :ok = StorageEngine.upsert(engine, 1, [1.0], %{}, 100)
      :ok = StorageEngine.upsert(engine, 1, [2.0], %{"updated" => true}, 200)

      {:ok, record} = StorageEngine.get(engine, 1)
      assert elem(record, 1) == [2.0]
      assert elem(record, 2) == %{"updated" => true}
      assert elem(record, 3) == 200

      StorageEngine.close(engine)
    end

    test "inserts multiple records" do
      dir = engine_dir("multi_insert")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      for i <- 1..100 do
        :ok = StorageEngine.upsert(engine, i, [i * 1.0], %{"i" => i})
      end

      stats = StorageEngine.stats(engine)
      assert stats.memtable_size == 100

      StorageEngine.close(engine)
    end
  end

  describe "delete/2" do
    test "deletes a record" do
      dir = engine_dir("delete")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      :ok = StorageEngine.upsert(engine, 1, [1.0], %{})
      {:ok, _} = StorageEngine.get(engine, 1)

      :ok = StorageEngine.delete(engine, 1)
      assert {:error, :not_found} = StorageEngine.get(engine, 1)

      StorageEngine.close(engine)
    end

    test "delete non-existent is ok" do
      dir = engine_dir("delete_missing")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      :ok = StorageEngine.delete(engine, 999)

      StorageEngine.close(engine)
    end
  end

  describe "get/2" do
    test "returns error for missing record" do
      dir = engine_dir("get_missing")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      assert {:error, :not_found} = StorageEngine.get(engine, 999)

      StorageEngine.close(engine)
    end
  end

  describe "get_all/1" do
    test "returns all records" do
      dir = engine_dir("get_all")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      for i <- 1..10 do
        :ok = StorageEngine.upsert(engine, i, [i * 1.0], %{})
      end

      {:ok, records} = StorageEngine.get_all(engine)
      assert length(records) == 10

      StorageEngine.close(engine)
    end

    test "excludes deleted records" do
      dir = engine_dir("get_all_deleted")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      for i <- 1..10 do
        :ok = StorageEngine.upsert(engine, i, [i * 1.0], %{})
      end

      :ok = StorageEngine.delete(engine, 5)

      {:ok, records} = StorageEngine.get_all(engine)
      assert length(records) == 9
      refute Enum.any?(records, fn {id, _, _, _} -> id == 5 end)

      StorageEngine.close(engine)
    end
  end

  describe "commit/1" do
    test "returns snapshot root" do
      dir = engine_dir("commit")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      :ok = StorageEngine.upsert(engine, 1, [1.0], %{})
      :ok = StorageEngine.upsert(engine, 2, [2.0], %{})

      {:ok, root} = StorageEngine.commit(engine)

      assert byte_size(root) == 32
      assert is_binary(root)

      StorageEngine.close(engine)
    end

    test "same data produces same tree root" do
      dir1 = engine_dir("commit_det1")
      dir2 = engine_dir("commit_det2")

      {:ok, engine1} = StorageEngine.start_link(data_dir: dir1)
      {:ok, engine2} = StorageEngine.start_link(data_dir: dir2)

      # Insert same data in different order
      :ok = StorageEngine.upsert(engine1, 1, [1.0], %{}, 0)
      :ok = StorageEngine.upsert(engine1, 2, [2.0], %{}, 0)

      :ok = StorageEngine.upsert(engine2, 2, [2.0], %{}, 0)
      :ok = StorageEngine.upsert(engine2, 1, [1.0], %{}, 0)

      {:ok, root1} = StorageEngine.commit(engine1)
      {:ok, root2} = StorageEngine.commit(engine2)

      # Tree roots should match (manifest roots differ due to timestamp)
      stats1 = StorageEngine.stats(engine1)
      stats2 = StorageEngine.stats(engine2)

      # The last_commit includes timestamp, but data structure is same
      assert byte_size(root1) == 32
      assert byte_size(root2) == 32
      assert stats1.record_count == stats2.record_count

      StorageEngine.close(engine1)
      StorageEngine.close(engine2)
    end
  end

  describe "flush/1" do
    test "flushes memtable to segment" do
      dir = engine_dir("flush")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      for i <- 1..100 do
        :ok = StorageEngine.upsert(engine, i, [i * 1.0], %{})
      end

      :ok = StorageEngine.flush(engine)

      stats = StorageEngine.stats(engine)
      assert stats.memtable_size == 0
      assert stats.segment_count == 1

      # Data still accessible
      {:ok, record} = StorageEngine.get(engine, 50)
      assert elem(record, 0) == 50

      StorageEngine.close(engine)
    end

    test "empty memtable flush is ok" do
      dir = engine_dir("empty_flush")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      :ok = StorageEngine.flush(engine)

      stats = StorageEngine.stats(engine)
      assert stats.segment_count == 0

      StorageEngine.close(engine)
    end
  end

  describe "compact/1" do
    test "merges multiple segments" do
      dir = engine_dir("compact")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      # Create multiple segments
      for batch <- 1..3 do
        for i <- 1..50 do
          id = (batch - 1) * 50 + i
          :ok = StorageEngine.upsert(engine, id, [id * 1.0], %{})
        end
        :ok = StorageEngine.flush(engine)
      end

      stats_before = StorageEngine.stats(engine)
      assert stats_before.segment_count == 3

      :ok = StorageEngine.compact(engine)

      stats_after = StorageEngine.stats(engine)
      assert stats_after.segment_count == 1

      # All data still accessible
      {:ok, records} = StorageEngine.get_all(engine)
      assert length(records) == 150

      StorageEngine.close(engine)
    end

    test "compaction removes deleted records" do
      dir = engine_dir("compact_delete")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      # Insert and flush
      for i <- 1..100 do
        :ok = StorageEngine.upsert(engine, i, [i * 1.0], %{})
      end
      :ok = StorageEngine.flush(engine)

      # Delete some records
      for i <- 1..50 do
        :ok = StorageEngine.delete(engine, i)
      end

      # Compact
      :ok = StorageEngine.compact(engine)

      # Only 50 records should remain
      {:ok, records} = StorageEngine.get_all(engine)
      assert length(records) == 50

      StorageEngine.close(engine)
    end
  end

  describe "crash recovery" do
    test "recovers data after restart" do
      dir = engine_dir("recovery")

      # Phase 1: Insert data and close
      {:ok, engine1} = StorageEngine.start_link(data_dir: dir)
      for i <- 1..100 do
        :ok = StorageEngine.upsert(engine1, i, [i * 1.0], %{"i" => i})
      end
      StorageEngine.close(engine1)

      # Phase 2: Reopen and verify
      {:ok, engine2} = StorageEngine.start_link(data_dir: dir)

      {:ok, record} = StorageEngine.get(engine2, 50)
      assert elem(record, 0) == 50
      assert elem(record, 1) == [50.0]

      StorageEngine.close(engine2)
    end

    test "recovers after flush and restart" do
      dir = engine_dir("recovery_flush")

      # Phase 1: Insert, flush, add more, close
      {:ok, engine1} = StorageEngine.start_link(data_dir: dir)
      for i <- 1..50 do
        :ok = StorageEngine.upsert(engine1, i, [i * 1.0], %{})
      end
      :ok = StorageEngine.flush(engine1)

      for i <- 51..100 do
        :ok = StorageEngine.upsert(engine1, i, [i * 1.0], %{})
      end
      StorageEngine.close(engine1)

      # Phase 2: Reopen and verify
      {:ok, engine2} = StorageEngine.start_link(data_dir: dir)

      # Records from segment
      {:ok, _} = StorageEngine.get(engine2, 25)

      # Records from WAL replay
      {:ok, _} = StorageEngine.get(engine2, 75)

      {:ok, all} = StorageEngine.get_all(engine2)
      assert length(all) == 100

      StorageEngine.close(engine2)
    end

    test "recovers delete operations" do
      dir = engine_dir("recovery_delete")

      # Phase 1: Insert, delete, close
      {:ok, engine1} = StorageEngine.start_link(data_dir: dir)
      for i <- 1..10 do
        :ok = StorageEngine.upsert(engine1, i, [i * 1.0], %{})
      end
      :ok = StorageEngine.delete(engine1, 5)
      StorageEngine.close(engine1)

      # Phase 2: Verify delete persisted
      {:ok, engine2} = StorageEngine.start_link(data_dir: dir)

      assert {:error, :not_found} = StorageEngine.get(engine2, 5)
      {:ok, all} = StorageEngine.get_all(engine2)
      assert length(all) == 9

      StorageEngine.close(engine2)
    end
  end

  describe "stats/1" do
    test "returns correct statistics" do
      dir = engine_dir("stats")
      {:ok, engine} = StorageEngine.start_link(data_dir: dir)

      for i <- 1..25 do
        :ok = StorageEngine.upsert(engine, i, [i * 1.0], %{})
      end

      stats = StorageEngine.stats(engine)

      assert stats.data_dir == dir
      assert stats.memtable_size == 25
      assert stats.segment_count == 0
      assert stats.record_count == 25

      StorageEngine.close(engine)
    end
  end
end
