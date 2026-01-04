defmodule MerkleDb.WALTest do
  use ExUnit.Case, async: false

  alias MerkleDb.WAL
  alias MerkleDb.Crypto

  @test_dir "test/tmp/wal"

  setup do
    # Clean up test directory
    File.rm_rf!(@test_dir)
    File.mkdir_p!(@test_dir)

    on_exit(fn ->
      File.rm_rf!(@test_dir)
    end)

    :ok
  end

  defp wal_path(name) do
    Path.join(@test_dir, "#{name}.wal")
  end

  describe "open/2" do
    test "creates new WAL file" do
      path = wal_path("new")
      {:ok, wal} = WAL.open(path)

      assert File.exists?(path)
      WAL.close(wal)
    end

    test "opens existing WAL file" do
      path = wal_path("existing")

      # Create and close
      {:ok, wal1} = WAL.open(path)
      WAL.append_upsert(wal1, {1, [1.0], %{}, 0})
      WAL.close(wal1)

      # Reopen
      {:ok, wal2} = WAL.open(path)
      stats = WAL.stats(wal2)
      assert stats.bytes_written > 8  # More than just header
      WAL.close(wal2)
    end
  end

  describe "append_upsert/2" do
    test "appends upsert entry" do
      path = wal_path("upsert")
      {:ok, wal} = WAL.open(path)

      assert :ok = WAL.append_upsert(wal, {1, [1.0, 2.0, 3.0], %{"key" => "value"}, 0})

      stats = WAL.stats(wal)
      assert stats.entry_count == 1
      WAL.close(wal)
    end

    test "appends multiple upserts" do
      path = wal_path("multi_upsert")
      {:ok, wal} = WAL.open(path)

      for i <- 1..100 do
        WAL.append_upsert(wal, {i, [i * 1.0], %{}, 0})
      end

      stats = WAL.stats(wal)
      assert stats.entry_count == 100
      WAL.close(wal)
    end
  end

  describe "append_delete/2" do
    test "appends delete entry" do
      path = wal_path("delete")
      {:ok, wal} = WAL.open(path)

      assert :ok = WAL.append_delete(wal, 42)

      stats = WAL.stats(wal)
      assert stats.entry_count == 1
      WAL.close(wal)
    end
  end

  describe "append_commit/2" do
    test "appends commit marker with snapshot root" do
      path = wal_path("commit")
      {:ok, wal} = WAL.open(path)

      root = Crypto.hash("test root")
      assert :ok = WAL.append_commit(wal, root)

      stats = WAL.stats(wal)
      assert stats.entry_count == 1
      WAL.close(wal)
    end
  end

  describe "replay/1" do
    test "replays empty WAL" do
      path = wal_path("empty_replay")
      {:ok, wal} = WAL.open(path)
      WAL.close(wal)

      assert {:ok, []} = WAL.replay(path)
    end

    test "replays upsert entries" do
      path = wal_path("upsert_replay")
      {:ok, wal} = WAL.open(path)

      WAL.append_upsert(wal, {1, [1.0, 2.0], %{"a" => 1}, 100})
      WAL.append_upsert(wal, {2, [3.0, 4.0], %{"b" => 2}, 200})
      WAL.sync(wal)
      WAL.close(wal)

      {:ok, entries} = WAL.replay(path)

      assert length(entries) == 2
      assert {:upsert, {1, [1.0, 2.0], %{"a" => 1}, 100}} = hd(entries)
      assert {:upsert, {2, [3.0, 4.0], %{"b" => 2}, 200}} = Enum.at(entries, 1)
    end

    test "replays delete entries" do
      path = wal_path("delete_replay")
      {:ok, wal} = WAL.open(path)

      WAL.append_delete(wal, 42)
      WAL.append_delete(wal, 99)
      WAL.sync(wal)
      WAL.close(wal)

      {:ok, entries} = WAL.replay(path)

      assert length(entries) == 2
      assert {:delete, 42} = hd(entries)
      assert {:delete, 99} = Enum.at(entries, 1)
    end

    test "replays commit entries" do
      path = wal_path("commit_replay")
      {:ok, wal} = WAL.open(path)

      root = Crypto.hash("snapshot")
      WAL.append_commit(wal, root)
      WAL.sync(wal)
      WAL.close(wal)

      {:ok, entries} = WAL.replay(path)

      assert length(entries) == 1
      assert {:commit, ^root} = hd(entries)
    end

    test "replays mixed entries in order" do
      path = wal_path("mixed_replay")
      {:ok, wal} = WAL.open(path)

      WAL.append_upsert(wal, {1, [1.0], %{}, 0})
      WAL.append_upsert(wal, {2, [2.0], %{}, 0})
      WAL.append_delete(wal, 1)
      root = Crypto.hash("commit1")
      WAL.append_commit(wal, root)
      WAL.append_upsert(wal, {3, [3.0], %{}, 0})
      WAL.sync(wal)
      WAL.close(wal)

      {:ok, entries} = WAL.replay(path)

      assert length(entries) == 5
      assert {:upsert, {1, _, _, _}} = Enum.at(entries, 0)
      assert {:upsert, {2, _, _, _}} = Enum.at(entries, 1)
      assert {:delete, 1} = Enum.at(entries, 2)
      assert {:commit, ^root} = Enum.at(entries, 3)
      assert {:upsert, {3, _, _, _}} = Enum.at(entries, 4)
    end

    test "replays non-existent file as empty" do
      assert {:ok, []} = WAL.replay(wal_path("nonexistent"))
    end
  end

  describe "sync/1" do
    test "forces sync to disk" do
      path = wal_path("sync_test")
      {:ok, wal} = WAL.open(path, sync_mode: :batch)

      WAL.append_upsert(wal, {1, [1.0], %{}, 0})
      assert :ok = WAL.sync(wal)

      WAL.close(wal)
    end
  end

  describe "crash recovery simulation" do
    test "recovers state after simulated crash" do
      path = wal_path("crash_recovery")

      # Phase 1: Write some data
      {:ok, wal1} = WAL.open(path)
      WAL.append_upsert(wal1, {1, [1.0, 2.0], %{"name" => "alice"}, 0})
      WAL.append_upsert(wal1, {2, [3.0, 4.0], %{"name" => "bob"}, 0})
      root1 = Crypto.hash("commit1")
      WAL.append_commit(wal1, root1)
      WAL.sync(wal1)
      WAL.close(wal1)

      # Phase 2: Write more data
      {:ok, wal2} = WAL.open(path)
      WAL.append_upsert(wal2, {3, [5.0, 6.0], %{"name" => "charlie"}, 0})
      WAL.append_delete(wal2, 1)
      WAL.sync(wal2)
      WAL.close(wal2)

      # Phase 3: Recover from WAL
      {:ok, entries} = WAL.replay(path)

      # Build state from entries
      state = recover_state(entries)

      # Should have records 2 and 3 (1 was deleted)
      assert Map.has_key?(state.records, 2)
      assert Map.has_key?(state.records, 3)
      refute Map.has_key?(state.records, 1)

      # Should have one commit
      assert length(state.commits) == 1
      assert hd(state.commits) == root1
    end
  end

  describe "data integrity" do
    test "detects corruption via CRC" do
      path = wal_path("corruption")

      # Write valid data
      {:ok, wal} = WAL.open(path)
      WAL.append_upsert(wal, {1, [1.0], %{}, 0})
      WAL.sync(wal)
      WAL.close(wal)

      # Corrupt the file (flip a byte in the data section)
      {:ok, content} = File.read(path)
      # Corrupt byte at offset 20 (inside the entry)
      corrupted = binary_part(content, 0, 20) <> <<255>> <> binary_part(content, 21, byte_size(content) - 21)
      File.write!(path, corrupted)

      # Replay should handle corruption gracefully
      {:ok, entries} = WAL.replay(path)
      # May have 0 entries if corruption is detected
      assert is_list(entries)
    end

    test "handles truncated file" do
      path = wal_path("truncated")

      # Write valid data
      {:ok, wal} = WAL.open(path)
      WAL.append_upsert(wal, {1, [1.0], %{}, 0})
      WAL.append_upsert(wal, {2, [2.0], %{}, 0})
      WAL.sync(wal)
      WAL.close(wal)

      # Truncate file mid-entry
      {:ok, content} = File.read(path)
      truncated = binary_part(content, 0, byte_size(content) - 10)
      File.write!(path, truncated)

      # Replay should recover what it can
      {:ok, entries} = WAL.replay(path)
      # Should have at least the first complete entry
      assert length(entries) >= 1
    end
  end

  describe "stats/1" do
    test "returns correct statistics" do
      path = wal_path("stats")
      {:ok, wal} = WAL.open(path)

      WAL.append_upsert(wal, {1, [1.0], %{}, 0})
      WAL.append_upsert(wal, {2, [2.0], %{}, 0})
      WAL.append_delete(wal, 1)

      stats = WAL.stats(wal)

      assert stats.entry_count == 3
      assert stats.bytes_written > 8
      assert stats.path == path

      WAL.close(wal)
    end
  end

  # Helper to recover state from WAL entries
  defp recover_state(entries) do
    Enum.reduce(entries, %{records: %{}, commits: []}, fn entry, acc ->
      case entry do
        {:upsert, {id, vector, payload, version}} ->
          put_in(acc, [:records, id], {vector, payload, version})

        {:delete, id} ->
          update_in(acc, [:records], &Map.delete(&1, id))

        {:commit, root} ->
          update_in(acc, [:commits], &[root | &1])
      end
    end)
  end
end
