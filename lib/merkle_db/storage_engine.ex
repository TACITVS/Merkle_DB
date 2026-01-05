defmodule MerkleDb.StorageEngine do
  @moduledoc """
  LSM-tree style storage engine for MerkleDB.

  Combines:
  - WAL for durability (all writes logged before applied)
  - Memtable for fast writes (in-memory sorted map)
  - Segments for persistent storage (immutable sorted files)

  ## Write Path
  1. Append to WAL
  2. Insert into memtable
  3. When memtable exceeds threshold, flush to segment
  4. Periodically compact segments

  ## Read Path
  1. Check memtable first (newest data)
  2. Check segments from newest to oldest
  3. Return first match or :not_found

  ## Recovery
  1. Load segment manifest
  2. Replay WAL from last checkpoint
  3. Rebuild memtable
  """

  use GenServer

  require Logger

  alias MerkleDb.{WAL, Segment, Snapshot, SnapshotStore, Crypto}

  @memtable_flush_threshold 10_000  # Records before flush
  @compaction_threshold 4           # Segments before compaction

  defmodule State do
    @moduledoc false
    defstruct [
      :data_dir,
      :wal,
      :memtable,           # %{id => {vector, payload, version, deleted?}}
      :segments,           # [{path, min_id, max_id, record_count}]
      :manifest_path,
      :next_segment_id,
      :record_count,
      :last_commit_root
    ]
  end

  # Client API

  @doc """
  Start the storage engine.

  Options:
  - :data_dir - directory for all data files (required)
  - :wal_sync - :sync or :batch (default: :batch)
  """
  def start_link(opts) do
    data_dir = Keyword.fetch!(opts, :data_dir)
    GenServer.start_link(__MODULE__, opts, name: via_name(data_dir))
  end

  defp via_name(data_dir) do
    {:via, Registry, {MerkleDb.StorageRegistry, data_dir}}
  end

  @doc """
  Get or start engine for a data directory.
  """
  def get_or_start(data_dir, opts \\ []) do
    opts = Keyword.put(opts, :data_dir, data_dir)

    case start_link(opts) do
      {:ok, pid} -> {:ok, pid}
      {:error, {:already_started, pid}} -> {:ok, pid}
      error -> error
    end
  end

  @doc """
  Insert or update a record.
  """
  def upsert(engine, id, vector, payload, version \\ nil) do
    version = version || System.system_time(:microsecond)
    GenServer.call(engine, {:upsert, id, vector, payload, version})
  end

  @doc """
  Delete a record.
  """
  def delete(engine, id) do
    GenServer.call(engine, {:delete, id})
  end

  @doc """
  Get a record by ID.
  """
  def get(engine, id) do
    GenServer.call(engine, {:get, id})
  end

  @doc """
  Commit current state and return snapshot root.
  """
  def commit(engine) do
    GenServer.call(engine, :commit, 30_000)
  end

  @doc """
  Get all records (for small datasets or testing).
  """
  def get_all(engine) do
    GenServer.call(engine, :get_all, 30_000)
  end

  @doc """
  Force flush memtable to segment.
  """
  def flush(engine) do
    GenServer.call(engine, :flush, 30_000)
  end

  @doc """
  Force compaction of segments.
  """
  def compact(engine) do
    GenServer.call(engine, :compact, 60_000)
  end

  @doc """
  Get engine statistics.
  """
  def stats(engine) do
    GenServer.call(engine, :stats)
  end

  @doc """
  Close the engine gracefully.
  """
  def close(engine) do
    GenServer.call(engine, :close)
  end

  # Snapshot API

  @doc """
  Generate an inclusion proof for a record in a snapshot.
  """
  def prove_inclusion(engine, record_id, snapshot_root \\ nil) do
    GenServer.call(engine, {:prove_inclusion, record_id, snapshot_root})
  end

  @doc """
  Get a record with its inclusion proof.
  Returns both the record and a proof that can be verified client-side.
  """
  def get_with_proof(engine, record_id, snapshot_root \\ nil) do
    GenServer.call(engine, {:get_with_proof, record_id, snapshot_root})
  end

  @doc """
  List all snapshots.
  """
  def list_snapshots(engine) do
    GenServer.call(engine, :list_snapshots)
  end

  @doc """
  Get snapshot info by root.
  """
  def get_snapshot_info(engine, snapshot_root) do
    GenServer.call(engine, {:get_snapshot_info, snapshot_root})
  end

  @doc """
  Garbage collect old snapshots.
  """
  def gc_snapshots(engine, opts \\ []) do
    GenServer.call(engine, {:gc_snapshots, opts})
  end

  @doc """
  Verify an inclusion proof (static function, no engine needed).
  Can be called client-side with no database access.
  """
  def verify_proof(record, proof) do
    Snapshot.verify_inclusion(record, proof)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    data_dir = Keyword.fetch!(opts, :data_dir)
    wal_sync = Keyword.get(opts, :wal_sync, :batch)

    # Ensure directories exist
    File.mkdir_p!(data_dir)
    File.mkdir_p!(Path.join(data_dir, "segments"))

    # Initialize or recover state
    case recover(data_dir, wal_sync) do
      {:ok, state} ->
        Logger.info("StorageEngine started: #{data_dir}, #{state.record_count} records")
        {:ok, state}

      {:error, reason} ->
        {:stop, reason}
    end
  end

  @impl true
  def handle_call({:upsert, id, vector, payload, version}, _from, state) do
    # Write to WAL first
    :ok = WAL.append_upsert(state.wal, {"default", id, vector, payload, version})

    # Update memtable
    new_memtable = Map.put(state.memtable, id, {vector, payload, version, false})
    new_state = %{state | memtable: new_memtable, record_count: state.record_count + 1}

    # Check if flush needed
    new_state = maybe_flush(new_state)

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:delete, id}, _from, state) do
    # Write to WAL
    :ok = WAL.append_delete(state.wal, {"default", id})

    # Mark as deleted in memtable (tombstone)
    new_memtable = Map.put(state.memtable, id, {[], %{}, 0, true})
    new_state = %{state | memtable: new_memtable}

    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call({:get, id}, _from, state) do
    result = do_get(id, state)
    {:reply, result, state}
  end

  @impl true
  def handle_call(:get_all, _from, state) do
    records = collect_all_records(state)
    {:reply, {:ok, records}, state}
  end

  @impl true
  def handle_call(:commit, _from, state) do
    case do_commit(state) do
      {:ok, snapshot_root, new_state} ->
        {:reply, {:ok, snapshot_root}, new_state}

      {:error, reason} ->
        {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:flush, _from, state) do
    case do_flush(state) do
      {:ok, new_state} -> {:reply, :ok, new_state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:compact, _from, state) do
    case do_compact(state) do
      {:ok, new_state} -> {:reply, :ok, new_state}
      {:error, reason} -> {:reply, {:error, reason}, state}
    end
  end

  @impl true
  def handle_call(:stats, _from, state) do
    stats = %{
      data_dir: state.data_dir,
      memtable_size: map_size(state.memtable),
      segment_count: length(state.segments),
      record_count: state.record_count,
      last_commit: state.last_commit_root && Crypto.to_hex(state.last_commit_root)
    }
    {:reply, stats, state}
  end

  @impl true
  def handle_call(:close, _from, state) do
    # Flush memtable
    {:ok, state} = do_flush(state)

    # Sync and close WAL
    WAL.sync(state.wal)
    WAL.close(state.wal)

    # Save manifest
    save_manifest(state)

    {:stop, :normal, :ok, state}
  end

  @impl true
  def handle_call({:prove_inclusion, record_id, snapshot_root}, _from, state) do
    root = snapshot_root || state.last_commit_root

    if is_nil(root) do
      {:reply, {:error, :no_snapshot}, state}
    else
      result = SnapshotStore.prove_inclusion(state.data_dir, root, record_id)
      {:reply, result, state}
    end
  end

  @impl true
  def handle_call({:get_with_proof, record_id, snapshot_root}, _from, state) do
    root = snapshot_root || state.last_commit_root

    if is_nil(root) do
      {:reply, {:error, :no_snapshot}, state}
    else
      with {:ok, record} <- do_get(record_id, state),
           {:ok, proof} <- SnapshotStore.prove_inclusion(state.data_dir, root, record_id) do
        {:reply, {:ok, record, proof}, state}
      else
        error -> {:reply, error, state}
      end
    end
  end

  @impl true
  def handle_call(:list_snapshots, _from, state) do
    result = SnapshotStore.list(state.data_dir)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:get_snapshot_info, snapshot_root}, _from, state) do
    result = SnapshotStore.info(state.data_dir, snapshot_root)
    {:reply, result, state}
  end

  @impl true
  def handle_call({:gc_snapshots, opts}, _from, state) do
    result = SnapshotStore.gc(state.data_dir, opts)
    {:reply, result, state}
  end

  # Internal Functions

  defp recover(data_dir, wal_sync) do
    manifest_path = Path.join(data_dir, "manifest.bin")
    wal_path = Path.join(data_dir, "wal.log")

    # Load manifest if exists
    {segments, next_segment_id, last_commit} = load_manifest(manifest_path)

    # Open or create WAL
    {:ok, wal} = WAL.open(wal_path, sync_mode: wal_sync)

    # Replay WAL to rebuild memtable
    {:ok, entries} = WAL.replay(wal_path)
    memtable = replay_to_memtable(entries)

    # Count records (memtable + segments)
    segment_count = Enum.reduce(segments, 0, fn {_, _, _, count}, acc -> acc + count end)
    memtable_count = Enum.count(memtable, fn {_, {_, _, _, deleted}} -> not deleted end)

    state = %State{
      data_dir: data_dir,
      wal: wal,
      memtable: memtable,
      segments: segments,
      manifest_path: manifest_path,
      next_segment_id: next_segment_id,
      record_count: segment_count + memtable_count,
      last_commit_root: last_commit
    }

    {:ok, state}
  end

  defp load_manifest(path) do
    case File.read(path) do
      {:ok, data} ->
        try do
          %{segments: segments, next_id: next_id, last_commit: last_commit} =
            :erlang.binary_to_term(data, [:safe])
          {segments, next_id, last_commit}
        rescue
          _ -> {[], 1, nil}
        end

      {:error, :enoent} ->
        {[], 1, nil}
    end
  end

  defp save_manifest(state) do
    data = :erlang.term_to_binary(%{
      segments: state.segments,
      next_id: state.next_segment_id,
      last_commit: state.last_commit_root
    })

    # Atomic write: write to temp, then rename
    temp_path = state.manifest_path <> ".tmp"
    File.write!(temp_path, data)
    File.rename!(temp_path, state.manifest_path)
  end

  defp replay_to_memtable(entries) do
    Enum.reduce(entries, %{}, fn entry, acc ->
      case entry do
        {:upsert, {_collection, id, vector, payload, version}} ->
          Map.put(acc, id, {vector, payload, version, false})

        {:delete, {_collection, id}} ->
          Map.put(acc, id, {[], %{}, 0, true})

        {:commit, _root} ->
          # Commit markers don't affect memtable
          acc
      end
    end)
  end

  defp do_get(id, state) do
    # Check memtable first
    case Map.get(state.memtable, id) do
      {_vector, _payload, _version, true} ->
        # Deleted
        {:error, :not_found}

      {vector, payload, version, false} ->
        {:ok, {id, vector, payload, version}}

      nil ->
        # Check segments (newest to oldest)
        search_segments(id, state.segments)
    end
  end

  defp search_segments(_id, []), do: {:error, :not_found}

  defp search_segments(id, [{path, min_id, max_id, _count} | rest]) do
    if id >= min_id and id <= max_id do
      case Segment.read_record(path, id) do
        {:ok, record} ->
          {:ok, {record.id, record.vector, record.payload, record.version}}

        {:error, :not_found} ->
          search_segments(id, rest)
      end
    else
      search_segments(id, rest)
    end
  end

  defp collect_all_records(state) do
    # Collect from segments (oldest to newest)
    segment_records =
      state.segments
      |> Enum.reverse()
      |> Enum.reduce(%{}, fn {path, _, _, _}, acc ->
        Segment.scan(path, fn record ->
          send(self(), {:segment_record, record})
        end)

        receive_segment_records(acc)
      end)

    # Overlay memtable (newest)
    all_records =
      Enum.reduce(state.memtable, segment_records, fn {id, {vector, payload, version, deleted}}, acc ->
        if deleted do
          Map.delete(acc, id)
        else
          Map.put(acc, id, {id, vector, payload, version})
        end
      end)

    Map.values(all_records)
  end

  defp receive_segment_records(acc) do
    receive do
      {:segment_record, record} ->
        new_acc = Map.put(acc, record.id, {record.id, record.vector, record.payload, record.version})
        receive_segment_records(new_acc)
    after
      0 -> acc
    end
  end

  defp do_commit(state) do
    # Get all current records
    records = collect_all_records(state)
    |> Enum.map(fn {id, vector, payload, version} -> {id, vector, payload, version} end)

    # Create snapshot
    case Snapshot.commit(records, collection_name: "default") do
      {:ok, snapshot_state} ->
        root = Snapshot.root(snapshot_state)

        # Save snapshot to store (for proof generation)
        :ok = SnapshotStore.save(state.data_dir, snapshot_state)

        # Write commit marker to WAL
        :ok = WAL.append_commit(state.wal, {"default", root})
        :ok = WAL.sync(state.wal)

        new_state = %{state | last_commit_root: root}

        Logger.info("Committed snapshot: #{Crypto.to_hex(root) |> String.slice(0, 16)}...")

        {:ok, root, new_state}

      error ->
        error
    end
  end

  defp maybe_flush(state) do
    if map_size(state.memtable) >= @memtable_flush_threshold do
      case do_flush(state) do
        {:ok, new_state} -> maybe_compact(new_state)
        {:error, _} -> state
      end
    else
      state
    end
  end

  defp do_flush(%{memtable: memtable} = state) when map_size(memtable) == 0 do
    {:ok, state}
  end

  defp do_flush(state) do
    # Convert memtable to records (excluding tombstones)
    records =
      state.memtable
      |> Enum.filter(fn {_, {_, _, _, deleted}} -> not deleted end)
      |> Enum.map(fn {id, {vector, payload, version, _}} -> {id, vector, payload, version} end)

    if records == [] do
      # Only tombstones, just clear memtable
      {:ok, %{state | memtable: %{}}}
    else
      # Write to new segment
      segment_path = segment_path(state.data_dir, state.next_segment_id)

      case Segment.write(segment_path, records) do
        {:ok, info} ->
          # Update state
          new_segment = {segment_path, info.min_id, info.max_id, info.record_count}
          new_segments = [new_segment | state.segments]

          new_state = %{state |
            memtable: %{},
            segments: new_segments,
            next_segment_id: state.next_segment_id + 1
          }

          # Save manifest
          save_manifest(new_state)

          # Truncate WAL (we can replay from segments now)
          :ok = WAL.reset(state.wal)

          Logger.info("Flushed memtable to segment #{state.next_segment_id}: #{info.record_count} records")

          {:ok, new_state}

        error ->
          error
      end
    end
  end

  defp maybe_compact(state) do
    if length(state.segments) >= @compaction_threshold do
      case do_compact(state) do
        {:ok, new_state} -> new_state
        {:error, _} -> state
      end
    else
      state
    end
  end

  defp do_compact(%{segments: segments} = state) when length(segments) < 2 do
    {:ok, state}
  end

  defp do_compact(state) do
    # Get deleted IDs from memtable (tombstones)
    deleted_ids = MapSet.new(
      state.memtable
      |> Enum.filter(fn {_, {_, _, _, deleted}} -> deleted end)
      |> Enum.map(fn {id, _} -> id end)
    )

    # Filter function to exclude deleted records
    filter_fn = fn {id, _, _, _} -> not MapSet.member?(deleted_ids, id) end

    # Merge all segments
    segment_paths = Enum.map(state.segments, fn {path, _, _, _} -> path end)
    new_segment_path = segment_path(state.data_dir, state.next_segment_id)

    case Segment.merge(new_segment_path, segment_paths, filter_fn) do
      {:ok, info} ->
        # Remove old segments
        Enum.each(segment_paths, &File.rm/1)

        new_segment = {new_segment_path, info.min_id, info.max_id, info.record_count}

        new_state = %{state |
          segments: [new_segment],
          next_segment_id: state.next_segment_id + 1,
          memtable: %{}  # Clear tombstones after compaction
        }

        save_manifest(new_state)

        Logger.info("Compacted #{length(segment_paths)} segments into 1: #{info.record_count} records")

        {:ok, new_state}

      {:error, :empty_result} ->
        # All records were deleted
        Enum.each(segment_paths, &File.rm/1)

        new_state = %{state |
          segments: [],
          next_segment_id: state.next_segment_id,
          memtable: %{},
          record_count: 0
        }

        save_manifest(new_state)
        {:ok, new_state}

      error ->
        error
    end
  end

  defp segment_path(data_dir, segment_id) do
    Path.join([data_dir, "segments", "seg_#{String.pad_leading(Integer.to_string(segment_id), 8, "0")}.seg"])
  end
end
