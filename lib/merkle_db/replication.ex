defmodule MerkleDb.Replication do
  @moduledoc """
  Replication and synchronization for MerkleDB.

  Provides append-only operation log (oplog) for tracking all changes,
  enabling snapshot-based full sync and incremental delta sync.

  ## Architecture

  1. **Oplog** - Append-only log of all operations (upsert, delete)
  2. **Sequence Numbers** - Monotonic IDs for ordering operations
  3. **Snapshots** - Full state export for initial sync
  4. **Deltas** - Incremental changes since a sequence number

  ## Usage

      # Record an operation
      Replication.record_upsert(key, vector, payload, version)

      # Get changes since sequence 1000
      {:ok, ops} = Replication.get_deltas(since: 1000)

      # Apply received operations
      Replication.apply_operations(ops)

      # Export snapshot
      {:ok, snapshot} = Replication.export_snapshot()
  """
  use GenServer

  alias MerkleDb.{KV, PayloadStore, TextStore, Tree}

  @oplog_table :replication_oplog
  @meta_table :replication_meta
  @dets_file "oplog.dets"

  defmodule Operation do
    @moduledoc "A single operation in the oplog"
    defstruct [:seq, :op, :key, :data, :timestamp]

    @type t :: %__MODULE__{
      seq: non_neg_integer(),
      op: :upsert | :delete,
      key: term(),
      data: map() | nil,
      timestamp: integer()
    }
  end

  # --- Client API ---

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Record an upsert operation.
  """
  @spec record_upsert(term(), binary(), map(), non_neg_integer()) :: {:ok, non_neg_integer()}
  def record_upsert(key, vector, payload, version) do
    GenServer.call(__MODULE__, {:record, :upsert, key, %{vector: vector, payload: payload, version: version}})
  end

  @doc """
  Record a delete operation.
  """
  @spec record_delete(term()) :: {:ok, non_neg_integer()}
  def record_delete(key) do
    GenServer.call(__MODULE__, {:record, :delete, key, nil})
  end

  @doc """
  Get operations since a sequence number.

  Options:
  - `:since` - Start sequence (exclusive, default 0)
  - `:limit` - Maximum operations to return (default 1000)
  """
  @spec get_deltas(keyword()) :: {:ok, [Operation.t()]}
  def get_deltas(opts \\ []) do
    GenServer.call(__MODULE__, {:get_deltas, opts})
  end

  @doc """
  Get current sequence number.
  """
  @spec current_seq() :: non_neg_integer()
  def current_seq do
    GenServer.call(__MODULE__, :current_seq)
  end

  @doc """
  Apply a list of operations from a remote replica.
  """
  @spec apply_operations([Operation.t() | map()]) :: {:ok, non_neg_integer()} | {:error, term()}
  def apply_operations(operations) do
    GenServer.call(__MODULE__, {:apply_operations, operations}, 60_000)
  end

  @doc """
  Export a full snapshot for initial sync.
  """
  @spec export_snapshot() :: {:ok, map()}
  def export_snapshot do
    GenServer.call(__MODULE__, :export_snapshot, 60_000)
  end

  @doc """
  Import a snapshot from a remote replica.
  """
  @spec import_snapshot(map()) :: :ok | {:error, term()}
  def import_snapshot(snapshot) do
    GenServer.call(__MODULE__, {:import_snapshot, snapshot}, 60_000)
  end

  @doc """
  Get replication status.
  """
  @spec status() :: map()
  def status do
    GenServer.call(__MODULE__, :status)
  end

  @doc """
  Compact the oplog by removing old entries.
  """
  @spec compact(keyword()) :: {:ok, non_neg_integer()}
  def compact(opts \\ []) do
    GenServer.call(__MODULE__, {:compact, opts})
  end

  # --- Server Callbacks ---

  @impl true
  def init(opts) do
    # Create ETS tables
    :ets.new(@oplog_table, [:named_table, :ordered_set, :public])
    :ets.new(@meta_table, [:named_table, :set, :public])

    # Initialize sequence counter
    :ets.insert(@meta_table, {:current_seq, 0})
    :ets.insert(@meta_table, {:created_at, System.system_time(:millisecond)})

    # Optionally load from DETS
    if Keyword.get(opts, :persist, false) do
      load_from_dets()
    end

    {:ok, %{persist: Keyword.get(opts, :persist, false)}}
  end

  @impl true
  def handle_call({:record, op, key, data}, _from, state) do
    seq = increment_seq()
    timestamp = System.system_time(:millisecond)

    operation = %Operation{
      seq: seq,
      op: op,
      key: key,
      data: data,
      timestamp: timestamp
    }

    :ets.insert(@oplog_table, {seq, operation})

    if state.persist do
      persist_operation(operation)
    end

    {:reply, {:ok, seq}, state}
  end

  @impl true
  def handle_call({:get_deltas, opts}, _from, state) do
    since = Keyword.get(opts, :since, 0)
    limit = Keyword.get(opts, :limit, 1000)

    # Get operations where seq > since
    operations =
      :ets.select(@oplog_table, [
        {{:"$1", :"$2"}, [{:>, :"$1", since}], [:"$2"]}
      ])
      |> Enum.sort_by(& &1.seq)
      |> Enum.take(limit)

    {:reply, {:ok, operations}, state}
  end

  @impl true
  def handle_call(:current_seq, _from, state) do
    [{:current_seq, seq}] = :ets.lookup(@meta_table, :current_seq)
    {:reply, seq, state}
  end

  @impl true
  def handle_call({:apply_operations, operations}, _from, state) do
    applied = apply_ops(operations)
    {:reply, {:ok, applied}, state}
  end

  @impl true
  def handle_call(:export_snapshot, _from, state) do
    tree = KV.snapshot()

    snapshot = %{
      type: :full_snapshot,
      timestamp: System.system_time(:millisecond),
      seq: get_current_seq(),
      tree_stats: Tree.stats(tree),
      vectors: export_vectors(tree),
      payloads: PayloadStore.get_all(),
      texts: TextStore.get_all()
    }

    {:reply, {:ok, snapshot}, state}
  end

  @impl true
  def handle_call({:import_snapshot, snapshot}, _from, state) do
    result = do_import_snapshot(snapshot)
    {:reply, result, state}
  end

  @impl true
  def handle_call(:status, _from, state) do
    [{:current_seq, seq}] = :ets.lookup(@meta_table, :current_seq)
    [{:created_at, created_at}] = :ets.lookup(@meta_table, :created_at)
    oplog_size = :ets.info(@oplog_table, :size)

    status = %{
      current_seq: seq,
      oplog_size: oplog_size,
      created_at: created_at,
      persist: state.persist
    }

    {:reply, status, state}
  end

  @impl true
  def handle_call({:compact, opts}, _from, state) do
    keep_count = Keyword.get(opts, :keep_last, 10_000)
    [{:current_seq, current_seq}] = :ets.lookup(@meta_table, :current_seq)

    cutoff = max(0, current_seq - keep_count)

    # Delete old entries
    deleted =
      :ets.select_delete(@oplog_table, [
        {{:"$1", :"$2"}, [{:"=<", :"$1", cutoff}], [true]}
      ])

    {:reply, {:ok, deleted}, state}
  end

  # --- Private Functions ---

  defp increment_seq do
    :ets.update_counter(@meta_table, :current_seq, {2, 1})
  end

  defp get_current_seq do
    [{:current_seq, seq}] = :ets.lookup(@meta_table, :current_seq)
    seq
  end

  defp apply_ops(operations) do
    Enum.reduce(operations, 0, fn op, count ->
      apply_single_op(op)
      count + 1
    end)
  end

  defp apply_single_op(%Operation{op: :upsert, key: key, data: data}) do
    vector = data.vector || data[:vector]
    payload = data.payload || data[:payload] || %{}

    if vector do
      :ok = KV.put(key, vector)
    end

    if payload && map_size(payload) > 0 do
      PayloadStore.put(key, payload)
    end
  end

  defp apply_single_op(%Operation{op: :delete, key: key}) do
    KV.delete(key)
    PayloadStore.delete(key)
  end

  defp apply_single_op(%{"op" => "upsert", "key" => key, "data" => data}) do
    apply_single_op(%Operation{op: :upsert, key: key, data: atomize_keys(data)})
  end

  defp apply_single_op(%{"op" => "delete", "key" => key}) do
    apply_single_op(%Operation{op: :delete, key: key, data: nil})
  end

  defp apply_single_op(_), do: :ignored

  defp atomize_keys(map) when is_map(map) do
    Map.new(map, fn
      {k, v} when is_binary(k) -> {String.to_existing_atom(k), v}
      {k, v} -> {k, v}
    end)
  rescue
    _ -> map
  end

  defp export_vectors(nil), do: []
  defp export_vectors(tree) do
    # Export all vectors with their keys
    if tree.count > 0 do
      Enum.map(0..(tree.count - 1), fn idx ->
        key = Map.get(tree.keys, idx)
        vector = get_vector_at_index(tree, idx)
        {key, vector}
      end)
    else
      []
    end
  end

  defp get_vector_at_index(tree, idx) do
    # Reconstruct vector from column-major storage
    dim = tree.dim

    Enum.map(0..(dim - 1), fn d ->
      col = elem(tree.columns, d)
      <<_::binary-size(idx * 8), val::little-float-size(64), _::binary>> = col
      val
    end)
    |> pack_f64()
  end

  defp pack_f64(values) do
    values
    |> Enum.map(fn v -> <<v::little-float-size(64)>> end)
    |> IO.iodata_to_binary()
  end

  defp do_import_snapshot(%{vectors: vectors, payloads: payloads, texts: texts}) do
    # Import vectors
    for {key, vector} <- vectors do
      KV.put(key, vector)
    end

    # Import payloads
    for {key, payload} <- payloads do
      PayloadStore.put(key, payload)
    end

    # Import texts
    for {key, text} <- texts do
      TextStore.put(key, text)
    end

    :ok
  end

  defp do_import_snapshot(_), do: {:error, :invalid_snapshot}

  # --- DETS Persistence ---

  defp load_from_dets do
    filename = String.to_charlist(@dets_file)

    case :dets.open_file(@oplog_table, file: filename, type: :set) do
      {:ok, _} ->
        :dets.to_ets(@oplog_table, @oplog_table)
        # Restore seq counter
        max_seq = :ets.foldl(fn {seq, _}, acc -> max(seq, acc) end, 0, @oplog_table)
        :ets.insert(@meta_table, {:current_seq, max_seq})
        :dets.close(@oplog_table)

      {:error, _} ->
        :ok
    end
  end

  defp persist_operation(operation) do
    filename = String.to_charlist(@dets_file)

    case :dets.open_file(@oplog_table, file: filename, type: :set) do
      {:ok, _} ->
        :dets.insert(@oplog_table, {operation.seq, operation})
        :dets.close(@oplog_table)

      {:error, _} ->
        :ok
    end
  end
end
