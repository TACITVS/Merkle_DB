defmodule MerkleDb.KV do
  @moduledoc """
  GenServer that acts as a proxy to the Raft-managed state.
  Ensures Strong Consistency by routing writes through Raft log.
  """
  use GenServer

  alias MerkleDb.{Tree, Persistence, WAL, Raft}

  require Logger

  # State is no longer needed here as truth lives in Raft machine
  def start_link(_), do: GenServer.start_link(__MODULE__, %{}, name: __MODULE__)

  @doc "Insert a single vector with optional metadata"
  def put(key, vector), do: put("default", key, vector, %{})
  def put(key, vector, metadata), do: put("default", key, vector, metadata)
  def put(collection, key, vector, metadata) do
    Raft.process_command({:put, collection, key, vector, metadata})
  end

  @doc "Batch insert vectors with optional metadata"
  def put_batch(key_vector_pairs), do: put_batch("default", key_vector_pairs)
  def put_batch(collection, key_vector_pairs) do
    Raft.process_command({:put_batch, collection, key_vector_pairs})
  end

  @doc "Delete a key (soft delete)"
  def delete(key), do: delete("default", key)
  def delete(collection, key) do
    Raft.process_command({:delete, collection, key})
  end

  @doc "Create a new collection with optional parameters (e.g. dim, precision)"
  def create_collection(name, opts \\ []) do
    Raft.process_command({:create_collection, name, opts})
  end

  @doc "Drop a collection"
  def drop_collection(name) do
    Raft.process_command({:drop_collection, name})
  end

  @doc "List all available collections"
  def list_collections do
    Raft.get_state() |> Map.keys()
  end

  @doc "Get tree snapshot for a collection"
  def snapshot, do: snapshot("default")
  def snapshot(collection) do
    Raft.get_state() |> Map.get(collection)
  end

  @doc "Replace entire tree"
  def set_tree(tree), do: set_tree("default", tree)
  def set_tree(collection, %Tree{} = tree) do
    retry_command({:set_tree, collection, tree}, 5)
  end

  def set_tree(collection, %Tree{} = tree) do
    retry_command({:set_tree, collection, tree}, 20)
  end

  defp retry_command(command, attempts) when attempts > 0 do
    case Raft.process_command(command) do
      :ok -> :ok
      {:error, :noproc} ->
        if rem(attempts, 5) == 0, do: Logger.debug("Waiting for Raft leader... (#{attempts} left)")
        Process.sleep(1000)
        retry_command(command, attempts - 1)
      err ->
        Logger.error("Raft command failed: #{inspect(err)}")
        err
    end
  end
  defp retry_command(_, _), do: {:error, :timeout}

  # ... other APIs simplified similarly ...

  @doc """
  Atomically update tree's IVF index if generation matches (optimistic locking).
  """
  def update_index(tree, expected_generation), do: update_index("default", tree, expected_generation)
  def update_index(collection, %Tree{} = new_tree, expected_generation) do
    GenServer.call(__MODULE__, {:update_index, collection, new_tree, expected_generation})
  end

  @doc "Get current tree generation"
  def generation, do: generation("default")
  def generation(collection) do
    GenServer.call(__MODULE__, {:generation, collection})
  end

  @doc "Reset a collection to empty state"
  def reset(collection \\ "default") do
    GenServer.call(__MODULE__, {:reset, collection})
  end

  @doc "Create a fast binary checkpoint for instant startup"
  def checkpoint(collection \\ "default") do
    GenServer.call(__MODULE__, {:checkpoint, collection})
  end

  # ==================== Callbacks ====================

  # ==================== Callbacks ====================
  @impl true
  def init(_) do
    # 1. Load all existing collections from disk
    collections = 
      Persistence.list_collections()
      |> Enum.reduce(%{}, fn name, acc ->
        # Try checkpoint first (V2 persistence), then snapshot (V1)
        tree_res = 
          case Persistence.load_checkpoint(name) do
            {:ok, tree} -> {:ok, tree}
            _ -> 
              case Persistence.load(collection: name) do
                {:ok, %{tree: tree}} -> {:ok, tree}
                _ -> :error
              end
          end

        case tree_res do
          {:ok, tree} -> Map.put(acc, name, tree)
          _ -> acc
        end
      end)
    
    # Ensure "default" exists
    collections = Map.put_new(collections, "default", Tree.new())

    # 2. Replay WAL to recover missing operations
    wal_path = Application.get_env(:merkle_db, :wal_path, "data/wal.bin")
    collections = 
      case WAL.replay(wal_path) do
        {:ok, entries} ->
          if length(entries) > 0 do
            Logger.info("Replaying #{length(entries)} WAL entries...")
            res = Enum.reduce(entries, collections, fn entry, acc -> apply_wal_entry(acc, entry) end)
            Logger.info("Replay finished.")
            res
          else
            collections
          end
        _ -> 
          collections
      end
    
    Logger.info("KV init finished.")
    {:ok, collections}
  end

  defp apply_wal_entry(collections, {:upsert, {collection, key, vector, metadata, version}}) do
    case get_tree(collections, collection) do
      {:ok, tree} ->
        if version > tree.last_wal_version do
          new_tree = Tree.insert(tree, key, vector, metadata)
          new_tree = %{new_tree | last_wal_version: version}
          Map.put(collections, collection, new_tree)
        else
          collections
        end
      _ ->
        # Create collection if it was mentioned in WAL but doesn't exist
        new_tree = Tree.insert(Tree.new(), key, vector, metadata)
        new_tree = %{new_tree | last_wal_version: version}
        Map.put(collections, collection, new_tree)
    end
  end

  defp apply_wal_entry(collections, {:delete, data}) do
    {collection, key, version} = case data do
      {c, k, v} -> {c, k, v}
      {c, k} -> {c, k, 0}
    end

    case get_tree(collections, collection) do
      {:ok, tree} ->
        if version == 0 or version > tree.last_wal_version do
          case Tree.delete(tree, key) do
            {:error, :not_found} -> collections
            new_tree -> 
              new_tree = if version > 0, do: %{new_tree | last_wal_version: version}, else: new_tree
              Map.put(collections, collection, new_tree)
          end
        else
          collections
        end
      _ -> collections
    end
  end

  defp apply_wal_entry(collections, {:commit, {_collection, _root}}) do
    collections
  end

  # --- Collection Management ---

  @impl true
  def handle_call({:create_collection, name, opts}, _from, collections) do
    if Map.has_key?(collections, name) do
      {:reply, {:error, :already_exists}, collections}
    else
      new_tree = Tree.new(opts)
      {:reply, :ok, Map.put(collections, name, new_tree)}
    end
  end

  @impl true
  def handle_call({:drop_collection, name}, _from, collections) do
    if name == "default" do
      # Cannot drop default, but can reset it?
      # For safety, let's just reset it to empty
      new_collections = Map.put(collections, "default", Tree.new())
      Persistence.delete("default")
      {:reply, :ok, new_collections}
    else
      new_collections = Map.delete(collections, name)
      Persistence.delete(name)
      {:reply, :ok, new_collections}
    end
  end

  @impl true
  def handle_call(:list_collections, _from, collections) do
    {:reply, Map.keys(collections), collections}
  end

  # --- Tree Operations ---

  @impl true
  def handle_call({:put, collection, key, vector, metadata}, _from, collections) do
    # 1. Log to WAL first for durability
    version = System.system_time(:microsecond)
    :ok = WAL.append_upsert(WAL, {collection, key, vector, metadata, version})

    with {:ok, tree} <- get_tree(collections, collection) do
      new_tree = Tree.insert(tree, key, vector, metadata)
      new_tree = %{new_tree | last_wal_version: version}
      
      check_auto_index(collection, new_tree)
      
      {:reply, :ok, Map.put(collections, collection, new_tree)}
    else
      err -> {:reply, err, collections}
    end
  end

  @impl true
  def handle_call({:put_batch, collection, pairs}, _from, collections) do
    # Log each pair to WAL without immediate sync
    version = System.system_time(:microsecond)
    Enum.each(pairs, fn
      {key, vec} -> WAL.append_upsert(WAL, {collection, key, vec, %{}, version}, sync: false)
      {key, vec, meta} -> WAL.append_upsert(WAL, {collection, key, vec, meta, version}, sync: false)
    end)
    
    # Sync WAL once for the whole batch
    WAL.sync()

    with {:ok, tree} <- get_tree(collections, collection) do
      new_tree = Tree.insert_batch(tree, pairs)
      new_tree = %{new_tree | last_wal_version: version}
      Logger.info("put_batch finished for #{collection}, count=#{length(pairs)}")
      
      check_auto_index(collection, new_tree)
      
      {:reply, :ok, Map.put(collections, collection, new_tree)}
    else
      err -> {:reply, err, collections}
    end
  end

  defp check_auto_index(collection, tree) do
    # Trigger indexing if > 1000 items and no index (HNSW or IVF)
    if tree.count > 1000 and tree.hnsw == nil and tree.centroids == nil do
      MerkleDb.IndexBuilder.trigger_auto_build(collection, tree.count)
    end
  end

  @impl true
  def handle_call({:delete, collection, key}, _from, collections) do
    # Log to WAL
    version = System.system_time(:microsecond)
    :ok = WAL.append_delete(WAL, {collection, key, version})

    with {:ok, tree} <- get_tree(collections, collection) do
      case Tree.delete(tree, key) do
        {:error, :not_found} ->
          {:reply, {:error, :not_found}, collections}
        new_tree ->
          new_tree = %{new_tree | last_wal_version: version}
          {:reply, :ok, Map.put(collections, collection, new_tree)}
      end
    else
      err -> {:reply, err, collections}
    end
  end

  @impl true
  def handle_call({:snapshot, collection}, _from, collections) do
    with {:ok, tree} <- get_tree(collections, collection) do
      {:reply, tree, collections}
    else
      err -> {:reply, err, collections}
    end
  end

  @impl true
  def handle_call({:set_tree, collection, new_tree}, _from, collections) do
    # Implicitly creates collection if not exists
    {:reply, :ok, Map.put(collections, collection, new_tree)}
  end

  @impl true
  def handle_call({:update_index, collection, new_tree, expected_gen}, _from, collections) do
    with {:ok, current_tree} <- get_tree(collections, collection) do
      if current_tree.generation == expected_gen do
        updated_tree = %{current_tree |
          centroids: new_tree.centroids,
          clusters: new_tree.clusters,
          hnsw: new_tree.hnsw, # Also update HNSW if present
          generation: current_tree.generation + 1
        }
        {:reply, :ok, Map.put(collections, collection, updated_tree)}
      else
        {:reply, {:error, :generation_mismatch}, collections}
      end
    else
      err -> {:reply, err, collections}
    end
  end

  @impl true
  def handle_call({:generation, collection}, _from, collections) do
    with {:ok, tree} <- get_tree(collections, collection) do
      {:reply, tree.generation, collections}
    else
      err -> {:reply, err, collections}
    end
  end

  @impl true
  def handle_call({:reset, collection}, _from, collections) do
    if Map.has_key?(collections, collection) do
      new_collections = Map.put(collections, collection, Tree.new())
      Persistence.delete(collection)
      {:reply, :ok, new_collections}
    else
      {:reply, {:error, :collection_not_found}, collections}
    end
  end

  @impl true
  def handle_call({:checkpoint, collection}, _from, collections) do
    with {:ok, tree} <- get_tree(collections, collection) do
      # 1. Flush WAL (ensure everything on disk is in the tree) - implicitly handled by memory state
      # 2. Save Checkpoint
      case Persistence.save_checkpoint(tree, collection) do
        {:ok, dir} -> 
          # 3. Truncate WAL logic would go here (update last_persisted_version)
          Logger.info("Checkpoint saved to #{dir}")
          {:reply, :ok, collections}
        err -> 
          {:reply, err, collections}
      end
    else
      err -> {:reply, err, collections}
    end
  end

  defp get_tree(collections, name) do
    case Map.get(collections, name) do
      nil -> {:error, :collection_not_found}
      tree -> {:ok, tree}
    end
  end
end
