defmodule MerkleDb.KV do
  @moduledoc """
  GenServer managing the current Tree state with optimistic locking support.
  """
  use GenServer

  alias MerkleDb.{Tree, Persistence, WAL}

  require Logger

  # State: map of collection_name -> %Tree{}
  def start_link(_), do: GenServer.start_link(__MODULE__, %{}, name: __MODULE__)

  @doc "Insert a single vector with optional metadata"
  def put(key, vector), do: put("default", key, vector, %{})
  def put(key, vector, metadata), do: put("default", key, vector, metadata)
  def put(collection, key, vector, metadata) do
    GenServer.call(__MODULE__, {:put, collection, key, vector, metadata})
  end

  @doc "Batch insert vectors with optional metadata"
  def put_batch(key_vector_pairs), do: put_batch("default", key_vector_pairs)
  def put_batch(collection, key_vector_pairs) do
    GenServer.call(__MODULE__, {:put_batch, collection, key_vector_pairs})
  end

  @doc "Delete a key (soft delete)"
  def delete(key), do: delete("default", key)
  def delete(collection, key) do
    GenServer.call(__MODULE__, {:delete, collection, key})
  end

  @doc "Create a new collection"
  def create_collection(name) do
    GenServer.call(__MODULE__, {:create_collection, name})
  end

  @doc "Drop a collection (delete from memory and disk)"
  def drop_collection(name) do
    GenServer.call(__MODULE__, {:drop_collection, name})
  end

  @doc "List all available collections"
  def list_collections do
    GenServer.call(__MODULE__, :list_collections)
  end

  @doc "Get tree snapshot for a collection"
  def snapshot, do: snapshot("default")
  def snapshot(collection) do
    GenServer.call(__MODULE__, {:snapshot, collection})
  end

  @doc "Replace entire tree (used by Bootstrap)"
  def set_tree(tree), do: set_tree("default", tree)
  def set_tree(collection, %Tree{} = tree) do
    GenServer.call(__MODULE__, {:set_tree, collection, tree})
  end

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

  # ==================== Callbacks ====================

  # ==================== Callbacks ====================
  @impl true
  def init(_) do
    # 1. Load all existing collections from disk
    collections = 
      Persistence.list_collections()
      |> Enum.reduce(%{}, fn name, acc ->
        case Persistence.load(collection: name) do
          {:ok, %{tree: tree}} -> Map.put(acc, name, tree)
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
            Enum.reduce(entries, collections, fn entry, acc -> apply_wal_entry(acc, entry) end)
          else
            collections
          end
        _ -> 
          collections
      end
    
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
  def handle_call({:create_collection, name}, _from, collections) do
    if Map.has_key?(collections, name) do
      {:reply, {:error, :already_exists}, collections}
    else
      new_tree = Tree.new()
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
    :ok = WAL.append_upsert({collection, key, vector, metadata, version})

    with {:ok, tree} <- get_tree(collections, collection) do
      new_tree = Tree.insert(tree, key, vector, metadata)
      new_tree = %{new_tree | last_wal_version: version}
      {:reply, :ok, Map.put(collections, collection, new_tree)}
    else
      err -> {:reply, err, collections}
    end
  end

  @impl true
  def handle_call({:put_batch, collection, pairs}, _from, collections) do
    # Log each pair to WAL
    version = System.system_time(:microsecond)
    Enum.each(pairs, fn
      {key, vec} -> WAL.append_upsert({collection, key, vec, %{}, version})
      {key, vec, meta} -> WAL.append_upsert({collection, key, vec, meta, version})
    end)

    with {:ok, tree} <- get_tree(collections, collection) do
      new_tree = Tree.insert_batch(tree, pairs)
      new_tree = %{new_tree | last_wal_version: version}
      {:reply, :ok, Map.put(collections, collection, new_tree)}
    else
      err -> {:reply, err, collections}
    end
  end

  @impl true
  def handle_call({:delete, collection, key}, _from, collections) do
    # Log to WAL
    version = System.system_time(:microsecond)
    :ok = WAL.append_delete({collection, key, version})

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

  defp get_tree(collections, name) do
    case Map.get(collections, name) do
      nil -> {:error, :collection_not_found}
      tree -> {:ok, tree}
    end
  end
end
