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
    retry_command({:set_tree, collection, tree}, 20)
  end

  # ... other APIs simplified similarly ...

  @doc """
  Atomically update tree's IVF index if generation matches (optimistic locking).
  """
  def update_index(tree, expected_generation), do: update_index("default", tree, expected_generation)
  def update_index(collection, %Tree{} = new_tree, expected_generation) do
    Raft.process_command({:update_index, collection, new_tree, expected_generation})
  end

  @doc "Get current tree generation"
  def generation, do: generation("default")
  def generation(collection) do
    case Raft.get_state() |> Map.get(collection) do
      nil -> {:error, :collection_not_found}
      tree -> tree.generation
    end
  end

  @doc "Reset a collection to empty state"
  def reset(collection \\ "default") do
    Raft.process_command({:reset, collection})
  end

  @doc "Create a fast binary checkpoint for instant startup"
  def checkpoint(collection \\ "default") do
    # Checkpoint is still somewhat local to the node's disk, 
    # but we should probably trigger it via Raft to ensure all nodes do it?
    # Or just let it be a local node operation.
    # For now, let's keep it as a GenServer call if it's meant to be local,
    # OR route it through Raft if we want a cluster-wide checkpoint.
    # Given the original code, it was a GenServer.call.
    GenServer.call(__MODULE__, {:checkpoint, collection})
  end

  # ==================== Callbacks ====================
  @impl true
  def init(_) do
    # KV now acts strictly as a proxy.
    # Truth lives in the Raft state machine.
    # We do NOT load collections or replay WAL here, 
    # because Raft handles its own log recovery.
    Logger.info("KV proxy initialized.")
    {:ok, %{}}
  end

  # --- Callbacks for proxying (legacy support or local ops) ---

  @impl true
  def handle_call(:list_collections, _from, _state) do
    {:reply, list_collections(), %{}}
  end

  @impl true
  def handle_call({:snapshot, collection}, _from, _state) do
    {:reply, snapshot(collection), %{}}
  end

  @impl true
  def handle_call({:generation, collection}, _from, _state) do
    {:reply, generation(collection), %{}}
  end

  @impl true
  def handle_call({:checkpoint, collection}, _from, state) do
    # Checkpoint is performed on the local node's state fetched from Raft
    case snapshot(collection) do
      %Tree{} = tree ->
        case Persistence.save_checkpoint(tree, collection) do
          {:ok, dir} -> 
            Logger.info("Checkpoint saved to #{dir}")
            {:reply, :ok, state}
          err -> 
            {:reply, err, state}
        end
      nil ->
        {:reply, {:error, :collection_not_found}, state}
    end
  end

  # Fallback for other calls
  @impl true
  def handle_call(msg, _from, state) do
    Logger.warn("KV received unhandled call: #{inspect(msg)}")
    {:reply, {:error, :unhandled}, state}
  end

  # ==================== Private Helpers ====================

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
end
