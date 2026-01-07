defmodule MerkleDb.KV do
  @moduledoc """
  Strongly Consistent Proxy to the Raft-managed state.
  All state-changing operations are serialized through the Raft log.
  Reads are performed via consistent queries to guarantee visibility of recent writes.
  """
  use GenServer
  alias MerkleDb.{Tree, Persistence, Raft}
  require Logger

  def start_link(_), do: GenServer.start_link(__MODULE__, %{}, name: __MODULE__)

  @doc "Insert a single vector with optional metadata"
  def put(key, vector), do: put("default", key, vector, %{})
  
  def put(key, vector, metadata) when is_map(metadata) do
    put("default", key, vector, metadata)
  end

  def put(collection, key, vector) do
    put(collection, key, vector, %{})
  end

  def put(collection, key, vector, metadata) do
    retry_command({:put, collection, key, vector, metadata}, 20)
  end

  @doc "Batch insert vectors with optional metadata"
  def put_batch(key_vector_pairs), do: put_batch("default", key_vector_pairs)
  def put_batch(collection, key_vector_pairs) do
    retry_command({:put_batch, collection, key_vector_pairs}, 20)
  end

  @doc "Delete a key (soft delete)"
  def delete(key), do: delete("default", key)
  def delete(collection, key) do
    retry_command({:delete, collection, key}, 20)
  end

  @doc "Create a new collection with optional parameters (e.g. dim, precision)"
  def create_collection(name, opts \\ []) do
    case retry_command({:create_collection, name, opts}, 20) do
      :ok -> 
        wait_until_created(name, 20)
      err -> err
    end
  end

  defp wait_until_created(name, attempts) when attempts > 0 do
    case snapshot(name) do
      %Tree{} -> :ok
      _ ->
        Process.sleep(200)
        wait_until_created(name, attempts - 1)
    end
  end
  defp wait_until_created(name, _), do: {:error, {:collection_initialization_timeout, name}}

  @doc "Drop a collection"
  def drop_collection(name) do
    retry_command({:drop_collection, name}, 20)
  end

  @doc "List all available collections"
  def list_collections do
    case :ra.consistent_query(Raft.server_id(), fn collections -> Map.keys(collections) end) do
      {:ok, keys, _leader} -> keys
      _ -> []
    end
  end

  @doc "Get tree snapshot for a collection"
  def snapshot, do: snapshot("default")
  def snapshot(collection) do
    # Linearizable read via Raft consistent_query (guarantees seeing recent writes)
    case :ra.consistent_query(Raft.server_id(), fn collections -> Map.get(collections, collection) end) do
      {:ok, %Tree{} = tree, _leader} -> tree
      {:ok, nil, _leader} -> nil
      _ -> nil
    end
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
    retry_command({:update_index, collection, new_tree, expected_generation}, 20)
  end

  @doc "Get current tree generation"
  def generation, do: generation("default")
  def generation(collection) do
    case snapshot(collection) do
      %Tree{} = tree -> tree.generation
      _ -> {:error, :collection_not_found}
    end
  end

  @doc "Reset a collection to empty state"
  def reset(collection \\ "default") do
    retry_command({:reset, collection}, 20)
  end


  @doc "Create a local binary checkpoint"
  def checkpoint(collection \\ "default") do
    GenServer.call(__MODULE__, {:checkpoint, collection})
  end

  # ==================== Callbacks ====================
  @impl true
  def init(_) do
    Logger.info("KV proxy initialized.")
    {:ok, %{}}
  end

  @impl true
  def handle_call({:checkpoint, collection}, _from, state) do
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

  @impl true
  def handle_call(msg, _from, state) do
    Logger.warn("KV received unhandled call: #{inspect(msg)}")
    {:reply, {:error, :unhandled}, state}
  end

  # ==================== Private Helpers ====================

  defp retry_command(command, attempts) when attempts > 0 do
    case Raft.process_command(command) do
      # SUCCESS: Raft leader processed it and state machine returned a result
      {:ok, result, _leader} ->
        case result do
          {:error, :collection_not_found} ->
            Process.sleep(500)
            retry_command(command, attempts - 1)
          
          # If it's any other error tuple from SM, we still retry a bit
          {:error, _reason} ->
            Process.sleep(500)
            retry_command(command, attempts - 1)

          # Actual success (e.g. :ok, %Tree{}, list)
          _ -> result
        end

      # SUCCESS: Simple :ok return
      :ok -> :ok

      # ERROR: Library reported no leader/no process
      {:error, :noproc} ->
        Process.sleep(1000)
        retry_command(command, attempts - 1)
      
      # ERROR: Library level collection error
      {:error, :collection_not_found} ->
        Process.sleep(500)
        retry_command(command, attempts - 1)

      # ERROR: Any other library error
      {:error, _reason} ->
        Process.sleep(500)
        retry_command(command, attempts - 1)

      # FALLBACK: Unexpected format, treat as success/result
      result -> result
    end
  end

  defp retry_command(_, _), do: {:error, :timeout}
end

  