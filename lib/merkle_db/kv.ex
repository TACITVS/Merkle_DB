defmodule MerkleDb.KV do
  @moduledoc """
  GenServer managing the current Tree state with optimistic locking support.
  """
  use GenServer

  alias MerkleDb.Tree

  def start_link(_), do: GenServer.start_link(__MODULE__, %{}, name: __MODULE__)

  @doc "Insert a single vector with optional metadata"
  def put(key, vector, metadata \\ %{}), do: GenServer.call(__MODULE__, {:put, key, vector, metadata})

  @doc "Batch insert vectors with optional metadata"
  def put_batch(key_vector_pairs), do: GenServer.call(__MODULE__, {:put_batch, key_vector_pairs})

  @doc "Delete a key (soft delete)"
  def delete(key), do: GenServer.call(__MODULE__, {:delete, key})

  @doc "Get current tree snapshot"
  def snapshot, do: GenServer.call(__MODULE__, :snapshot)

  @doc "Replace entire tree (used by Bootstrap)"
  def set_tree(%Tree{} = tree), do: GenServer.call(__MODULE__, {:set_tree, tree})

  @doc """
  Atomically update tree's IVF index if generation matches (optimistic locking).
  Used by IndexBuilder to safely apply index after async computation.
  Returns :ok | {:error, :generation_mismatch}
  """
  def update_index(%Tree{} = new_tree, expected_generation) do
    GenServer.call(__MODULE__, {:update_index, new_tree, expected_generation})
  end

  @doc "Get current tree generation"
  def generation, do: GenServer.call(__MODULE__, :generation)

  # ==================== Callbacks ====================

  @impl true
  def init(_) do
    {:ok, Tree.new()}
  end

  @impl true
  def handle_call({:put, key, vector, metadata}, _from, current_tree) do
    new_tree = Tree.insert(current_tree, key, vector, metadata)
    {:reply, :ok, new_tree}
  end

  @impl true
  def handle_call({:put_batch, pairs}, _from, current_tree) do
    new_tree = Tree.insert_batch(current_tree, pairs)
    {:reply, :ok, new_tree}
  end

  @impl true
  def handle_call({:delete, key}, _from, current_tree) do
    case Tree.delete(current_tree, key) do
      {:error, :not_found} ->
        {:reply, {:error, :not_found}, current_tree}
      new_tree ->
        {:reply, :ok, new_tree}
    end
  end

  @impl true
  def handle_call(:snapshot, _from, current_tree) do
    {:reply, current_tree, current_tree}
  end

  @impl true
  def handle_call({:set_tree, new_tree}, _from, _current_tree) do
    {:reply, :ok, new_tree}
  end

  @impl true
  def handle_call({:update_index, new_tree, expected_generation}, _from, current_tree) do
    if current_tree.generation == expected_generation do
      # Apply the new index (centroids + clusters) while keeping current data
      updated_tree = %{current_tree |
        centroids: new_tree.centroids,
        clusters: new_tree.clusters,
        generation: current_tree.generation + 1
      }
      {:reply, :ok, updated_tree}
    else
      {:reply, {:error, :generation_mismatch}, current_tree}
    end
  end

  @impl true
  def handle_call(:generation, _from, current_tree) do
    {:reply, current_tree.generation, current_tree}
  end
end