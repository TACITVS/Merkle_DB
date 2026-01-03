defmodule MerkleDb.KV do
  use GenServer

  def start_link(_), do: GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  def put(key, vector), do: GenServer.call(__MODULE__, {:put, key, vector})
  def delete(key), do: GenServer.call(__MODULE__, {:delete, key})
  def snapshot, do: GenServer.call(__MODULE__, :snapshot)
  def set_tree(tree), do: GenServer.call(__MODULE__, {:set_tree, tree})
  def reset, do: GenServer.call(__MODULE__, :reset)
  def update_index(tree, expected_generation),
    do: GenServer.call(__MODULE__, {:update_index, tree, expected_generation})

  @impl true
  def init(_) do
    {:ok, MerkleDb.Tree.new()}
  end

  @impl true
  def handle_call({:put, key, vector}, _from, current_tree) do
    # Use our new Columnar Insert
    new_tree = MerkleDb.Tree.insert(current_tree, key, vector)
    {:reply, :ok, new_tree}
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
      {:reply, :ok, %{new_tree | generation: current_tree.generation}}
    else
      {:reply, {:error, :stale_tree}, current_tree}
    end
  end

  @impl true
  def handle_call({:delete, key}, _from, current_tree) do
    # Soft delete: remove key from keys map (leaves data in columns for compaction later)
    inverted = current_tree.keys |> Map.new(fn {idx, k} -> {k, idx} end)
    case Map.get(inverted, key) do
      nil ->
        {:reply, :ok, current_tree}
      idx ->
        new_keys = Map.delete(current_tree.keys, idx)
        {:reply, :ok, %{current_tree | keys: new_keys, generation: current_tree.generation + 1}}
    end
  end

  @impl true
  def handle_call(:reset, _from, _current_tree) do
    {:reply, :ok, MerkleDb.Tree.new()}
  end
end
