defmodule MerkleDb.KV do
  use GenServer

  def start_link(_), do: GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  def put(key, vector), do: GenServer.call(__MODULE__, {:put, key, vector})
  def snapshot, do: GenServer.call(__MODULE__, :snapshot)

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
end