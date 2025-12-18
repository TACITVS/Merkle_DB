defmodule MerkleDb.TextStore do
  use GenServer

  @table :bible_text_storage
  @filename "bible_text.dets"

  # --- Client API ---

  def start_link(_) do
    GenServer.start_link(__MODULE__, [], name: __MODULE__)
  end

  def put(key, text) do
    GenServer.cast(__MODULE__, {:put, key, text})
  end

  def get(key) do
    GenServer.call(__MODULE__, {:get, key})
  end

  # NEW: Retrieve all texts for analysis
  def get_all do
    GenServer.call(__MODULE__, :get_all)
  end

  # --- Server Callbacks ---

  @impl true
  def init(_) do
    filename = String.to_charlist(@filename)
    # Open file once
    {:ok, _ref} = :dets.open_file(@table, [file: filename, type: :set])
    {:ok, %{}}
  end

  @impl true
  def handle_cast({:put, key, text}, state) do
    :dets.insert(@table, {key, text})
    {:noreply, state}
  end

  @impl true
  def handle_call({:get, key}, _from, state) do
    result = case :dets.lookup(@table, key) do
      [{^key, text}] -> text
      [] -> "Text not found"
    end
    {:reply, result, state}
  end

  # NEW: Iterate over the entire database to return all text
  @impl true
  def handle_call(:get_all, _from, state) do
    # Efficiently fold over the Dets table
    all_data = :dets.foldl(fn {key, val}, acc -> [{key, val} | acc] end, [], @table)
    {:reply, all_data, state}
  end

  @impl true
  def terminate(_reason, _state) do
    :dets.close(@table)
  end
end