defmodule MerkleDb.Progress do
  use GenServer

  # --- Client API ---

  def start_link(_opts \\ []) do
    GenServer.start_link(__MODULE__, %{status: :idle}, name: __MODULE__)
  end

  # Called by the Cluster Engine to update status
  def report(topics_found, verses_scanned, total_verses) do
    GenServer.cast(__MODULE__, {:update, topics_found, verses_scanned, total_verses})
  end

  # Called when job finishes
  def complete(result_json) do
    GenServer.cast(__MODULE__, {:complete, result_json})
  end

  # Called by the Web Router to send JSON to frontend
  def get_status do
    GenServer.call(__MODULE__, :get_status)
  end

  # Helper: Ensure it's running (Lazy Start)
  def ensure_started do
    if Process.whereis(__MODULE__) == nil do
      start_link()
    end
  end

  # --- Server Callbacks ---

  @impl true
  def init(state), do: {:ok, state}

  @impl true
  def handle_cast({:update, t, v, total}, _state) do
    percent = if total == 0, do: 0, else: Float.round((v / total) * 100, 1)
    {:noreply, %{status: :running, topics: t, scanned: v, total: total, percent: percent}}
  end

  @impl true
  def handle_cast({:complete, json}, state) do
    # Keep the final stats but mark as done
    new_state = Map.put(state, :status, :done) |> Map.put(:result, json)
    {:noreply, new_state}
  end

  @impl true
  def handle_call(:get_status, _from, state) do
    {:reply, state, state}
  end
end