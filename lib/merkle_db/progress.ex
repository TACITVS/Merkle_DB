defmodule MerkleDb.Progress do
  use GenServer

  @default_state %{
    status: :idle,
    phase: :idle,
    iter: 0,
    max_iter: 0,
    percent: 0.0,
    message: "",
    started_at_ms: nil,
    updated_at_ms: nil,
    elapsed_ms: 0
  }

  # --- Client API ---

  def start_link(_opts \\ []) do
    GenServer.start_link(__MODULE__, @default_state, name: __MODULE__)
  end

  def report(update) when is_map(update) do
    GenServer.cast(__MODULE__, {:report, update})
  end

  # Called when job finishes
  def complete(result_json) do
    GenServer.cast(__MODULE__, {:complete, result_json})
  end

  # Called by the Web Router to send JSON to frontend
  def get_status do
    GenServer.call(__MODULE__, :get_status)
  end

  # --- Server Callbacks ---

  @impl true
  def init(state), do: {:ok, state}

  @impl true
  def handle_cast({:report, update}, state) do
    merged =
      state
      |> Map.merge(update)
      |> normalize()

    {:noreply, merged}
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

  defp normalize(state) do
    max_iter = Map.get(state, :max_iter, 0) || 0
    iter = Map.get(state, :iter, 0) || 0

    percent =
      cond do
        state.status == :done -> 100.0
        max_iter > 0 -> Float.round(iter / max_iter * 100.0, 1)
        true -> 0.0
      end

    started_at_ms = Map.get(state, :started_at_ms)
    updated_at_ms = Map.get(state, :updated_at_ms, started_at_ms)

    elapsed_ms =
      if started_at_ms && updated_at_ms do
        max(updated_at_ms - started_at_ms, 0)
      else
        Map.get(state, :elapsed_ms, 0)
      end

    state
    |> Map.put(:percent, percent)
    |> Map.put(:elapsed_ms, elapsed_ms)
  end
end
