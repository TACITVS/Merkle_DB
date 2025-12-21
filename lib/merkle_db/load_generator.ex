defmodule MerkleDb.LoadGenerator do
  @moduledoc """
  Automated load generator for stress testing the vector database.
  Simulates concurrent queries, monitors system health, and tests resilience.
  """
  use GenServer
  alias MerkleDb.{KV, Query, TextEmbedding}

  @status_table :merkle_load_status
  @status_key :status
  @refresh_interval_ms 5000
  @min_in_flight 32
  @in_flight_multiplier 4

  @query_pool [
    "faith and hope",
    "love and compassion",
    "creation and light",
    "wisdom and understanding",
    "justice and righteousness",
    "peace and truth",
    "strength and courage",
    "grace and mercy",
    "redemption and salvation",
    "blessing and prosperity"
  ]

  # Client API

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  @doc """
  Start load generation with specified QPS (queries per second).
  """
  def start_load(target_qps) do
    GenServer.call(__MODULE__, {:start_load, target_qps})
  end

  @doc """
  Stop load generation.
  """
  def stop_load do
    GenServer.call(__MODULE__, :stop_load)
  end

  def status_snapshot do
    case :ets.whereis(@status_table) do
      :undefined -> default_status()
      _ ->
        case :ets.lookup(@status_table, @status_key) do
          [{@status_key, status}] -> status
          _ -> default_status()
        end
    end
  end

  def stop_if_active do
    case Process.whereis(__MODULE__) do
      nil -> :ok
      _pid ->
        case status_snapshot() do
          %{active: true} -> stop_load()
          _ -> :ok
        end
    end
  end

  def active? do
    case status_snapshot() do
      %{active: active} -> active
      _ -> false
    end
  end

  @doc """
  Get current load generation status.
  """
  def get_status do
    status_snapshot()
  end

  # Server callbacks

  @impl true
  def init(_) do
    if :ets.whereis(@status_table) == :undefined do
      :ets.new(@status_table, [:set, :public, :named_table, read_concurrency: true])
    end

    state = %{
      active: false,
      target_qps: 0,
      current_qps: 0,
      tick_ref: nil,
      refresh_ref: nil,
      tick_interval_ms: nil,
      run_id: 0,
      in_flight: 0,
      max_in_flight: max(@min_in_flight, System.schedulers_online() * @in_flight_multiplier),
      queries_sent: 0,
      errors: 0,
      start_time: nil,
      cached_tree: nil  # Cache tree snapshot to avoid KV timeout under load
    }

    _ = write_status(state)
    {:ok, state}
  end

  @impl true
  def handle_call({:start_load, target_qps}, _from, state) do
    # Stop existing timers if any
    if state.tick_ref, do: Process.cancel_timer(state.tick_ref)
    if state.refresh_ref, do: Process.cancel_timer(state.refresh_ref)

    # Calculate interval in milliseconds
    interval = max(trunc(1000 / max(target_qps, 1)), 1)

    # Start periodic query generation with a single scheduled tick
    tick_ref = Process.send_after(self(), {:tick, interval}, interval)

    # Start snapshot refresh every 5 seconds
    refresh_ref = Process.send_after(self(), :refresh_snapshot, @refresh_interval_ms)

    # Trigger immediate snapshot refresh (async)
    send(self(), :refresh_snapshot)

    run_id = state.run_id + 1
    new_state = %{state |
      active: true,
      target_qps: target_qps,
      tick_ref: tick_ref,
      refresh_ref: refresh_ref,
      tick_interval_ms: interval,
      run_id: run_id,
      in_flight: 0,
      queries_sent: 0,
      errors: 0,
      start_time: System.monotonic_time(:second)
    }

    _ = write_status(new_state)
    {:reply, {:ok, "Load generation started at #{target_qps} QPS"}, new_state}
  end

  @impl true
  def handle_call(:stop_load, _from, state) do
    if state.tick_ref, do: Process.cancel_timer(state.tick_ref)
    if state.refresh_ref, do: Process.cancel_timer(state.refresh_ref)

    new_state = %{state |
      active: false,
      tick_ref: nil,
      refresh_ref: nil,
      tick_interval_ms: nil,
      target_qps: 0,
      cached_tree: nil,
      run_id: state.run_id + 1,
      start_time: nil,
      current_qps: 0
    }

    _ = write_status(new_state)
    {:reply, {:ok, "Load generation stopped"}, new_state}
  end

  @impl true
  def handle_call(:get_status, _from, state) do
    status = write_status(state)
    {:reply, status, state}
  end

  @impl true
  def handle_info({:tick, interval_ms}, state) do
    # Execute query asynchronously using cached snapshot to avoid KV timeout
    if state.active and interval_ms == state.tick_interval_ms do
      new_state =
        if state.cached_tree && state.in_flight < state.max_in_flight do
          run_id = state.run_id
          Task.start(fn -> execute_random_query(state.cached_tree, run_id) end)
          %{state | in_flight: state.in_flight + 1, queries_sent: state.queries_sent + 1}
        else
          state
        end

      # Update current QPS estimate
      elapsed = System.monotonic_time(:second) - state.start_time
      current_qps = if elapsed > 0, do: state.queries_sent / elapsed, else: 0.0

      tick_ref = Process.send_after(self(), {:tick, interval_ms}, interval_ms)
      updated = %{new_state | current_qps: Float.round(current_qps * 1.0, 2), tick_ref: tick_ref}

      _ = write_status(updated)
      {:noreply, updated}
    else
      {:noreply, state}
    end
  end

  @impl true
  def handle_info(:refresh_snapshot, state) do
    # Refresh cached tree snapshot every 5 seconds
    updated_state =
      try do
        cached_tree = KV.snapshot()
        %{state | cached_tree: cached_tree}
      rescue
        _ ->
          # Keep old snapshot if refresh fails
          state
      end

    if updated_state.active do
      refresh_ref = Process.send_after(self(), :refresh_snapshot, @refresh_interval_ms)
      updated_state = %{updated_state | refresh_ref: refresh_ref}
      _ = write_status(updated_state)
      {:noreply, updated_state}
    else
      {:noreply, updated_state}
    end
  end

  # Private helpers

  defp execute_random_query(tree, run_id) do
    try do
      status = status_snapshot()

      if status.active != true or status.run_id != run_id do
        :skip
      else
        if tree.count > 0 do
          # Pick random query
          query_text = Enum.random(@query_pool)
          query_vec = TextEmbedding.embed(query_text)

          # Execute with appropriate mode based on indexing
          mode = if tree.centroids, do: [:knn, query_vec, 10, 0.3, :cached],
                                   else: [:knn, query_vec, 10, 0.3]

          _results = Query.execute(tree, mode)
          :ok
        else
          :skip  # Skip if no data
        end
      end
    rescue
      e ->
        IO.puts("Query error: #{inspect(e)}")
        GenServer.cast(__MODULE__, {:increment_error, run_id})
        :error
    after
      GenServer.cast(__MODULE__, {:task_done, run_id})
    end
  end

  @impl true
  def handle_cast({:increment_error, run_id}, state) do
    updated =
      if run_id == state.run_id do
        %{state | errors: state.errors + 1}
      else
        state
      end

    _ = write_status(updated)
    {:noreply, updated}
  end

  @impl true
  def handle_cast({:task_done, _run_id}, state) do
    updated = %{state | in_flight: max(state.in_flight - 1, 0)}
    _ = write_status(updated)
    {:noreply, updated}
  end

  defp write_status(state) do
    status = build_status(state)
    :ets.insert(@status_table, {@status_key, status})
    status
  end

  defp build_status(state) do
    elapsed =
      if state.start_time do
        System.monotonic_time(:second) - state.start_time
      else
        0
      end

    actual_qps =
      if state.active and elapsed > 0 do
        Float.round(state.queries_sent / elapsed, 2)
      else
        0.0
      end

    %{
      active: state.active,
      target_qps: state.target_qps,
      actual_qps: actual_qps,
      queries_sent: state.queries_sent,
      errors: state.errors,
      in_flight: state.in_flight,
      max_in_flight: state.max_in_flight,
      run_id: state.run_id,
      success_rate:
        if state.queries_sent > 0 do
          Float.round((state.queries_sent - state.errors) / state.queries_sent * 100, 2)
        else
          100.0
        end,
      elapsed_seconds: elapsed,
      updated_at_ms: System.monotonic_time(:millisecond)
    }
  end

  defp default_status do
    %{
      active: false,
      target_qps: 0,
      actual_qps: 0.0,
      queries_sent: 0,
      errors: 0,
      in_flight: 0,
      max_in_flight: max(@min_in_flight, System.schedulers_online() * @in_flight_multiplier),
      run_id: 0,
      success_rate: 100.0,
      elapsed_seconds: 0,
      updated_at_ms: nil
    }
  end
end
