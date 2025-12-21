defmodule MerkleDb.TelemetryAggregator do
  @moduledoc """
  GenServer that aggregates telemetry metrics for real-time dashboard.
  Maintains rolling windows of query performance, cache stats, and system metrics.
  """
  use GenServer
  alias MerkleDb.{Bootstrap, KV, LoadGenerator, Progress, Tree, VectorCache}

  @window_size 100  # Keep last 100 queries for percentile calculations
  @table_name :telemetry_aggregator
  @metrics_table :telemetry_metrics_snapshot
  @metrics_key :snapshot
  @metrics_timeout_ms 200

  # Client API

  def start_link(_opts) do
    GenServer.start_link(__MODULE__, %{}, name: __MODULE__)
  end

  @doc """
  Get current metrics snapshot for the dashboard.
  """
  def get_metrics do
    case Process.whereis(__MODULE__) do
      nil -> fetch_cached_metrics(true)
      _ ->
        try do
          GenServer.call(__MODULE__, :get_metrics, @metrics_timeout_ms)
        catch
          :exit, _ -> fetch_cached_metrics(true)
        end
    end
  end

  # Server Callbacks

  @impl true
  def init(_) do
    # Attach to telemetry events
    :telemetry.attach_many(
      "aggregator-handler",
      [
        [:merkle_db, :query, :execute],
        [:merkle_db, :cache, :hit],
        [:merkle_db, :cache, :miss]
      ],
      &handle_telemetry_event/4,
      nil
    )

    # Initialize ETS table for fast concurrent reads
    :ets.new(@table_name, [:set, :public, :named_table, read_concurrency: true])
    if :ets.whereis(@metrics_table) == :undefined do
      :ets.new(@metrics_table, [:set, :public, :named_table, read_concurrency: true])
    end

    initial_state = %{
      query_durations: :queue.new(),  # Rolling window of last 100 query durations
      total_queries: 0,
      cache_hits: 0,
      cache_misses: 0,
      last_qps_timestamp: System.monotonic_time(:second),
      queries_since_last_check: 0,
      start_time: System.system_time(:second)
    }

    _ = cache_metrics(default_metrics(false))
    {:ok, initial_state}
  end

  @impl true
  def handle_call(:get_metrics, _from, state) do
    metrics =
      try do
        # Extract durations from queue
        durations_list = :queue.to_list(state.query_durations)

        # Get tree snapshot for vector count
        tree = safe_kv_snapshot()

        # Calculate metrics
        snapshot = %{
          timestamp: System.system_time(:millisecond),
          query_metrics: build_query_metrics(state, durations_list),
          cache_metrics: build_cache_metrics(state),
          system_metrics: build_system_metrics(tree, state),
          index_build: fetch_index_build(),
          bootstrap: fetch_bootstrap(tree),
          load_status: fetch_load_status()
        }

        cache_metrics(snapshot)
      rescue
        e ->
          cached = fetch_cached_metrics(true)
          Map.put(cached, :error, Exception.message(e))
      end

    {:reply, metrics, state}
  end

  @impl true
  def handle_cast({:record_query, duration_ms}, state) do
    # Add to rolling window
    new_queue = :queue.in(duration_ms, state.query_durations)

    # Keep only last @window_size queries
    trimmed_queue = if :queue.len(new_queue) > @window_size do
      {_, q} = :queue.out(new_queue)
      q
    else
      new_queue
    end

    new_state = %{state |
      query_durations: trimmed_queue,
      total_queries: state.total_queries + 1,
      queries_since_last_check: state.queries_since_last_check + 1
    }

    {:noreply, new_state}
  end

  @impl true
  def handle_cast(:cache_hit, state) do
    {:noreply, %{state | cache_hits: state.cache_hits + 1}}
  end

  @impl true
  def handle_cast(:cache_miss, state) do
    {:noreply, %{state | cache_misses: state.cache_misses + 1}}
  end

  # Telemetry Event Handlers

  def handle_telemetry_event([:merkle_db, :query, :execute], measurements, _metadata, _config) do
    duration_ms = measurements.duration / 1_000_000
    GenServer.cast(__MODULE__, {:record_query, duration_ms})
  end

  def handle_telemetry_event([:merkle_db, :cache, :hit], _measurements, _metadata, _config) do
    GenServer.cast(__MODULE__, :cache_hit)
  end

  def handle_telemetry_event([:merkle_db, :cache, :miss], _measurements, _metadata, _config) do
    GenServer.cast(__MODULE__, :cache_miss)
  end

  # Private Helpers

  defp build_query_metrics(state, durations_list) do
    current_qps = calculate_qps(state)

    if length(durations_list) > 0 do
      sorted_durations = Enum.sort(durations_list)

      %{
        total_queries: state.total_queries,
        qps_current: current_qps,
        avg_latency_ms: avg(durations_list),
        median_latency_ms: percentile(sorted_durations, 0.50),
        p95_latency_ms: percentile(sorted_durations, 0.95),
        p99_latency_ms: percentile(sorted_durations, 0.99),
        slowest_query_ms: Enum.max(durations_list)
      }
    else
      %{
        total_queries: state.total_queries,
        qps_current: current_qps,
        avg_latency_ms: 0.0,
        median_latency_ms: 0.0,
        p95_latency_ms: 0.0,
        p99_latency_ms: 0.0,
        slowest_query_ms: 0.0
      }
    end
  end

  defp build_cache_metrics(state) do
    total_requests = state.cache_hits + state.cache_misses
    hit_rate = if total_requests > 0, do: state.cache_hits / total_requests, else: 0.0

    cache_stats =
      case :ets.whereis(:merkle_vector_cache) do
        :undefined -> %{size: 0, memory_mb: 0.0}
        _ -> VectorCache.stats()
      end

    %{
      hit_rate: Float.round(hit_rate, 3),
      total_hits: state.cache_hits,
      total_misses: state.cache_misses,
      size_mb: Map.get(cache_stats, :memory_mb, 0.0),
      size_entries: Map.get(cache_stats, :size, 0)
    }
  end

  defp build_system_metrics(tree, state) do
    uptime_seconds = System.system_time(:second) - state.start_time
    memory_bytes = :erlang.memory(:total)

    %{
      memory_mb: Float.round(memory_bytes / (1024 * 1024), 2),
      memory_gb: Float.round(memory_bytes / (1024 * 1024 * 1024), 3),
      vector_count: tree.count,
      dimensions: tree.dim,
      indexed: tree.centroids != nil,
      index_type: if(tree.centroids, do: "ivf", else: "flat"),
      cluster_count: if(tree.centroids, do: map_size(tree.clusters), else: 0),
      uptime_seconds: uptime_seconds
    }
  end

  defp fetch_index_build do
    try do
      case Process.whereis(Progress) do
        nil -> %{status: :idle}
        _ -> Progress.get_status()
      end
    catch
      :exit, _ -> %{status: :unknown}
    end
  end

  defp fetch_bootstrap(tree) do
    try do
      case Process.whereis(Bootstrap) do
        nil -> %{status: :idle}
        _ -> Bootstrap.status(tree_stats: Tree.stats(tree))
      end
    catch
      :exit, _ -> %{status: :unknown}
    end
  end

  defp fetch_load_status do
    LoadGenerator.status_snapshot()
  end

  defp safe_kv_snapshot do
    try do
      GenServer.call(KV, :snapshot, 100)
    rescue
      _ -> Tree.new()
    catch
      :exit, _ -> Tree.new()
    end
  end

  defp calculate_qps(state) do
    current_time = System.monotonic_time(:second)
    elapsed = max(current_time - state.last_qps_timestamp, 1)

    Float.round(state.queries_since_last_check / elapsed, 2)
  end

  defp avg([]), do: 0.0
  defp avg(list) do
    Float.round(Enum.sum(list) / length(list), 2)
  end

  defp percentile([], _p), do: 0.0
  defp percentile(sorted_list, p) do
    index = trunc(length(sorted_list) * p)
    clamped_index = min(index, length(sorted_list) - 1)
    Float.round(Enum.at(sorted_list, clamped_index), 2)
  end

  defp cache_metrics(metrics) do
    updated =
      metrics
      |> Map.put(:stale, false)
      |> Map.put(:updated_at_ms, System.monotonic_time(:millisecond))

    :ets.insert(@metrics_table, {@metrics_key, updated})
    updated
  end

  defp fetch_cached_metrics(stale?) do
    case :ets.whereis(@metrics_table) do
      :undefined -> default_metrics(stale?)
      _ ->
        case :ets.lookup(@metrics_table, @metrics_key) do
          [{@metrics_key, metrics}] -> Map.put(metrics, :stale, stale?)
          _ -> default_metrics(stale?)
        end
    end
  end

  defp default_metrics(stale?) do
    %{
      timestamp: System.system_time(:millisecond),
      query_metrics: %{
        total_queries: 0,
        qps_current: 0.0,
        avg_latency_ms: 0.0,
        median_latency_ms: 0.0,
        p95_latency_ms: 0.0,
        p99_latency_ms: 0.0,
        slowest_query_ms: 0.0
      },
      cache_metrics: %{
        hit_rate: 0.0,
        total_hits: 0,
        total_misses: 0,
        size_mb: 0.0,
        size_entries: 0
      },
      system_metrics: %{
        memory_mb: Float.round(:erlang.memory(:total) / (1024 * 1024), 2),
        memory_gb: Float.round(:erlang.memory(:total) / (1024 * 1024 * 1024), 3),
        vector_count: 0,
        dimensions: 0,
        indexed: false,
        index_type: "flat",
        cluster_count: 0,
        uptime_seconds: 0
      },
      index_build: %{status: :idle},
      bootstrap: %{status: :idle},
      load_status: LoadGenerator.status_snapshot(),
      stale: stale?,
      updated_at_ms: nil
    }
  end
end
