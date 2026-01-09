defmodule MerkleDb.Web.PrometheusExporter do
  @moduledoc """
  Prometheus metrics exporter for MerkleDB.
  Exposes metrics in Prometheus text format for scraping.
  """

  alias MerkleDb.TelemetryAggregator

  @doc """
  Generate Prometheus-compatible metrics text.
  """
  def export_metrics do
    metrics = TelemetryAggregator.get_metrics()

    """
    # HELP merkledb_queries_total Total number of queries executed
    # TYPE merkledb_queries_total counter
    merkledb_queries_total #{metrics.query_metrics.total_queries}

    # HELP merkledb_query_duration_seconds Query execution duration
    # TYPE merkledb_query_duration_seconds histogram
    merkledb_query_duration_seconds_sum #{metrics.query_metrics.total_duration_ms / 1000}
    merkledb_query_duration_seconds_count #{metrics.query_metrics.total_queries}

    # HELP merkledb_query_results_total Number of results returned
    # TYPE merkledb_query_results_total counter
    merkledb_query_results_total #{metrics.query_metrics.total_results}

    # HELP merkledb_cache_hits_total Cache hit count
    # TYPE merkledb_cache_hits_total counter
    merkledb_cache_hits_total #{metrics.cache_metrics.hits}

    # HELP merkledb_cache_misses_total Cache miss count
    # TYPE merkledb_cache_misses_total counter
    merkledb_cache_misses_total #{metrics.cache_metrics.misses}

    # HELP merkledb_cache_hit_rate Cache hit rate (0-1)
    # TYPE merkledb_cache_hit_rate gauge
    merkledb_cache_hit_rate #{safe_divide(metrics.cache_metrics.hits, metrics.cache_metrics.hits + metrics.cache_metrics.misses)}

    # HELP merkledb_vector_count Total vectors in database
    # TYPE merkledb_vector_count gauge
    merkledb_vector_count #{metrics.tree_stats.count}

    # HELP merkledb_memory_usage_bytes Memory usage in bytes
    # TYPE merkledb_memory_usage_bytes gauge
    merkledb_memory_usage_bytes #{metrics.tree_stats.memory_bytes}

    # HELP merkledb_load_active Load generator active status
    # TYPE merkledb_load_active gauge
    merkledb_load_active #{if metrics.load_status.active, do: 1, else: 0}

    # HELP merkledb_uptime_seconds Application uptime in seconds
    # TYPE merkledb_uptime_seconds counter
    merkledb_uptime_seconds #{System.system_time(:second) - get_start_time()}

    # HELP merkledb_raft_leader Raft leadership status (1 if leader)
    # TYPE merkledb_raft_leader gauge
    merkledb_raft_leader #{if check_raft_leader(), do: 1, else: 0}

    # HELP merkledb_health_status Overall health status (1 if healthy)
    # TYPE merkledb_health_status gauge
    merkledb_health_status #{if check_health(), do: 1, else: 0}
    """
  end

  defp safe_divide(_num, 0), do: 0.0
  defp safe_divide(num, denom), do: num / denom

  defp get_start_time do
    :persistent_term.get(:merkledb_start_time, System.system_time(:second))
  end

  defp check_raft_leader do
    case :ra.members({:merkle_db_server, node()}) do
      {:ok, _, leader} -> leader == {:merkle_db_server, node()}
      _ -> false
    end
  end

  defp check_health do
    # Check if all critical services are running
    Process.whereis(MerkleDb.KV) != nil &&
    Process.whereis(TelemetryAggregator) != nil
  end
end
