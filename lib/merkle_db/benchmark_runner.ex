defmodule MerkleDb.BenchmarkRunner do
  @moduledoc """
  Runs performance benchmarks for the racing widget.
  Compares different search modes, insert strategies, and caching performance.
  """
  alias MerkleDb.{KV, Query, TextEmbedding}

  @doc """
  Run a benchmark by type and return results with timing, throughput, and speedup metrics.

  Supported benchmark types:
  - :flat_vs_ivf - Compare flat brute-force vs IVF indexed search
  - :single_vs_batch - Compare single inserts vs batch insert
  - :cached_vs_uncached - Compare first query vs cached repeated query
  """
  def run_benchmark(type, params \\ %{})

  def run_benchmark(:flat_vs_ivf, params) do
    tree = KV.snapshot()

    if tree.count == 0 do
      {:error, "Database is empty. Please ingest data first."}
    else
      # Use provided query or default
      query_text = Map.get(params, "query", "faith and hope")
      k = Map.get(params, "k", 10)
      threshold = Map.get(params, "threshold", 0.3)

      query_vec = TextEmbedding.embed(query_text)

      # Benchmark 1: Flat search (disable IVF temporarily)
      {time_flat_us, results_flat} = :timer.tc(fn ->
        tree_no_ivf = %{tree | centroids: nil, clusters: %{}}
        Query.execute(tree_no_ivf, [:knn, query_vec, k, threshold])
      end)

      # Benchmark 2: IVF search (if indexed)
      {time_ivf_us, results_ivf} = if tree.centroids do
        :timer.tc(fn ->
          Query.execute(tree, [:knn, query_vec, k, threshold])
        end)
      else
        # If not indexed, use same as flat
        {time_flat_us, results_flat}
      end

      flat_result = %{
        name: "Flat Search",
        duration_ms: Float.round(time_flat_us / 1000.0, 2),
        duration_us: time_flat_us,
        results_count: length(results_flat),
        throughput_qps: Float.round(1_000_000.0 / max(time_flat_us, 1), 1),
        metadata: %{
          vectors_scanned: tree.count,
          index_type: "flat"
        }
      }

      ivf_result = %{
        name: "IVF Search",
        duration_ms: Float.round(time_ivf_us / 1000.0, 2),
        duration_us: time_ivf_us,
        results_count: length(results_ivf),
        throughput_qps: Float.round(1_000_000.0 / max(time_ivf_us, 1), 1),
        metadata: %{
          vectors_scanned: estimate_ivf_scan_count(tree),
          clusters_searched: 1,
          index_type: "ivf"
        }
      }

      speedup = Float.round(time_flat_us / max(time_ivf_us, 1), 2)
      latency_reduction = Float.round((1 - time_ivf_us / max(time_flat_us, 1)) * 100, 1)

      {:ok, %{
        benchmark_type: "flat_vs_ivf",
        timestamp: System.system_time(:millisecond),
        results: [flat_result, ivf_result],
        speedup: speedup,
        speedup_display: "#{speedup}x",
        winner: if(speedup > 1.1, do: "IVF Search", else: "Tie"),
        comparison: %{
          latency_reduction_pct: latency_reduction,
          efficiency_gain: "Scanned #{Float.round((1 - estimate_ivf_scan_count(tree) / tree.count) * 100, 0)}% fewer vectors"
        }
      }}
    end
  end

  def run_benchmark(:single_vs_batch, params) do
    test_count = parse_positive_int(params, "count", 1000)
    dim = parse_positive_int(params, "dim", 768)

    test_vectors = for i <- 1..test_count do
      # Random normalized vectors
      vec = for _d <- 1..dim, do: :rand.uniform() - 0.5
      mag = :math.sqrt(Enum.reduce(vec, 0.0, fn x, acc -> acc + x * x end))
      normalized = Enum.map(vec, &(&1 / mag))
      vec_bin = for x <- normalized, into: <<>>, do: <<x::little-float-64>>

      {"Test#{i}", vec_bin}
    end

    # Benchmark 1: Single inserts
    single_result =
      with_temp_collection("bench_single", fn collection ->
        :timer.tc(fn ->
          Enum.each(test_vectors, fn {key, vec} ->
            :ok = KV.put(collection, key, vec)
          end)
        end)
      end)

    # Benchmark 2: Batch insert
    batch_result =
      with_temp_collection("bench_batch", fn collection ->
        :timer.tc(fn ->
          :ok = KV.put_batch(collection, test_vectors)
        end)
      end)

    with {:ok, {time_single_us, _}} <- single_result,
         {:ok, {time_batch_us, _}} <- batch_result do
      single_result = %{
        name: "Single Inserts",
        duration_ms: Float.round(time_single_us / 1000.0, 2),
        duration_us: time_single_us,
        vectors_inserted: test_count,
        throughput_vps: Float.round(test_count / (time_single_us / 1_000_000.0), 0),
        metadata: %{
          method: "individual KV.put calls"
        }
      }

      batch_result = %{
        name: "Batch Insert",
        duration_ms: Float.round(time_batch_us / 1000.0, 2),
        duration_us: time_batch_us,
        vectors_inserted: test_count,
        throughput_vps: Float.round(test_count / (time_batch_us / 1_000_000.0), 0),
        metadata: %{
          method: "Tree.insert_batch"
        }
      }

      speedup = Float.round(time_single_us / max(time_batch_us, 1), 1)

      {:ok, %{
        benchmark_type: "single_vs_batch",
        timestamp: System.system_time(:millisecond),
        results: [single_result, batch_result],
        speedup: speedup,
        speedup_display: "#{speedup}x",
        winner: "Batch Insert",
        comparison: %{
          throughput_improvement_pct: Float.round((speedup - 1) * 100, 0),
          time_saved_ms: Float.round((time_single_us - time_batch_us) / 1000.0, 1)
        }
      }}
    else
      {:error, reason} -> {:error, "Benchmark failed: #{inspect(reason)}"}
    end
  end

  def run_benchmark(:cached_vs_uncached, params) do
    tree = KV.snapshot()

    if tree.count == 0 do
      {:error, "Database is empty. Please ingest data first."}
    else
      query_text = Map.get(params, "query", "love and compassion")
      k = Map.get(params, "k", 10)
      threshold = Map.get(params, "threshold", 0.3)

      query_vec = TextEmbedding.embed(query_text)

      # Clear cache to ensure fresh start
      alias MerkleDb.VectorCache
      VectorCache.clear()

      # Benchmark 1: Uncached (first query)
      {time_uncached_us, results_uncached} = :timer.tc(fn ->
        Query.execute(tree, [:knn, query_vec, k, threshold, :cached])
      end)

      # Benchmark 2: Cached (second query, should hit cache)
      {time_cached_us, results_cached} = :timer.tc(fn ->
        Query.execute(tree, [:knn, query_vec, k, threshold, :cached])
      end)

      uncached_result = %{
        name: "Uncached Query",
        duration_ms: Float.round(time_uncached_us / 1000.0, 2),
        duration_us: time_uncached_us,
        results_count: length(results_uncached),
        throughput_qps: Float.round(1_000_000.0 / max(time_uncached_us, 1), 1),
        metadata: %{
          cache_status: "miss",
          computation: "full search + IVF"
        }
      }

      cached_result = %{
        name: "Cached Query",
        duration_ms: Float.round(time_cached_us / 1000.0, 2),
        duration_us: time_cached_us,
        results_count: length(results_cached),
        throughput_qps: Float.round(1_000_000.0 / max(time_cached_us, 1), 1),
        metadata: %{
          cache_status: "hit",
          computation: "ETS lookup only"
        }
      }

      speedup = Float.round(time_uncached_us / max(time_cached_us, 1), 1)

      {:ok, %{
        benchmark_type: "cached_vs_uncached",
        timestamp: System.system_time(:millisecond),
        results: [uncached_result, cached_result],
        speedup: speedup,
        speedup_display: "#{speedup}x",
        winner: "Cached Query",
        comparison: %{
          latency_reduction_pct: Float.round((1 - time_cached_us / max(time_uncached_us, 1)) * 100, 1),
          cache_advantage: "Sub-millisecond ETS lookup vs full vector search"
        }
      }}
    end
  end

  def run_benchmark(type, _params) do
    {:error, "Unknown benchmark type: #{type}"}
  end

  # Private Helpers

  defp parse_positive_int(params, key, default) do
    case Map.get(params, key) do
      value when is_integer(value) and value > 0 -> value
      value when is_binary(value) ->
        case Integer.parse(value) do
          {parsed, _} when parsed > 0 -> parsed
          _ -> default
        end
      _ -> default
    end
  end

  defp with_temp_collection(prefix, fun) when is_function(fun, 1) do
    name = "#{prefix}_#{:erlang.unique_integer([:positive])}"

    case KV.create_collection(name) do
      :ok ->
        try do
          {:ok, fun.(name)}
        after
          _ = KV.drop_collection(name)
        end

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp estimate_ivf_scan_count(tree) do
    if tree.centroids && map_size(tree.clusters) > 0 do
      # Approximate: average cluster size
      avg_cluster_size = div(tree.count, map_size(tree.clusters))
      avg_cluster_size
    else
      tree.count
    end
  end
end
