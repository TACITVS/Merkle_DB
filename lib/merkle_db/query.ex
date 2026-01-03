defmodule MerkleDb.Query do
  @moduledoc """
  Vector query execution with IVF indexing, parallel search, and payload filtering.
  """

  alias MerkleDb.{Tree, ASM, Filter, PayloadStore, Telemetry, VectorCache}

  @doc """
  Execute a query against the tree.

  ## Supported Queries
  - [:knn, query_vec, k, threshold] - K-nearest neighbors with similarity threshold
  - [:knn, query_vec, k, threshold, :parallel] - Parallel IVF search across top clusters
  - [:knn, query_vec, k, threshold, :cached] - Cache-aware search
  - [:knn, query_vec, k, threshold, filter] - KNN with payload filtering
  - [:knn, query_vec, k, threshold, filter, :parallel] - Parallel KNN with filtering

  ## Filter Format
  Filter is a list of conditions:
      [{"field", "op", value}, ...]

  Supported operators: ==, !=, >, >=, <, <=, in, not_in, contains, starts_with, exists

  ## Options
  - parallel: Search top N clusters in parallel (requires IVF index)
  - cached: Use VectorCache for repeated queries
  """
  def execute(%Tree{} = tree, [:knn, query_vec, k, threshold]) do
    execute_with_telemetry(tree, query_vec, k, threshold, :auto, fn ->
      execute_knn(tree, query_vec, k, threshold)
    end)
  end

  def execute(%Tree{} = tree, [:knn, query_vec, k, threshold, :parallel]) do
    execute_with_telemetry(tree, query_vec, k, threshold, :parallel, fn ->
      execute_knn_parallel(tree, query_vec, k, threshold)
    end)
  end

  def execute(%Tree{} = tree, [:knn, query_vec, k, threshold, :cached]) do
    execute_with_telemetry(tree, query_vec, k, threshold, :cached, fn ->
      cache_key = {:knn, :crypto.hash(:sha256, query_vec), k, threshold}

      VectorCache.get_or_compute(cache_key, fn ->
        execute_knn(tree, query_vec, k, threshold)
      end)
    end)
  end

  # KNN with payload filter
  def execute(%Tree{} = tree, [:knn, query_vec, k, threshold, filter])
      when is_list(filter) do
    execute_with_telemetry(tree, query_vec, k, threshold, :filtered, fn ->
      execute_knn_filtered(tree, query_vec, k, threshold, filter)
    end)
  end

  # Parallel KNN with payload filter
  def execute(%Tree{} = tree, [:knn, query_vec, k, threshold, filter, :parallel])
      when is_list(filter) do
    execute_with_telemetry(tree, query_vec, k, threshold, :filtered_parallel, fn ->
      execute_knn_parallel_filtered(tree, query_vec, k, threshold, filter)
    end)
  end

  # ==================== IVF Search (Single Cluster) ====================

  defp execute_knn(tree, query_vec, k, threshold) do
    if tree.count == 0, do: []

    if tree.centroids do
      execute_ivf(tree, query_vec, k, threshold)
    else
      execute_flat(tree, query_vec, k, threshold)
    end
  end

  defp execute_knn_parallel(tree, query_vec, k, threshold) do
    if tree.count == 0, do: []

    if tree.centroids do
      execute_ivf_parallel(tree, query_vec, k, threshold)
    else
      execute_flat(tree, query_vec, k, threshold)
    end
  end

  defp execute_ivf(tree, query_vec, k, threshold) do
    # 1. Find nearest centroid
    num_clusters = map_size(tree.clusters)

    cluster_id = find_nearest_cluster(tree.centroids, num_clusters, tree.dim, query_vec)

    # 2. Get indices in this cluster
    indices = Map.get(tree.clusters, cluster_id, [])

    # 3. Perform search ONLY on these indices
    execute_flat(tree, query_vec, k, threshold, indices)
  end

  # ==================== IVF Parallel Search (Top N Clusters) ====================

  # Parallel IVF search across top N candidate clusters.
  # Significantly faster for high-dimensional data with good clustering.
  defp execute_ivf_parallel(tree, query_vec, k, threshold, n_clusters \\ 5) do
    num_clusters = map_size(tree.clusters)

    # 1. Find top N nearest clusters (not just 1)
    top_clusters = find_top_n_clusters(tree.centroids, num_clusters, tree.dim, query_vec, n_clusters)

    # 2. Search each cluster in parallel
    results =
      top_clusters
      |> Task.async_stream(
        fn cluster_id ->
          indices = Map.get(tree.clusters, cluster_id, [])
          execute_flat(tree, query_vec, k * 2, threshold, indices)  # Get more results per cluster
        end,
        max_concurrency: System.schedulers_online(),
        timeout: 10_000
      )
      |> Enum.flat_map(fn
        {:ok, res} -> res
        {:exit, _reason} -> []
      end)

    # 3. Merge and re-rank results from all clusters
    results
    |> Enum.sort_by(fn {_, score} -> score end, :desc)
    |> Enum.take(k)
  end

  # ==================== Flat Search (Brute Force) ====================

  defp execute_flat(tree, query_vec, k, threshold, row_indices \\ nil) do
    count = tree.count
    q_floats = for <<x::little-float-size(64) <- query_vec>>, do: x
    q_mag = :math.sqrt(Enum.reduce(q_floats, 0.0, fn x, acc -> acc + x*x end))
    q_norm = if q_mag == 0, do: q_floats, else: Enum.map(q_floats, &(&1 / q_mag))

    output_size = count * 8
    accumulator = ASM.fp_replicate_f64(output_size, count, 0.0)

    final_scores_bin =
      q_norm
      |> Enum.with_index()
      |> Enum.reduce(accumulator, fn {q_val, dim_idx}, acc_bin ->
        column_bin = elem(tree.columns, dim_idx)
        ASM.fp_map_axpy_f64(column_bin, acc_bin, output_size, count, q_val)
      end)

    scores_list = for <<s::little-float-size(64) <- final_scores_bin>>, do: s

    # Negative threshold = "k-only" mode (return k results without filtering)
    k_only_mode = threshold < 0

    scores_list
    |> Stream.with_index()
    |> Stream.filter(fn {_score, idx} ->
      if row_indices, do: idx in row_indices, else: true
    end)
    |> Stream.map(fn {score, idx} -> {Map.get(tree.keys, idx), score} end)
    |> Stream.filter(fn {_, score} ->
      # Skip threshold filter in k-only mode
      k_only_mode or score >= threshold
    end)
    |> Enum.sort_by(fn {_, score} -> score end, :desc)
    |> Enum.take(k)
  end

  # ==================== Filtered Search ====================

  defp execute_knn_filtered(tree, query_vec, k, threshold, filter) do
    if tree.count == 0, do: []

    # First, get more candidates (we'll filter some out)
    # Request k * 3 to account for filtering
    candidate_k = min(k * 3, tree.count)

    candidates =
      if tree.centroids do
        execute_ivf(tree, query_vec, candidate_k, threshold)
      else
        execute_flat(tree, query_vec, candidate_k, threshold)
      end

    # Apply payload filter to candidates
    apply_payload_filter(candidates, filter, k)
  end

  defp execute_knn_parallel_filtered(tree, query_vec, k, threshold, filter) do
    if tree.count == 0, do: []

    # Get more candidates to account for filtering
    candidate_k = min(k * 3, tree.count)

    candidates =
      if tree.centroids do
        execute_ivf_parallel(tree, query_vec, candidate_k, threshold)
      else
        execute_flat(tree, query_vec, candidate_k, threshold)
      end

    # Apply payload filter to candidates
    apply_payload_filter(candidates, filter, k)
  end

  defp apply_payload_filter(candidates, [], k) do
    # No filter, just take k results
    Enum.take(candidates, k)
  end

  defp apply_payload_filter(candidates, filter, k) do
    # Get keys from candidates
    keys = Enum.map(candidates, fn {key, _score} -> key end)

    # Fetch payloads from PayloadStore
    payloads = PayloadStore.get_many(keys)

    # Filter candidates by payload match
    candidates
    |> Enum.filter(fn {key, _score} ->
      case Map.get(payloads, key) do
        nil -> false  # No payload = doesn't match
        payload -> Filter.matches?(payload, filter)
      end
    end)
    |> Enum.take(k)
  end

  # ==================== Cluster Finding Helpers ====================

  defp find_nearest_cluster(centroids_bin, num_clusters, dim, query_vec) do
    # Simple linear scan of centroids (K is small, e.g. 100)
    0..(num_clusters - 1)
    |> Enum.max_by(fn c_idx ->
      c_vec = binary_part(centroids_bin, c_idx * dim * 8, dim * 8)
      ASM.fp_fold_dotp_f64(c_vec, query_vec, dim)
    end)
  end

  defp find_top_n_clusters(centroids_bin, num_clusters, dim, query_vec, n) do
    # Find top N clusters by similarity score
    n = min(n, num_clusters)  # Can't get more clusters than exist

    0..(num_clusters - 1)
    |> Enum.map(fn c_idx ->
      c_vec = binary_part(centroids_bin, c_idx * dim * 8, dim * 8)
      score = ASM.fp_fold_dotp_f64(c_vec, query_vec, dim)
      {c_idx, score}
    end)
    |> Enum.sort_by(fn {_idx, score} -> score end, :desc)
    |> Enum.take(n)
    |> Enum.map(fn {idx, _score} -> idx end)
  end

  # ==================== Query Validation ====================

  @doc """
  Validate query parameters before execution.
  Returns :ok | {:error, reason}
  """
  def validate_query(%Tree{} = tree, [:knn, query_vec, k, threshold | _opts]) do
    with :ok <- validate_tree(tree),
         :ok <- validate_query_vector(query_vec, tree.dim),
         :ok <- validate_k(k),
         :ok <- validate_threshold(threshold) do
      :ok
    end
  end

  defp validate_tree(%Tree{count: 0}), do: {:error, :empty_tree}
  defp validate_tree(%Tree{dim: 0}), do: {:error, :uninitialized_tree}
  defp validate_tree(_tree), do: :ok

  defp validate_query_vector(vec_bin, expected_dim) do
    if is_binary(vec_bin) do
      actual_dim = div(byte_size(vec_bin), 8)
      if actual_dim == expected_dim do
        :ok
      else
        {:error, "query vector dimension mismatch: expected #{expected_dim}, got #{actual_dim}"}
      end
    else
      {:error, "query vector must be a binary"}
    end
  end

  defp validate_k(k) when is_integer(k) and k > 0, do: :ok
  defp validate_k(_k), do: {:error, "k must be a positive integer"}

  defp validate_threshold(t) when is_float(t) or is_integer(t), do: :ok
  defp validate_threshold(_t), do: {:error, "threshold must be a number"}

  defp execute_with_telemetry(_tree, _query_vec, k, threshold, mode, fun) do
    Telemetry.span([:merkle_db, :query, :execute], %{k: k, threshold: threshold, mode: mode}, fn ->
      result = fun.()
      {result, %{result_count: length(result)}}
    end)
  end
end
