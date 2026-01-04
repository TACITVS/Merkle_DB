defmodule MerkleDb.Query do
  @moduledoc """
  Vector query execution with IVF indexing and parallel search.
  """

  alias MerkleDb.{Tree, ASM, VectorCache}

  @doc """
  Execute a query against the tree.

  ## Supported Queries

  ### KNN (K-Nearest Neighbors)
  - [:knn, query_vec, k, threshold] - K-nearest neighbors with similarity threshold
  - [:knn, query_vec, k, threshold, :parallel] - Parallel IVF search across top clusters
  - [:knn, query_vec, k, threshold, :cached] - Cache-aware search

  ### Range Queries
  - [:range, query_vec, min_sim, max_sim] - All vectors with similarity in [min_sim, max_sim]
  - [:range, query_vec, min_sim, max_sim, :parallel] - Parallel IVF range search
  - [:range, query_vec, min_sim, max_sim, limit] - Range with max results limit

  ## Options
  - parallel: Search top N clusters in parallel (requires IVF index)
  - cached: Use VectorCache for repeated queries
  - limit: Maximum number of results (for range queries)
  """
  def execute(%Tree{} = tree, [:knn, query_vec, k, threshold]) do
    if tree.count == 0, do: []

    if tree.centroids do
      execute_ivf(tree, query_vec, k, threshold)
    else
      execute_flat(tree, query_vec, k, threshold)
    end
  end

  def execute(%Tree{} = tree, [:knn, query_vec, k, threshold, :parallel]) do
    if tree.count == 0, do: []

    if tree.centroids do
      execute_ivf_parallel(tree, query_vec, k, threshold)
    else
      execute_flat(tree, query_vec, k, threshold)
    end
  end

  def execute(%Tree{} = tree, [:knn, query_vec, k, threshold, :cached]) do
    cache_key = {:knn, ASM.fp_blake3_hash(query_vec), k, threshold}

    VectorCache.get_or_compute(cache_key, fn ->
      execute(tree, [:knn, query_vec, k, threshold])
    end)
  end

  # ==================== Range Queries ====================

  # Range query: returns all vectors with similarity score in [min_sim, max_sim].
  # Unlike KNN which returns top-K, this returns ALL matches in the range.
  def execute(%Tree{} = tree, [:range, query_vec, min_sim, max_sim]) do
    execute_range(tree, query_vec, min_sim, max_sim, nil)
  end

  def execute(%Tree{count: 0}, [:range, _query_vec, _min_sim, _max_sim, :parallel]), do: []

  def execute(%Tree{} = tree, [:range, query_vec, min_sim, max_sim, :parallel]) do
    if tree.centroids do
      execute_range_parallel(tree, query_vec, min_sim, max_sim, nil)
    else
      execute_range(tree, query_vec, min_sim, max_sim, nil)
    end
  end

  def execute(%Tree{} = tree, [:range, query_vec, min_sim, max_sim, limit])
      when is_integer(limit) and limit > 0 do
    execute_range(tree, query_vec, min_sim, max_sim, limit)
  end

  def execute(%Tree{count: 0}, [:range, _query_vec, _min_sim, _max_sim, limit, :parallel])
      when is_integer(limit) and limit > 0,
      do: []

  def execute(%Tree{} = tree, [:range, query_vec, min_sim, max_sim, limit, :parallel])
      when is_integer(limit) and limit > 0 do
    if tree.centroids do
      execute_range_parallel(tree, query_vec, min_sim, max_sim, limit)
    else
      execute_range(tree, query_vec, min_sim, max_sim, limit)
    end
  end

  # ==================== IVF Search (Single Cluster) ====================

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
    dim = tree.dim
    tombstones = tree.tombstones || MapSet.new()

    # Normalize query vector
    q_floats = for <<x::little-float-size(64) <- query_vec>>, do: x
    q_mag = :math.sqrt(Enum.reduce(q_floats, 0.0, fn x, acc -> acc + x*x end))
    q_norm_list = if q_mag == 0, do: q_floats, else: Enum.map(q_floats, &(&1 / q_mag))
    q_norm_bin = for q <- q_norm_list, into: <<>>, do: <<q::little-float-size(64)>>

    if row_indices do
      # IVF path: use indexed GEMV - O(num_indices * dim) instead of O(count * dim)
      # First, filter out tombstones from the candidate indices
      valid_indices = Enum.filter(row_indices, fn idx -> not MapSet.member?(tombstones, idx) end)

      if valid_indices == [] do
        []
      else
        # Convert indices to int32 binary
        indices_bin = for idx <- valid_indices, into: <<>>, do: <<idx::little-signed-32>>

        # Compute scores ONLY for candidate indices
        scores_bin = ASM.fp_query_gemv_indexed(tree.columns, q_norm_bin, indices_bin, count, dim)

        # Parse scores and pair with original indices
        scores_list = for <<s::little-float-size(64) <- scores_bin>>, do: s

        Enum.zip(valid_indices, scores_list)
        |> Enum.filter(fn {_idx, score} -> score >= threshold end)
        |> Enum.sort_by(fn {_idx, score} -> score end, :desc)
        |> Enum.take(k)
        |> Enum.map(fn {idx, score} -> {Map.get(tree.keys, idx), score} end)
      end
    else
      # Flat path: compute all scores, then native top-k selection
      # We request K * 1.5 to account for potential deleted items in the top results
      # This is a heuristic; strictly correct would be to filter inside ASM or fetch more if needed
      search_k = trunc(k * 1.5) + 1
      
      scores_bin = ASM.fp_query_gemv_columnar(tree.columns, q_norm_bin, count, dim)

      {result_count, indices_bin, result_scores_bin} =
        ASM.fp_query_topk(scores_bin, count, search_k, threshold)

      if result_count == 0 do
        []
      else
        indices = for <<i::little-signed-32 <- indices_bin>>, do: i
        scores = for <<s::little-float-size(64) <- result_scores_bin>>, do: s

        Enum.zip(indices, scores)
        |> Enum.reject(fn {idx, _score} -> MapSet.member?(tombstones, idx) end)
        |> Enum.take(k)
        |> Enum.map(fn {idx, score} -> {Map.get(tree.keys, idx), score} end)
      end
    end
  end

  # ==================== Range Query Implementation ====================

  defp execute_range(tree, query_vec, min_sim, max_sim, limit) do
    if tree.count == 0, do: []

    count = tree.count
    dim = tree.dim
    tombstones = tree.tombstones || MapSet.new()

    # Normalize query vector
    q_floats = for <<x::little-float-size(64) <- query_vec>>, do: x
    q_mag = :math.sqrt(Enum.reduce(q_floats, 0.0, fn x, acc -> acc + x*x end))
    q_norm_list = if q_mag == 0, do: q_floats, else: Enum.map(q_floats, &(&1 / q_mag))
    q_norm_bin = for q <- q_norm_list, into: <<>>, do: <<q::little-float-size(64)>>

    # Compute all scores
    scores_bin = ASM.fp_query_gemv_columnar(tree.columns, q_norm_bin, count, dim)
    scores_list = for <<s::little-float-size(64) <- scores_bin>>, do: s

    # Filter by range [min_sim, max_sim] AND check tombstones
    results =
      scores_list
      |> Enum.with_index()
      |> Enum.reject(fn {_score, idx} -> MapSet.member?(tombstones, idx) end)
      |> Enum.filter(fn {score, _idx} -> score >= min_sim and score <= max_sim end)
      |> Enum.sort_by(fn {score, _idx} -> score end, :desc)
      |> Enum.map(fn {score, idx} -> {Map.get(tree.keys, idx), score} end)

    # Apply limit if specified
    if limit, do: Enum.take(results, limit), else: results
  end

  # Parallel range query using IVF - searches all clusters in parallel
  defp execute_range_parallel(tree, query_vec, min_sim, max_sim, limit, n_clusters \\ 10) do
    num_clusters = map_size(tree.clusters)

    # Normalize query
    q_floats = for <<x::little-float-size(64) <- query_vec>>, do: x
    q_mag = :math.sqrt(Enum.reduce(q_floats, 0.0, fn x, acc -> acc + x*x end))
    q_norm_list = if q_mag == 0, do: q_floats, else: Enum.map(q_floats, &(&1 / q_mag))
    q_norm_bin = for q <- q_norm_list, into: <<>>, do: <<q::little-float-size(64)>>

    # Find top N clusters that might contain matches
    top_clusters = find_top_n_clusters(tree.centroids, num_clusters, tree.dim, q_norm_bin, min(n_clusters, num_clusters))

    # Search each cluster in parallel
    results =
      top_clusters
      |> Task.async_stream(
        fn cluster_id ->
          indices = Map.get(tree.clusters, cluster_id, [])
          execute_range_indexed(tree, q_norm_bin, min_sim, max_sim, indices)
        end,
        max_concurrency: System.schedulers_online(),
        timeout: 10_000
      )
      |> Enum.flat_map(fn
        {:ok, res} -> res
        {:exit, _reason} -> []
      end)

    # Merge, sort, and apply limit
    sorted = Enum.sort_by(results, fn {_key, score} -> score end, :desc)
    if limit, do: Enum.take(sorted, limit), else: sorted
  end

  # Range search on specific indices (for IVF)
  defp execute_range_indexed(tree, q_norm_bin, min_sim, max_sim, row_indices) do
    tombstones = tree.tombstones || MapSet.new()
    valid_indices = Enum.filter(row_indices, fn idx -> not MapSet.member?(tombstones, idx) end)
    
    if valid_indices == [] do
      []
    else
      count = tree.count
      dim = tree.dim

      # Convert indices to int32 binary
      indices_bin = for idx <- valid_indices, into: <<>>, do: <<idx::little-signed-32>>

      # Compute scores for candidates
      scores_bin = ASM.fp_query_gemv_indexed(tree.columns, q_norm_bin, indices_bin, count, dim)
      scores_list = for <<s::little-float-size(64) <- scores_bin>>, do: s

      # Filter by range
      Enum.zip(valid_indices, scores_list)
      |> Enum.filter(fn {_idx, score} -> score >= min_sim and score <= max_sim end)
      |> Enum.map(fn {idx, score} -> {Map.get(tree.keys, idx), score} end)
    end
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

  def validate_query(%Tree{} = tree, [:range, query_vec, min_sim, max_sim | _opts]) do
    with :ok <- validate_tree(tree),
         :ok <- validate_query_vector(query_vec, tree.dim),
         :ok <- validate_range(min_sim, max_sim) do
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

  defp validate_range(min_sim, max_sim) when is_number(min_sim) and is_number(max_sim) do
    cond do
      min_sim > max_sim -> {:error, "min_sim must be <= max_sim"}
      min_sim < -1.0 or min_sim > 1.0 -> {:error, "min_sim must be in [-1, 1] for cosine similarity"}
      max_sim < -1.0 or max_sim > 1.0 -> {:error, "max_sim must be in [-1, 1] for cosine similarity"}
      true -> :ok
    end
  end

  defp validate_range(_min, _max), do: {:error, "min_sim and max_sim must be numbers"}
end