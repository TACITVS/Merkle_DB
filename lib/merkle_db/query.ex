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
  - [:knn, query_vec, k, threshold, {:where, filters}] - KNN with metadata filtering

  ### Range Queries
  - [:range, query_vec, min_sim, max_sim] - All vectors with similarity in [min_sim, max_sim]
  - [:range, query_vec, min_sim, max_sim, :parallel] - Parallel IVF range search
  - [:range, query_vec, min_sim, max_sim, limit] - Range with max results limit
  - [:range, query_vec, min_sim, max_sim, {:where, filters}] - Range with metadata filtering

  ## Metadata Filtering Syntax
  Filters is a list of conditions: `[{"field", :eq, value}, {"field", :>, value}, ...]`
  Supported operators: :eq, :neq, :gt, :lt, :gte, :lte, :in

  ## Options
  - parallel: Search top N clusters in parallel (requires IVF index)
  - cached: Use VectorCache for repeated queries
  - limit: Maximum number of results (for range queries)
  - {:where, filters}: Filter by metadata
  """
  def execute(%Tree{} = tree, query) do
    case query do
      [:knn, query_vec, k, threshold | opts] ->
        if tree.count == 0, do: [], else: do_knn(tree, query_vec, k, threshold, opts)

      [:range, query_vec, min_sim, max_sim | opts] ->
        if tree.count == 0, do: [], else: do_range(tree, query_vec, min_sim, max_sim, opts)

      _ ->
        {:error, :unsupported_query}
    end
  end

  defp do_knn(tree, query_vec, k, threshold, opts) do
    where_filter = extract_where(opts)
    parallel = :parallel in opts
    cached = :cached in opts

    if cached do
      cache_key = {:knn, ASM.fp_blake3_hash(query_vec), k, threshold, where_filter}
      VectorCache.get_or_compute(cache_key, fn ->
        do_knn(tree, query_vec, k, threshold, List.delete(opts, :cached))
      end)
    else
      cond do
        parallel and tree.centroids != nil ->
          execute_ivf_parallel(tree, query_vec, k, threshold, 5, where_filter)

        tree.centroids != nil and where_filter == nil ->
          execute_ivf(tree, query_vec, k, threshold)

        tree.hnsw != nil and where_filter == nil ->
          # Use HNSW if available and no filter
          execute_hnsw(tree, query_vec, k, threshold)

        tree.quantized != nil and where_filter == nil ->
          # Use quantized search if available and no filter
          execute_quantized(tree, query_vec, k, threshold)

        true ->
          execute_flat(tree, query_vec, k, threshold, nil, where_filter)
      end
    end
  end

  # ==================== HNSW Search ====================

  defp execute_hnsw(tree, query_vec, k, threshold) do
    # Normalize query
    q_floats = for <<x::little-float-size(64) <- query_vec>>, do: x
    q_mag = :math.sqrt(Enum.reduce(q_floats, 0.0, fn x, acc -> acc + x*x end))
    q_norm_bin = if q_mag == 0, do: query_vec, else: (for q <- q_floats, into: <<>>, do: <<q/q_mag::little-float-64>>)

    ef_search = max(k * 2, 32)
    
    # 1. Call NIF
    {result_count, indices_bin, scores_bin} = 
      ASM.fp_hnsw_search(tree.hnsw, q_norm_bin, k, ef_search, tree.columns, tree.count)

    if result_count == 0 do
      []
    else
      indices = for <<i::little-signed-32 <- indices_bin>>, do: i
      scores = for <<s::little-float-size(64) <- scores_bin>>, do: s
      tombstones = tree.tombstones || MapSet.new()

      Enum.zip(indices, scores)
      |> Enum.reject(fn {idx, _score} -> MapSet.member?(tombstones, idx) end)
      |> Enum.filter(fn {_idx, score} -> score >= threshold end)
      |> Enum.map(fn {idx, score} -> {Map.get(tree.keys, idx), score} end)
    end
  end

  # ==================== Quantized Search ====================

  defp execute_quantized(tree, query_vec, k, threshold) do
    count = tree.count
    dim = tree.dim
    tombstones = tree.tombstones || MapSet.new()
    %{columns: q_cols, params: q_params} = tree.quantized

    # 1. Normalize and parse query
    q_floats = for <<x::little-float-size(64) <- query_vec>>, do: x
    q_mag = :math.sqrt(Enum.reduce(q_floats, 0.0, fn x, acc -> acc + x*x end))
    q_norm_list = if q_mag == 0, do: q_floats, else: Enum.map(q_floats, &(&1 / q_mag))

    # 2. Pre-process query for quantized dot product
    # Dot product logic:
    # float_val = (uint8_val * scale) + min
    # sum(float_val_d * q_d) = sum(((uint8_val_d * scale_d) + min_d) * q_d)
    #                        = sum(uint8_val_d * (scale_d * q_d) + (min_d * q_d))
    #                        = sum(uint8_val_d * scaled_q_d) + sum(min_d * q_d)
    # where scaled_q_d = q_d / inv_scale_d
    #       bias = sum(min_d * q_d)

    {scaled_q_list, bias} = 
      Enum.zip(q_norm_list, q_params)
      |> Enum.reduce({[], 0.0}, fn {q_d, {min_d, inv_scale_d}}, {s_acc, b_acc} ->
        scale_d = 1.0 / inv_scale_d
        {[q_d * scale_d | s_acc], b_acc + (min_d * q_d)}
      end)
    
    scaled_q_bin = for s <- Enum.reverse(scaled_q_list), into: <<>>, do: <<s::little-float-64>>

    # 3. Call NIF
    scores_bin = ASM.fp_query_gemv_quantized(q_cols, scaled_q_bin, bias, count, dim)

    # 4. Top-K selection (same as flat)
    {result_count, indices_bin, result_scores_bin} =
      ASM.fp_query_topk(scores_bin, count, k, threshold)

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

  defp do_range(tree, query_vec, min_sim, max_sim, opts) do
    where_filter = extract_where(opts)
    parallel = :parallel in opts
    limit = Enum.find(opts, &is_integer/1)

    if parallel and tree.centroids != nil do
      execute_range_parallel(tree, query_vec, min_sim, max_sim, limit, 10, where_filter)
    else
      execute_range(tree, query_vec, min_sim, max_sim, limit, where_filter)
    end
  end

  defp extract_where(opts) do
    Enum.find_value(opts, fn
      {:where, filters} -> filters
      _ -> nil
    end)
  end

  # ==================== IVF Search (Single Cluster) ====================

  defp execute_ivf(tree, query_vec, k, threshold) do
    # 1. Find nearest centroid
    num_clusters = map_size(tree.clusters)

    cluster_id = find_nearest_cluster(tree.centroids, num_clusters, tree.dim, query_vec)

    # 2. Get indices in this cluster
    indices = Map.get(tree.clusters, cluster_id, [])

    # 3. Perform search ONLY on these indices
    execute_flat(tree, query_vec, k, threshold, indices, nil)
  end

  # ==================== IVF Parallel Search (Top N Clusters) ====================

  # Parallel IVF search across top N candidate clusters.
  # Significantly faster for high-dimensional data with good clustering.
  defp execute_ivf_parallel(tree, query_vec, k, threshold, n_clusters, where_filter) do
    num_clusters = map_size(tree.clusters)

    # 1. Find top N nearest clusters (not just 1)
    top_clusters = find_top_n_clusters(tree.centroids, num_clusters, tree.dim, query_vec, n_clusters)

    # 2. Search each cluster in parallel
    results =
      top_clusters
      |> Task.async_stream(
        fn cluster_id ->
          indices = Map.get(tree.clusters, cluster_id, [])
          execute_flat(tree, query_vec, k * 2, threshold, indices, where_filter)  # Get more results per cluster
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

  defp execute_flat(tree, query_vec, k, threshold, row_indices, where_filter) do
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
      # First, filter out tombstones AND metadata from the candidate indices
      valid_indices = 
        row_indices 
        |> Enum.reject(fn idx -> MapSet.member?(tombstones, idx) end)
        |> Enum.filter(fn idx -> matches_where?(tree, idx, where_filter) end)

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
      # If we have a filter, we might need to look at more than just top-K
      # If NO filter, we use top-K optimization
      if where_filter == nil do
        search_k = trunc(k * 1.5) + 1
        scores_bin = ASM.fp_query_gemv_columnar(tree.columns, q_norm_bin, count, dim)
        {result_count, indices_bin, result_scores_bin} = ASM.fp_query_topk(scores_bin, count, search_k, threshold)

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
      else
        # With metadata filter, we scan all scores
        # We use a row-major fallback for filtered queries
        # because the columnar GEMV NIF is currently unstable.
        
        # 1. Flatten tree to row-major for efficient per-vector dot product
        flat_data = Tree.flatten(tree)
        
        # 2. Compute scores for each row
        scores_list = 
          for idx <- 0..(count - 1) do
            vec_bin = binary_part(flat_data, idx * dim * 8, dim * 8)
            ASM.fp_fold_dotp_f64(vec_bin, q_norm_bin, dim)
          end
        
        scores_list
        |> Enum.with_index()
        |> Enum.reject(fn {_score, idx} -> MapSet.member?(tombstones, idx) end)
        |> Enum.filter(fn {score, idx} -> score >= threshold and matches_where?(tree, idx, where_filter) end)
        |> Enum.sort_by(fn {score, _idx} -> score end, :desc)
        |> Enum.take(k)
        |> Enum.map(fn {score, idx} -> {Map.get(tree.keys, idx), score} end)
      end
    end
  end

  # ==================== Range Query Implementation ====================

  defp execute_range(tree, query_vec, min_sim, max_sim, limit, where_filter) do
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

    # Filter by range [min_sim, max_sim] AND check tombstones AND metadata
    results =
      scores_list
      |> Enum.with_index()
      |> Enum.reject(fn {_score, idx} -> MapSet.member?(tombstones, idx) end)
      |> Enum.filter(fn {score, idx} -> 
           score >= min_sim and score <= max_sim and matches_where?(tree, idx, where_filter)
         end)
      |> Enum.sort_by(fn {score, _idx} -> score end, :desc)
      |> Enum.map(fn {score, idx} -> {Map.get(tree.keys, idx), score} end)

    # Apply limit if specified
    if limit, do: Enum.take(results, limit), else: results
  end

  # Parallel range query using IVF - searches all clusters in parallel
  defp execute_range_parallel(tree, query_vec, min_sim, max_sim, limit, n_clusters, where_filter) do
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
          execute_range_indexed(tree, q_norm_bin, min_sim, max_sim, indices, where_filter)
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
  defp execute_range_indexed(tree, q_norm_bin, min_sim, max_sim, row_indices, where_filter) do
    tombstones = tree.tombstones || MapSet.new()
    
    valid_indices = 
      row_indices 
      |> Enum.reject(fn idx -> MapSet.member?(tombstones, idx) end)
      |> Enum.filter(fn idx -> matches_where?(tree, idx, where_filter) end)
    
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

  # ==================== Metadata Filter Engine ====================

  defp matches_where?(_tree, _idx, nil), do: true
  defp matches_where?(tree, idx, filters) when is_list(filters) do
    meta = Map.get(tree.metadata, idx, %{})
    Enum.all?(filters, fn filter -> match_filter?(meta, filter) end)
  end

  defp match_filter?(meta, {field, :eq, value}), do: Map.get(meta, field) == value
  defp match_filter?(meta, {field, :neq, value}), do: Map.get(meta, field) != value
  defp match_filter?(meta, {field, :gt, value}), do: Map.get(meta, field) > value
  defp match_filter?(meta, {field, :lt, value}), do: Map.get(meta, field) < value
  defp match_filter?(meta, {field, :gte, value}), do: Map.get(meta, field) >= value
  defp match_filter?(meta, {field, :lte, value}), do: Map.get(meta, field) <= value
  defp match_filter?(meta, {field, :in, values}) when is_list(values), do: Map.get(meta, field) in values
  defp match_filter?(_meta, _), do: false

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
