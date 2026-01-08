defmodule MerkleDb.Query do
  @moduledoc """
  Vector query execution with IVF indexing, parallel search, and payload filtering.
  """

  alias MerkleDb.{Tree, ASM, Telemetry, VectorCache, PayloadStore, TextEmbedding}

  @doc """
  Execute a query against the tree.

  ## Supported Queries

  ### KNN (K-Nearest Neighbors)
  - [:knn, query_vec, k, threshold] - K-nearest neighbors with similarity threshold
  - [:knn, query_vec, k, threshold, :parallel] - Parallel IVF search across top clusters
  - [:knn, query_vec, k, threshold, :cached] - Cache-aware search
  - [:knn, query_vec, k, threshold, {:where, filters}] - KNN with metadata filtering

  ### Semantic Search
  - [:semantic, "text query", k, threshold] - Automated text-to-vector embedding using GloVe

  ### Range Queries
  - [:range, query_vec, min_sim, max_sim] - All vectors with similarity in [min_sim, max_sim]
  - [:range, query_vec, min_sim, max_sim, :parallel] - Parallel IVF range search
  - [:range, query_vec, min_sim, max_sim, limit] - Range with max results limit
  - [:range, query_vec, min_sim, max_sim, {:where, filters}] - Range with metadata filtering

  ## Metadata Filtering Syntax
  Filters is a list of conditions: `[{"field", :eq, value}, {"field", "==", value}, ...]`
  Supported operators: :eq, :neq, :gt, :lt, :gte, :lte, :in, :not_in, :contains,
  :starts_with, :exists and their string forms ("==", "!=", ">", "<", ">=", "<=",
  "in", "not_in", "contains", "starts_with", "exists")

  ## Options
  - parallel: Search top N clusters in parallel (requires IVF index)
  - cached: Use VectorCache for repeated queries
  - limit: Maximum number of results (for range queries)
  - {:where, filters}: Filter by metadata
  """
  def execute(%Tree{} = tree, query) do
    execute_with_telemetry(tree, query, fn ->
      case query do
        [:knn, query_vec, k, threshold | opts] ->
          if tree.count == 0, do: [], else: do_knn(tree, query_vec, k, threshold, opts)

        [:semantic, text, k, threshold | opts] ->
          if tree.count == 0 do
            []
          else
            # 1. Embed text to f32
            query_vec = TextEmbedding.embed(text)
            
            # 2. Convert to f64 ONLY if tree is f64
            query_vec = 
              if tree.precision == :f64 do
                TextEmbedding.to_f64(query_vec)
              else
                query_vec
              end

            # 3. Execute as KNN
            do_knn(tree, query_vec, k, threshold, opts)
          end

        [:range, query_vec, min_sim, max_sim | opts] ->
          if tree.count == 0, do: [], else: do_range(tree, query_vec, min_sim, max_sim, opts)

        [:sparse, sparse_query, k, threshold] ->
          if tree.count == 0, do: [], else: do_sparse(tree, sparse_query, k, threshold)

        [:hybrid, query_vec, sparse_query, k, threshold | opts] ->
          if tree.count == 0, do: [], else: do_hybrid(tree, query_vec, sparse_query, k, threshold, opts)

        _ ->
          {:error, :unsupported_query}
      end
    end)
  end

  @doc """
  Execute multiple KNN queries in a single batch for high throughput.
  Amortizes NIF overhead across all queries in the batch.

  ## Parameters
  - tree: %Tree{}
  - query_vectors: List of binary query vectors
  - k: Number of top results per query
  - threshold: Similarity threshold

  Returns: List of result lists (one per query vector)
  """
  def execute_batch(%Tree{count: 0}, _, _, _), do: []
  def execute_batch(%Tree{} = tree, query_vectors, k, threshold) when is_list(query_vectors) do
    batch_count = length(query_vectors)
    dim = tree.dim
    tombstones = tree.tombstones || MapSet.new()

    # 1. Prepare queries binary
    # Normalize each query vector
    norm_queries = 
      query_vectors
      |> Enum.map(fn q_vec ->
        q_floats = for <<x::little-float-size(64) <- q_vec>>, do: x
        q_mag = :math.sqrt(Enum.reduce(q_floats, 0.0, fn x, acc -> acc + x*x end))
        if q_mag == 0, do: q_vec, else: (for q <- q_floats, into: <<>>, do: <<q/q_mag::little-float-64>>)
      end)
    
    queries_bin = IO.iodata_to_binary(norm_queries)

    # 2. Call batch NIF
    scores_bin = ASM.fp_query_gemv_columnar_batch(tree.columns, queries_bin, batch_count, tree.count, dim)
    
    # 3. Process results for each query in batch
    for b <- 0..(batch_count - 1) do
      # Extract score slice for this query
      offset = b * tree.count * 8
      len = tree.count * 8
      b_scores_bin = binary_part(scores_bin, offset, len)

      # Use top-k selection (same logic as flat search)
      search_k = trunc(k * 1.5) + 1
      {result_count, indices_bin, result_scores_bin} = ASM.fp_query_topk(b_scores_bin, tree.count, search_k, threshold)

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

  defp execute_with_telemetry(_tree, query, fun) do
    # Extract basic info for telemetry
    {type, k, threshold} =
      case query do
        [:knn, _query_vec, k, th | _] -> {:knn, k, th}
        [:semantic, _text, k, th | _] -> {:semantic, k, th}
        [:range, _query_vec, _min_sim, max_sim | _] -> {:range, 0, max_sim}
        [:sparse, _sparse_query, k, th] -> {:sparse, k, th}
        [:hybrid, _query_vec, _sparse_query, k, th | _] -> {:hybrid, k, th}
        _ -> {:unknown, 0, 0}
      end

    Telemetry.span([:merkle_db, :query, :execute], %{type: type, k: k, threshold: threshold}, fn ->
      result = fun.()
      # Handle potential error tuple result
      count = if is_list(result), do: length(result), else: 0
      {result, %{result_count: count}}
    end)
  end

  defp do_sparse(tree, sparse_query, k, threshold) do
    # 1. Prepare query
    q = to_sparse_struct(sparse_query)
    tombstones = tree.tombstones || MapSet.new()

    # 2. Linear scan (optimized via NIF)
    # Note: For production, an inverted index for sparse vectors would be better.
    results = 
      tree.sparse_vectors
      |> Enum.reject(fn {idx, _vec} -> MapSet.member?(tombstones, idx) end)
      |> Enum.map(fn {idx, vec} ->
        score = ASM.fp_sparse_dotp(q.indices, q.values, vec.indices, vec.values)
        {idx, score}
      end)
      |> Enum.filter(fn {_idx, score} -> score >= threshold end)
      |> Enum.sort_by(fn {_idx, score} -> score end, :desc)
      |> Enum.take(k)
      |> Enum.map(fn {idx, score} -> {Map.get(tree.keys, idx), score} end)
    
    results
  end

  defp do_hybrid(tree, query_vec, sparse_query, k, _threshold, opts) do
    # 1. Get dense and sparse results (oversample for better fusion)
    # We take k*2 to have enough overlap for RRF
    dense_results = do_knn(tree, query_vec, k * 2, 0.0, opts)
    sparse_results = do_sparse(tree, sparse_query, k * 2, 0.0)
    
    # 2. Combine results using Reciprocal Rank Fusion (RRF)
    # score = sum(1 / (k_rrf + rank))
    rrf_k = Keyword.get(opts, :rrf_k, 60)
    
    merged_results = rrf_merge(dense_results, sparse_results, rrf_k)
    
    # 3. Filter by threshold and take top K
    # Note: RRF scores are in [0, 1] range, different from cosine similarity.
    # If a threshold was provided, we might need to normalize or just apply it to raw scores
    # but usually RRF is treated as a new ranking signal.
    merged_results
    |> Enum.sort_by(fn {_key, score} -> score end, :desc)
    |> Enum.take(k)
  end

  defp rrf_merge(dense, sparse, k_rrf) do
    # Create rank maps
    dense_ranks = 
      dense 
      |> Enum.with_index(1) 
      |> Enum.into(%{}, fn {{key, _score}, rank} -> {key, rank} end)

    sparse_ranks = 
      sparse 
      |> Enum.with_index(1) 
      |> Enum.into(%{}, fn {{key, _score}, rank} -> {key, rank} end)

    all_keys = MapSet.union(MapSet.new(Map.keys(dense_ranks)), MapSet.new(Map.keys(sparse_ranks)))

    all_keys
    |> Enum.map(fn key ->
      d_rank = Map.get(dense_ranks, key)
      s_rank = Map.get(sparse_ranks, key)

      score = rrf_score(d_rank, k_rrf) + rrf_score(s_rank, k_rrf)
      {key, score}
    end)
  end

  defp rrf_score(nil, _k), do: 0.0
  defp rrf_score(rank, k), do: 1.0 / (k + rank)

  defp to_sparse_struct(%MerkleDb.SparseVector{} = sv), do: sv
  defp to_sparse_struct({pairs, dim}) when is_list(pairs) do
    sorted = Enum.sort_by(pairs, fn {i, _v} -> i end)
    ind = for {i, _v} <- sorted, into: <<>>, do: <<i::little-signed-32>>
    val = for {_i, v} <- sorted, into: <<>>, do: <<v::little-float-64>>
    %MerkleDb.SparseVector{indices: ind, values: val, dim: dim}
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
    q_floats = 
      case tree.precision do
        :f32 -> for <<x::little-float-size(32) <- query_vec>>, do: x
        :f64 -> for <<x::little-float-size(64) <- query_vec>>, do: x
      end

    q_mag = :math.sqrt(Enum.reduce(q_floats, 0.0, fn x, acc -> acc + x*x end))
    q_norm_list = if q_mag == 0, do: q_floats, else: Enum.map(q_floats, &(&1 / q_mag))
    
    # HNSW currently only supports f64 search in NIF
    q_norm_bin = for q <- q_norm_list, into: <<>>, do: <<q::little-float-64>>

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

      results =
        Enum.zip(indices, scores)
        |> Enum.reject(fn {idx, _score} -> MapSet.member?(tombstones, idx) end)
        |> Enum.filter(fn {_idx, score} -> score >= threshold end)
        |> Enum.map(fn {idx, score} -> {Map.get(tree.keys, idx), score} end)
      results
    end
  end

  # ==================== Quantized Search ====================

  defp execute_quantized(tree, query_vec, k, threshold) do
    count = tree.count
    dim = tree.dim
    tombstones = tree.tombstones || MapSet.new()
    %{columns: q_cols, params: q_params} = tree.quantized

    # 1. Normalize and parse query (handles both f32 and f64 input)
    q_floats = 
      case tree.precision do
        :f32 -> for <<x::little-float-size(32) <- query_vec>>, do: x
        :f64 -> for <<x::little-float-size(64) <- query_vec>>, do: x
      end
    
    q_mag = :math.sqrt(Enum.reduce(q_floats, 0.0, fn x, acc -> acc + x*x end))
    q_norm_list = if q_mag == 0, do: q_floats, else: Enum.map(q_floats, &(&1 / q_mag))

    # 2. Pre-process query for quantized dot product
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
    
    # 3. Call appropriate NIF based on precision
    scores_bin = 
      if tree.precision == :f32 do
        scaled_q_bin = for s <- Enum.reverse(scaled_q_list), into: <<>>, do: <<s::little-float-32>>
        ASM.fp_query_gemv_quantized_f32(q_cols, scaled_q_bin, bias, count, dim)
      else
        scaled_q_bin = for s <- Enum.reverse(scaled_q_list), into: <<>>, do: <<s::little-float-64>>
        ASM.fp_query_gemv_quantized(q_cols, scaled_q_bin, bias, count, dim)
      end

    # 4. Top-K selection (f32/f64 result conversion)
    scores_list = 
      if tree.precision == :f32 do
        for <<s::little-float-32 <- scores_bin>>, do: (double = s; double)
      else
        for <<s::little-float-64 <- scores_bin>>, do: s
      end

    results = 
      scores_list
      |> Enum.with_index()
      |> Enum.reject(fn {_score, idx} -> MapSet.member?(tombstones, idx) end)
      |> Enum.filter(fn {score, _idx} -> score >= threshold end)
      |> Enum.sort_by(fn {score, _idx} -> score end, :desc)
      |> Enum.take(k)
      |> Enum.map(fn {score, idx} -> {Map.get(tree.keys, idx), score} end)
    
    results
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
    end) || if looks_like_filter?(opts), do: opts, else: nil
  end

  defp looks_like_filter?([]), do: false
  defp looks_like_filter?(filters) when is_list(filters) do
    Enum.all?(filters, fn
      {field, op, _} when (is_binary(field) or is_atom(field)) and (is_binary(op) or is_atom(op)) -> true
      [field, op, _] when (is_binary(field) or is_atom(field)) and (is_binary(op) or is_atom(op)) -> true
      _ -> false
    end)
  end
  defp looks_like_filter?(_), do: false

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
    elem_size = if tree.precision == :f32, do: 4, else: 8

    # Normalize query vector (handles both f32 and f64 input)
    q_floats = 
      case tree.precision do
        :f32 -> for <<x::little-float-size(32) <- query_vec>>, do: x
        :f64 -> for <<x::little-float-size(64) <- query_vec>>, do: x
      end

    q_mag = :math.sqrt(Enum.reduce(q_floats, 0.0, fn x, acc -> acc + x*x end))
    q_norm_list = if q_mag == 0, do: q_floats, else: Enum.map(q_floats, &(&1 / q_mag))
    
    q_norm_bin = 
      case tree.precision do
        :f32 -> for q <- q_norm_list, into: <<>>, do: <<q::little-float-size(32)>>
        :f64 -> for q <- q_norm_list, into: <<>>, do: <<q::little-float-size(64)>>
      end

    if row_indices do
      # IVF path: use indexed GEMV
      # (Note: fp_query_gemv_indexed currently only supports f64 in NIF, 
      # we might need to add f32 version later if using IVF with f32)
      valid_indices = 
        row_indices 
        |> Enum.reject(fn idx -> MapSet.member?(tombstones, idx) end)
        |> Enum.filter(fn idx -> matches_where?(tree, idx, where_filter) end)

      if valid_indices == [] do
        []
      else
        indices_bin = for idx <- valid_indices, into: <<>>, do: <<idx::little-signed-32>>

        # Compute scores for indexed vectors (supports both f32 and f64)
        scores_list =
          if tree.precision == :f32 do
             # F32 indexed search: compute dot products for each indexed vector individually
             flat_data = Tree.flatten(tree)

             for idx <- valid_indices do
               vec_bin = binary_part(flat_data, idx * dim * elem_size, dim * elem_size)
               ASM.fp_fold_dotp_f32(vec_bin, q_norm_bin, dim)
             end
          else
             scores_bin = ASM.fp_query_gemv_indexed(tree.columns, q_norm_bin, indices_bin, count, dim)
             for <<s::little-float-size(64) <- scores_bin>>, do: s
          end

        Enum.zip(valid_indices, scores_list)
        |> Enum.filter(fn {_idx, score} -> score >= threshold end)
        |> Enum.sort_by(fn {_idx, score} -> score end, :desc)
        |> Enum.take(k)
        |> Enum.map(fn {idx, score} -> {Map.get(tree.keys, idx), score} end)
      end
    else
      # Flat path: compute all scores
      if where_filter == nil do
        _search_k = trunc(k * 1.5) + 1
        
        # Check if columns are mmap resources (references) or binaries
        is_mmap = is_reference(elem(tree.columns, 0))

        scores_bin = 
          cond do
            is_mmap and tree.precision == :f32 ->
              ASM.fp_query_gemv_mmap_f32(tree.columns, q_norm_bin, count, dim)
            
            is_mmap ->
              raise "Mmap search only supported for f32 precision in V1"

            tree.precision == :f32 ->
              # Use our new f32 batch kernel (batch of 1)
              # Tree.flatten returns f32 if tree is f32
              ASM.fp_query_gemv_f32_batch(Tree.flatten(tree), q_norm_bin, count, dim)
            
            true ->
              ASM.fp_query_gemv_columnar(tree.columns, q_norm_bin, count, dim)
          end

        # Top-K selection NIF expects f64 or f32? 
        # fp_query_topk_f64 is what we have.
        # We need to ensure scores_bin is converted to f64 list if it was f32
        scores_list = 
          if tree.precision == :f32 do
            for <<s::little-float-32 <- scores_bin>>, do: (double = s; double)
          else
            for <<s::little-float-64 <- scores_bin>>, do: s
          end
        
        scores_list
        |> Enum.with_index()
        |> Enum.reject(fn {_score, idx} -> MapSet.member?(tombstones, idx) end)
        |> Enum.filter(fn {score, _idx} -> score >= threshold end)
        |> Enum.sort_by(fn {score, _idx} -> score end, :desc)
        |> Enum.take(k)
        |> Enum.map(fn {score, idx} -> {Map.get(tree.keys, idx), score} end)
      else
        # Metadata filter path - try bitmap optimization for equality filters
        case try_bitmap_optimization(tree, where_filter, count) do
          {:ok, bitmap} when tree.precision == :f32 ->
            # Use bitmap-optimized search for f32
            scores_bin = ASM.fp_query_gemv_bitmasked_f32(tree.columns, q_norm_bin, bitmap, count, dim)

            for <<s::little-float-32 <- scores_bin>>, do: (double = s; double)
            |> Enum.with_index()
            |> Enum.reject(fn {_score, idx} -> MapSet.member?(tombstones, idx) end)
            |> Enum.filter(fn {score, _idx} -> score >= threshold end)
            |> Enum.sort_by(fn {score, _idx} -> score end, :desc)
            |> Enum.take(k)
            |> Enum.map(fn {score, idx} -> {Map.get(tree.keys, idx), score} end)

          _ ->
            # Fallback path: Linear Scan (for non-equality filters or f64)
            flat_data = Tree.flatten(tree)

            scores_list =
              for idx <- 0..(count - 1) do
                vec_bin = binary_part(flat_data, idx * dim * elem_size, dim * elem_size)
                if tree.precision == :f32 do
                  ASM.fp_fold_dotp_f32(vec_bin, q_norm_bin, dim)
                else
                  ASM.fp_fold_dotp_f64(vec_bin, q_norm_bin, dim)
                end
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
  end

  # ==================== Range Query Implementation ====================

  defp execute_range(%{count: 0}, _query_vec, _min_sim, _max_sim, _limit, _where_filter), do: []
  defp execute_range(tree, query_vec, min_sim, max_sim, limit, where_filter) do
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
    payload =
      case Map.get(tree.keys, idx) do
        nil ->
          %{}

        key ->
          if :ets.whereis(:payload_store) == :undefined do
            %{}
          else
            case PayloadStore.get(key) do
              payload when is_map(payload) -> payload
              _ -> %{}
            end
          end
      end

    data = if payload == %{}, do: meta, else: Map.merge(payload, meta)
    Enum.all?(filters, fn filter -> match_filter?(data, filter) end)
  end

  defp match_filter?(meta, [field, op, value]), do: match_filter?(meta, {field, op, value})

  defp match_filter?(meta, {field, op, value}) do
    field_value = get_meta_value(meta, field)

    case op do
      :exists -> if value == true, do: field_value != nil, else: field_value == nil
      "exists" -> if value == true, do: field_value != nil, else: field_value == nil
      _ ->
        if field_value == nil do
          false
        else
          compare_field(field_value, op, value)
        end
    end
  end

  defp compare_field(field_value, op, value) do
    case op do
      :eq -> field_value == value
      "==" -> field_value == value
      :neq -> field_value != value
      "!=" -> field_value != value
      :gt -> is_number(field_value) and is_number(value) and field_value > value
      ">" -> is_number(field_value) and is_number(value) and field_value > value
      :lt -> is_number(field_value) and is_number(value) and field_value < value
      "<" -> is_number(field_value) and is_number(value) and field_value < value
      :gte -> is_number(field_value) and is_number(value) and field_value >= value
      ">=" -> is_number(field_value) and is_number(value) and field_value >= value
      :lte -> is_number(field_value) and is_number(value) and field_value <= value
      "<=" -> is_number(field_value) and is_number(value) and field_value <= value
      :in -> is_list(value) and field_value in value
      "in" -> is_list(value) and field_value in value
      :not_in -> is_list(value) and field_value not in value
      "not_in" -> is_list(value) and field_value not in value
      :contains -> is_binary(field_value) and is_binary(value) and String.contains?(field_value, value)
      "contains" -> is_binary(field_value) and is_binary(value) and String.contains?(field_value, value)
      :starts_with -> is_binary(field_value) and is_binary(value) and String.starts_with?(field_value, value)
      "starts_with" -> is_binary(field_value) and is_binary(value) and String.starts_with?(field_value, value)
      _ -> false
    end
  end

  defp get_meta_value(meta, field) when is_atom(field) do
    get_meta_value(meta, Atom.to_string(field))
  end

  defp get_meta_value(meta, field) when is_binary(field) do
    parts = String.split(field, ".")
    get_in_path(meta, parts)
  end

  defp get_meta_value(_meta, _field), do: nil

  defp get_in_path(value, []), do: value
  defp get_in_path(nil, _parts), do: nil
  defp get_in_path(map, [key | rest]) when is_map(map) do
    case Map.fetch(map, key) do
      {:ok, value} -> get_in_path(value, rest)
      :error ->
        case safe_existing_atom(key) do
          nil -> nil
          atom_key -> get_in_path(Map.get(map, atom_key), rest)
        end
    end
  end
  defp get_in_path(_value, _parts), do: nil

  # ==================== Bitmap Indexing ====================

  # Try to build a combined bitmap from equality filters using the inverted index.
  # Returns {:ok, bitmap} if all filters are equality-based and have indexed values,
  # otherwise returns :fallback to use linear scan.
  defp try_bitmap_optimization(tree, filters, count) when is_list(filters) do
    inverted_index = tree.inverted_index || %{}

    # Check if all filters are equality-based and can use the inverted index
    bitmaps =
      Enum.reduce_while(filters, [], fn filter, acc ->
        case extract_equality_filter(filter) do
          {:ok, field, value} ->
            field_str = if is_atom(field), do: Atom.to_string(field), else: field
            case get_in(inverted_index, [field_str, value]) do
              nil ->
                # Also try atom key
                case get_in(inverted_index, [field, value]) do
                  nil -> {:halt, :fallback}
                  bitmap -> {:cont, [bitmap | acc]}
                end
              bitmap -> {:cont, [bitmap | acc]}
            end
          :not_equality ->
            {:halt, :fallback}
        end
      end)

    case bitmaps do
      :fallback -> :fallback
      [] -> :fallback
      [single] ->
        # Ensure bitmap is properly sized
        {:ok, ensure_bitmap_size(single, count)}
      [first | rest] ->
        # AND all bitmaps together
        combined = Enum.reduce(rest, first, fn b, acc -> ASM.fp_bitmap_and(acc, b) end)
        {:ok, ensure_bitmap_size(combined, count)}
    end
  end
  defp try_bitmap_optimization(_, _, _), do: :fallback

  # Extract field and value from equality filters only
  defp extract_equality_filter({field, op, value}) when op in [:eq, "=="], do: {:ok, field, value}
  defp extract_equality_filter([field, op, value]) when op in [:eq, "=="], do: {:ok, field, value}
  defp extract_equality_filter(_), do: :not_equality

  # Ensure bitmap is padded to the expected size
  defp ensure_bitmap_size(bitmap, count) do
    expected_bytes = div(count + 63, 64) * 8
    current_size = byte_size(bitmap)
    if current_size >= expected_bytes do
      bitmap
    else
      <<bitmap::binary, 0::size((expected_bytes - current_size) * 8)>>
    end
  end

  defp safe_existing_atom(value) do
    try do
      String.to_existing_atom(value)
    rescue
      ArgumentError -> nil
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
         :ok <- validate_query_vector(query_vec, tree.dim, tree.precision),
         :ok <- validate_k(k),
         :ok <- validate_threshold(threshold) do
      :ok
    end
  end

  def validate_query(%Tree{} = tree, [:range, query_vec, min_sim, max_sim | _opts]) do
    with :ok <- validate_tree(tree),
         :ok <- validate_query_vector(query_vec, tree.dim, tree.precision),
         :ok <- validate_range(min_sim, max_sim) do
      :ok
    end
  end

  defp validate_tree(%Tree{count: 0}), do: {:error, :empty_tree}
  defp validate_tree(%Tree{dim: 0}), do: {:error, :uninitialized_tree}
  defp validate_tree(_tree), do: :ok

  defp validate_query_vector(vec_bin, expected_dim, precision) do
    if is_binary(vec_bin) do
      elem_size = if precision == :f32, do: 4, else: 8
      actual_dim = div(byte_size(vec_bin), elem_size)
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
