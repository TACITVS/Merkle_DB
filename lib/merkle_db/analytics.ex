defmodule MerkleDb.Analytics do
  alias MerkleDb.{FPDispatcher, Telemetry, Tree}

  @doc """
  Clusters the vectors in the tree using K-Means and updates the tree with IVF index.
  Uses ASM directly with telemetry.
  """
  def build_ivf_index(tree, k, max_iter \\ 100, opts \\ []) do
    if tree.count < k do
      {tree, %{converged: false, skipped: true}}
    else
      start_time = Keyword.get(opts, :start_time, System.monotonic_time())
      tol = Keyword.get(opts, :tol, 1.0e-4)
      seed = Keyword.get(opts, :seed, 42)

      res =
        case Keyword.get(opts, :kmeans_result) do
          nil ->
            data_bin = Tree.flatten(tree)
            FPDispatcher.call(:fp_kmeans_f64, [data_bin, tree.count, tree.dim, k, max_iter, tol, seed])
          {:ok, kmeans_res} ->
            kmeans_res
          kmeans_res ->
            kmeans_res
        end

      {new_tree, metadata} = build_ivf_from_result(tree, k, res)

      # Manual telemetry
      duration = System.monotonic_time() - start_time
      :telemetry.execute(
        [:merkle_db, :analytics, :build_ivf],
        %{duration: duration},
        %{count: tree.count, k: k, converged: metadata.converged}
      )

      {new_tree, metadata}
    end
  end

  @doc false
  def build_ivf_from_result(tree, k, res) do
    # 1. Extract Centroids (k * dim * 8 bytes)
    centroids_bin = FPDispatcher.call(:get_KMeansResult_centroids, [res, k * tree.dim * 8])

    # 2. Extract Assignments (n * 4 bytes for int*)
    assignments_bin = FPDispatcher.call(:get_KMeansResult_assignments, [res, tree.count * 4])

    # 3. Get convergence info
    converged = FPDispatcher.call(:get_KMeansResult_converged, [res])

    # 4. Group Keys into Clusters
    assignments = for <<cluster_id::little-32 <- assignments_bin>>, do: cluster_id

    clusters =
      assignments
      |> Enum.with_index()
      |> Enum.reduce(%{}, fn {cluster_id, vec_idx}, acc ->
        Map.update(acc, cluster_id, [vec_idx], &[vec_idx | &1])
      end)

    new_tree = %{tree | centroids: centroids_bin, clusters: clusters}

    {new_tree, %{converged: converged == 1}}
  end

  @doc """
  Performs PCA on the vectors to reduce dimensionality.
  Returns {:ok, %PCAResult} | {:error, reason}
  """
  def reduce_dimensions(tree, n_components, max_iter \\ 100, seed \\ 42) do
    Telemetry.span([:merkle_db, :analytics, :pca], %{count: tree.count, n_components: n_components}, fn ->
      data_bin = Tree.flatten(tree)

      # Call ASM directly (Safe wrapper not available)
      result =
        FPDispatcher.call(:fp_pca_fit, [
          data_bin,
          tree.count,
          tree.dim,
          n_components,
          max_iter,
          1.0e-6,
          seed
        ])

      converged = FPDispatcher.call(:get_PCAResult_converged, [result])

      {result, %{converged: converged == 1}}
    end)
  end

  @doc """
  Calculates summary statistics for a specific dimension (column).
  Returns %{mean, min, max, count} or {:error, reason}
  """
  def column_stats(tree, dim_idx) do
    if dim_idx >= tree.dim do
      {:error, "dimension index #{dim_idx} out of range (tree has #{tree.dim} dimensions)"}
    else
      col_bin = elem(tree.columns, dim_idx)

      # Using reduce functions from ASM (these are fast, no need for Safe wrapper)
      with sum when is_number(sum) <- FPDispatcher.call(:fp_reduce_add_f64, [col_bin, tree.count]),
           min_val when is_number(min_val) <- FPDispatcher.call(:fp_reduce_min_f64, [col_bin, tree.count]),
           max_val when is_number(max_val) <- FPDispatcher.call(:fp_reduce_max_f64, [col_bin, tree.count]) do
        {:ok, %{
          mean: sum / tree.count,
          min: min_val,
          max: max_val,
          count: tree.count,
          dimension: dim_idx
        }}
      else
        error -> {:error, "ASM operation failed: #{inspect(error)}"}
      end
    end
  end

  @doc """
  Extract embeddings for visualization by projecting to first N dimensions.

  NOTE: This is a simplified projection (not true PCA transformation).
  For now, we project each vector to its first N components for visualization.
  Future enhancement: Add fp_pca_transform NIF for true PCA projection.

  ## Parameters
  - _pca_result: Binary PCAResult from fp_pca_fit (unused for now)
  - tree: The original tree (for data and cluster assignments)
  - n_components: Number of dimensions to project to (2 or 3)
  - limit: Maximum number of embeddings to return (default 500)

  ## Returns
  List of maps: `[%{id: "Chunk 0", coords: [x, y, z], cluster: 0}, ...]`
  """
  def extract_pca_embeddings(_pca_result, tree, n_components, limit \\ 500) do
    # Simple projection: Take first n_components dimensions from each vector
    # This provides approximate visualization (not true PCA, but good enough for demo)

    coords_per_vector = for i <- 0..(min(tree.count, limit) - 1) do
      # Extract first n_components values from this vector
      coords = for dim_idx <- 0..(n_components - 1) do
        if dim_idx < tree.dim do
          col_bin = elem(tree.columns, dim_idx)
          <<value::little-float-64>> = binary_part(col_bin, i * 8, 8)
          Float.round(value, 4)
        else
          0.0
        end
      end

      coords
    end

    # Get cluster assignments if tree is indexed
    cluster_assignments = if tree.clusters do
      # Build reverse map: vec_idx -> cluster_id
      tree.clusters
      |> Enum.flat_map(fn {cluster_id, vec_indices} ->
        Enum.map(vec_indices, fn idx -> {idx, cluster_id} end)
      end)
      |> Map.new()
    else
      %{}
    end

    # Combine keys, coordinates, and cluster info
    embeddings =
      coords_per_vector
      |> Enum.with_index()
      |> Enum.map(fn {coords, idx} ->
        key = Map.get(tree.keys, idx, "Unknown")
        cluster = Map.get(cluster_assignments, idx, 0)

        %{
          id: key,
          coords: coords,
          cluster: cluster
        }
      end)

    embeddings
  end

end
