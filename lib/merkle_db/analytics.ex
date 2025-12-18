defmodule MerkleDb.Analytics do
  alias MerkleDb.{ASM, Telemetry}
  alias MerkleDb.ASM.Safe

  @doc """
  Clusters the vectors in the tree using K-Means and updates the tree with IVF index.
  Uses safe wrapper with validation and telemetry.
  """
  def build_ivf_index(tree, k, max_iter \\ 100) do
    if tree.count < k do
      tree
    else
      Telemetry.span([:merkle_db, :analytics, :build_ivf], %{count: tree.count, k: k}, fn ->
        # 1. Run K-Means with safe wrapper
        data_bin = flatten_tree(tree)

        case Safe.fp_kmeans_f64(data_bin, tree.count, tree.dim, k, max_iter, 1.0e-4, 42) do
          {:ok, res} ->
            # 2. Extract Centroids (k * dim * 8 bytes)
            centroids_bin = ASM.get_KMeansResult_centroids(res, k * tree.dim * 8)

            # 3. Extract Assignments (n * 4 bytes for int*)
            assignments_bin = ASM.get_KMeansResult_assignments(res, tree.count * 4)

            # 4. Get convergence info
            converged = ASM.get_KMeansResult_converged(res)

            # 5. Group Keys into Clusters
            assignments = for <<cluster_id::little-32 <- assignments_bin>>, do: cluster_id

            clusters =
              assignments
              |> Enum.with_index()
              |> Enum.reduce(%{}, fn {cluster_id, vec_idx}, acc ->
                Map.update(acc, cluster_id, [vec_idx], &[vec_idx | &1])
              end)

            new_tree = %{tree | centroids: centroids_bin, clusters: clusters}
            {new_tree, %{converged: converged == 1}}

          {:error, reason} ->
            raise "K-Means failed: #{inspect(reason)}"
        end
      end)
    end
  end

  @doc """
  Performs PCA on the vectors to reduce dimensionality.
  Returns {:ok, %PCAResult} | {:error, reason}
  """
  def reduce_dimensions(tree, n_components, max_iter \\ 100, seed \\ 42) do
    Telemetry.span([:merkle_db, :analytics, :pca], %{count: tree.count, n_components: n_components}, fn ->
      data_bin = flatten_tree(tree)

      case Safe.fp_pca_fit(data_bin, tree.count, tree.dim, n_components, max_iter, 1.0e-6, seed) do
        {:ok, result} ->
          converged = ASM.get_PCAResult_converged(result)
          {result, %{converged: converged == 1}}

        {:error, reason} ->
          raise "PCA failed: #{inspect(reason)}"
      end
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
      with sum when is_number(sum) <- ASM.fp_reduce_add_f64(col_bin, tree.count),
           min_val when is_number(min_val) <- ASM.fp_reduce_min_f64(col_bin, tree.count),
           max_val when is_number(max_val) <- ASM.fp_reduce_max_f64(col_bin, tree.count) do
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

  # Helper: Transpose Columnar -> Row-Major
  defp flatten_tree(tree) do
    for i <- 0..(tree.count - 1), into: <<>> do
      for d <- 0..(tree.dim - 1), into: <<>> do
        col = elem(tree.columns, d)
        binary_part(col, i * 8, 8)
      end
    end
  end
end