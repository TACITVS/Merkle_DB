defmodule MerkleDb.Analytics do
  alias MerkleDb.ASM

  @doc """
  Clusters the vectors in the tree using K-Means and updates the tree with IVF index.
  """
  def build_ivf_index(tree, k, max_iter \\ 100) do
    # 1. Run K-Means
    data_bin = flatten_tree(tree)
    res = ASM.fp_kmeans_f64(data_bin, tree.count, tree.dim, k, max_iter, 1.0e-4, 42)
    
    # 2. Extract Centroids (k * dim * 8 bytes)
    centroids_bin = ASM.get_KMeansResult_centroids(res, k * tree.dim * 8)
    
    # 3. Extract Assignments (n * 4 bytes for int*)
    assignments_bin = ASM.get_KMeansResult_assignments(res, tree.count * 4)
    
    # 4. Group Keys into Clusters
    # We'll build a map: ClusterID -> List of Vector Indices
    assignments = for <<cluster_id::little-32 <- assignments_bin>>, do: cluster_id
    
    clusters = 
      assignments
      |> Enum.with_index()
      |> Enum.reduce(%{}, fn {cluster_id, vec_idx}, acc -> 
        Map.update(acc, cluster_id, [vec_idx], &[vec_idx | &1])
      end)

    %{tree | centroids: centroids_bin, clusters: clusters}
  end

  @doc """
  Calculates summary statistics for a specific dimension (column).
  """
  def column_stats(tree, dim_idx) do
    if dim_idx >= tree.dim, do: raise "Dimension index out of bounds"
    col_bin = elem(tree.columns, dim_idx)
    
    mean = ASM.fp_reduce_add_f64(col_bin, tree.count) / tree.count
    min_val = ASM.fp_reduce_min_f64(col_bin, tree.count)
    max_val = ASM.fp_reduce_max_f64(col_bin, tree.count)
    
    %{
      mean: mean,
      min: min_val,
      max: max_val,
      count: tree.count
    }
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
