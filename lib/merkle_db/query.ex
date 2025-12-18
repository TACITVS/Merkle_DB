defmodule MerkleDb.Query do
  alias MerkleDb.{Tree, ASM}

  # Fixed: We now match on %Tree{} to use the alias and ensure type safety.
  def execute(%Tree{} = tree, [:knn, query_vec, k, threshold]) do
    if tree.count == 0, do: []

    if tree.centroids do
      execute_ivf(tree, query_vec, k, threshold)
    else
      execute_flat(tree, query_vec, k, threshold)
    end
  end

  defp execute_ivf(tree, query_vec, k, threshold) do
    # 1. Find nearest centroid
    # centroids is k_clusters * dim doubles
    num_clusters = map_size(tree.clusters)
    
    # We find the ID of the centroid with highest cosine similarity
    # (Since centroids are usually normalized during training or we can just use dot product)
    cluster_id = find_nearest_cluster(tree.centroids, num_clusters, tree.dim, query_vec)
    
    # 2. Get indices in this cluster
    indices = Map.get(tree.clusters, cluster_id, [])
    
    # 3. Perform search ONLY on these indices
    # For IVF, we usually do a flat search on the cluster subset
    # Since our storage is columnar, it's actually easier to just filter the final scores
    # but for true IVF speedup we should only process rows in the cluster.
    # For now, let's keep the columnar speed and just filter for correctness.
    execute_flat(tree, query_vec, k, threshold, indices)
  end

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
    
    scores_list
    |> Stream.with_index()
    |> Stream.filter(fn {_score, idx} -> 
      if row_indices, do: idx in row_indices, else: true
    end)
    |> Stream.map(fn {score, idx} -> {Map.get(tree.keys, idx), score} end)
    |> Stream.filter(fn {_, score} -> score >= threshold end)
    |> Enum.sort_by(fn {_, score} -> score end, :desc)
    |> Enum.take(k)
  end

  defp find_nearest_cluster(centroids_bin, num_clusters, dim, query_vec) do
    # Simple linear scan of centroids (K is small, e.g. 100)
    # We use ASM dot product for each centroid
    0..(num_clusters - 1)
    |> Enum.max_by(fn c_idx -> 
      c_vec = binary_part(centroids_bin, c_idx * dim * 8, dim * 8)
      ASM.fp_fold_dotp_f64(c_vec, query_vec, dim)
    end)
  end
end