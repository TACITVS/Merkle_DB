defmodule MerkleDb.Query do
  alias MerkleDb.{Tree, ASM}

  # Fixed: We now match on %Tree{} to use the alias and ensure type safety.
  def execute(%Tree{} = tree, [:knn, query_vec, k, threshold]) do
    count = tree.count
    # If database is empty, return empty list immediately
    if count == 0, do: []

    # 1. Normalize Query Vector (Unit Vector) 
    #    We do this in Elixir because it's only 64 floats (fast).
    #    Dot Product of normalized vectors == Cosine Similarity.
    q_floats = for <<x::little-float-size(64) <- query_vec>>, do: x
    q_mag = :math.sqrt(Enum.reduce(q_floats, 0.0, fn x, acc -> acc + x*x end))
    q_norm = if q_mag == 0, do: q_floats, else: Enum.map(q_floats, &(&1 / q_mag))

    # 2. Initialize Accumulator (Zeroed Array of length Count)
    #    This binary will hold the running score for EVERY verse in the Bible.
    output_size = count * 8 # 8 bytes per double
    accumulator = ASM.fp_replicate_f64(output_size, count, 0.0) 

    # 3. HIGH-SPEED COLUMNAR LOOP (The "Forest" Approach)
    #    We loop over the actual dimensions stored in the tree.
    
    final_scores_bin = 
      q_norm
      |> Enum.with_index()
      |> Enum.reduce(accumulator, fn {q_val, dim_idx}, acc_bin ->
        # Get the massive binary column for this dimension
        column_bin = elem(tree.columns, dim_idx)
        
        # ASM Call: Out = (Column * Scalar) + Accumulator
        ASM.fp_map_axpy_f64(column_bin, acc_bin, output_size, count, q_val)
      end)

    # 4. Extract Scores and Sort
    scores_list = for <<s::little-float-size(64) <- final_scores_bin>>, do: s
    
    scores_list
    |> Stream.with_index()
    |> Stream.map(fn {score, idx} -> 
       {Map.get(tree.keys, idx), score} 
    end)
    |> Stream.filter(fn {_, score} -> score >= threshold end)
    |> Enum.sort_by(fn {_, score} -> score end, :desc)
    |> Enum.take(k)
  end
end