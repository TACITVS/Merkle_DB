defmodule MerkleDb.TextEmbedding do
  alias MerkleDb.FPDispatcher

  @dim 64
  @bytes_per_float 8
  @vec_size @dim * @bytes_per_float

  def embed(text) do
    # 1. Tokenize and count (Accumulate)
    # We do this in Elixir for simplicity, though C would be faster.
    raw_counts = 
      text
      |> String.downcase()
      |> String.split(~r/[^a-z0-9]+/, trim: true)
      |> Enum.reduce(%{}, fn word, acc -> 
        idx = :erlang.phash2(word, @dim)
        Map.update(acc, idx, 1.0, &(&1 + 1.0))
      end)
    
    # 2. Construct binary vector
    # This is O(dim), fast enough for 64 dims.
    vec_bin = 
      for i <- 0..(@dim-1), into: <<>> do
        val = Map.get(raw_counts, i, 0.0)
        <<val::little-float-size(64)>>
      end

    # 3. Normalize (L2 Norm) using ASM
    # norm = sqrt(sum(x^2))
    # We use dot product with itself to get sum of squares
    sum_sq = FPDispatcher.call(:fp_fold_dotp_f64, [vec_bin, vec_bin, @dim])
    
    if sum_sq > 0.0 do
      norm = :math.sqrt(sum_sq)
      scale = 1.0 / norm
      
      # Use ASM to scale
      # fp_map_scale_f64(in, size, n, scale) -> returns binary
      FPDispatcher.call(:fp_map_scale_f64, [vec_bin, @vec_size, @dim, scale])
    else
      vec_bin
    end
  end
end
