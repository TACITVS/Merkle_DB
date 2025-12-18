defmodule MerkleDb.Tree do
  # COLUMNAR STORAGE: A structure optimized for AXPY batch processing.
  # columns: Tuple of binaries. Each binary holds N doubles.
  # keys: Map from Index -> ID (to reconstruct results).
  # count: Total number of vectors.
  # dim: Number of dimensions.
  
  defstruct columns: nil, keys: %{}, count: 0, dim: 0, centroids: nil, clusters: %{}

  def new do
    %MerkleDb.Tree{columns: nil, keys: %{}, count: 0, dim: 0, centroids: nil, clusters: %{}}
  end

  def insert(tree, key, vector_bin) do
    # 1. Parse the incoming vector (little-endian floats)
    floats = for <<x::little-float-size(64) <- vector_bin>>, do: x
    dim = length(floats)

    # 2. Initialize or verify dimensions
    tree = if tree.columns == nil do
      %{tree | dim: dim, columns: List.to_tuple(for _ <- 1..dim, do: <<>>)}
    else
      if tree.dim != dim, do: raise "Dimension mismatch: expected #{tree.dim}, got #{dim}"
      tree
    end
    
    # 3. Append each dimension to its respective Column
    new_cols = 
      tree.columns
      |> Tuple.to_list()
      |> Enum.zip(floats)
      |> Enum.map(fn {col_bin, val} -> 
         <<col_bin::binary, val::little-float-size(64)>> 
      end)
      |> List.to_tuple()

    # 4. Store Key Mapping
    new_keys = Map.put(tree.keys, tree.count, key)

    %{tree | columns: new_cols, keys: new_keys, count: tree.count + 1}
  end
end