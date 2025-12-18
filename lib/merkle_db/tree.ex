defmodule MerkleDb.Tree do
  # COLUMNAR STORAGE: A structure optimized for AXPY batch processing.
  # columns: Tuple of 64 binaries. Each binary holds N doubles.
  # keys: Map from Index -> Verse ID (to reconstruct results).
  # count: Total number of vectors.
  
  defstruct columns: {}, keys: %{}, count: 0

  def new do
    # Initialize 64 empty binaries
    empty_cols = List.to_tuple(for _ <- 1..64, do: <<>>)
    %MerkleDb.Tree{columns: empty_cols, keys: %{}, count: 0}
  end

  def insert(tree, key, vector_bin) do
    # 1. Parse the incoming 64-float vector
    floats = for <<x::little-float-size(64) <- vector_bin>>, do: x
    
    # 2. Append each dimension to its respective Column
    #    (This effectively transposes the data on insert)
    new_cols = 
      tree.columns
      |> Tuple.to_list()
      |> Enum.zip(floats)
      |> Enum.map(fn {col_bin, val} -> 
         <<col_bin::binary, val::little-float-size(64)>> 
      end)
      |> List.to_tuple()

    # 3. Store Key Mapping
    new_keys = Map.put(tree.keys, tree.count, key)

    %{tree | columns: new_cols, keys: new_keys, count: tree.count + 1}
  end
end