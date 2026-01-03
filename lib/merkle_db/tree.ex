defmodule MerkleDb.Tree do
  @moduledoc """
  COLUMNAR STORAGE: A structure optimized for AXPY batch processing.
  - columns: Tuple of binaries. Each binary holds N doubles.
  - keys: Map from Index -> ID (to reconstruct results).
  - count: Total number of vectors.
  - dim: Number of dimensions.
  - centroids: IVF index centroids (optional).
  - clusters: IVF index cluster assignments (optional).
  """

  defstruct columns: nil,
            keys: %{},
            count: 0,
            dim: 0,
            centroids: nil,
            clusters: %{},
            generation: 0

  # Memory limits
  @max_tree_size_gb 10
  @max_vector_count 10_000_000

  def new do
    %MerkleDb.Tree{
      columns: nil,
      keys: %{},
      count: 0,
      dim: 0,
      centroids: nil,
      clusters: %{},
      generation: 0
    }
  end

  @doc """
  Insert a single vector into the tree.
  Returns updated tree or raises on error.
  """
  def insert(tree, key, vector_bin) do
    # Check limits
    if tree.count >= @max_vector_count do
      raise ArgumentError, "Tree size limit reached: #{@max_vector_count} vectors"
    end

    # 1. Parse the incoming vector (little-endian floats)
    floats = for <<x::little-float-size(64) <- vector_bin>>, do: x
    dim = length(floats)

    # 2. Initialize or verify dimensions
    tree = if tree.columns == nil do
      %{tree | dim: dim, columns: List.to_tuple(for _ <- 1..dim, do: <<>>)}
    else
      if tree.dim != dim, do: raise ArgumentError, "Dimension mismatch: expected #{tree.dim}, got #{dim}"
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

    # 5. Check memory usage
    estimated_mb = estimate_memory_mb(%{tree | count: tree.count + 1, columns: new_cols})
    if estimated_mb > @max_tree_size_gb * 1024 do
      raise ArgumentError, "Tree memory limit reached: #{estimated_mb}MB > #{@max_tree_size_gb}GB"
    end

    %{tree | columns: new_cols, keys: new_keys, count: tree.count + 1, generation: tree.generation + 1}
  end

  @doc """
  Batch insert multiple vectors. ~50x faster than individual inserts.

  ## Example
      tree = Tree.insert_batch(tree, [
        {"key1", vector_bin1},
        {"key2", vector_bin2},
        {"key3", vector_bin3}
      ])
  """
  def insert_batch(tree, []), do: tree
  def insert_batch(tree, key_vector_pairs) when is_list(key_vector_pairs) do
    batch_size = length(key_vector_pairs)

    # Check limits upfront
    new_count = tree.count + batch_size
    if new_count > @max_vector_count do
      raise ArgumentError, "Batch would exceed tree size limit: #{new_count} > #{@max_vector_count}"
    end

    # Parse first vector to get/verify dimensions
    {_first_key, first_vec_bin} = List.first(key_vector_pairs)
    floats = for <<x::little-float-size(64) <- first_vec_bin>>, do: x
    dim = length(floats)

    # Initialize or verify dimensions
    tree = if tree.columns == nil do
      %{tree | dim: dim, columns: List.to_tuple(for _ <- 1..dim, do: <<>>)}
    else
      if tree.dim != dim, do: raise ArgumentError, "Dimension mismatch: expected #{tree.dim}, got #{dim}"
      tree
    end

    # Build column updates (one pass through all vectors)
    # This is the key optimization: we build iolists for each column
    column_updates =
      for dim_idx <- 0..(tree.dim - 1) do
        col_bin = elem(tree.columns, dim_idx)

        # Collect all values for this dimension
        new_values = for {_key, vec_bin} <- key_vector_pairs do
          binary_part(vec_bin, dim_idx * 8, 8)
        end

        # Single concat operation per column
        IO.iodata_to_binary([col_bin | new_values])
      end

    # Update keys map
    new_keys =
      key_vector_pairs
      |> Enum.with_index(tree.count)
      |> Enum.reduce(tree.keys, fn {{key, _vec}, idx}, acc ->
        Map.put(acc, idx, key)
      end)

    new_tree = %{tree |
      columns: List.to_tuple(column_updates),
      keys: new_keys,
      count: new_count,
      generation: tree.generation + 1
    }

    # Check memory
    estimated_mb = estimate_memory_mb(new_tree)
    if estimated_mb > @max_tree_size_gb * 1024 do
      raise ArgumentError, "Batch would exceed memory limit: #{estimated_mb}MB > #{@max_tree_size_gb}GB"
    end

    new_tree
  end

  @doc """
  Estimate memory usage in MB.
  """
  def estimate_memory_mb(%__MODULE__{count: count, dim: dim}) do
    # columns: count * dim * 8 bytes (f64)
    # keys: count * ~50 bytes (average key size + map overhead)
    # centroids + clusters: negligible compared to main data
    columns_mb = (count * dim * 8) / (1024 * 1024)
    keys_mb = (count * 50) / (1024 * 1024)
    Float.round(columns_mb + keys_mb, 2)
  end

  @doc """
  Get tree statistics.
  """
  def stats(%__MODULE__{} = tree) do
    %{
      count: tree.count,
      dimensions: tree.dim,
      memory_mb: estimate_memory_mb(tree),
      has_ivf_index: tree.centroids != nil,
      cluster_count: map_size(tree.clusters)
    }
  end

  @doc """
  Flatten columnar storage into row-major binary (f64).
  """
  def flatten(%__MODULE__{columns: nil}), do: <<>>
  def flatten(%__MODULE__{count: 0}), do: <<>>
  def flatten(%__MODULE__{count: count, dim: dim} = tree) when count > 0 and dim > 0 do
    for i <- 0..(count - 1), into: <<>> do
      for d <- 0..(dim - 1), into: <<>> do
        col = elem(tree.columns, d)
        binary_part(col, i * 8, 8)
      end
    end
  end
end
