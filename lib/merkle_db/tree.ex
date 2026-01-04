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

  defstruct columns: nil, keys: %{}, key_index: %{}, tombstones: nil, count: 0, dim: 0, centroids: nil, clusters: %{}, generation: 0

  # Memory limits
  @max_tree_size_gb 10
  @max_vector_count 10_000_000

  # L2-normalize a vector for cosine similarity
  defp normalize_vector(floats) do
    magnitude = :math.sqrt(Enum.reduce(floats, 0.0, fn x, acc -> acc + x * x end))
    if magnitude == 0.0 do
      floats
    else
      Enum.map(floats, fn x -> x / magnitude end)
    end
  end

  # Convert normalized floats back to binary
  defp floats_to_binary(floats) do
    for f <- floats, into: <<>>, do: <<f::little-float-size(64)>>
  end

  def new do
    %MerkleDb.Tree{
      columns: nil,
      keys: %{},
      key_index: %{},
      tombstones: MapSet.new(),
      count: 0,
      dim: 0,
      centroids: nil,
      clusters: %{},
      generation: 0
    }
  end

  @doc """
  Increment tree generation (used for optimistic locking).
  """
  def bump_generation(%__MODULE__{} = tree) do
    %{tree | generation: tree.generation + 1}
  end

  @doc """
  Ensure auxiliary data structures (key_index, tombstones) are present.
  Used when loading snapshots from older versions.
  """
  def rebuild_aux_data(%__MODULE__{} = tree) do
    tree
    |> ensure_tombstones()
    |> ensure_key_index()
  end

  defp ensure_tombstones(%{tombstones: nil} = tree), do: %{tree | tombstones: MapSet.new()}
  defp ensure_tombstones(tree), do: tree

  defp ensure_key_index(%{key_index: nil} = tree) do
    # Rebuild key_index from keys map
    # keys is %{index => key}
    # We want %{key => index}.
    # If duplicates exist (shouldn't in pure append, but logical update implies latest index wins),
    # we take the max index for each key.
    
    new_key_index =
      tree.keys
      |> Enum.reduce(%{}, fn {idx, key}, acc ->
        # If key exists, keep the larger index (newer version)
        Map.update(acc, key, idx, fn old_idx -> max(old_idx, idx) end)
      end)
      
    %{tree | key_index: new_key_index}
  end
  defp ensure_key_index(tree) do
    if map_size(tree.key_index) == 0 and map_size(tree.keys) > 0 do
       # Try to populate if empty but keys exist
       ensure_key_index(%{tree | key_index: nil})
    else
       tree
    end
  end

  @doc """
  Insert a single vector into the tree.
  Vectors are L2-normalized at insert time for proper cosine similarity.
  If the key already exists, the old index is tombstoned (soft delete) and the new vector is appended.
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

    # 2. L2-normalize for proper cosine similarity
    floats = normalize_vector(floats)

    # 2. Initialize or verify dimensions
    tree = if tree.columns == nil do
      %{tree | dim: dim, columns: List.to_tuple(for _ <- 1..dim, do: <<>>)}
    else
      if tree.dim != dim, do: raise ArgumentError, "Dimension mismatch: expected #{tree.dim}, got #{dim}"
      tree
    end

    # 3. Handle Updates: Check if key exists
    tree =
      case Map.get(tree.key_index, key) do
        nil -> tree
        old_idx ->
          # Tombstone the old index
          %{tree | tombstones: MapSet.put(tree.tombstones, old_idx)}
      end

    # 4. Append each dimension to its respective Column
    new_cols =
      tree.columns
      |> Tuple.to_list()
      |> Enum.zip(floats)
      |> Enum.map(fn {col_bin, val} ->
         <<col_bin::binary, val::little-float-size(64)>>
      end)
      |> List.to_tuple()

    # 5. Update Key Mappings
    new_idx = tree.count
    new_keys = Map.put(tree.keys, new_idx, key)
    new_key_index = Map.put(tree.key_index, key, new_idx)

    # 6. Check memory usage
    # Note: Pass updated fields for estimation
    temp_tree = %{tree | count: new_idx + 1, columns: new_cols, keys: new_keys, key_index: new_key_index}
    estimated_mb = estimate_memory_mb(temp_tree)
    if estimated_mb > @max_tree_size_gb * 1024 do
      raise ArgumentError, "Tree memory limit reached: #{estimated_mb}MB > #{@max_tree_size_gb}GB"
    end

    %{temp_tree | generation: tree.generation + 1}
  end

  @doc """
  Batch insert multiple vectors. ~50x faster than individual inserts.
  Handles updates by tombstoning old indices for existing keys.

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

    # Pre-normalize all vectors for cosine similarity
    normalized_pairs =
      for {key, vec_bin} <- key_vector_pairs do
        floats = for <<x::little-float-size(64) <- vec_bin>>, do: x
        norm_floats = normalize_vector(floats)
        norm_bin = floats_to_binary(norm_floats)
        {key, norm_bin}
      end

    # Handle Updates: Identify old indices to tombstone
    # We also need to handle duplicates WITHIN the batch.
    # The last occurrence in the batch should win.
    
    # First, let's process the batch to find unique keys (last wins) 
    # but we must append ALL vectors to keep alignment with row indices?
    # Actually, MerkleDB seems to rely on Append-Only Log. 
    # If we append 3 versions of "key1", the last one is valid.
    # The previous 2 should be tombstoned.
    # Existing "key1" in tree should also be tombstoned.
    
    # It's simpler to append ALL, and then update key_index and tombstones.
    
    # 1. Find existing keys that are being updated
    batch_keys = Enum.map(key_vector_pairs, fn {k, _} -> k end)
    
    updated_tombstones = 
      Enum.reduce(batch_keys, tree.tombstones, fn key, acc ->
        case Map.get(tree.key_index, key) do
          nil -> acc
          old_idx -> MapSet.put(acc, old_idx)
        end
      end)
      
    # 2. Append all new vectors (same as before)
    column_updates =
      for dim_idx <- 0..(tree.dim - 1) do
        col_bin = elem(tree.columns, dim_idx)

        # Collect all values for this dimension
        new_values = for {_key, vec_bin} <- normalized_pairs do
          binary_part(vec_bin, dim_idx * 8, 8)
        end

        # Single concat operation per column
        IO.iodata_to_binary([col_bin | new_values])
      end

    # 3. Update keys and key_index
    # We iterate through the batch, assigning new indices starting from tree.count
    {new_keys, new_key_index, final_tombstones} =
      normalized_pairs
      |> Enum.with_index(tree.count)
      |> Enum.reduce({tree.keys, tree.key_index, updated_tombstones}, fn {{key, _vec}, idx}, {keys_acc, key_index_acc, tombs_acc} ->
        # If this key was seen earlier IN THIS BATCH (and thus put into key_index_acc),
        # we need to tombstone that earlier batch index too.
        # Check if key is already in key_index_acc AND the index is >= tree.count (meaning it's from this batch)
        
        tombs_acc = 
          case Map.get(key_index_acc, key) do
            nil -> tombs_acc
            prev_idx -> MapSet.put(tombs_acc, prev_idx) # Tombstone the previous version (whether old or intra-batch)
          end
          
        {
          Map.put(keys_acc, idx, key),
          Map.put(key_index_acc, key, idx),
          tombs_acc
        }
      end)

    new_tree = %{tree |
      columns: List.to_tuple(column_updates),
      keys: new_keys,
      key_index: new_key_index,
      tombstones: final_tombstones,
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
  Soft delete a key by adding its index to tombstones.
  Returns updated tree or {:error, :not_found} if key doesn't exist.
  """
  def delete(tree, key) do
    case Map.get(tree.key_index, key) do
      nil -> {:error, :not_found}
      idx ->
        new_tombstones = MapSet.put(tree.tombstones, idx)
        new_key_index = Map.delete(tree.key_index, key)
        # We don't remove from 'keys' or 'columns' to preserve append-only structure/indices
        
        %{tree | 
          tombstones: new_tombstones, 
          key_index: new_key_index,
          generation: tree.generation + 1
        }
    end
  end

  @doc """
  Estimate memory usage in MB.
  """
  def estimate_memory_mb(%__MODULE__{count: count, dim: dim} = tree) do
    # columns: count * dim * 8 bytes (f64)
    # keys: count * ~50 bytes (average key size + map overhead)
    # key_index: count * ~50 bytes (inverse of keys)
    # tombstones: count * 8 bytes (worst case all deleted)
    
    columns_mb = (count * dim * 8) / (1024 * 1024)
    keys_mb = (count * 50) / (1024 * 1024)
    key_index_mb = (count * 50) / (1024 * 1024)
    
    tombstones_count = if tree.tombstones, do: MapSet.size(tree.tombstones), else: 0
    tombstones_mb = (tombstones_count * 32) / (1024 * 1024) # Approximate overhead for set
    
    Float.round(columns_mb + keys_mb + key_index_mb + tombstones_mb, 2)
  end

  @doc """
  Get tree statistics.
  """
  def stats(%__MODULE__{} = tree) do
    %{
      count: tree.count,
      active_count: tree.count - MapSet.size(tree.tombstones || MapSet.new()),
      dimensions: tree.dim,
      memory_mb: estimate_memory_mb(tree),
      has_ivf_index: tree.centroids != nil,
      cluster_count: map_size(tree.clusters),
      tombstones: MapSet.size(tree.tombstones || MapSet.new())
    }
  end

  @doc """
  Convert columnar storage to row-major binary format.
  Used by KMeans and PCA which expect row-major data.

  Returns: binary of count * dim * 8 bytes (row-major f64)
  """
  def flatten(%__MODULE__{count: 0}), do: <<>>
  def flatten(%__MODULE__{columns: nil}), do: <<>>
  def flatten(%__MODULE__{columns: columns, count: count, dim: dim}) do
    # Convert columnar to row-major
    # Column d has all values for dimension d: [v0_d, v1_d, v2_d, ...]
    # Row-major needs: [v0_0, v0_1, ..., v0_dim, v1_0, v1_1, ...]

    # Parse all columns into lists
    column_lists =
      for d <- 0..(dim - 1) do
        col_bin = elem(columns, d)
        for <<x::little-float-size(64) <- col_bin>>, do: x
      end

    # Build row-major binary
    for row_idx <- 0..(count - 1), into: <<>> do
      for col_idx <- 0..(dim - 1), into: <<>> do
        val = column_lists |> Enum.at(col_idx) |> Enum.at(row_idx)
        <<val::little-float-size(64)>>
      end
    end
  end
end