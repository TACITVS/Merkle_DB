defmodule MerkleDb.SparseVector do
  @moduledoc """
  Sparse vector representation.
  - indices: binary of int32 indices
  - values: binary of float64 values
  - dim: total logical dimension
  """
  defstruct [:indices, :values, :dim]
end

defmodule MerkleDb.Tree do
  @moduledoc """
  COLUMNAR STORAGE: A structure optimized for AXPY batch processing.
  - columns: Tuple of binaries. Each binary holds N doubles.
  - keys: Map from Index -> ID (to reconstruct results).
  - key_index: Map from ID -> Index (for updates/deletes).
  - tombstones: MapSet of deleted indices.
  - metadata: Map from Index -> %{field => value}.
  - quantized: Optional Int8 scalar quantization data.
  - hnsw: Optional HNSW index resource.
  - sparse_vectors: Map from Index -> %SparseVector{}.
  - count: Total number of vectors.
  - dim: Number of dimensions.
  - centroids: IVF index centroids (optional).
  - clusters: IVF index cluster assignments (optional).
  """

  defstruct columns: nil, keys: %{}, key_index: %{}, tombstones: nil, metadata: %{}, inverted_index: %{}, quantized: nil, hnsw: nil, sparse_vectors: %{}, count: 0, dim: 0, precision: :f64, centroids: nil, clusters: %{}, generation: 0, last_wal_version: 0

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
  defp floats_to_binary(floats, precision) do
    case precision do
      :f32 -> for f <- floats, into: <<>>, do: <<f::little-float-size(32)>>
      :f64 -> for f <- floats, into: <<>>, do: <<f::little-float-size(64)>>
    end
  end

  def new(opts \\ []) do
    dim = Keyword.get(opts, :dim, 0)
    precision = Keyword.get(opts, :precision, :f64)
    cols = if dim > 0, do: List.to_tuple(for _ <- 1..dim, do: <<>>), else: nil

    %MerkleDb.Tree{
      columns: cols,
      keys: %{},
      key_index: %{},
      tombstones: MapSet.new(),
      metadata: %{},
      inverted_index: %{},
      quantized: nil,
      hnsw: nil,
      sparse_vectors: %{},
      count: 0,
      dim: dim,
      precision: precision,
      centroids: nil,
      clusters: %{},
      generation: 0,
      last_wal_version: 0
    }
  end

  @doc """
  Increment tree generation (used for optimistic locking).
  """
  def bump_generation(%__MODULE__{} = tree) do
    %{tree | generation: tree.generation + 1}
  end

  @doc """
  Ensure auxiliary data structures (key_index, tombstones, metadata) are present.
  Used when loading snapshots from older versions.
  """
  def rebuild_aux_data(%__MODULE__{} = tree) do
    tree
    |> ensure_tombstones()
    |> ensure_key_index()
    |> ensure_metadata()
    |> ensure_quantized()
    |> ensure_hnsw()
    |> ensure_sparse()
  end

  defp ensure_tombstones(%{tombstones: nil} = tree), do: %{tree | tombstones: MapSet.new()}
  defp ensure_tombstones(tree), do: tree

  defp ensure_metadata(%{metadata: nil} = tree), do: %{tree | metadata: %{}}
  defp ensure_metadata(tree), do: tree

  defp ensure_quantized(%{quantized: nil} = tree), do: %{tree | quantized: nil}
  defp ensure_quantized(tree), do: tree

  defp ensure_hnsw(%{hnsw: nil} = tree), do: %{tree | hnsw: nil}
  defp ensure_hnsw(tree), do: tree

  defp ensure_sparse(%{sparse_vectors: nil} = tree), do: %{tree | sparse_vectors: %{}}
  defp ensure_sparse(tree), do: tree

  defp ensure_key_index(%{key_index: nil} = tree) do
    new_key_index = 
      tree.keys
      |> Enum.reduce(%{}, fn {idx, key}, acc ->
        Map.update(acc, key, idx, fn old_idx -> max(old_idx, idx) end)
      end)
    %{tree | key_index: new_key_index}
  end
  defp ensure_key_index(tree) do
    if Map.get(tree, :key_index) == nil or (map_size(tree.key_index) == 0 and map_size(tree.keys) > 0) do
       ensure_key_index(%{tree | key_index: nil})
    else
       tree
    end
  end

  @doc """
  Quantize all vectors in the tree to 8-bit integers.
  This significantly reduces memory usage and can speed up searches.
  """
  def quantize(%__MODULE__{count: 0} = tree), do: tree
  def quantize(%__MODULE__{columns: nil} = tree), do: tree
  def quantize(%__MODULE__{} = tree) do
    q_params = 
      for d <- 0..(tree.dim - 1) do
        col_bin = elem(tree.columns, d)
        floats = 
          case tree.precision do
            :f32 -> for <<x::little-float-32 <- col_bin>>, do: x
            :f64 -> for <<x::little-float-64 <- col_bin>>, do: x
          end
        min_v = Enum.min(floats)
        max_v = Enum.max(floats)
        range = max_v - min_v
        inv_scale = if range == 0, do: 1.0, else: 255.0 / range
        {min_v, inv_scale}
      end

    quantized_cols = 
      for d <- 0..(tree.dim - 1) do
        col_bin = elem(tree.columns, d)
        {min_v, inv_scale} = Enum.at(q_params, d)
        if tree.precision == :f32 do
          MerkleDb.ASM.fp_quantize_f32_to_u8(col_bin, min_v, inv_scale)
        else
          MerkleDb.ASM.fp_quantize_f64_to_u8(col_bin, min_v, inv_scale)
        end
      end

    new_quantized = %{
      columns: List.to_tuple(quantized_cols),
      params: q_params
    }

    %{tree | quantized: new_quantized, generation: tree.generation + 1}
  end

  @doc """
  Build a Hierarchical Navigable Small World (HNSW) index for the tree.
  HNSW provides very fast approximate nearest neighbor search without full scans.
  """
  def build_hnsw(tree, opts \\ [])
  def build_hnsw(%__MODULE__{count: 0} = tree, _opts), do: tree
  def build_hnsw(%__MODULE__{} = tree, opts) do
    m = Keyword.get(opts, :m, 16)
    ef_construction = Keyword.get(opts, :ef_construction, 64)
    
    # 1. Create native resource
    hnsw_res = MerkleDb.ASM.fp_hnsw_create(tree.dim, m, ef_construction, tree.count + 1000)
    
    # 2. Insert all vectors
    flat_data = flatten(tree)
    elem_size = if tree.precision == :f32, do: 4, else: 8
    
    Enum.each(0..(tree.count - 1), fn idx ->
      vec_bin = binary_part(flat_data, idx * tree.dim * elem_size, tree.dim * elem_size)
      if tree.precision == :f32 do
        # HNSW implementation in merkle_nif.c currently expects f64 binaries for insertion.
        # We must convert to f64 for the HNSW indexer even if stored as f32 in tree.
        vec_f64 = for <<f::float-little-32 <- vec_bin>>, into: <<>>, do: <<f::float-little-64>>
        MerkleDb.ASM.fp_hnsw_insert(hnsw_res, idx, vec_f64, tree.columns, tree.count)
      else
        MerkleDb.ASM.fp_hnsw_insert(hnsw_res, idx, vec_bin, tree.columns, tree.count)
      end
    end)
    
    %{tree | hnsw: hnsw_res, generation: tree.generation + 1}
  end

  @doc """
  Add a sparse vector representation for an existing vector ID.
  - key: database key (must already exist)
  - pairs: list of {dimension_index, value}
  - dim: total logical dimension
  """
  def insert_sparse(tree, key, pairs, dim) do
    with {:ok, idx} <- find_vector_index(tree, key) do
      # Sort pairs by index (required for native intersection kernel)
      sorted = Enum.sort_by(pairs, fn {i, _v} -> i end)
      
      indices_bin = for {i, _v} <- sorted, into: <<>>, do: <<i::little-signed-32>>
      values_bin = for {_i, v} <- sorted, into: <<>>, do: <<v::little-float-64>>
      
      sparse_vec = %MerkleDb.SparseVector{
        indices: indices_bin,
        values: values_bin,
        dim: dim
      }
      
      new_sparse_vectors = Map.put(tree.sparse_vectors, idx, sparse_vec)
      %{tree | sparse_vectors: new_sparse_vectors, generation: tree.generation + 1}
    else
      {:error, :not_found} -> raise ArgumentError, "Key not found: #{key}"
    end
  end

  defp find_vector_index(tree, key) do
    case Map.get(tree.key_index, key) do
      nil -> {:error, :not_found}
      idx -> {:ok, idx}
    end
  end

  @doc """
  Insert a single vector into the tree with optional metadata.
  Vectors are L2-normalized at insert time for proper cosine similarity.
  If the key already exists, the old index is tombstoned (soft delete) and the new vector is appended.
  Returns updated tree or raises on error.
  """
  def insert(tree, key, vector_input, meta \\ %{}) do
    if tree.count >= @max_vector_count do
      raise ArgumentError, "Tree size limit reached: #{@max_vector_count} vectors"
    end

    # Handle f32/f64 binary conversion
    floats = 
      cond do
        is_list(vector_input) -> 
          vector_input
        is_binary(vector_input) and tree.dim > 0 and byte_size(vector_input) == tree.dim * 4 ->
          for <<f::float-little-32 <- vector_input>>, do: f
        is_binary(vector_input) ->
          for <<f::float-little-64 <- vector_input>>, do: f
        true ->
          raise ArgumentError, "Invalid vector input"
      end

    dim = length(floats)
    floats = normalize_vector(floats)

    tree = if tree.columns == nil do
      %{tree | dim: dim, columns: List.to_tuple(for _ <- 1..dim, do: <<>>)}
    else
      if tree.dim != dim, do: raise ArgumentError, "Dimension mismatch: expected #{tree.dim}, got #{dim}"
      tree
    end

    tree = 
      case Map.get(tree.key_index, key) do
        nil -> tree
        old_idx ->
          %{tree | tombstones: MapSet.put(tree.tombstones, old_idx)}
      end

    new_cols = 
      tree.columns
      |> Tuple.to_list()
      |> Enum.zip(floats)
      |> Enum.map(fn {col_bin, val} ->
         case tree.precision do
           :f32 -> <<col_bin::binary, val::little-float-size(32)>>
           :f64 -> <<col_bin::binary, val::little-float-size(64)>>
         end
      end)
      |> List.to_tuple()

    new_idx = tree.count
    new_keys = Map.put(tree.keys, new_idx, key)
    new_key_index = Map.put(tree.key_index, key, new_idx)
    new_metadata = if meta == %{}, do: tree.metadata, else: Map.put(tree.metadata, new_idx, meta)
    
    new_inverted = update_inverted_index(tree.inverted_index, new_idx, meta)

    temp_tree = %{tree | count: new_idx + 1, columns: new_cols, keys: new_keys, key_index: new_key_index, metadata: new_metadata, inverted_index: new_inverted}
    estimated_mb = estimate_memory_mb(temp_tree)
    if estimated_mb > @max_tree_size_gb * 1024 do
      raise ArgumentError, "Tree memory limit reached: #{estimated_mb}MB > #{@max_tree_size_gb}GB"
    end

    %{temp_tree | generation: tree.generation + 1}
  end

  defp update_inverted_index(index, idx, metadata) do
    Enum.reduce(metadata, index, fn {field, value}, acc_index ->
      # Ensure field map exists
      field_map = Map.get(acc_index, field, %{})
      
      # Ensure bitmap exists (default empty)
      bitmap = Map.get(field_map, value, <<>>)
      
      # Update bitmap (Copy-on-Write via NIF)
      new_bitmap = MerkleDb.ASM.fp_bitmap_set(bitmap, idx)
      
      # Update nested structure
      new_field_map = Map.put(field_map, value, new_bitmap)
      Map.put(acc_index, field, new_field_map)
    end)
  end

  defp to_f64_list(input, expected_dim) do
    cond do
      is_list(input) -> input
      is_binary(input) and expected_dim > 0 and byte_size(input) == expected_dim * 4 ->
        for <<f::float-little-32 <- input>>, do: f
      is_binary(input) ->
        for <<f::float-little-64 <- input>>, do: f
      true -> nil
    end
  end

  @doc """
  Batch insert multiple vectors. ~50x faster than individual inserts.
  Handles updates by tombstoning old indices for existing keys.
  """
  def insert_batch(tree, []), do: tree
  def insert_batch(tree, key_vector_pairs) when is_list(key_vector_pairs) do
    batch_size = length(key_vector_pairs)
    new_count = tree.count + batch_size
    if new_count > @max_vector_count do
      raise ArgumentError, "Batch would exceed tree size limit: #{new_count} > #{@max_vector_count}"
    end

    # Determine dim from first vector if not already set
    {_k, first_v, _m} = case List.first(key_vector_pairs) do
      {k, v} -> {k, v, %{}}
      {k, v, m} -> {k, v, m}
    end
    
    first_floats = 
      if tree.dim > 0 do
        to_f64_list(first_v, tree.dim)
      else
        # Try f32 first if size matches common embed dims
        if is_binary(first_v) and byte_size(first_v) in [1200, 1536, 3072, 4096, 6144] do
           for <<f::float-little-32 <- first_v>>, do: f
        else
           to_f64_list(first_v, 0)
        end
      end
    
    dim = length(first_floats)

    tree = if tree.columns == nil do
      %{tree | dim: dim, columns: List.to_tuple(for _ <- 1..dim, do: <<>>)}
    else
      if tree.dim != dim, do: raise ArgumentError, "Dimension mismatch: expected #{tree.dim}, got #{dim}"
      tree
    end

    elem_size = if tree.precision == :f32, do: 4, else: 8

    normalized_triplets = 
      for item <- key_vector_pairs do
        {key, vec_input, meta} = case item do
          {k, v} -> {k, v, %{}}
          {k, v, m} -> {k, v, m}
        end
        floats = to_f64_list(vec_input, tree.dim)
        if length(floats) != tree.dim, do: raise ArgumentError, "Dimension mismatch in batch"
        
        norm_floats = normalize_vector(floats)
        norm_bin = floats_to_binary(norm_floats, tree.precision)
        {key, norm_bin, meta}
      end

    # 1. Handle existing keys in DB
    updated_tombstones = 
      Enum.reduce(normalized_triplets, tree.tombstones, fn {key, _, _}, acc ->
        case Map.get(tree.key_index, key) do
          nil -> acc
          old_idx -> MapSet.put(acc, old_idx)
        end
      end)

    # 2. Handle duplicates WITHIN the batch
    {intra_batch_tombstones, _seen} = 
      normalized_triplets
      |> Enum.with_index(tree.count)
      |> Enum.reduce({updated_tombstones, %{}}, fn {{key, _, _}, idx}, {tombs, seen} ->
        case Map.get(seen, key) do
          nil -> {tombs, Map.put(seen, key, idx)}
          prev_idx -> {MapSet.put(tombs, prev_idx), Map.put(seen, key, idx)}
        end
      end)
      
    column_updates = 
      for dim_idx <- 0..(tree.dim - 1) do
        col_bin = elem(tree.columns, dim_idx)
        new_values = for {_key, vec_bin, _meta} <- normalized_triplets do
          binary_part(vec_bin, dim_idx * elem_size, elem_size)
        end
        IO.iodata_to_binary([col_bin | new_values])
      end

    {new_keys, new_key_index, final_metadata, final_inverted} = 
      normalized_triplets
      |> Enum.with_index(tree.count)
      |> Enum.reduce({tree.keys, tree.key_index, tree.metadata, tree.inverted_index}, 
         fn {{key, _vec, meta}, idx}, {keys_acc, key_index_acc, meta_acc, inverted_acc} ->
        
        meta_acc = if meta == %{}, do: meta_acc, else: Map.put(meta_acc, idx, meta)
        inverted_acc = if meta == %{}, do: inverted_acc, else: update_inverted_index(inverted_acc, idx, meta)

        {
          Map.put(keys_acc, idx, key),
          Map.put(key_index_acc, key, idx),
          meta_acc,
          inverted_acc
        }
      end)

    new_tree = %{tree |
      columns: List.to_tuple(column_updates),
      keys: new_keys,
      key_index: new_key_index,
      tombstones: intra_batch_tombstones,
      metadata: final_metadata,
      inverted_index: final_inverted,
      count: new_count,
      generation: tree.generation + 1
    }

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
        new_metadata = Map.delete(tree.metadata, idx)
        
        %{tree | 
          tombstones: new_tombstones, 
          key_index: new_key_index,
          metadata: new_metadata,
          generation: tree.generation + 1
        }
    end
  end

  @doc """
  Estimate memory usage in MB.
  """
  def estimate_memory_mb(%__MODULE__{count: count, dim: dim, precision: prec} = tree) do
    # columns: count * dim * size bytes
    elem_size = if prec == :f32, do: 4, else: 8
    columns_mb = (count * dim * elem_size) / (1024 * 1024)
    keys_mb = (count * 50) / (1024 * 1024)
    key_index_mb = (count * 50) / (1024 * 1024)
    
    metadata_count = if tree.metadata, do: map_size(tree.metadata), else: 0
    metadata_mb = (metadata_count * 100) / (1024 * 1024)
    tombstones_count = if tree.tombstones, do: MapSet.size(tree.tombstones), else: 0
    tombstones_mb = (tombstones_count * 32) / (1024 * 1024)
    quantized_mb = if tree.quantized, do: (count * dim) / (1024 * 1024), else: 0
    
    hnsw_mb = 
      if tree.hnsw do
        (count * 16 * 8) / (1024 * 1024) 
      else
        0
      end

    sparse_count = if tree.sparse_vectors, do: map_size(tree.sparse_vectors), else: 0
    sparse_mb = (sparse_count * 200) / (1024 * 1024) # Approximate size per sparse vector
    
    Float.round(columns_mb + keys_mb + key_index_mb + metadata_mb + tombstones_mb + quantized_mb + hnsw_mb + sparse_mb, 2)
  end

  @doc "Return human-readable stats"
  def stats(nil), do: %{count: 0, dim: 0, precision: :unknown, has_ivf_index: false, has_hnsw_index: false}
  def stats(%__MODULE__{} = tree) do
    %{ 
      count: tree.count,
      active_count: tree.count - MapSet.size(tree.tombstones || MapSet.new()),
      dimensions: tree.dim,
      precision: tree.precision,
      memory_mb: estimate_memory_mb(tree),
      has_ivf_index: tree.centroids != nil,
      cluster_count: map_size(tree.clusters),
      tombstones: MapSet.size(tree.tombstones || MapSet.new()),
      metadata_entries: map_size(tree.metadata || %{}),
      quantized: tree.quantized != nil,
      has_hnsw_index: tree.hnsw != nil,
      sparse_vector_count: map_size(tree.sparse_vectors || %{})
    }
  end

  @doc """
  Flatten columnar storage into row-major binary.
  """
  def flatten(%__MODULE__{columns: nil}), do: <<>>
  def flatten(%__MODULE__{count: 0}), do: <<>>
  def flatten(%__MODULE__{count: count, dim: dim, precision: prec} = tree) when count > 0 and dim > 0 do
    elem_size = if prec == :f32, do: 4, else: 8
    for i <- 0..(count - 1), into: <<>> do
      for d <- 0..(dim - 1), into: <<>> do
        col = elem(tree.columns, d)
        binary_part(col, i * elem_size, elem_size)
      end
    end
  end
end