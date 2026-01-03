defmodule MerkleDb.Snapshot do
  @moduledoc """
  Snapshot management for MerkleDB.

  A snapshot is an immutable, cryptographically committed view of:
  - All records (vectors + payloads)
  - Index state (centroids, posting lists)
  - Collection schema

  The snapshot_root hash uniquely identifies this state.
  """

  alias MerkleDb.{Crypto, Canonical, Merkle}

  @encoding_version 1

  defmodule Manifest do
    @moduledoc "Snapshot manifest structure"
    @enforce_keys [:snapshot_root, :tree_root, :index_state_hash, :record_count, :timestamp, :schema_hash]
    defstruct [
      :snapshot_root,
      :tree_root,
      :index_state_hash,
      :record_count,
      :timestamp,
      :schema_hash,
      version: 1
    ]

    @type t :: %__MODULE__{
            version: pos_integer(),
            snapshot_root: <<_::256>>,
            tree_root: <<_::256>>,
            index_state_hash: <<_::256>>,
            record_count: non_neg_integer(),
            timestamp: non_neg_integer(),
            schema_hash: <<_::256>>
          }
  end

  defmodule State do
    @moduledoc "Full snapshot state including tree and index data"
    @enforce_keys [:manifest, :tree]
    defstruct [:manifest, :tree, :index_state, :schema]

    @type t :: %__MODULE__{
            manifest: Manifest.t(),
            tree: Merkle.Tree.t(),
            index_state: map() | nil,
            schema: map() | nil
          }
  end

  @doc """
  Commit a collection to create an immutable snapshot.

  ## Parameters
    - records: list of {id, vector, payload, version} tuples
    - opts: keyword list of options
      - :collection_name - string, collection name
      - :dim - integer, vector dimension
      - :metric - atom, distance metric (:cosine, :dot, :l2)
      - :index_state - map with :type, :centroids, :postings (optional)

  ## Returns
    - {:ok, %State{}} with snapshot_root and full state
    - {:error, reason} on failure
  """
  @spec commit(
          [{non_neg_integer(), [float()], map(), non_neg_integer()}],
          keyword()
        ) :: {:ok, State.t()} | {:error, atom()}
  def commit(records, opts \\ []) do
    collection_name = Keyword.get(opts, :collection_name, "default")
    dim = Keyword.get(opts, :dim, 0)
    metric = Keyword.get(opts, :metric, :cosine)
    index_state = Keyword.get(opts, :index_state, nil)

    # Infer dimension from first record if not provided
    dim =
      if dim == 0 and length(records) > 0 do
        {_, vector, _, _} = hd(records)
        length(vector)
      else
        dim
      end

    with {:ok, tree} <- Merkle.build_tree(records),
         {:ok, index_hash} <- compute_index_hash(index_state),
         schema_hash <- Canonical.schema_hash(collection_name, dim, metric) do
      timestamp = System.system_time(:microsecond)

      # Compute manifest hash (this is the snapshot_root)
      manifest_data =
        Canonical.encode_manifest(
          @encoding_version,
          tree.root,
          index_hash,
          tree.leaf_count,
          timestamp,
          schema_hash
        )

      snapshot_root = Crypto.hash_manifest(manifest_data)

      manifest = %Manifest{
        version: @encoding_version,
        snapshot_root: snapshot_root,
        tree_root: tree.root,
        index_state_hash: index_hash,
        record_count: tree.leaf_count,
        timestamp: timestamp,
        schema_hash: schema_hash
      }

      state = %State{
        manifest: manifest,
        tree: tree,
        index_state: index_state,
        schema: %{name: collection_name, dim: dim, metric: metric}
      }

      {:ok, state}
    end
  end

  @doc """
  Get the snapshot root hash from a state.
  """
  @spec root(State.t()) :: <<_::256>>
  def root(%State{manifest: %Manifest{snapshot_root: r}}), do: r

  @doc """
  Get the tree root from a state.
  """
  @spec tree_root(State.t()) :: <<_::256>>
  def tree_root(%State{manifest: %Manifest{tree_root: r}}), do: r

  @doc """
  Generate an inclusion proof for a record in the snapshot.
  """
  @spec prove_inclusion(State.t(), non_neg_integer()) ::
          {:ok, Merkle.Proof.t()} | {:error, atom()}
  def prove_inclusion(%State{tree: tree}, record_id) do
    Merkle.prove_inclusion(tree, record_id)
  end

  @doc """
  Verify an inclusion proof.

  The proof contains the tree_root it was generated against.
  This function verifies:
  1. The record hashes correctly to a leaf
  2. The proof path leads to the tree_root in the proof

  For full snapshot verification, also check that proof.snapshot_root
  matches your expected tree_root (from manifest.tree_root).

  This is a static function that can run client-side.
  """
  @spec verify_inclusion(
          {non_neg_integer(), [float()], map(), non_neg_integer()},
          Merkle.Proof.t()
        ) :: boolean()
  def verify_inclusion(record, proof) do
    # Use the tree_root embedded in the proof
    Merkle.verify_inclusion(proof.snapshot_root, record, proof)
  end

  @doc """
  Verify an inclusion proof against a specific tree root.

  Use this when you want to verify against a known tree_root
  (e.g., from a trusted manifest).
  """
  @spec verify_inclusion(
          <<_::256>>,
          {non_neg_integer(), [float()], map(), non_neg_integer()},
          Merkle.Proof.t()
        ) :: boolean()
  def verify_inclusion(tree_root, record, proof) do
    Merkle.verify_inclusion(tree_root, record, proof)
  end

  # Compute hash of index state
  defp compute_index_hash(nil) do
    # No index - use empty marker
    {:ok, Crypto.hash_empty()}
  end

  defp compute_index_hash(%{type: type, centroids: centroids, postings: postings}) do
    # Hash each centroid
    centroid_hashes =
      centroids
      |> Enum.with_index()
      |> Enum.map(fn {vector, idx} ->
        encoded = Canonical.encode_centroid(idx, vector)
        Crypto.hash_centroid(encoded)
      end)

    # Hash each posting list
    posting_hashes =
      postings
      |> Enum.with_index()
      |> Enum.map(fn {ids, cluster_id} ->
        encoded = Canonical.encode_posting(cluster_id, ids)
        Crypto.hash_posting(encoded)
      end)

    # Combine into index state hash
    encoded = Canonical.encode_index_state(type, centroid_hashes, posting_hashes)
    {:ok, Crypto.hash_index(encoded)}
  end

  defp compute_index_hash(_invalid) do
    {:error, :invalid_index_state}
  end

  @doc """
  Serialize a snapshot state for persistence.
  """
  @spec serialize(State.t()) :: binary()
  def serialize(%State{} = state) do
    # Use Erlang term_to_binary for now (can optimize later)
    :erlang.term_to_binary(state, [:compressed])
  end

  @doc """
  Deserialize a snapshot state.
  """
  @spec deserialize(binary()) :: {:ok, State.t()} | {:error, :invalid_data}
  def deserialize(data) when is_binary(data) do
    try do
      state = :erlang.binary_to_term(data, [:safe])

      case state do
        %State{} -> {:ok, state}
        _ -> {:error, :invalid_data}
      end
    rescue
      _ -> {:error, :invalid_data}
    end
  end

  @doc """
  Get snapshot info for display.
  """
  @spec info(State.t()) :: map()
  def info(%State{manifest: m, schema: s}) do
    %{
      snapshot_root: Crypto.to_hex(m.snapshot_root),
      tree_root: Crypto.to_hex(m.tree_root),
      index_hash: Crypto.to_hex(m.index_state_hash),
      record_count: m.record_count,
      timestamp: DateTime.from_unix!(m.timestamp, :microsecond),
      schema: s
    }
  end
end
