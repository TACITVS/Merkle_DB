defmodule MerkleDb.Merkle do
  @moduledoc """
  Merkle tree implementation for MerkleDB.

  Provides:
  - Tree construction from sorted records
  - Inclusion proof generation
  - Proof verification (client-side, no DB access needed)
  - Snapshot root computation

  Tree structure:
  - Leaves are sorted by record ID
  - Internal nodes hash their children with domain separation
  - Odd leaves are promoted (not duplicated)
  """

  alias MerkleDb.Crypto
  alias MerkleDb.Canonical

  defmodule Proof do
    @moduledoc "Inclusion proof structure"
    @enforce_keys [:snapshot_root, :record_id, :leaf_index, :path]
    defstruct [
      :snapshot_root,
      :record_id,
      :leaf_index,
      :path,
      version: 1
    ]

    @type direction :: :left | :right
    @type path_element :: {direction(), <<_::256>>}
    @type t :: %__MODULE__{
            version: pos_integer(),
            snapshot_root: <<_::256>>,
            record_id: non_neg_integer(),
            leaf_index: non_neg_integer(),
            path: [path_element()]
          }
  end

  defmodule Tree do
    @moduledoc "Internal tree structure"
    @enforce_keys [:root, :leaf_count, :levels]
    defstruct [:root, :leaf_count, :levels, :leaf_hashes, :leaf_ids]

    @type t :: %__MODULE__{
            root: <<_::256>>,
            leaf_count: non_neg_integer(),
            levels: [[<<_::256>>]],
            leaf_hashes: [<<_::256>>],
            leaf_ids: [non_neg_integer()]
          }
  end

  @doc """
  Build a Merkle tree from a list of records.

  Records are automatically sorted by ID for deterministic tree structure.

  ## Parameters
    - records: list of {id, vector, payload, version} tuples

  ## Returns
    - {:ok, %Tree{}} with root hash and structure for proofs
    - {:ok, empty_tree} if no records
  """
  @spec build_tree([{non_neg_integer(), [float()], map(), non_neg_integer()}]) ::
          {:ok, Tree.t()} | {:error, atom()}
  def build_tree([]) do
    {:ok,
     %Tree{
       root: Crypto.hash_empty(),
       leaf_count: 0,
       levels: [],
       leaf_hashes: [],
       leaf_ids: []
     }}
  end

  def build_tree(records) when is_list(records) do
    # Sort by ID for deterministic ordering
    sorted_records = Enum.sort_by(records, fn {id, _, _, _} -> id end)

    # Compute leaf hashes
    leaf_results =
      Enum.map(sorted_records, fn {id, vector, payload, version} ->
        case Canonical.record_hash(id, vector, payload, version) do
          {:ok, hash} -> {:ok, id, hash}
          {:error, reason} -> {:error, reason}
        end
      end)

    # Check for errors
    case Enum.find(leaf_results, &match?({:error, _}, &1)) do
      {:error, reason} ->
        {:error, reason}

      nil ->
        leaf_data = Enum.map(leaf_results, fn {:ok, id, hash} -> {id, hash} end)
        leaf_ids = Enum.map(leaf_data, fn {id, _} -> id end)
        leaf_hashes = Enum.map(leaf_data, fn {_, hash} -> hash end)

        # Build tree levels bottom-up
        levels = build_levels([leaf_hashes], leaf_hashes)
        root = hd(hd(levels))

        {:ok,
         %Tree{
           root: root,
           leaf_count: length(leaf_hashes),
           levels: levels,
           leaf_hashes: leaf_hashes,
           leaf_ids: leaf_ids
         }}
    end
  end

  # Build tree levels from bottom to top
  defp build_levels(acc, [_single]) do
    # Single node at this level = root
    acc
  end

  defp build_levels(acc, current_level) do
    next_level = build_next_level(current_level, [])
    build_levels([next_level | acc], next_level)
  end

  defp build_next_level([], acc), do: Enum.reverse(acc)

  defp build_next_level([single], acc) do
    # Odd node: promote without hashing
    Enum.reverse([single | acc])
  end

  defp build_next_level([left, right | rest], acc) do
    parent = Crypto.hash_internal(left, right)
    build_next_level(rest, [parent | acc])
  end

  @doc """
  Get the root hash of a tree.
  """
  @spec root(Tree.t()) :: <<_::256>>
  def root(%Tree{root: r}), do: r

  @doc """
  Generate an inclusion proof for a record.

  ## Parameters
    - tree: the Merkle tree
    - record_id: ID of the record to prove

  ## Returns
    - {:ok, %Proof{}} if record exists
    - {:error, :not_found} if record not in tree
  """
  @spec prove_inclusion(Tree.t(), non_neg_integer()) ::
          {:ok, Proof.t()} | {:error, :not_found | :empty_tree}
  def prove_inclusion(%Tree{leaf_count: 0}, _record_id) do
    {:error, :empty_tree}
  end

  def prove_inclusion(%Tree{} = tree, record_id) do
    case find_leaf_index(tree.leaf_ids, record_id) do
      nil ->
        {:error, :not_found}

      leaf_index ->
        path = build_proof_path(tree, leaf_index)

        {:ok,
         %Proof{
           snapshot_root: tree.root,
           record_id: record_id,
           leaf_index: leaf_index,
           path: path
         }}
    end
  end

  defp find_leaf_index(leaf_ids, target_id) do
    Enum.find_index(leaf_ids, fn id -> id == target_id end)
  end

  # Build the sibling path from leaf to root
  defp build_proof_path(%Tree{levels: levels, leaf_hashes: leaf_hashes}, leaf_index) do
    # levels[0] = root (single element)
    # levels[-1] or last = one level above leaves
    # We traverse from leaves upward

    leaf_level = leaf_hashes
    build_path_acc(leaf_index, leaf_level, tl(Enum.reverse(levels)), [])
  end

  defp build_path_acc(_index, _current_level, [], acc) do
    # Reached root
    Enum.reverse(acc)
  end

  defp build_path_acc(index, current_level, [_parent_level | rest_levels], acc) do
    # Determine sibling
    sibling_index =
      if rem(index, 2) == 0 do
        index + 1
      else
        index - 1
      end

    # Direction: where is the sibling relative to current?
    # If index is even, sibling is to the right
    # If index is odd, sibling is to the left
    direction = if rem(index, 2) == 0, do: :right, else: :left

    # Get sibling hash (if it exists)
    path_element =
      if sibling_index < length(current_level) do
        sibling_hash = Enum.at(current_level, sibling_index)
        [{direction, sibling_hash}]
      else
        # No sibling (odd promotion), no path element needed
        []
      end

    # Move up: parent index is index div 2
    parent_index = div(index, 2)

    # Get next level (parent level becomes current)
    # We need to reconstruct parent level from current
    next_level = build_next_level(current_level, [])

    build_path_acc(parent_index, next_level, rest_levels, path_element ++ acc)
  end

  @doc """
  Verify an inclusion proof.

  This function can run client-side with no database access.
  It only needs the record data, proof, and expected root.

  ## Parameters
    - snapshot_root: expected root hash (32 bytes)
    - record: {id, vector, payload, version} tuple
    - proof: %Proof{} structure

  ## Returns
    - true if proof is valid
    - false if proof is invalid
  """
  @spec verify_inclusion(<<_::256>>, {non_neg_integer(), [float()], map(), non_neg_integer()}, Proof.t()) ::
          boolean()
  def verify_inclusion(snapshot_root, {id, vector, payload, version}, %Proof{} = proof) do
    # Step 1: Verify record ID matches proof
    if proof.record_id != id do
      false
    else
      # Step 2: Compute leaf hash from record
      case Canonical.record_hash(id, vector, payload, version) do
        {:ok, leaf_hash} ->
          # Step 3: Walk up the tree using sibling path
          computed_root = walk_proof_path(leaf_hash, proof.path)

          # Step 4: Compare with expected root
          Crypto.secure_compare(computed_root, snapshot_root)

        {:error, _} ->
          false
      end
    end
  end

  defp walk_proof_path(current_hash, []) do
    current_hash
  end

  defp walk_proof_path(current_hash, [{direction, sibling_hash} | rest]) do
    parent_hash =
      case direction do
        :left ->
          # Sibling is on the left, current is on the right
          Crypto.hash_internal(sibling_hash, current_hash)

        :right ->
          # Sibling is on the right, current is on the left
          Crypto.hash_internal(current_hash, sibling_hash)
      end

    walk_proof_path(parent_hash, rest)
  end

  @doc """
  Encode a proof to binary format for storage/transport.
  """
  @spec encode_proof(Proof.t()) :: binary()
  def encode_proof(%Proof{} = proof) do
    path_length = length(proof.path)

    path_binary =
      proof.path
      |> Enum.map(fn {direction, hash} ->
        dir_byte = if direction == :left, do: 0, else: 1
        <<dir_byte::8, hash::binary-32>>
      end)
      |> IO.iodata_to_binary()

    <<
      proof.version::8,
      proof.snapshot_root::binary-32,
      proof.record_id::little-unsigned-128,
      proof.leaf_index::little-unsigned-64,
      path_length::8,
      path_binary::binary
    >>
  end

  @doc """
  Decode a proof from binary format.
  """
  @spec decode_proof(binary()) :: {:ok, Proof.t()} | {:error, :invalid_proof}
  def decode_proof(<<
        version::8,
        snapshot_root::binary-32,
        record_id::little-unsigned-128,
        leaf_index::little-unsigned-64,
        path_length::8,
        path_binary::binary
      >>) do
    case decode_path(path_binary, path_length, []) do
      {:ok, path} ->
        {:ok,
         %Proof{
           version: version,
           snapshot_root: snapshot_root,
           record_id: record_id,
           leaf_index: leaf_index,
           path: path
         }}

      :error ->
        {:error, :invalid_proof}
    end
  end

  def decode_proof(_), do: {:error, :invalid_proof}

  defp decode_path(<<>>, 0, acc), do: {:ok, Enum.reverse(acc)}

  defp decode_path(<<dir::8, hash::binary-32, rest::binary>>, remaining, acc) when remaining > 0 do
    direction = if dir == 0, do: :left, else: :right
    decode_path(rest, remaining - 1, [{direction, hash} | acc])
  end

  defp decode_path(_, _, _), do: :error
end
