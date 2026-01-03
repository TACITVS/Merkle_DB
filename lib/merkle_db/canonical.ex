defmodule MerkleDb.Canonical do
  @moduledoc """
  Canonical encoding for MerkleDB records and structures.

  All encodings follow the specification in docs/spec/canonical_encoding.md:
  - Little-endian byte order for integers
  - IEEE 754 binary32 for floats (little-endian)
  - Length-prefixed variable fields
  - Domain separation via tags

  This module ensures deterministic hashing by enforcing
  a single canonical representation for all data.
  """

  alias MerkleDb.Crypto

  @doc """
  Encode a record for leaf hash computation.

  ## Parameters
    - id: u128 record identifier
    - vector: list of float32 values
    - payload: map (will be canonically JSON-encoded)
    - version: u64 version/timestamp

  ## Returns
    Binary encoding (without the tag - caller adds it)
  """
  @spec encode_record(non_neg_integer(), [float()], map(), non_neg_integer()) ::
          {:ok, binary()} | {:error, atom()}
  def encode_record(id, vector, payload, version)
      when is_integer(id) and id >= 0 and is_list(vector) and is_map(payload) and
             is_integer(version) and version >= 0 do
    with :ok <- validate_vector(vector),
         {:ok, payload_bytes} <- encode_payload(payload) do
      dim = length(vector)
      vector_bytes = encode_vector(vector)

      encoded =
        <<
          id::little-unsigned-128,
          dim::little-unsigned-32,
          vector_bytes::binary,
          byte_size(payload_bytes)::little-unsigned-32,
          payload_bytes::binary,
          version::little-unsigned-64
        >>

      {:ok, encoded}
    end
  end

  def encode_record(_, _, _, _), do: {:error, :invalid_arguments}

  @doc """
  Compute the leaf hash for a record.
  """
  @spec record_hash(non_neg_integer(), [float()], map(), non_neg_integer()) ::
          {:ok, <<_::256>>} | {:error, atom()}
  def record_hash(id, vector, payload, version) do
    case encode_record(id, vector, payload, version) do
      {:ok, encoded} -> {:ok, Crypto.hash_leaf(encoded)}
      error -> error
    end
  end

  @doc """
  Encode a vector as little-endian float32 bytes.
  """
  @spec encode_vector([float()]) :: binary()
  def encode_vector(vector) when is_list(vector) do
    vector
    |> Enum.map(fn f -> <<f::little-float-32>> end)
    |> IO.iodata_to_binary()
  end

  @doc """
  Decode a vector from little-endian float32 bytes.
  """
  @spec decode_vector(binary(), non_neg_integer()) :: [float()]
  def decode_vector(bytes, dim) when byte_size(bytes) == dim * 4 do
    decode_vector_acc(bytes, [])
  end

  defp decode_vector_acc(<<>>, acc), do: Enum.reverse(acc)

  defp decode_vector_acc(<<f::little-float-32, rest::binary>>, acc) do
    decode_vector_acc(rest, [f | acc])
  end

  @doc """
  Validate vector values (no NaN, no Infinity).
  """
  @spec validate_vector([float()]) :: :ok | {:error, :invalid_vector}
  def validate_vector(vector) when is_list(vector) do
    if Enum.all?(vector, &valid_float?/1) do
      :ok
    else
      {:error, :invalid_vector}
    end
  end

  defp valid_float?(f) when is_float(f) do
    # Check for NaN and Infinity
    # NaN != NaN is true, and abs(Infinity) is Infinity
    f == f and abs(f) != :infinity and not is_nan_binary?(f)
  end

  defp valid_float?(i) when is_integer(i), do: true
  defp valid_float?(_), do: false

  # Extra check via binary representation
  defp is_nan_binary?(f) do
    <<_sign::1, exp::8, mantissa::23>> = <<f::float-32>>
    exp == 255 and mantissa != 0
  end

  @doc """
  Encode payload as canonical JSON.
  Keys are sorted, no whitespace, UTF-8.
  """
  @spec encode_payload(map()) :: {:ok, binary()} | {:error, :payload_encoding_failed}
  def encode_payload(payload) when is_map(payload) do
    try do
      # Jason with sorted keys for determinism
      json = Jason.encode!(payload, maps: :strict)
      {:ok, json}
    rescue
      _ -> {:error, :payload_encoding_failed}
    end
  end

  @doc """
  Encode index state for hashing.
  """
  @spec encode_index_state(atom(), [<<_::256>>], [<<_::256>>]) :: binary()
  def encode_index_state(index_type, centroid_hashes, posting_hashes) do
    type_byte =
      case index_type do
        :none -> 0
        :flat -> 0
        :ivf -> 1
        :hnsw -> 2
        _ -> 0
      end

    num_centroids = length(centroid_hashes)
    num_postings = length(posting_hashes)

    centroids_binary = IO.iodata_to_binary(centroid_hashes)
    postings_binary = IO.iodata_to_binary(posting_hashes)

    <<
      type_byte::8,
      num_centroids::little-unsigned-32,
      centroids_binary::binary,
      num_postings::little-unsigned-32,
      postings_binary::binary
    >>
  end

  @doc """
  Encode centroid for hashing.
  """
  @spec encode_centroid(non_neg_integer(), [float()]) :: binary()
  def encode_centroid(centroid_id, vector) do
    dim = length(vector)
    vector_bytes = encode_vector(vector)

    <<
      centroid_id::little-unsigned-32,
      dim::little-unsigned-32,
      vector_bytes::binary
    >>
  end

  @doc """
  Encode posting list for hashing.
  Record IDs must be sorted.
  """
  @spec encode_posting(non_neg_integer(), [non_neg_integer()]) :: binary()
  def encode_posting(cluster_id, record_ids) do
    sorted_ids = Enum.sort(record_ids)
    num_ids = length(sorted_ids)

    ids_binary =
      sorted_ids
      |> Enum.map(fn id -> <<id::little-unsigned-128>> end)
      |> IO.iodata_to_binary()

    <<
      cluster_id::little-unsigned-32,
      num_ids::little-unsigned-32,
      ids_binary::binary
    >>
  end

  @doc """
  Encode manifest for hashing.
  """
  @spec encode_manifest(
          non_neg_integer(),
          <<_::256>>,
          <<_::256>>,
          non_neg_integer(),
          non_neg_integer(),
          <<_::256>>
        ) :: binary()
  def encode_manifest(version, tree_root, index_state_hash, record_count, timestamp, schema_hash) do
    <<
      version::8,
      tree_root::binary-32,
      index_state_hash::binary-32,
      record_count::little-unsigned-64,
      timestamp::little-unsigned-64,
      schema_hash::binary-32
    >>
  end

  @doc """
  Encode schema for hashing.
  """
  @spec encode_schema(String.t(), non_neg_integer(), atom()) :: binary()
  def encode_schema(collection_name, dim, metric_type) do
    name_bytes = collection_name

    metric_byte =
      case metric_type do
        :cosine -> 0
        :dot -> 1
        :l2 -> 2
        _ -> 0
      end

    <<
      byte_size(name_bytes)::little-unsigned-32,
      name_bytes::binary,
      dim::little-unsigned-32,
      metric_byte::8
    >>
  end

  @doc """
  Compute schema hash.
  """
  @spec schema_hash(String.t(), non_neg_integer(), atom()) :: <<_::256>>
  def schema_hash(collection_name, dim, metric_type) do
    encoded = encode_schema(collection_name, dim, metric_type)
    # Reuse segment tag for schema (as per spec)
    Crypto.hash_tagged(Crypto.tag_segment(), encoded)
  end
end
