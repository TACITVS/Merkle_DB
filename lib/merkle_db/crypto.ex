defmodule MerkleDb.Crypto do
  @moduledoc """
  Cryptographic primitives for MerkleDB.

  Uses BLAKE3 by default via FP_ASM_LIB's AVX2-optimized implementation.
  BLAKE3 is ~3x faster than SHA-256 while maintaining equivalent security.

  Falls back to SHA-256 if BLAKE3 NIF is not loaded.

  All hashes are 32 bytes (256 bits).
  """

  # Domain separation tags (1 byte each)
  @tag_leaf 0x00
  @tag_internal 0x01
  @tag_index 0x02
  @tag_manifest 0x03
  @tag_segment 0x04
  @tag_centroid 0x05
  @tag_posting 0x06
  @tag_empty 0xFF

  @hash_size 32

  # Public tag accessors
  def tag_leaf, do: @tag_leaf
  def tag_internal, do: @tag_internal
  def tag_index, do: @tag_index
  def tag_manifest, do: @tag_manifest
  def tag_segment, do: @tag_segment
  def tag_centroid, do: @tag_centroid
  def tag_posting, do: @tag_posting
  def tag_empty, do: @tag_empty
  def hash_size, do: @hash_size

  @doc """
  Compute hash of binary data.
  Returns a 32-byte binary.

  Uses BLAKE3 if the NIF is available, otherwise falls back to SHA-256.
  """
  @spec hash(binary()) :: <<_::256>>
  def hash(data) when is_binary(data) do
    if blake3_available?() do
      MerkleDb.Blake3.hash(data)
    else
      :crypto.hash(:sha256, data)
    end
  end

  @doc """
  Check if BLAKE3 NIF is available and working.
  """
  @spec blake3_available?() :: boolean()
  def blake3_available? do
    MerkleDb.Blake3.available?()
  end

  @doc """
  Compute hash with domain separation tag.
  """
  @spec hash_tagged(non_neg_integer(), binary()) :: <<_::256>>
  def hash_tagged(tag, data) when is_integer(tag) and tag >= 0 and tag <= 255 do
    hash(<<tag::8, data::binary>>)
  end

  @doc """
  Hash for leaf node (record).
  """
  @spec hash_leaf(binary()) :: <<_::256>>
  def hash_leaf(encoded_record) do
    hash_tagged(@tag_leaf, encoded_record)
  end

  @doc """
  Hash for internal Merkle tree node.
  Combines two 32-byte child hashes.
  """
  @spec hash_internal(<<_::256>>, <<_::256>>) :: <<_::256>>
  def hash_internal(left, right)
      when byte_size(left) == 32 and byte_size(right) == 32 do
    hash_tagged(@tag_internal, <<left::binary-32, right::binary-32>>)
  end

  @doc """
  Hash for empty tree.
  """
  @spec hash_empty() :: <<_::256>>
  def hash_empty do
    hash(<<@tag_empty::8>>)
  end

  @doc """
  Hash for index state.
  """
  @spec hash_index(binary()) :: <<_::256>>
  def hash_index(encoded_index) do
    hash_tagged(@tag_index, encoded_index)
  end

  @doc """
  Hash for manifest.
  """
  @spec hash_manifest(binary()) :: <<_::256>>
  def hash_manifest(encoded_manifest) do
    hash_tagged(@tag_manifest, encoded_manifest)
  end

  @doc """
  Hash for centroid.
  """
  @spec hash_centroid(binary()) :: <<_::256>>
  def hash_centroid(encoded_centroid) do
    hash_tagged(@tag_centroid, encoded_centroid)
  end

  @doc """
  Hash for posting list.
  """
  @spec hash_posting(binary()) :: <<_::256>>
  def hash_posting(encoded_posting) do
    hash_tagged(@tag_posting, encoded_posting)
  end

  @doc """
  Convert hash to hex string for display.
  """
  @spec to_hex(binary()) :: String.t()
  def to_hex(hash) when byte_size(hash) == 32 do
    Base.encode16(hash, case: :lower)
  end

  @doc """
  Parse hex string to hash binary.
  """
  @spec from_hex(String.t()) :: {:ok, <<_::256>>} | {:error, :invalid_hex}
  def from_hex(hex) when is_binary(hex) and byte_size(hex) == 64 do
    case Base.decode16(hex, case: :mixed) do
      {:ok, hash} when byte_size(hash) == 32 -> {:ok, hash}
      _ -> {:error, :invalid_hex}
    end
  end

  def from_hex(_), do: {:error, :invalid_hex}

  @doc """
  Constant-time comparison of two hashes.
  Prevents timing attacks.
  """
  @spec secure_compare(binary(), binary()) :: boolean()
  def secure_compare(a, b) when byte_size(a) == byte_size(b) do
    :crypto.hash_equals(a, b)
  end

  def secure_compare(_, _), do: false
end
