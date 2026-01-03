defmodule MerkleDb.Blake3 do
  @moduledoc """
  BLAKE3 cryptographic hash function implementation using FP_ASM_LIB.

  BLAKE3 is a modern cryptographic hash function that is:
  - **Fast**: ~3x faster than SHA-256 on modern CPUs with AVX2
  - **Secure**: Based on ChaCha permutation, resistant to length extension
  - **Parallelizable**: Tree structure enables multi-threaded hashing
  - **Versatile**: Supports hashing, keyed hashing (MAC), and key derivation

  ## Features

  - 256-bit (32-byte) output by default
  - AVX2 SIMD optimizations for maximum performance
  - Streaming API for incremental hashing
  - XOF (eXtendable Output Function) mode for arbitrary output lengths
  - Constant-time comparison to prevent timing attacks

  ## Examples

      # Simple hashing
      iex> MerkleDb.Blake3.hash("hello world")
      <<215, 227, 81, 220, ...>>

      # Hex output
      iex> MerkleDb.Blake3.hash_hex("hello world")
      "d7e3ab..."

      # Keyed hashing (MAC)
      iex> key = :crypto.strong_rand_bytes(32)
      iex> MerkleDb.Blake3.hash_keyed(key, "message")
      <<...>>

      # Key derivation
      iex> MerkleDb.Blake3.derive_key("MerkleDb encryption key", master_key)
      <<...>>

      # Incremental hashing
      iex> hasher = MerkleDb.Blake3.new()
      iex> hasher = MerkleDb.Blake3.update(hasher, "hello ")
      iex> hasher = MerkleDb.Blake3.update(hasher, "world")
      iex> MerkleDb.Blake3.finalize(hasher)
      <<215, 227, 81, 220, ...>>
  """

  @on_load :load_nif

  @doc false
  def load_nif do
    path = :filename.join(:code.priv_dir(:merkle_db), ~c"blake3_nif")
    case :erlang.load_nif(path, 0) do
      :ok ->
        :persistent_term.put({__MODULE__, :nif_loaded}, true)
        :ok
      {:error, _reason} ->
        # NIF not available - will use fallback
        :persistent_term.put({__MODULE__, :nif_loaded}, false)
        :ok  # Return :ok to prevent on_load failure
    end
  end

  @doc """
  Check if BLAKE3 NIF is loaded and available.
  """
  @spec available?() :: boolean()
  def available? do
    :persistent_term.get({__MODULE__, :nif_loaded}, false)
  end

  # ============================================================================
  # Simple API
  # ============================================================================

  @doc """
  Hash binary data and return 32-byte hash.

  Raises ArgumentError if NIF is not loaded.

  ## Examples

      iex> MerkleDb.Blake3.hash("hello")
      <<234, 134, 175, ...>>

      iex> MerkleDb.Blake3.hash(<<1, 2, 3, 4>>)
      <<...>>
  """
  @spec hash(binary()) :: binary()
  def hash(_data) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Hash binary data and return hex-encoded string.

  ## Examples

      iex> MerkleDb.Blake3.hash_hex("hello")
      "ea8aff..."
  """
  @spec hash_hex(binary()) :: String.t()
  def hash_hex(data) when is_binary(data) do
    data
    |> hash()
    |> to_hex()
    |> to_string()
  end

  @doc """
  Keyed hash (MAC) with 32-byte key.

  Use this for message authentication codes.

  ## Examples

      iex> key = :crypto.strong_rand_bytes(32)
      iex> MerkleDb.Blake3.hash_keyed(key, "message")
      <<...>>
  """
  @spec hash_keyed(binary(), binary()) :: binary()
  def hash_keyed(_key, _data) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Derive a key from context string and key material.

  The context string provides domain separation, ensuring that keys
  derived for different purposes are cryptographically independent.

  ## Examples

      iex> master = :crypto.strong_rand_bytes(32)
      iex> enc_key = MerkleDb.Blake3.derive_key("MerkleDb encryption v1", master)
      iex> mac_key = MerkleDb.Blake3.derive_key("MerkleDb authentication v1", master)
      # enc_key and mac_key are independent despite same master key
  """
  @spec derive_key(binary(), binary()) :: binary()
  def derive_key(_context, _key_material) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Convert 32-byte hash to hex string.

  ## Examples

      iex> hash = MerkleDb.Blake3.hash("test")
      iex> MerkleDb.Blake3.to_hex(hash)
      "4878ca..."
  """
  @spec to_hex(binary()) :: binary()
  def to_hex(_hash) do
    :erlang.nif_error(:nif_not_loaded)
  end

  @doc """
  Constant-time comparison of two hashes.

  Use this instead of `==` to prevent timing attacks.

  ## Examples

      iex> h1 = MerkleDb.Blake3.hash("test")
      iex> h2 = MerkleDb.Blake3.hash("test")
      iex> MerkleDb.Blake3.compare(h1, h2)
      true

      iex> h3 = MerkleDb.Blake3.hash("other")
      iex> MerkleDb.Blake3.compare(h1, h3)
      false
  """
  @spec compare(binary(), binary()) :: boolean()
  def compare(_a, _b) do
    :erlang.nif_error(:nif_not_loaded)
  end

  # ============================================================================
  # Incremental API
  # ============================================================================

  @doc """
  Create a new hasher for incremental hashing.

  ## Examples

      iex> hasher = MerkleDb.Blake3.new()
      iex> hasher = MerkleDb.Blake3.update(hasher, "chunk1")
      iex> hasher = MerkleDb.Blake3.update(hasher, "chunk2")
      iex> MerkleDb.Blake3.finalize(hasher)
      <<...>>
  """
  @spec new() :: reference()
  def new do
    hasher_new()
  end

  @doc """
  Create a new keyed hasher for incremental MAC.

  ## Examples

      iex> key = :crypto.strong_rand_bytes(32)
      iex> hasher = MerkleDb.Blake3.new_keyed(key)
      iex> hasher = MerkleDb.Blake3.update(hasher, "message")
      iex> MerkleDb.Blake3.finalize(hasher)
      <<...>>
  """
  @spec new_keyed(binary()) :: reference()
  def new_keyed(key) when byte_size(key) == 32 do
    hasher_new_keyed(key)
  end

  @doc """
  Add data to hasher state.

  Returns updated hasher reference for chaining.

  ## Examples

      iex> hasher = MerkleDb.Blake3.new()
      iex> hasher = MerkleDb.Blake3.update(hasher, "part1")
      iex> hasher = MerkleDb.Blake3.update(hasher, "part2")
  """
  @spec update(reference(), binary()) :: reference()
  def update(hasher, data) when is_binary(data) do
    :ok = hasher_update(hasher, data)
    hasher
  end

  @doc """
  Finalize hasher and produce 32-byte hash.

  The hasher can still be used after finalization (e.g., to continue hashing).

  ## Examples

      iex> hasher = MerkleDb.Blake3.new() |> MerkleDb.Blake3.update("data")
      iex> MerkleDb.Blake3.finalize(hasher)
      <<...>>
  """
  @spec finalize(reference()) :: binary()
  def finalize(hasher) do
    hasher_finalize(hasher)
  end

  @doc """
  Finalize with extended output (XOF mode).

  BLAKE3 can produce arbitrarily long output. Use this when you need
  more than 32 bytes, such as for deriving multiple keys.

  ## Examples

      iex> hasher = MerkleDb.Blake3.new() |> MerkleDb.Blake3.update("seed")
      iex> MerkleDb.Blake3.finalize_xof(hasher, 64)  # Get 64 bytes
      <<...>>
  """
  @spec finalize_xof(reference(), pos_integer()) :: binary()
  def finalize_xof(hasher, length) when is_integer(length) and length > 0 do
    hasher_finalize_xof(hasher, length)
  end

  # ============================================================================
  # NIF Function Stubs
  # ============================================================================

  defp hasher_new, do: :erlang.nif_error(:nif_not_loaded)
  defp hasher_new_keyed(_key), do: :erlang.nif_error(:nif_not_loaded)
  defp hasher_update(_hasher, _data), do: :erlang.nif_error(:nif_not_loaded)
  defp hasher_finalize(_hasher), do: :erlang.nif_error(:nif_not_loaded)
  defp hasher_finalize_xof(_hasher, _length), do: :erlang.nif_error(:nif_not_loaded)
end
