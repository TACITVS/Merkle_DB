defmodule MerkleDb.Bitmap do
  @moduledoc """
  Pure functional wrapper for dense bitmaps.
  Uses Copy-on-Write NIFs for mutation to ensure immutability at the Elixir level.
  """

  alias MerkleDb.ASM
  import Bitwise

  @doc """
  Create a new empty bitmap capable of holding `size` bits.
  Returns a binary of zeros.
  """
  def new(size) do
    bytes = div(size + 63, 64) * 8
    :binary.copy(<<0>>, bytes)
  end

  @doc """
  Set a bit at `index`.
  Returns a NEW bitmap binary (Copy-on-Write).
  """
  def set(bitmap, index) do
    ASM.fp_bitmap_set(bitmap, index)
  end

  @doc """
  Check if a bit is set at `index`.
  Pure Elixir implementation for speed (no NIF overhead for single bit check).
  """
  def test(bitmap, index) do
    byte_idx = div(index, 8)
    if byte_idx < byte_size(bitmap) do
      bit_offset = rem(index, 8)
      <<_::binary-size(byte_idx), byte, _::binary>> = bitmap
      (byte &&& (1 <<< bit_offset)) != 0
    else
      false
    end
  end
end
