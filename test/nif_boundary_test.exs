defmodule MerkleDb.NifBoundaryTest do
  use ExUnit.Case
  alias MerkleDb.ASM

  describe "Columnar GEMV NIF" do
    test "fails gracefully on empty columns" do
      # Mismatched tuple size vs dim
      assert_raise ArgumentError, fn -> 
        ASM.fp_query_gemv_columnar({}, <<0::float-64>>, 0, 1) 
      end
    end

    test "fails gracefully on binary size mismatch" do
      # dim=2, but query bin is only 8 bytes (should be 16)
      cols = {<<0.0::float-64>>, <<0.0::float-64>>}
      query = <<1.0::float-64>> # 8 bytes
      assert_raise ArgumentError, fn -> 
        ASM.fp_query_gemv_columnar(cols, query, 1, 2) 
      end
    end

    test "handles large dimensionality sanity check" do
      # We added a 100,000 limit in the NIF
      cols = {}
      query = <<>>
      assert_raise ArgumentError, fn ->
        ASM.fp_query_gemv_columnar(cols, query, 0, 100_001)
      end
    end
  end

  describe "HNSW NIF" do
    test "create hnsw fails on invalid capacity" do
      # Negative capacity if cast incorrectly or zero
      # Capacity 0 is handled in fp_hnsw_create (fp_lib)
      assert_raise ArgumentError, fn ->
        ASM.fp_hnsw_create(128, 16, 64, 0)
      end
    end
  end

  describe "Quantization NIF" do
    test "f32 to u8 size validation" do
      # 4 floats = 16 bytes
      input = <<1.0::float-32, 2.0::float-32, 3.0::float-32, 4.0::float-32>>
      res = ASM.fp_quantize_f32_to_u8(input, 0.0, 1.0)
      assert byte_size(res) == 4
      
      # Mismatched size (not multiple of 4)
      assert_raise ArgumentError, fn ->
        ASM.fp_quantize_f32_to_u8(<<1, 2, 3>>, 0.0, 1.0)
      end
    end
  end
end
