# example_usage.exs

defmodule ExampleUsage do
  alias MerkleDb.ASM

  def run do
    IO.puts "=== FP ASM LIBRARY - ELIXIR BRIDGE EXAMPLE ==="

    # 1. Create a binary vector of Int64s
    data = [10, 20, 30, 40, 50]
    binary = for x <- data, into: <<>>, do: <<x::little-integer-size(64)>>
    count = length(data)

    # 2. Call fp_reduce_add_i64
    sum = ASM.fp_reduce_add_i64(binary, count)
    IO.puts "Sum of #{inspect(data)}: #{sum}"

    # 3. Call fp_reduce_max_i64
    max = ASM.fp_reduce_max_i64(binary, count)
    IO.puts "Max of #{inspect(data)}: #{max}"

    # 4. Floating point example
    floats = [1.5, 2.5, 3.5]
    float_bin = for x <- floats, into: <<>>, do: <<x::little-float-size(64)>>
    float_count = length(floats)
    
    f_sum = ASM.fp_reduce_add_f64(float_bin, float_count)
    IO.puts "Sum of #{inspect(floats)}: #{f_sum}"

    IO.puts "=== DONE ==="
  end
end

ExampleUsage.run()
