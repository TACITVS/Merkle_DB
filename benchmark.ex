# 1. Setup: Create a large dataset (1 Million Integers)
# 8MB of raw data
count = 1_000_000
# Note: little-endian 1 as 64-bit int
data = :binary.copy(<<1::little-64>>, count)

# 2. Define the Contenders

# CONTENDER A: Pure Elixir (Enum.reduce)
elixir_sum_sq = fn binary ->
  for <<x::little-64 <- binary>>, reduce: 0 do
    acc -> acc + (x * x)
  end
end

# CONTENDER B: Your Assembly Kernel via ASM module
asm_sum_sq = fn binary ->
  MerkleDb.ASM.fp_fold_sumsq_i64(binary, count)
end

# 3. The Race
IO.puts "--- RACING: 1 Million Integers (Sum of Squares) ---"

{time_elixir, res_elixir} = :timer.tc(fn -> elixir_sum_sq.(data) end)
IO.puts "Elixir:   #{time_elixir} microseconds (Result: #{res_elixir})"

{time_asm, res_asm} = :timer.tc(fn -> asm_sum_sq.(data) end)
IO.puts "Assembly: #{time_asm} microseconds    (Result: #{res_asm})"

# 4. The Verdict
speedup = time_elixir / time_asm
IO.puts "\nSpeedup: #{Float.round(speedup, 2)}x FASTER"

# --- AXPY BENCHMARK (The Bottleneck) ---
IO.puts "\n--- BENCHMARKING AXPY: 10 Million Doubles (80MB) ---"
f_count = 10_000_000
x_bin = :binary.copy(<<1.0::little-float-64>>, f_count)
y_bin = :binary.copy(<<2.0::little-float-64>>, f_count)
output_size = f_count * 8

{time_axpy, _res} = :timer.tc(fn -> 
  # fp_map_axpy_f64(x, y, out_size, n, c) -> out = c*x + y
  MerkleDb.ASM.fp_map_axpy_f64(x_bin, y_bin, output_size, f_count, 2.0)
end)
IO.puts "ASM AXPY: #{time_axpy} microseconds"
IO.puts "Throughput: #{Float.round((f_count * 8 * 3) / (time_axpy / 1_000_000) / 1024 / 1024, 2)} MB/s (In+In+Out)"
