# 1. Setup: Create a large dataset (1 Million Integers)
# 8MB of raw data
count = 1_000_000
data = for _ <- 1..count, into: <<>>, do: <<1::64-native>>

# 2. Define the Contenders

# CONTENDER A: Pure Elixir (Enum.reduce)
elixir_sum_sq = fn binary ->
  # We have to unpack the binary first, which is expensive
  for <<x::64-native <- binary>>, reduce: 0 do
    acc -> acc + (x * x)
  end
end

# CONTENDER B: Your Assembly Kernel via NIF
# (We use the internal function for raw speed comparison)
asm_sum_sq = fn binary ->
  MerkleDb.Native.sum_sq_vectors(binary)
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