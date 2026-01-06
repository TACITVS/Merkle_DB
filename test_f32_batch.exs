# test_f32_batch.exs
alias MerkleDb.ASM

# Prepare data
# db_vectors: 3 chunks of 4 floats each
# Chunk 1: [1.0, 1.0, 1.0, 1.0]
# Chunk 2: [2.0, 2.0, 2.0, 2.0]
# Chunk 3: [3.0, 3.0, 3.0, 3.0]
db_vectors = <<
  1.0::float-little-32, 1.0::float-little-32, 1.0::float-little-32, 1.0::float-little-32,
  2.0::float-little-32, 2.0::float-little-32, 2.0::float-little-32, 2.0::float-little-32,
  3.0::float-little-32, 3.0::float-little-32, 3.0::float-little-32, 3.0::float-little-32
>>

# Query: [0.5, 0.5, 0.5, 0.5]
query = <<
  0.5::float-little-32, 0.5::float-little-32, 0.5::float-little-32, 0.5::float-little-32
>>

count = 3
dim = 4

# Expected scores:
# Chunk 1: 4 * (1.0 * 0.5) = 2.0
# Chunk 2: 4 * (2.0 * 0.5) = 4.0
# Chunk 3: 4 * (3.0 * 0.5) = 6.0

IO.puts "Calling fp_query_gemv_f32_batch..."
scores_bin = ASM.fp_query_gemv_f32_batch(db_vectors, query, count, dim)

# Decode scores (3 floats)
scores = for <<f::float-little-32 <- scores_bin>>, do: f

IO.inspect(scores, label: "Scores")

expected = [2.0, 4.0, 6.0]
if scores == expected do
  IO.puts "ELIXIR TEST PASSED!"
else
  IO.puts "ELIXIR TEST FAILED!"
  exit({:error, :failed})
end
