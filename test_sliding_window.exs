# test_sliding_window.exs
alias MerkleDb.Ingestor

text = "one two three four five six seven eight nine ten"

IO.puts "Testing Sliding Window Chunking..."

# Chunk size 4, overlap 2
# Expected:
# 1: one two three four
# 2: three four five six
# 3: five six seven eight
# 4: seven eight nine ten
chunks = Ingestor.chunk_text(text, 4, 2)

IO.inspect(chunks, label: "Generated Chunks")

expected = [
  "one two three four",
  "three four five six",
  "five six seven eight",
  "seven eight nine ten"
]

if chunks == expected do
  IO.puts "✅ SLIDING WINDOW TEST PASSED!"
else
  IO.puts "❌ SLIDING WINDOW TEST FAILED!"
  exit({:error, :failed})
end
