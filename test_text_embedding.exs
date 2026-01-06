# test_text_embedding.exs

# Patch the module to use our test file
defmodule MerkleDb.TextEmbeddingTestPatch do
  def run do
    # We can't easily change @constants, but we can set the env if the module uses it
    # For now, let's just use a simple trick: rename the file or rely on manual loading logic
    # Actually, let's just make MerkleDb.TextEmbedding load from glove_test.txt by modifying it briefly
  end
end

# Better: Change the file path in the actual module for the test
content = File.read!("lib/merkle_db/text_embedding.ex")
File.write!("lib/merkle_db/text_embedding.ex", String.replace(content, "data/glove.6B.300d.txt", "data/glove_test.txt"))

alias MerkleDb.TextEmbedding

IO.puts "Initializing TextEmbedding..."
TextEmbedding.init()

IO.puts "Embedding 'philosophy logic'..."
vec_bin = TextEmbedding.embed("philosophy logic")

# Decode first 5 floats
floats = for <<f::float-little-32 <- vec_bin>>, do: f
IO.inspect(Enum.take(floats, 5), label: "First 5 dimensions")

# Average of 1.0 and 2.0 should be 1.5
if Enum.all?(floats, fn f -> abs(f - 1.5) < 1.0e-5 end) do
  IO.puts "TEXT EMBEDDING TEST PASSED!"
else
  IO.puts "TEXT EMBEDDING TEST FAILED!"
  exit({:error, :failed})
end

# Restore original file path
File.write!("lib/merkle_db/text_embedding.ex", content)
