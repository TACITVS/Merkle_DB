# test_auto_index.exs
alias MerkleDb.{KV, IndexBuilder}

System.put_env("GLOVE_FILE", "data/glove_test.txt")
Application.ensure_all_started(:merkle_db)

collection = "auto_index_test"
KV.drop_collection(collection)
KV.create_collection(collection, dim: 5, precision: :f32)

IO.puts "1. Inserting 1500 vectors to trigger auto-indexing..."
# Create a batch of 1500 zero vectors (dim 5)
vec = <<0::float-little-32, 0::float-little-32, 0::float-little-32, 0::float-little-32, 0::float-little-32>>
batch = Enum.map(1..1500, fn i -> {"k#{i}", vec} end)

KV.put_batch(collection, batch)

IO.puts "2. Waiting for IndexBuilder to start..."
# Poll for 5 seconds
wait_result = 
  Enum.reduce_while(1..50, :idle, fn _, _ ->
    status = IndexBuilder.status()
    # Status is a map %{status: :idle | :preparing | :running ...}
    # But Progress.get_status might return different structure depending on implementation
    # IndexBuilder.status calls Progress.get_status().
    # Let's inspect it.
    
    if status.status != :idle do
      IO.inspect(status, label: "IndexBuilder Status")
      {:halt, :started}
    else
      Process.sleep(100)
      {:cont, :idle}
    end
  end)

if wait_result == :started do
  IO.puts "✅ Auto-Indexing Triggered!"
else
  IO.puts "❌ Auto-Indexing NOT Triggered (Timeout)"
  exit({:error, :failed})
end
