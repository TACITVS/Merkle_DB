# test_checkpoint.exs
alias MerkleDb.{KV, Persistence}

collection = "checkpoint_test"
IO.puts "1. Setting up collection..."
KV.create_collection(collection, dim: 10, precision: :f32)

IO.puts "2. Inserting data..."
vec = for _ <- 1..10, into: <<>>, do: <<1.0::float-little-32>>
res = KV.put(collection, "key1", vec, %{})
IO.inspect(res, label: "KV.put result")

IO.puts "3. Creating Checkpoint..."
case KV.checkpoint(collection) do
  :ok -> IO.puts "✅ Checkpoint created successfully."
  err -> IO.inspect(err, label: "❌ Checkpoint failed")
end

IO.puts "4. Verifying Checkpoint existence..."
dir = Persistence.checkpoint_dir(collection)
if File.exists?(Path.join(dir, "col_0.bin")) and File.exists?(Path.join(dir, "metadata.term")) do
  IO.puts "✅ Checkpoint files found."
else
  IO.puts "❌ Checkpoint files missing."
  exit({:error, :failed})
end

IO.puts "5. Simulating Load..."
case Persistence.load_checkpoint(collection) do
  {:ok, %{count: count, dim: dim}} ->
    IO.puts "✅ Loaded Checkpoint: count=#{count}, dim=#{dim}"
    if count == 1 and dim == 10 do
      IO.puts "✅ Data integrity verified."
    else
      IO.puts "❌ Data integrity mismatch."
    end
  err -> 
    IO.inspect(err, label: "❌ Load failed")
end
