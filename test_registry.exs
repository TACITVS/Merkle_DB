# test_registry.exs
alias MerkleDb.{KV, Persistence}

# Ensure app is started
Application.ensure_all_started(:merkle_db)

collection = "registry_test"

IO.puts "1. Setting up collection..."
# Check if it exists from previous run and delete if so, to ensure clean test
if collection in Persistence.list_collections() do
  File.rm_rf(Persistence.checkpoint_dir(collection))
end

KV.create_collection(collection, dim: 5, precision: :f32)

IO.puts "2. Inserting data..."
vec = <<0.0::float-little-32, 0.0::float-little-32, 0.0::float-little-32, 0.0::float-little-32, 0.0::float-little-32>>
KV.put(collection, "key1", vec, %{})

IO.puts "3. Creating Checkpoint..."
KV.checkpoint(collection)

IO.puts "4. Verifying Persistence.list_collections..."
collections = Persistence.list_collections()
IO.inspect(collections, label: "Available Collections")

if collection in collections do
  IO.puts "✅ Collection '#{collection}' found in registry."
else
  IO.puts "❌ Collection '#{collection}' NOT found in registry."
  exit({:error, :failed})
end
