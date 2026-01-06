# test_api_v1.exs
use Plug.Test
alias MerkleDb.Web.Router
alias MerkleDb.KV

# Mock start the KV and storage if needed (Application.ensure_all_started(:merkle_db) usually does this)
# But here we just want to test the routing logic and side effects on KV

# Ensure app is started
System.put_env("GLOVE_FILE", "data/glove_test.txt")

# Clean up corrupted state from previous runs
File.rm("data/wal.bin")
File.rm_rf("data/checkpoint-api_test")
File.rm("data/snapshot-api_test-current.bin")

Application.ensure_all_started(:merkle_db)

# Clean up previous runs
KV.drop_collection("api_test")

# Create collection
KV.create_collection("api_test", dim: 300, precision: :f32)

IO.puts "1. Testing Ingest (Authorized)..."
vec_zeros = for _ <- 1..300, do: 0.0
body = Jason.encode!([
  %{id: "vec1", vector: vec_zeros},
  %{id: "vec2", text: "philosophy"} # This will use the dummy embedding (300d)
])

conn = conn(:post, "/v1/api_test/vectors", body)
       |> put_req_header("content-type", "application/json")
       |> put_req_header("authorization", "Bearer secret")
       |> Router.call(Router.init([]))

IO.inspect(conn.status, label: "Ingest Status")
IO.inspect(conn.resp_body, label: "Ingest Body")

if conn.status == 200 do
  IO.puts "✅ Ingest OK"
else
  IO.puts "❌ Ingest Failed"
end

IO.puts "1b. Testing Ingest (Unauthorized)..."
conn = conn(:post, "/v1/api_test/vectors", body)
       |> put_req_header("content-type", "application/json")
       |> Router.call(Router.init([]))

IO.inspect(conn.status, label: "Unauthorized Status")
if conn.status == 401 do
  IO.puts "✅ Security OK"
else
  IO.puts "❌ Security Failed"
end

IO.puts "2. Testing Search..."
search_body = Jason.encode!(%{
  text: "philosophy",
  k: 5
})

conn = conn(:post, "/v1/api_test/search", search_body)
       |> put_req_header("content-type", "application/json")
       |> put_req_header("authorization", "Bearer secret")
       |> Router.call(Router.init([]))

IO.inspect(conn.status, label: "Search Status")
IO.inspect(conn.resp_body, label: "Search Body")

if conn.status == 200 do
  IO.puts "✅ Search OK"
else
  IO.puts "❌ Search Failed"
end

IO.puts "3. Testing Checkpoint..."
conn = conn(:post, "/v1/api_test/checkpoint", "")
       |> put_req_header("authorization", "Bearer secret")
       |> Router.call(Router.init([]))
