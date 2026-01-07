# test_raft_consensus.exs
alias MerkleDb.{KV, Tree}

IO.puts "Starting Raft Cluster on node: #{node()}"
# Clean up previous data
File.rm_rf("data/raft")
File.rm("data/wal.bin")

# Ensure app is started
Application.ensure_all_started(:merkle_db)

# Polling loop to wait for leader election
defmodule Waiter do
  def wait_for_leader(attempts) when attempts > 0 do
    server_id = {:merkle_db_server, node()}
    # Try consistent_query to check if leader exists
    case :ra.consistent_query(server_id, fn _ -> :ok end) do
      {:ok, :ok, _leader} -> 
        IO.puts "✅ Leader elected."
        :ok
      res ->
        IO.inspect(res, label: "consistent_query result")
        IO.write "."
        Process.sleep(1000)
        wait_for_leader(attempts - 1)
    end
  end
  def wait_for_leader(_), do: {:error, :timeout}
end

case Waiter.wait_for_leader(60) do
  :ok ->
    collection = "raft_test"
    IO.puts "\n1. Creating collection via Raft..."
    case KV.create_collection(collection, dim: 5, precision: :f32) do
      :ok -> IO.puts "✅ Collection created."
      err -> IO.inspect(err, label: "❌ Creation failed")
    end

    IO.puts "2. Inserting data via Raft..."
    vec = <<1.0::float-little-32, 0.0::float-little-32, 0.0::float-little-32, 0.0::float-little-32, 0.0::float-little-32>>
    case KV.put(collection, "key1", vec, %{"tag" => "raft"}) do
      :ok -> IO.puts "✅ Vector inserted."
      err -> IO.inspect(err, label: "❌ Insert failed")
    end

    IO.puts "3. Verifying state via Raft snapshot..."
    tree = KV.snapshot(collection)
    if tree do
      IO.inspect(tree.count, label: "Tree Count")
      if tree.count == 1 do
        IO.puts "✅ Raft Consensus Verified!"
      else
        IO.puts "❌ Raft State Mismatch."
        exit({:error, :failed})
      end
    else
      IO.puts "❌ Tree not found in Raft state."
      exit({:error, :failed})
    end

  err ->
    IO.puts "\n❌ Raft Cluster failed to form: #{inspect(err)}"
    exit({:error, :timeout})
end