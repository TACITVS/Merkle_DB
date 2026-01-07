ExUnit.start()

# Wait for Raft Cluster to form
defmodule RaftWaiter do
  def wait(attempts) when attempts > 0 do
    # Try a simple consistent query
    case :ra.consistent_query({:merkle_db_server, node()}, fn _ -> :ok end) do
      {:ok, :ok, _leader} -> 
        IO.puts "✅ Raft Cluster Ready for tests."
        :ok
      _ ->
        IO.write "waiting for raft leader... "
        Process.sleep(1000)
        wait(attempts - 1)
    end
  end
  def wait(_), do: {:error, :timeout}
end

# Ensure application is started
{:ok, _} = Application.ensure_all_started(:merkle_db)
:ok = RaftWaiter.wait(30)
