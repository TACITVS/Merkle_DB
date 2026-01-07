defmodule MerkleDb.Raft do
  @moduledoc """
  Public API for the Raft-based distributed consensus layer.
  """

  @cluster_name :merkle_db_cluster
  
  def server_id, do: {:merkle_db_server, node()}

  def start_link do
    # 1. Ensure ra application is started
    Application.ensure_all_started(:ra)

    # 2. Define the machine
    machine = {:module, MerkleDb.Raft.Machine, %{}}

    # 3. Start the Raft server on this node
    data_dir = Path.join(File.cwd!(), "data/raft")
    File.mkdir_p!(data_dir)
    
    # Configure ra to use our data directory
    Application.put_env(:ra, :data_dir, String.to_charlist(data_dir))

    # Explicitly start ra system (required by newer versions)
    :ra_system.start_default()

    try_start_cluster(machine, 5)
  end

  defp try_start_cluster(machine, attempts) when attempts > 0 do
    case :ra.start_cluster(:default, @cluster_name, machine, [server_id()]) do
      {:ok, _, _} -> :ok
      {:error, {:already_exists, _}} -> :ok
      {:error, :system_not_started} ->
        Process.sleep(500)
        try_start_cluster(machine, attempts - 1)
      err ->
        IO.inspect(err, label: "Raft Cluster Start Failed")
        err
    end
  end
  defp try_start_cluster(_, _), do: {:error, :timeout}

  @doc """
  Send a command to the Raft cluster leader.
  """
  def process_command(command) do
    case :ra.process_command(server_id(), command) do
      {:ok, result, _leader} -> result
      err ->
        # If no leader, we might be in election
        Process.sleep(1000)
        case :ra.process_command(server_id(), command) do
          {:ok, res, _} -> res
          _ -> err
        end
    end
  end

  @doc """
  Get the current state from the Raft cluster (Strongly Consistent).
  """
  def get_state do
    case :ra.consistent_query(server_id(), fn state -> state end) do
      {:ok, state, _leader} -> state
      _ -> %{}
    end
  end

end
