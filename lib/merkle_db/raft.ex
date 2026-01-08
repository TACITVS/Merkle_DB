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
    base_dir = Path.join(File.cwd!(), "data/raft")
    # Add server-specific subdir to avoid path conflicts and segment writer enoent
    server_dir = Path.join(base_dir, to_string(node()))
    File.mkdir_p!(server_dir)
    
    # Raft Tuning Parameters
    # Set data_dir to the server-specific directory
    Application.put_env(:ra, :data_dir, String.to_charlist(server_dir))
    
    # Raft timing configuration (configurable via config)
    election_min = Application.get_env(:merkle_db, :raft_election_timeout_min, 1000)
    election_max = Application.get_env(:merkle_db, :raft_election_timeout_max, 2000)
    heartbeat = Application.get_env(:merkle_db, :raft_heartbeat_interval, 200)

    Application.put_env(:ra, :election_timeout_min, election_min)
    Application.put_env(:ra, :election_timeout_max, election_max)
    Application.put_env(:ra, :heartbeat_interval, heartbeat)

    # Explicitly start ra system (required by newer versions)
    # Use a case to handle already_started if it happens
    case :ra_system.start_default() do
      {:ok, _} -> :ok
      {:error, {:already_started, _}} -> :ok
      err -> IO.inspect(err, label: "Ra System Start Failed")
    end

    try_start_cluster(machine, 10)
  end

  defp try_start_cluster(machine, attempts) when attempts > 0 do
    case :ra.start_cluster(:default, @cluster_name, machine, [server_id()]) do
      {:ok, _, _} -> :ok
      {:error, {:already_exists, _}} -> :ok
      {:error, :system_not_started} ->
        Process.sleep(1000)
        try_start_cluster(machine, attempts - 1)
      err ->
        IO.inspect(err, label: "Raft Cluster Start Failed")
        Process.sleep(1000)
        try_start_cluster(machine, attempts - 1)
    end
  end
  defp try_start_cluster(_, _), do: {:error, :timeout}

  @doc """
  Join an existing cluster.
  """
  def join_cluster(peer_node) do
    peer_id = {:merkle_db_server, peer_node}
    :ra.add_member(peer_id, server_id())
  end

  @doc """
  Send a command to the Raft cluster leader.
  """
  def process_command(command) do
    case :ra.process_command(server_id(), command) do
      {:ok, result, leader} -> {:ok, result, leader}
      {:error, :leader_not_known} ->
        # Retry once after a short wait if leader is not known
        Process.sleep(500)
        :ra.process_command(server_id(), command)
      err -> err
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
