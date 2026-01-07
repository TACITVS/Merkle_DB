defmodule MerkleDb.Raft.Supervisor do
  use Supervisor

  def start_link(opts) do
    Supervisor.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @impl true
  def init(_opts) do
    # Raft cluster initialization is usually done via a Task 
    # to avoid blocking the supervisor if the cluster is waiting for quorum.
    children = [
      {Task, fn -> MerkleDb.Raft.start_link() end}
    ]

    Supervisor.init(children, strategy: :one_for_one)
  end
end
