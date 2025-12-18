defmodule MerkleDb.Application do
  use Application

  @impl true
  def start(_type, _args) do
    # Initialize auxiliary storage if needed
    MerkleDb.Storage.init()

    children = [
      MerkleDb.KV,
      MerkleDb.TextStore,
      MerkleDb.Progress,
      MerkleDb.JobScheduler,
      {Plug.Cowboy, scheme: :http, plug: MerkleDb.Web.Router, options: [port: 4000]}
    ]

    opts = [strategy: :one_for_one, name: MerkleDb.Supervisor]
    Supervisor.start_link(children, opts)
  end
end