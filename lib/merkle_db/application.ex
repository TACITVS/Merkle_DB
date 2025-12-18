defmodule MerkleDb.Application do
  use Application

  @impl true
  def start(_type, _args) do
    # Initialize auxiliary storage if needed
    MerkleDb.Storage.init()

    children = [
      # 1. The Vector Database (Holds the numbers in RAM)
      MerkleDb.KV,

      # 2. The Text Database (Holds the verses on Disk)
      MerkleDb.TextStore,

      # 3. The Web Server
      {Plug.Cowboy, scheme: :http, plug: MerkleDb.Web.Router, options: [port: 4000]}
    ]

    opts = [strategy: :one_for_one, name: MerkleDb.Supervisor]
    Supervisor.start_link(children, opts)
  end
end