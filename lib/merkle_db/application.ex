defmodule MerkleDb.Application do
  use Application

  @impl true
  def start(_type, _args) do
    # Initialize auxiliary storage if needed
    MerkleDb.Storage.init()

    # Ensure ingestion flag is clear on boot
    Application.put_env(:merkle_db, :ingesting, false)

    # Attach telemetry handlers
    MerkleDb.Telemetry.attach_handlers()

    children = [
      # Core storage
      MerkleDb.KV,
      MerkleDb.TextStore,

      # Performance: ETS-backed cache
      MerkleDb.VectorCache,

      # Background jobs
      MerkleDb.Progress,
      MerkleDb.JobScheduler,

      # Web server
      {Plug.Cowboy, scheme: :http, plug: MerkleDb.Web.Router, options: [port: 4000]}
    ]

    opts = [strategy: :one_for_one, name: MerkleDb.Supervisor]

    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        # Log startup info
        IO.puts("""
        ╔════════════════════════════════════════════════════════════╗
        ║  MerkleDB Vector Database - Ready                          ║
        ║  AVX2-Accelerated • Zero-Copy NIFs • IVF Indexing         ║
        ║  Server: http://localhost:4000                             ║
        ╚════════════════════════════════════════════════════════════╝
        """)

        {:ok, pid}

      error ->
        error
    end
  end
end