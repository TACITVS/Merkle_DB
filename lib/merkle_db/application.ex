defmodule MerkleDb.Application do
  @moduledoc """
  MerkleDb Application module.

  Manages the application lifecycle including:
  - Service startup and supervision
  - HTTPS/TLS configuration
  - Graceful shutdown with WAL flush
  - Configuration validation
  """
  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    # Validate configuration in production
    if Application.get_env(:merkle_db, :env) == :prod do
      MerkleDb.ConfigValidator.validate!()
    end

    # Initialize auxiliary storage if needed
    MerkleDb.Storage.init()

    # Ensure ingestion flag is clear on boot
    Application.put_env(:merkle_db, :ingesting, false)

    # Attach telemetry handlers
    MerkleDb.Telemetry.attach_handlers()

    # Build children list with security modules
    children = build_children_list()

    opts = [strategy: :one_for_one, name: MerkleDb.Supervisor]

    case Supervisor.start_link(children, opts) do
      {:ok, pid} ->
        # Run auto-bootstrap if configured
        if Application.get_env(:merkle_db, :auto_bootstrap, true) do
          _ = MerkleDb.Bootstrap.start(mode: :auto)
        end

        # Log startup info
        log_startup_banner()

        {:ok, pid}

      error ->
        error
    end
  end

  @impl true
  def stop(_state) do
    Logger.info("MerkleDb shutting down...")

    # Graceful shutdown sequence
    shutdown_sequence()

    :ok
  end

  @doc """
  Performs graceful shutdown sequence.
  Called automatically on application stop.
  """
  def shutdown_sequence do
    Logger.info("Starting graceful shutdown sequence...")

    # 1. Stop accepting new connections
    Logger.debug("Stopping HTTP listeners...")
    stop_listeners()

    # 2. Drain existing requests (give them time to complete)
    drain_timeout = Application.get_env(:merkle_db, :shutdown_drain_timeout_ms, 5_000)
    Logger.debug("Draining requests (#{drain_timeout}ms timeout)...")
    Process.sleep(drain_timeout)

    # 3. Flush WAL to ensure durability
    Logger.debug("Flushing WAL...")
    flush_wal()

    # 4. Save final snapshot if configured
    if Application.get_env(:merkle_db, :save_snapshot_on_shutdown, true) do
      Logger.debug("Saving snapshot...")
      save_snapshot()
    end

    # 5. Leave Raft cluster gracefully
    Logger.debug("Leaving Raft cluster...")
    leave_raft_cluster()

    Logger.info("Graceful shutdown complete")
  end

  # Private functions

  defp build_children_list do
    http_port = Application.get_env(:merkle_db, :http_port, 4000)
    https_enabled = Application.get_env(:merkle_db, :enable_https, false)

    base_children = [
      {Task.Supervisor, name: MerkleDb.TaskSupervisor},
      {Registry, keys: :unique, name: MerkleDb.StorageRegistry},

      # Security modules (start early)
      MerkleDb.ApiKeyStore,
      MerkleDb.RateLimiter,

      # Core infrastructure
      MerkleDb.Raft.Supervisor,
      MerkleDb.WAL,
      MerkleDb.FPDispatcher,

      # Core storage
      MerkleDb.KV,
      MerkleDb.TextStore,
      MerkleDb.PayloadStore,
      MerkleDb.Replication,

      # Performance: ETS-backed cache
      MerkleDb.VectorCache,

      # Telemetry: Real-time metrics aggregation
      {MerkleDb.TelemetryAggregator, nil},

      # Load Testing: Automated stress testing
      MerkleDb.LoadGenerator,

      # Background jobs
      MerkleDb.Bootstrap,
      MerkleDb.Progress,
      MerkleDb.IndexBuilder,
      MerkleDb.JobScheduler,

      # HTTP server
      {Plug.Cowboy,
        scheme: :http,
        plug: MerkleDb.Web.Router,
        options: [
          port: http_port,
          transport_options: [num_acceptors: 100]
        ]}
    ]

    # Add HTTPS server if enabled
    if https_enabled do
      https_port = Application.get_env(:merkle_db, :https_port, 4443)
      certfile = Application.get_env(:merkle_db, :ssl_certfile)
      keyfile = Application.get_env(:merkle_db, :ssl_keyfile)

      https_child = {Plug.Cowboy,
        scheme: :https,
        plug: MerkleDb.Web.Router,
        options: [
          port: https_port,
          certfile: certfile,
          keyfile: keyfile,
          transport_options: [num_acceptors: 100]
        ]}

      base_children ++ [https_child]
    else
      base_children
    end
  end

  defp log_startup_banner do
    http_port = Application.get_env(:merkle_db, :http_port, 4000)
    https_enabled = Application.get_env(:merkle_db, :enable_https, false)
    https_port = Application.get_env(:merkle_db, :https_port, 4443)
    env = Application.get_env(:merkle_db, :env, :dev)
    require_auth = Application.get_env(:merkle_db, :require_auth, false)

    Logger.info("""

    ================================================================================
      MerkleDB Vector Database - Ready
      AVX2-Accelerated | Zero-Copy NIFs | IVF Indexing | Raft Consensus
    ================================================================================
      Environment: #{env}
      HTTP Server: http://localhost:#{http_port}
      #{if https_enabled, do: "HTTPS Server: https://localhost:#{https_port}", else: "HTTPS: Disabled"}
      Authentication: #{if require_auth, do: "Required", else: "Optional (dev mode)"}
      Health Check: http://localhost:#{http_port}/health/ready
    ================================================================================
    """)
  end

  defp stop_listeners do
    # Stop HTTP listener
    try do
      :ok = Plug.Cowboy.shutdown(MerkleDb.Web.Router.HTTP)
    rescue
      _ -> :ok
    end

    # Stop HTTPS listener if running
    try do
      :ok = Plug.Cowboy.shutdown(MerkleDb.Web.Router.HTTPS)
    rescue
      _ -> :ok
    end
  end

  defp flush_wal do
    try do
      MerkleDb.WAL.sync()
    rescue
      e -> Logger.warning("WAL flush failed: #{inspect(e)}")
    end
  end

  defp save_snapshot do
    try do
      tree = MerkleDb.KV.snapshot()
      if tree.count > 0 do
        MerkleDb.Persistence.save_sync(tree, label: "shutdown")
        Logger.info("Shutdown snapshot saved with #{tree.count} vectors")
      end
    rescue
      e -> Logger.warning("Snapshot save failed: #{inspect(e)}")
    end
  end

  defp leave_raft_cluster do
    try do
      # Notify the cluster we're leaving
      if Process.whereis(MerkleDb.Raft) do
        # If we're the leader, trigger election before leaving
        case MerkleDb.Raft.status() do
          %{role: :leader} ->
            Logger.info("Stepping down as Raft leader before shutdown")
            # Give followers time to receive final logs
            Process.sleep(500)
          _ ->
            :ok
        end
      end
    rescue
      e -> Logger.warning("Raft cleanup failed: #{inspect(e)}")
    end
  end
end
