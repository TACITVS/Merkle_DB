# This file is responsible for configuring your application
# and its dependencies with the aid of the Config module.
import Config

# MerkleDb Core Configuration
config :merkle_db,
  # Network
  http_port: 4000,
  https_port: 4443,
  enable_https: false,

  # Security
  api_keys: [],  # Loaded from runtime.exs in production
  require_auth: false,  # Set to true in production
  rate_limit_per_minute: 100,
  rate_limit_burst: 20,
  max_request_body_bytes: 10_485_760,  # 10MB

  # SSL/TLS (for HTTPS)
  ssl_certfile: nil,
  ssl_keyfile: nil,

  # Storage
  data_dir: "data",
  wal_path: "data/wal.bin",
  wal_sync_mode: :normal,  # :sync for fsync on every write, :normal for batched
  snapshot_dir: "data/snapshots",

  # Raft Consensus
  raft_election_timeout_min: 1000,
  raft_election_timeout_max: 2000,
  raft_heartbeat_interval: 200,

  # Performance
  cache_max_entries: 10_000,
  cache_ttl_seconds: 3600,
  query_timeout_ms: 30_000,

  # Telemetry
  telemetry_window_size: 100,
  telemetry_retention_seconds: 3600,

  # Features
  auto_bootstrap: true,
  enable_web_dashboard: true,
  enable_text_embedding: true

# Logger configuration
config :logger, :console,
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id, :collection, :operation]

# Import environment specific config
import_config "#{config_env()}.exs"
