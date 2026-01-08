import Config

# Production configuration - secure defaults
config :merkle_db,
  # Security: Authentication required
  require_auth: true,

  # Security: HTTPS enabled by default
  enable_https: true,
  https_port: 4443,
  http_port: 4000,  # Can be used for health checks behind load balancer

  # Security: Strict rate limiting
  rate_limit_per_minute: 100,
  rate_limit_burst: 20,
  max_request_body_bytes: 10_485_760,

  # Storage: Production paths (overridden in runtime.exs)
  data_dir: "data",
  wal_path: "data/wal.bin",
  wal_sync_mode: :sync,  # Ensure durability
  snapshot_dir: "data/snapshots",

  # Performance: Production tuning
  cache_max_entries: 50_000,
  cache_ttl_seconds: 3600,
  query_timeout_ms: 30_000,

  # Features
  auto_bootstrap: true,
  enable_web_dashboard: true

# Production logging: info level, no debug noise
config :logger, level: :info

config :logger, :console,
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id, :collection, :operation, :remote_ip]
