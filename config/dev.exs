import Config

# Development-specific configuration
config :merkle_db,
  # No auth required in development
  require_auth: false,

  # HTTP only in development
  enable_https: false,
  http_port: 4000,

  # Relaxed rate limits for development
  rate_limit_per_minute: 1000,
  rate_limit_burst: 100,

  # Local storage
  data_dir: "data",
  wal_path: "data/wal.bin",
  snapshot_dir: "data/snapshots",

  # Auto-bootstrap for easy development
  auto_bootstrap: true,
  enable_web_dashboard: true

# More verbose logging in development
config :logger, level: :debug
