import Config

# Test-specific configuration
config :merkle_db,
  # No auth in tests (unless testing auth specifically)
  require_auth: false,

  # HTTP only in tests
  enable_https: false,
  http_port: 4001,  # Different port to avoid conflicts

  # No rate limiting in tests
  rate_limit_per_minute: 100_000,
  rate_limit_burst: 10_000,

  # Isolated test storage
  data_dir: "test/tmp/data",
  wal_path: "test/tmp/wal.bin",
  snapshot_dir: "test/tmp/snapshots",

  # Faster timeouts for tests
  raft_election_timeout_min: 200,
  raft_election_timeout_max: 500,
  query_timeout_ms: 5_000,

  # Disable auto-bootstrap in tests (tests manage their own setup)
  auto_bootstrap: false,
  enable_web_dashboard: false

# Only show warnings and errors in tests
config :logger, level: :warning
