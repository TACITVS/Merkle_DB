# MerkleDb Configuration Guide

This document provides a complete reference for all MerkleDb configuration options.

---

## Configuration Methods

MerkleDb supports configuration through:

1. **Environment Variables** (recommended for production)
2. **Config Files** (for development)
3. **Runtime Configuration** (via `config/runtime.exs`)

Environment variables take precedence over config file values.

---

## Environment Variables

### Security Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MERKLE_DB_API_KEY` | **Yes** (prod) | - | Primary API key for authentication |
| `MERKLE_DB_API_KEYS` | No | - | Multiple API keys with scopes (see format below) |

#### Multiple API Keys Format

```bash
# Format: key:name:scope1|scope2,key2:name2:scope
MERKLE_DB_API_KEYS=abc123:primary:read|write|admin,xyz789:readonly:read,def456:writer:read|write
```

Each key entry has three parts separated by colons:
1. **key**: The actual API key string
2. **name**: Human-readable identifier
3. **scopes**: Pipe-separated list of permissions (`read`, `write`, `admin`)

### Network Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MERKLE_DB_HTTP_PORT` | 4000 | HTTP server port |
| `MERKLE_DB_HTTPS_PORT` | 4443 | HTTPS server port |
| `MERKLE_DB_ENABLE_HTTPS` | true | Enable HTTPS listener |
| `MERKLE_DB_SSL_CERT` | priv/ssl/cert.pem | Path to SSL certificate |
| `MERKLE_DB_SSL_KEY` | priv/ssl/key.pem | Path to SSL private key |

### Storage Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MERKLE_DB_DATA_DIR` | data | Base directory for all data |
| `MERKLE_DB_WAL_DIR` | data/wal | Write-Ahead Log directory |
| `MERKLE_DB_SNAPSHOT_DIR` | data/snapshots | Snapshot directory |
| `MERKLE_DB_RAFT_DIR` | data/raft | Raft consensus data |

### Performance Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MERKLE_DB_CACHE_SIZE` | 50000 | Maximum vector cache entries |
| `MERKLE_DB_CACHE_TTL` | 3600 | Cache TTL in seconds |
| `MERKLE_DB_QUERY_TIMEOUT_MS` | 30000 | Query timeout in milliseconds |
| `MERKLE_DB_MAX_BATCH_SIZE` | 10000 | Maximum vectors per batch insert |
| `MERKLE_DB_TELEMETRY_WINDOW` | 100 | Telemetry rolling window size |

### Rate Limiting Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MERKLE_DB_RATE_LIMIT` | 100 | Requests per minute per client |
| `MERKLE_DB_RATE_BURST` | 20 | Burst capacity (tokens) |

### Raft Consensus Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MERKLE_DB_RAFT_ELECTION_MIN` | 1000 | Minimum election timeout (ms) |
| `MERKLE_DB_RAFT_ELECTION_MAX` | 2000 | Maximum election timeout (ms) |
| `MERKLE_DB_RAFT_HEARTBEAT` | 200 | Heartbeat interval (ms) |

### Operational Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MERKLE_DB_AUTO_BOOTSTRAP` | true | Auto-bootstrap on startup |
| `MERKLE_DB_LOG_LEVEL` | info | Log level (debug, info, warning, error) |

---

## Config File Reference

### config/config.exs

Base configuration shared across all environments:

```elixir
import Config

config :merkle_db,
  # HTTP settings
  http_port: 4000,
  https_port: 4443,
  enable_https: false,

  # Storage paths
  data_dir: "data",
  wal_dir: "data/wal",
  snapshot_dir: "data/snapshots",

  # Cache settings
  cache_max_entries: 50_000,
  cache_ttl_seconds: 3600,

  # Query settings
  query_timeout_ms: 30_000,
  max_batch_size: 10_000,

  # Rate limiting
  rate_limit_requests_per_minute: 100,
  rate_limit_burst: 20,

  # Raft consensus
  raft_election_timeout_min: 1000,
  raft_election_timeout_max: 2000,
  raft_heartbeat_interval: 200,

  # Telemetry
  telemetry_window_size: 100,

  # Operations
  auto_bootstrap: true

# Logger configuration
config :logger,
  level: :info

config :logger, :console,
  format: "$time $metadata[$level] $message\n",
  metadata: [:request_id, :collection]
```

### config/dev.exs

Development-specific overrides:

```elixir
import Config

config :merkle_db,
  # Disable auth in development
  require_auth: false,

  # More verbose logging
  log_level: :debug,

  # Smaller cache for dev
  cache_max_entries: 1_000

config :logger,
  level: :debug
```

### config/test.exs

Test environment configuration:

```elixir
import Config

config :merkle_db,
  # Isolated test data
  data_dir: "test/data",
  wal_dir: "test/data/wal",

  # Disable auth for tests
  require_auth: false,

  # Fast timeouts for tests
  query_timeout_ms: 5_000,

  # Minimal cache
  cache_max_entries: 100

config :logger,
  level: :warning
```

### config/prod.exs

Production defaults:

```elixir
import Config

config :merkle_db,
  # Security enabled
  require_auth: true,
  enable_https: true,

  # Production cache size
  cache_max_entries: 100_000,

  # Conservative timeouts
  query_timeout_ms: 60_000

config :logger,
  level: :info,
  compile_time_purge_matching: [
    [level_lower_than: :info]
  ]
```

### config/runtime.exs

Runtime configuration from environment variables:

```elixir
import Config

if config_env() == :prod do
  # Security - Required in production
  api_key = System.get_env("MERKLE_DB_API_KEY") ||
    raise "MERKLE_DB_API_KEY environment variable is required in production"

  config :merkle_db, api_key: api_key

  # Parse multiple API keys if provided
  if api_keys_raw = System.get_env("MERKLE_DB_API_KEYS") do
    api_keys = parse_api_keys(api_keys_raw)
    config :merkle_db, api_keys: api_keys
  end
end

# Network settings (all environments)
if port = System.get_env("MERKLE_DB_HTTP_PORT") do
  config :merkle_db, http_port: String.to_integer(port)
end

if https_port = System.get_env("MERKLE_DB_HTTPS_PORT") do
  config :merkle_db, https_port: String.to_integer(https_port)
end

if enable_https = System.get_env("MERKLE_DB_ENABLE_HTTPS") do
  config :merkle_db, enable_https: enable_https == "true"
end

# Storage settings
if data_dir = System.get_env("MERKLE_DB_DATA_DIR") do
  config :merkle_db,
    data_dir: data_dir,
    wal_dir: Path.join(data_dir, "wal"),
    snapshot_dir: Path.join(data_dir, "snapshots")
end

# Cache settings
if cache_size = System.get_env("MERKLE_DB_CACHE_SIZE") do
  config :merkle_db, cache_max_entries: String.to_integer(cache_size)
end

if cache_ttl = System.get_env("MERKLE_DB_CACHE_TTL") do
  config :merkle_db, cache_ttl_seconds: String.to_integer(cache_ttl)
end

# Rate limiting
if rate_limit = System.get_env("MERKLE_DB_RATE_LIMIT") do
  config :merkle_db, rate_limit_requests_per_minute: String.to_integer(rate_limit)
end

if rate_burst = System.get_env("MERKLE_DB_RATE_BURST") do
  config :merkle_db, rate_limit_burst: String.to_integer(rate_burst)
end

# Query settings
if timeout = System.get_env("MERKLE_DB_QUERY_TIMEOUT_MS") do
  config :merkle_db, query_timeout_ms: String.to_integer(timeout)
end

# Raft settings
if election_min = System.get_env("MERKLE_DB_RAFT_ELECTION_MIN") do
  config :merkle_db, raft_election_timeout_min: String.to_integer(election_min)
end

if election_max = System.get_env("MERKLE_DB_RAFT_ELECTION_MAX") do
  config :merkle_db, raft_election_timeout_max: String.to_integer(election_max)
end

if heartbeat = System.get_env("MERKLE_DB_RAFT_HEARTBEAT") do
  config :merkle_db, raft_heartbeat_interval: String.to_integer(heartbeat)
end
```

---

## Configuration Validation

MerkleDb validates configuration on startup in production mode. Invalid configurations will prevent the server from starting.

### Validation Rules

#### Security Validations
- `api_key` must be set in production
- `api_key` must be at least 16 characters
- SSL certificate and key must exist if HTTPS is enabled

#### Storage Validations
- `data_dir` must be writable
- Parent directories are created automatically

#### Network Validations
- Ports must be valid (1-65535)
- HTTP and HTTPS ports must be different

#### Performance Validations
- `cache_max_entries` must be > 0
- `query_timeout_ms` must be > 0
- `rate_limit` must be > 0

### Validation Messages

On startup, you'll see validation results:

```
[info] Configuration validation passed
[warning] HTTPS enabled but using self-signed certificates
```

Or on failure:

```
[error] Configuration validation failed:
  - api_key is required in production
  - cache_max_entries must be greater than 0
```

---

## Configuration Examples

### Minimal Production Setup

```bash
# Windows
set MERKLE_DB_API_KEY=your_secure_32_character_key_here
set MIX_ENV=prod
mix run --no-halt
```

```bash
# Linux/Mac
export MERKLE_DB_API_KEY=your_secure_32_character_key_here
export MIX_ENV=prod
mix run --no-halt
```

### High-Performance Configuration

```bash
# Large cache, fast timeouts
set MERKLE_DB_API_KEY=your_key
set MERKLE_DB_CACHE_SIZE=200000
set MERKLE_DB_CACHE_TTL=7200
set MERKLE_DB_QUERY_TIMEOUT_MS=10000
set MERKLE_DB_RATE_LIMIT=1000
set MERKLE_DB_RATE_BURST=100
set MIX_ENV=prod
```

### Multi-Tenant Configuration

```bash
# Multiple API keys with different permissions
set MERKLE_DB_API_KEYS=admin123:admin:read|write|admin,user456:user1:read|write,reader789:readonly:read
set MIX_ENV=prod
```

### Secure HTTPS Configuration

```bash
# Full HTTPS setup
set MERKLE_DB_API_KEY=your_key
set MERKLE_DB_ENABLE_HTTPS=true
set MERKLE_DB_SSL_CERT=C:\certs\merkledb.crt
set MERKLE_DB_SSL_KEY=C:\certs\merkledb.key
set MERKLE_DB_HTTP_PORT=80
set MERKLE_DB_HTTPS_PORT=443
set MIX_ENV=prod
```

### Custom Data Directory

```bash
# Store data on separate drive
set MERKLE_DB_API_KEY=your_key
set MERKLE_DB_DATA_DIR=D:\MerkleDb\data
set MIX_ENV=prod
```

### Raft Cluster Configuration

For multi-node deployments, tune Raft timeouts based on network latency:

```bash
# Low-latency local network
set MERKLE_DB_RAFT_ELECTION_MIN=500
set MERKLE_DB_RAFT_ELECTION_MAX=1000
set MERKLE_DB_RAFT_HEARTBEAT=100

# Higher-latency WAN
set MERKLE_DB_RAFT_ELECTION_MIN=3000
set MERKLE_DB_RAFT_ELECTION_MAX=6000
set MERKLE_DB_RAFT_HEARTBEAT=500
```

---

## Generating Secure API Keys

Use the built-in generator for cryptographically secure keys:

```bash
mix run -e "IO.puts(:crypto.strong_rand_bytes(32) |> Base.url_encode64())"
```

This generates a 43-character URL-safe base64 key.

For even stronger keys:

```bash
# 48-byte key (64 characters)
mix run -e "IO.puts(:crypto.strong_rand_bytes(48) |> Base.url_encode64())"
```

---

## SSL Certificate Setup

### Self-Signed Certificates (Development/Testing)

```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 \
  -keyout priv/ssl/key.pem \
  -out priv/ssl/cert.pem \
  -days 365 -nodes \
  -subj "/CN=localhost"
```

### Let's Encrypt (Production)

1. Obtain certificates using certbot or similar
2. Point to the certificate files:

```bash
set MERKLE_DB_SSL_CERT=/etc/letsencrypt/live/yourdomain/fullchain.pem
set MERKLE_DB_SSL_KEY=/etc/letsencrypt/live/yourdomain/privkey.pem
```

---

## Tuning Guidelines

### Memory Usage

| Vectors | Dimensions | Precision | Recommended Cache |
|---------|------------|-----------|-------------------|
| < 10K | Any | Any | 10,000 |
| 10K-100K | < 512 | f32 | 50,000 |
| 10K-100K | 512-1024 | f32 | 25,000 |
| 100K-1M | Any | f32 | 10,000 |
| > 1M | Any | int8 | 5,000 |

### Rate Limiting

| Use Case | Rate Limit | Burst |
|----------|------------|-------|
| Single user | 100 | 20 |
| Small team | 500 | 50 |
| Production API | 1000 | 100 |
| High-traffic | 5000 | 500 |

### Query Timeouts

| Query Type | Recommended Timeout |
|------------|---------------------|
| Cached queries | 5,000 ms |
| Simple KNN | 10,000 ms |
| Filtered search | 30,000 ms |
| Index building | 300,000 ms |

---

## See Also

- [Quick Start Guide](QUICKSTART.md)
- [API Reference](API.md)
- [Deployment Guide](DEPLOYMENT.md)
