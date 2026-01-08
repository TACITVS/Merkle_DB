import Config

# Runtime configuration - loaded at runtime, can use System.get_env
# This file is executed at runtime, not compile time

if config_env() == :prod do
  # =============================================================================
  # REQUIRED: API Key Configuration
  # =============================================================================
  # At least one API key must be set in production
  api_key = System.get_env("MERKLE_DB_API_KEY")

  unless api_key do
    raise """
    Environment variable MERKLE_DB_API_KEY is required in production.

    Generate a secure API key:
      mix run -e "IO.puts(:crypto.strong_rand_bytes(32) |> Base.url_encode64())"

    Then set it:
      set MERKLE_DB_API_KEY=your_generated_key
    """
  end

  # Parse multiple API keys if provided (comma-separated)
  # Format: key1:name1:scopes1,key2:name2:scopes2
  # Example: abc123:primary:read,write,admin,xyz789:readonly:read
  api_keys =
    case System.get_env("MERKLE_DB_API_KEYS") do
      nil ->
        # Single key mode - full access
        [%{key: api_key, name: "primary", scopes: [:read, :write, :admin]}]

      multi_keys ->
        multi_keys
        |> String.split(",")
        |> Enum.map(fn key_spec ->
          case String.split(key_spec, ":") do
            [key, name, scopes_str] ->
              scopes = scopes_str |> String.split("|") |> Enum.map(&String.to_atom/1)
              %{key: key, name: name, scopes: scopes}
            [key, name] ->
              %{key: key, name: name, scopes: [:read, :write]}
            [key] ->
              %{key: key, name: "unnamed", scopes: [:read, :write]}
          end
        end)
    end

  config :merkle_db, api_keys: api_keys

  # =============================================================================
  # HTTPS/TLS Configuration
  # =============================================================================
  enable_https = System.get_env("MERKLE_DB_ENABLE_HTTPS", "true") == "true"

  if enable_https do
    ssl_certfile = System.get_env("MERKLE_DB_SSL_CERT", "priv/ssl/cert.pem")
    ssl_keyfile = System.get_env("MERKLE_DB_SSL_KEY", "priv/ssl/key.pem")

    unless File.exists?(ssl_certfile) and File.exists?(ssl_keyfile) do
      raise """
      HTTPS is enabled but SSL certificate files not found.

      Expected:
        Certificate: #{ssl_certfile}
        Key: #{ssl_keyfile}

      Generate self-signed certificates for testing:
        openssl req -x509 -newkey rsa:4096 -keyout priv/ssl/key.pem -out priv/ssl/cert.pem -days 365 -nodes -subj "/CN=localhost"

      Or disable HTTPS:
        set MERKLE_DB_ENABLE_HTTPS=false
      """
    end

    config :merkle_db,
      enable_https: true,
      ssl_certfile: ssl_certfile,
      ssl_keyfile: ssl_keyfile,
      https_port: String.to_integer(System.get_env("MERKLE_DB_HTTPS_PORT", "4443"))
  else
    config :merkle_db, enable_https: false
  end

  # =============================================================================
  # Network Configuration
  # =============================================================================
  config :merkle_db,
    http_port: String.to_integer(System.get_env("MERKLE_DB_HTTP_PORT", "4000"))

  # =============================================================================
  # Storage Configuration
  # =============================================================================
  data_dir = System.get_env("MERKLE_DB_DATA_DIR", "data")

  config :merkle_db,
    data_dir: data_dir,
    wal_path: System.get_env("MERKLE_DB_WAL_PATH", Path.join(data_dir, "wal.bin")),
    snapshot_dir: System.get_env("MERKLE_DB_SNAPSHOT_DIR", Path.join(data_dir, "snapshots"))

  # =============================================================================
  # Rate Limiting Configuration
  # =============================================================================
  config :merkle_db,
    rate_limit_per_minute: String.to_integer(System.get_env("MERKLE_DB_RATE_LIMIT", "100")),
    rate_limit_burst: String.to_integer(System.get_env("MERKLE_DB_RATE_BURST", "20"))

  # =============================================================================
  # Performance Configuration
  # =============================================================================
  config :merkle_db,
    cache_max_entries: String.to_integer(System.get_env("MERKLE_DB_CACHE_SIZE", "50000")),
    query_timeout_ms: String.to_integer(System.get_env("MERKLE_DB_QUERY_TIMEOUT_MS", "30000"))
end

# Development/Test runtime configuration
if config_env() in [:dev, :test] do
  # Optional API key for development
  if api_key = System.get_env("MERKLE_DB_API_KEY") do
    config :merkle_db,
      api_keys: [%{key: api_key, name: "dev", scopes: [:read, :write, :admin]}],
      require_auth: true
  end
end
