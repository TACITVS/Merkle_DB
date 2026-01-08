defmodule MerkleDb.ConfigValidator do
  @moduledoc """
  Validates MerkleDb configuration at startup.

  Ensures all required settings are present and valid before
  the application starts accepting requests.
  """
  require Logger

  @doc """
  Validate all configuration. Raises on invalid config.
  """
  @spec validate!() :: :ok
  def validate! do
    Logger.info("Validating configuration...")

    validate_security!()
    validate_storage!()
    validate_network!()
    validate_performance!()

    Logger.info("Configuration validation passed")
    :ok
  end

  @doc """
  Validate configuration without raising. Returns {:ok, warnings} or {:error, errors}.
  """
  @spec validate() :: {:ok, [String.t()]} | {:error, [String.t()]}
  def validate do
    errors = []
    warnings = []

    {errors, warnings} = validate_security(errors, warnings)
    {errors, warnings} = validate_storage(errors, warnings)
    {errors, warnings} = validate_network(errors, warnings)
    {errors, warnings} = validate_performance(errors, warnings)

    if errors == [] do
      {:ok, warnings}
    else
      {:error, errors}
    end
  end

  # Validation with raising

  defp validate_security! do
    if prod?() do
      # In production, require proper API key configuration
      api_keys = get_config(:api_keys, [])

      if api_keys == [] do
        raise_config_error("No API keys configured. Set MERKLE_DB_API_KEY environment variable.")
      end

      # Check for default/weak keys
      Enum.each(api_keys, fn key_config ->
        key = Map.get(key_config, :key, "")
        if key in ["secret", "password", "admin", "test", ""] do
          raise_config_error("Weak or default API key detected: '#{key}'. Use a strong random key.")
        end

        if byte_size(key) < 16 do
          raise_config_error("API key too short (min 16 characters for security).")
        end
      end)

      # If HTTPS is enabled, validate certificates
      if get_config(:enable_https, false) do
        certfile = get_config(:ssl_certfile)
        keyfile = get_config(:ssl_keyfile)

        unless certfile && File.exists?(certfile) do
          raise_config_error("HTTPS enabled but SSL certificate not found: #{certfile}")
        end

        unless keyfile && File.exists?(keyfile) do
          raise_config_error("HTTPS enabled but SSL key not found: #{keyfile}")
        end
      end
    end

    :ok
  end

  defp validate_storage! do
    data_dir = get_config(:data_dir, "data")

    # Ensure data directory exists or can be created
    case File.mkdir_p(data_dir) do
      :ok -> :ok
      {:error, reason} ->
        raise_config_error("Cannot create data directory '#{data_dir}': #{reason}")
    end

    # Check write permissions
    test_file = Path.join(data_dir, ".write_test")
    case File.write(test_file, "test") do
      :ok ->
        File.rm(test_file)
        :ok
      {:error, reason} ->
        raise_config_error("Data directory '#{data_dir}' is not writable: #{reason}")
    end

    # Ensure snapshot directory exists
    snapshot_dir = get_config(:snapshot_dir, Path.join(data_dir, "snapshots"))
    case File.mkdir_p(snapshot_dir) do
      :ok -> :ok
      {:error, reason} ->
        raise_config_error("Cannot create snapshot directory '#{snapshot_dir}': #{reason}")
    end

    :ok
  end

  defp validate_network! do
    http_port = get_config(:http_port, 4000)
    https_port = get_config(:https_port, 4443)

    unless is_integer(http_port) and http_port > 0 and http_port < 65536 do
      raise_config_error("Invalid HTTP port: #{http_port}")
    end

    unless is_integer(https_port) and https_port > 0 and https_port < 65536 do
      raise_config_error("Invalid HTTPS port: #{https_port}")
    end

    if http_port == https_port do
      raise_config_error("HTTP and HTTPS ports cannot be the same")
    end

    :ok
  end

  defp validate_performance! do
    cache_size = get_config(:cache_max_entries, 10_000)
    query_timeout = get_config(:query_timeout_ms, 30_000)

    unless is_integer(cache_size) and cache_size > 0 do
      raise_config_error("Invalid cache_max_entries: #{cache_size}")
    end

    unless is_integer(query_timeout) and query_timeout > 0 do
      raise_config_error("Invalid query_timeout_ms: #{query_timeout}")
    end

    :ok
  end

  # Validation without raising (for health checks)

  defp validate_security(errors, warnings) do
    api_keys = get_config(:api_keys, [])

    errors =
      if prod?() and api_keys == [] do
        ["No API keys configured" | errors]
      else
        errors
      end

    warnings =
      if not prod?() and api_keys == [] do
        ["Running without API key authentication (development mode)" | warnings]
      else
        warnings
      end

    warnings =
      if get_config(:require_auth, false) == false and prod?() do
        ["Authentication disabled in production" | warnings]
      else
        warnings
      end

    {errors, warnings}
  end

  defp validate_storage(errors, warnings) do
    data_dir = get_config(:data_dir, "data")

    errors =
      unless File.dir?(data_dir) do
        case File.mkdir_p(data_dir) do
          :ok -> errors
          {:error, _} -> ["Cannot access data directory: #{data_dir}" | errors]
        end
      else
        errors
      end

    # Check disk space (warning only)
    warnings =
      case disk_free_mb(data_dir) do
        {:ok, free_mb} when free_mb < 1000 ->
          ["Low disk space: #{free_mb}MB remaining" | warnings]
        _ ->
          warnings
      end

    {errors, warnings}
  end

  defp validate_network(errors, warnings) do
    http_port = get_config(:http_port, 4000)

    errors =
      unless is_integer(http_port) and http_port > 0 do
        ["Invalid HTTP port configuration" | errors]
      else
        errors
      end

    {errors, warnings}
  end

  defp validate_performance(errors, warnings) do
    cache_size = get_config(:cache_max_entries, 10_000)

    warnings =
      if cache_size < 1000 do
        ["Cache size is very small (#{cache_size}), may impact performance" | warnings]
      else
        warnings
      end

    {errors, warnings}
  end

  # Helpers

  defp prod? do
    Application.get_env(:merkle_db, :env, Mix.env()) == :prod
  end

  defp get_config(key, default \\ nil) do
    Application.get_env(:merkle_db, key, default)
  end

  defp raise_config_error(message) do
    raise """

    ╔══════════════════════════════════════════════════════════════╗
    ║                  CONFIGURATION ERROR                         ║
    ╠══════════════════════════════════════════════════════════════╣
    ║ #{String.pad_trailing(message, 60)} ║
    ╚══════════════════════════════════════════════════════════════╝

    Please fix the configuration and restart.
    """
  end

  defp disk_free_mb(path) do
    # Cross-platform disk space check
    try do
      case :os.type() do
        {:win32, _} ->
          # Windows: use wmic
          {output, 0} = System.cmd("wmic", ["logicaldisk", "where", "DeviceID='#{String.first(Path.absname(path))}:'", "get", "FreeSpace", "/value"])
          case Regex.run(~r/FreeSpace=(\d+)/, output) do
            [_, bytes] -> {:ok, div(String.to_integer(bytes), 1_048_576)}
            _ -> {:error, :parse_failed}
          end

        _ ->
          # Unix: use df
          {output, 0} = System.cmd("df", ["-m", path])
          lines = String.split(output, "\n")
          if length(lines) >= 2 do
            [_, _, _, avail | _] = String.split(Enum.at(lines, 1), ~r/\s+/, trim: true)
            {:ok, String.to_integer(avail)}
          else
            {:error, :parse_failed}
          end
      end
    rescue
      _ -> {:error, :command_failed}
    end
  end
end
