defmodule MerkleDb.RateLimiter do
  @moduledoc """
  ETS-based rate limiter using the token bucket algorithm.

  Supports per-IP and per-API-key rate limiting with configurable
  limits and burst capacity.

  ## Configuration

      config :merkle_db,
        rate_limit_per_minute: 100,
        rate_limit_burst: 20

  ## Usage

      case RateLimiter.check_rate(client_id) do
        :ok -> proceed_with_request()
        {:error, :rate_limited, retry_after} -> return_429(retry_after)
      end
  """
  use GenServer
  require Logger

  @table :merkle_db_rate_limits
  @cleanup_interval_ms 60_000  # Clean up expired entries every minute

  # Client API

  def start_link(opts \\ []) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  @doc """
  Check if a request from the given identifier should be allowed.

  Returns:
  - :ok - Request allowed
  - {:error, :rate_limited, retry_after_ms} - Rate limited, wait this many ms
  """
  @spec check_rate(String.t(), keyword()) :: :ok | {:error, :rate_limited, non_neg_integer()}
  def check_rate(identifier, opts \\ []) do
    limit = Keyword.get(opts, :limit, get_config(:rate_limit_per_minute, 100))
    burst = Keyword.get(opts, :burst, get_config(:rate_limit_burst, 20))
    window_seconds = Keyword.get(opts, :window_seconds, 60)

    now = System.monotonic_time(:millisecond)
    refill_rate = limit / (window_seconds * 1000)  # tokens per millisecond

    case :ets.lookup(@table, identifier) do
      [{^identifier, {tokens, last_update}}] ->
        # Calculate tokens added since last update
        elapsed = now - last_update
        new_tokens = min(tokens + (elapsed * refill_rate), limit + burst)

        if new_tokens >= 1.0 do
          # Consume one token
          :ets.insert(@table, {identifier, {new_tokens - 1.0, now}})
          :ok
        else
          # Rate limited - calculate when a token will be available
          tokens_needed = 1.0 - new_tokens
          retry_after = trunc(tokens_needed / refill_rate)
          {:error, :rate_limited, retry_after}
        end

      [] ->
        # First request from this identifier
        initial_tokens = limit + burst - 1.0  # Start with full bucket minus this request
        :ets.insert(@table, {identifier, {initial_tokens, now}})
        :ok
    end
  end

  @doc """
  Check rate limit using IP address from connection.
  """
  @spec check_rate_ip(Plug.Conn.t(), keyword()) :: :ok | {:error, :rate_limited, non_neg_integer()}
  def check_rate_ip(conn, opts \\ []) do
    ip = get_client_ip(conn)
    check_rate("ip:#{ip}", opts)
  end

  @doc """
  Check rate limit using API key.
  Allows different limits per key tier.
  """
  @spec check_rate_key(String.t(), keyword()) :: :ok | {:error, :rate_limited, non_neg_integer()}
  def check_rate_key(api_key, opts \\ []) do
    key_hash = :crypto.hash(:sha256, api_key) |> Base.encode16(case: :lower)
    check_rate("key:#{key_hash}", opts)
  end

  @doc """
  Get current rate limit status for an identifier.
  Returns remaining tokens and reset time.
  """
  @spec status(String.t()) :: {:ok, map()} | {:error, :not_found}
  def status(identifier) do
    limit = get_config(:rate_limit_per_minute, 100)
    burst = get_config(:rate_limit_burst, 20)

    case :ets.lookup(@table, identifier) do
      [{^identifier, {tokens, last_update}}] ->
        now = System.monotonic_time(:millisecond)
        elapsed = now - last_update
        refill_rate = limit / 60_000
        current_tokens = min(tokens + (elapsed * refill_rate), limit + burst)

        {:ok, %{
          remaining: trunc(current_tokens),
          limit: limit,
          burst: burst,
          reset_in_ms: if(current_tokens < limit, do: trunc((limit - current_tokens) / refill_rate), else: 0)
        }}

      [] ->
        {:ok, %{remaining: limit + burst, limit: limit, burst: burst, reset_in_ms: 0}}
    end
  end

  @doc """
  Reset rate limit for an identifier.
  """
  @spec reset(String.t()) :: :ok
  def reset(identifier) do
    :ets.delete(@table, identifier)
    :ok
  end

  # Server Callbacks

  @impl true
  def init(_opts) do
    # Create ETS table with concurrent read access
    :ets.new(@table, [:set, :public, :named_table, write_concurrency: true, read_concurrency: true])

    # Schedule periodic cleanup
    schedule_cleanup()

    Logger.info("RateLimiter initialized")
    {:ok, %{}}
  end

  @impl true
  def handle_info(:cleanup, state) do
    cleanup_old_entries()
    schedule_cleanup()
    {:noreply, state}
  end

  # Private Functions

  defp schedule_cleanup do
    Process.send_after(self(), :cleanup, @cleanup_interval_ms)
  end

  defp cleanup_old_entries do
    now = System.monotonic_time(:millisecond)
    # Remove entries that haven't been updated in 5 minutes (fully refilled)
    cutoff = now - 300_000

    # Use match_delete for efficient bulk deletion
    # This deletes entries where last_update < cutoff
    :ets.select_delete(@table, [
      {{:"$1", {:"$2", :"$3"}}, [{:<, :"$3", cutoff}], [true]}
    ])
  end

  defp get_client_ip(conn) do
    # Check for forwarded IP (behind proxy/load balancer)
    forwarded_for =
      conn
      |> Plug.Conn.get_req_header("x-forwarded-for")
      |> List.first()

    case forwarded_for do
      nil ->
        # Direct connection
        conn.remote_ip |> :inet.ntoa() |> to_string()

      ip_list ->
        # Get first IP from comma-separated list
        ip_list
        |> String.split(",")
        |> List.first()
        |> String.trim()
    end
  end

  defp get_config(key, default) do
    Application.get_env(:merkle_db, key, default)
  end
end
