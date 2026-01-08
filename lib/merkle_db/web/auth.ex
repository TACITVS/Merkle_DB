defmodule MerkleDb.Web.Auth do
  @moduledoc """
  Authentication plug for MerkleDb API requests.

  Integrates with ApiKeyStore for multi-key support with scopes.
  Optionally integrates with RateLimiter for per-key rate limiting.

  ## Usage

      # Require any valid API key
      plug MerkleDb.Web.Auth

      # Require specific scope
      plug MerkleDb.Web.Auth, scope: :write

      # Require admin scope
      plug MerkleDb.Web.Auth, scope: :admin
  """
  import Plug.Conn
  require Logger

  alias MerkleDb.ApiKeyStore
  alias MerkleDb.RateLimiter

  def init(opts), do: opts

  def call(conn, opts) do
    required_scope = Keyword.get(opts, :scope, :read)
    rate_limit = Keyword.get(opts, :rate_limit, true)

    # Check if authentication is required
    unless require_auth?() do
      # In dev/test without auth requirement, allow all requests
      assign(conn, :api_key_info, %{name: "anonymous", scopes: [:read, :write, :admin]})
    else
      case authenticate(conn, required_scope) do
        {:ok, conn, key_info} ->
          # Optionally apply rate limiting per API key
          if rate_limit do
            apply_rate_limit(conn, key_info)
          else
            conn
          end

        {:error, reason} ->
          send_error(conn, reason)
      end
    end
  end

  @doc """
  Authenticate a request and check scope.
  Returns {:ok, conn, key_info} or {:error, reason}.
  """
  def authenticate(conn, required_scope \\ :read) do
    case get_api_key(conn) do
      nil ->
        {:error, :missing_token}

      api_key ->
        case ApiKeyStore.validate(api_key, required_scope) do
          {:ok, key_info} ->
            conn =
              conn
              |> assign(:api_key_info, key_info)
              |> assign(:api_key_hash, hash_key(api_key))

            {:ok, conn, key_info}

          {:error, :invalid_key} ->
            Logger.warning("Invalid API key attempt from #{client_ip(conn)}")
            {:error, :invalid_key}

          {:error, :insufficient_scope} ->
            {:error, :insufficient_scope}
        end
    end
  end

  @doc """
  Check if the current connection has the required scope.
  Use after authentication in pipelines that need to verify scope.
  """
  def has_scope?(conn, scope) do
    case conn.assigns[:api_key_info] do
      nil -> false
      %{scopes: scopes} -> scope in scopes
      _ -> false
    end
  end

  @doc """
  Require a specific scope. Halts with 403 if not authorized.
  """
  def require_scope(conn, scope) do
    if has_scope?(conn, scope) do
      conn
    else
      send_error(conn, :insufficient_scope)
    end
  end

  # Private Functions

  defp get_api_key(conn) do
    # Check Authorization header first
    case get_req_header(conn, "authorization") do
      ["Bearer " <> token] ->
        String.trim(token)

      # Also accept X-API-Key header
      _ ->
        case get_req_header(conn, "x-api-key") do
          [key] -> String.trim(key)
          _ -> nil
        end
    end
  end

  defp apply_rate_limit(conn, key_info) do
    key_hash = conn.assigns[:api_key_hash]
    identifier = "key:#{Base.encode16(key_hash, case: :lower)}"

    # Get tier-specific limits if configured
    opts = get_tier_limits(key_info)

    case RateLimiter.check_rate(identifier, opts) do
      :ok ->
        add_rate_limit_headers(conn, identifier)

      {:error, :rate_limited, retry_after} ->
        conn
        |> put_resp_header("retry-after", to_string(div(retry_after, 1000)))
        |> put_resp_header("x-ratelimit-remaining", "0")
        |> send_error(:rate_limited)
    end
  end

  defp add_rate_limit_headers(conn, identifier) do
    case RateLimiter.status(identifier) do
      {:ok, status} ->
        conn
        |> put_resp_header("x-ratelimit-limit", to_string(status.limit))
        |> put_resp_header("x-ratelimit-remaining", to_string(status.remaining))
        |> put_resp_header("x-ratelimit-reset", to_string(status.reset_in_ms))

      _ ->
        conn
    end
  end

  defp get_tier_limits(%{tier: :premium}), do: [limit: 1000, burst: 100]
  defp get_tier_limits(%{tier: :enterprise}), do: [limit: 10000, burst: 500]
  defp get_tier_limits(_), do: []  # Use defaults

  defp send_error(conn, :missing_token) do
    conn
    |> put_resp_content_type("application/json")
    |> put_resp_header("www-authenticate", "Bearer")
    |> send_resp(401, Jason.encode!(%{
      error: "unauthorized",
      message: "Missing API key. Include 'Authorization: Bearer <key>' header."
    }))
    |> halt()
  end

  defp send_error(conn, :invalid_key) do
    conn
    |> put_resp_content_type("application/json")
    |> put_resp_header("www-authenticate", "Bearer error=\"invalid_token\"")
    |> send_resp(401, Jason.encode!(%{
      error: "unauthorized",
      message: "Invalid API key."
    }))
    |> halt()
  end

  defp send_error(conn, :insufficient_scope) do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(403, Jason.encode!(%{
      error: "forbidden",
      message: "Insufficient permissions for this operation."
    }))
    |> halt()
  end

  defp send_error(conn, :rate_limited) do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(429, Jason.encode!(%{
      error: "rate_limited",
      message: "Too many requests. Please slow down."
    }))
    |> halt()
  end

  defp require_auth? do
    Application.get_env(:merkle_db, :require_auth, false)
  end

  defp hash_key(key) do
    :crypto.hash(:sha256, key)
  end

  defp client_ip(conn) do
    case get_req_header(conn, "x-forwarded-for") do
      [ip | _] -> ip
      _ -> conn.remote_ip |> :inet.ntoa() |> to_string()
    end
  end
end
