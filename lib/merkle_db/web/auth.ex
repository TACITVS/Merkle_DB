defmodule MerkleDb.Web.Auth do
  import Plug.Conn

  def init(opts), do: opts

  def call(conn, _opts) do
    # Get token from environment or config, default to "secret" for dev
    expected_token = System.get_env("MERKLE_DB_API_KEY") || "secret"

    case get_req_header(conn, "authorization") do
      ["Bearer " <> token] ->
        if token == expected_token do
          conn
        else
          send_unauthorized(conn)
        end
      _ ->
        send_unauthorized(conn)
    end
  end

  defp send_unauthorized(conn) do
    conn
    |> put_resp_content_type("application/json")
    |> send_resp(401, Jason.encode!(%{error: "Unauthorized"}))
    |> halt()
  end
end
