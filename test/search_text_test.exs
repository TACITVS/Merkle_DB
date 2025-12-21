defmodule MerkleDb.SearchTextTest do
  use ExUnit.Case, async: false

  import Plug.Test

  defp ensure_started(child_spec) do
    case start_supervised(child_spec) do
      {:ok, pid} -> pid
      {:error, {:already_started, pid}} -> pid
      {:error, {{:already_started, pid}, _child}} -> pid
      {:error, {:already_present, pid}} -> pid
      {:error, {{:already_present, pid}, _child}} -> pid
    end
  end

  setup do
    Application.put_env(:merkle_db, :ingesting, false)
    ensure_started(MerkleDb.KV)
    ensure_started(MerkleDb.TextStore)
    ensure_started(MerkleDb.Bootstrap)
    :ok
  end

  test "search payload includes text" do
    key = "__test__#{System.unique_integer([:positive])}"
    text = "alpha beta"
    vec = MerkleDb.TextEmbedding.embed(text)

    :ok = MerkleDb.KV.put(key, vec)

    conn = conn(:get, "/search?q=#{URI.encode(text)}&limit=1&threshold=-1")
    conn = MerkleDb.Web.Router.call(conn, [])

    assert conn.status == 200
    {:ok, payload} = Jason.decode(conn.resp_body)
    assert [%{"text" => "Text not found"} | _] = payload
  end
end
