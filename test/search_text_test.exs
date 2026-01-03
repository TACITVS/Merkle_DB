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
    # Reset KV to empty tree to avoid dimension conflicts from other tests
    MerkleDb.KV.reset()
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
    {:ok, response} = Jason.decode(conn.resp_body)

    # New response format includes results array and indexed flag
    assert is_map(response)
    assert Map.has_key?(response, "results")
    assert Map.has_key?(response, "indexed")
    assert Map.has_key?(response, "count")
    assert is_list(response["results"])
  end
end
