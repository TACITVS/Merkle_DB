defmodule MerkleDb.ReplicationHttpTest do
  use ExUnit.Case, async: false

  import Plug.Test
  import Plug.Conn

  alias MerkleDb.{Replication, KV}

  setup do
    # Clear oplog before each test
    :ets.delete_all_objects(:replication_oplog)
    :ets.insert(:replication_meta, {:current_seq, 0})
    KV.reset()
    :ok
  end

  describe "GET /replication/status" do
    test "returns replication status" do
      conn = conn(:get, "/replication/status")
      conn = MerkleDb.Web.Router.call(conn, [])

      assert conn.status == 200
      {:ok, body} = Jason.decode(conn.resp_body)

      assert Map.has_key?(body, "current_seq")
      assert Map.has_key?(body, "oplog_size")
      assert Map.has_key?(body, "created_at")
    end
  end

  describe "GET /replication/deltas" do
    test "returns empty deltas when no operations" do
      conn = conn(:get, "/replication/deltas?since=0")
      conn = MerkleDb.Web.Router.call(conn, [])

      assert conn.status == 200
      {:ok, body} = Jason.decode(conn.resp_body)

      assert body["count"] == 0
      assert body["operations"] == []
    end

    test "returns operations since sequence number" do
      vector = <<1.0::little-float-64, 2.0::little-float-64>>
      {:ok, _} = Replication.record_upsert("key1", vector, %{}, 1)
      {:ok, _} = Replication.record_upsert("key2", vector, %{}, 1)

      conn = conn(:get, "/replication/deltas?since=0")
      conn = MerkleDb.Web.Router.call(conn, [])

      assert conn.status == 200
      {:ok, body} = Jason.decode(conn.resp_body)

      assert body["count"] == 2
      assert length(body["operations"]) == 2
    end

    test "respects limit parameter" do
      vector = <<1.0::little-float-64>>
      for i <- 1..10 do
        {:ok, _} = Replication.record_upsert("key#{i}", vector, %{}, 1)
      end

      conn = conn(:get, "/replication/deltas?since=0&limit=5")
      conn = MerkleDb.Web.Router.call(conn, [])

      assert conn.status == 200
      {:ok, body} = Jason.decode(conn.resp_body)

      assert body["count"] == 5
    end
  end

  describe "POST /replication/apply" do
    test "applies upsert operations" do
      operations = [
        %{
          "op" => "upsert",
          "key" => "applied_key",
          "data" => %{"vector" => Base.encode64(<<1.0::little-float-64, 2.0::little-float-64>>), "payload" => %{}}
        }
      ]

      body = Jason.encode!(%{operations: operations})
      conn = conn(:post, "/replication/apply", body)
      conn = put_req_header(conn, "content-type", "application/json")
      conn = MerkleDb.Web.Router.call(conn, [])

      assert conn.status == 200
      {:ok, resp} = Jason.decode(conn.resp_body)
      assert resp["status"] == "applied"
      assert resp["count"] == 1
    end

    test "returns error for missing operations" do
      body = Jason.encode!(%{})
      conn = conn(:post, "/replication/apply", body)
      conn = put_req_header(conn, "content-type", "application/json")
      conn = MerkleDb.Web.Router.call(conn, [])

      assert conn.status == 400
      {:ok, resp} = Jason.decode(conn.resp_body)
      assert resp["error"] == "Missing operations array"
    end
  end

  describe "GET /replication/snapshot" do
    test "returns snapshot metadata" do
      conn = conn(:get, "/replication/snapshot")
      conn = MerkleDb.Web.Router.call(conn, [])

      assert conn.status == 200
      {:ok, body} = Jason.decode(conn.resp_body)

      assert body["type"] == "full_snapshot"
      assert Map.has_key?(body, "timestamp")
      assert Map.has_key?(body, "tree_stats")
    end
  end

  describe "POST /replication/compact" do
    test "compacts oplog" do
      vector = <<1.0::little-float-64>>
      for i <- 1..100 do
        {:ok, _} = Replication.record_upsert("key#{i}", vector, %{}, 1)
      end

      body = Jason.encode!(%{keep_last: 10})
      conn = conn(:post, "/replication/compact", body)
      conn = put_req_header(conn, "content-type", "application/json")
      conn = MerkleDb.Web.Router.call(conn, [])

      assert conn.status == 200
      {:ok, resp} = Jason.decode(conn.resp_body)
      assert resp["status"] == "compacted"
      assert resp["deleted"] == 90
    end
  end
end
