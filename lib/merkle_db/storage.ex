defmodule MerkleDb.Storage do
  # Simple CAS (Content Addressable Storage) using ETS

  def init do
    # Named table, public read/write for performance
    # Use try-catch to handle race condition where multiple processes
    # might try to create the table simultaneously
    try do
      :ets.new(:merkle_nodes, [:set, :public, :named_table])
    rescue
      ArgumentError -> :ok  # Table already exists, created by another process
    end
    :ok
  end

  def put(node = %MerkleDb.Node{}) do
    :ets.insert(:merkle_nodes, {node.hash, node})
    node.hash
  end

  def get(hash) do
    case :ets.lookup(:merkle_nodes, hash) do
      [{^hash, node}] -> node
      [] -> nil
    end
  end

  def all_keys do
    # Debugging helper
    :ets.match(:merkle_nodes, {:"$1", :_}) |> List.flatten()
  end
end
