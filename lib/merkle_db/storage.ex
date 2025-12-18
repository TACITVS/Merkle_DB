defmodule MerkleDb.Storage do
  # Simple CAS (Content Addressable Storage) using ETS
  
  def init do
    # Named table, public read/write for performance
    if :ets.info(:merkle_nodes) == :undefined do
      :ets.new(:merkle_nodes, [:set, :public, :named_table])
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
