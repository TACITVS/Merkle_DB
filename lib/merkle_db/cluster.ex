defmodule MerkleDb.Cluster do
  alias MerkleDb.{Query, TextStore}

  # Analyzes a SINGLE candidate index.
  # Returns: {:found, topic_data, new_visited_set} OR {:skip, new_visited_set}
  def analyze_step(tree, candidate_idx, visited) do
    if MapSet.member?(visited, candidate_idx) do
      {:skip, visited}
    else
      # 1. Reconstruct Vector
      leader_vec = get_vector_at(tree, candidate_idx)
      leader_key = Map.get(tree.keys, candidate_idx)

      # 2. ASM Search (Fast Batch Scan)
      #    Strictness 0.35, find up to 2000 neighbors
      matches = Query.execute(tree, [:knn, leader_vec, 2000, 0.35])

      # 3. Convert Keys to Indices
      cluster_indices = 
        matches 
        |> Enum.map(fn {key, _, _} -> find_index_by_key(tree, key) end)

      # 4. Filter Overlaps
      new_members = Enum.reject(cluster_indices, &MapSet.member?(visited, &1))

      # 5. Result
      if length(new_members) > 3 do
        # Valid Topic Found!
        topic_label = TextStore.get(leader_key) |> String.slice(0, 60)
        topic = %{
          id: leader_key,
          label: topic_label <> "...",
          count: length(new_members)
        }
        
        # Mark all as visited
        updated_visited = Enum.reduce(new_members, visited, &MapSet.put(&2, &1))
        
        {:found, topic, updated_visited}
      else
        # Noise / Too Small
        {:skip, visited}
      end
    end
  end

  defp get_vector_at(tree, idx) do
    offset = idx * 8
    for i <- 0..63 do
      col = elem(tree.columns, i)
      <<_::binary-size(offset), val::little-float-size(64), _::binary>> = col
      val
    end
    |> Enum.map(fn x -> <<x::little-float-size(64)>> end)
    |> IO.iodata_to_binary()
  end
  
  defp find_index_by_key(tree, key) do
    {idx, _} = Enum.find(tree.keys, fn {_, k} -> k == key end)
    idx
  end
end