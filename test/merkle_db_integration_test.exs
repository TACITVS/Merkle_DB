defmodule MerkleDb.IntegrationTest do
  use ExUnit.Case
  alias MerkleDb.{Tree, Query, KV}

  # Helper to create a binary vector of floats
  defp vec(floats) do
    for f <- floats, into: <<>>, do: <<f::little-float-64>>
  end

  setup do
    collection = "test_#{:erlang.unique_integer([:positive])}"
    :ok = KV.create_collection(collection)
    
    # Wait for collection to be visible
    wait_for_collection(collection, 10)

    on_exit(fn -> 
      KV.drop_collection(collection) 
    end)
    {:ok, collection: collection}
  end

  defp wait_for_collection(name, attempts) when attempts > 0 do
    # Actually try to fetch it via snapshot to be 100% sure state machine applied it
    case KV.snapshot(name) do
      %MerkleDb.Tree{} -> :ok
      _ ->
        Process.sleep(500)
        wait_for_collection(name, attempts - 1)
    end
  end
  defp wait_for_collection(_, _), do: :timeout

  describe "Core Operations" do
    test "basic insert and knn search", %{collection: coll} do
      v1 = vec([1.0, 0.0, 0.0])
      v2 = vec([0.0, 1.0, 0.0])
      v3 = vec([0.0, 0.0, 1.0])

      :ok = KV.put(coll, "v1", v1)
      :ok = KV.put(coll, "v2", v2)
      :ok = KV.put(coll, "v3", v3)

      # Refresh snapshot
      tree = KV.snapshot(coll)
      
      # Search for v1 (exact match)
      results = Query.execute(tree, [:knn, v1, 3, 0.0])
      assert length(results) == 3
      {id, score} = hd(results)
      assert id == "v1"
      assert_in_delta score, 1.0, 0.0001
    end

    test "delete and update", %{collection: coll} do
      v1 = vec([1.0, 0.0])
      v1_new = vec([0.0, 1.0])

      :ok = KV.put(coll, "key1", v1)
      tree = KV.snapshot(coll)
      assert [{ "key1", _ }] = Query.execute(tree, [:knn, v1, 1, 0.9])

      # Delete
      :ok = KV.delete(coll, "key1")
      tree = KV.snapshot(coll)
      assert [] == Query.execute(tree, [:knn, v1, 1, 0.0])

      # Update (re-insert)
      :ok = KV.put(coll, "key1", v1_new)
      tree = KV.snapshot(coll)
      
      # Should find new vector
      results = Query.execute(tree, [:knn, v1_new, 1, 0.9])
      assert [{ "key1", _ }] = results
      
      # Should NOT find old vector (if orthogonal)
      results_old = Query.execute(tree, [:knn, v1, 1, 0.9])
      assert [] == results_old
    end
  end

  describe "Metadata Filtering" do
    test "filter by equality and range", %{collection: coll} do
      v = vec([1.0, 0.0])
      
      :ok = KV.put(coll, "p1", v, %{"cat" => "A", "price" => 10})
      :ok = KV.put(coll, "p2", v, %{"cat" => "B", "price" => 20})
      :ok = KV.put(coll, "p3", v, %{"cat" => "A", "price" => 30})

      tree = KV.snapshot(coll)

      # Filter: cat == "A"
      results = Query.execute(tree, [:knn, v, 10, 0.0, {:where, [{"cat", :eq, "A"}]}])
      ids = Enum.map(results, fn {id, _} -> id end) |> Enum.sort()
      assert ids == ["p1", "p3"]

      # Filter: price > 15
      results = Query.execute(tree, [:knn, v, 10, 0.0, {:where, [{"price", :gt, 15}]}])
      ids = Enum.map(results, fn {id, _} -> id end) |> Enum.sort()
      assert ids == ["p2", "p3"]

      # Filter: cat == "A" AND price < 20
      results = Query.execute(tree, [:knn, v, 10, 0.0, {:where, [{"cat", :eq, "A"}, {"price", :lt, 20}]}])
      assert [{ "p1", _ }] = results
    end
  end

  describe "Advanced Indexing" do
    test "HNSW index build and search", %{collection: coll} do
      # Insert enough vectors to make HNSW useful
      data = for i <- 1..100 do
         # Random-ish vectors
         x = :math.sin(i / 10.0)
         y = :math.cos(i / 10.0)
         { "k#{i}", vec([x, y]) }
      end
      
      :ok = KV.put_batch(coll, data)
      tree = KV.snapshot(coll)
      
      # Build HNSW
      tree_hnsw = Tree.build_hnsw(tree, m: 32, ef_construction: 128)
      assert tree_hnsw.hnsw != nil

      # Update KV with the HNSW tree
      :ok = KV.update_index(coll, tree_hnsw, tree.generation)
      
      # Verify search works
      q = vec([1.0, 0.0])
      results = Query.execute(tree_hnsw, [:knn, q, 5, 0.0])
      assert length(results) == 5
    end

    test "Int8 Quantization", %{collection: coll} do
      v1 = vec([0.1, 0.2, 0.3, 0.4])
      v2 = vec([-0.1, -0.2, -0.3, -0.4])
      
      :ok = KV.put(coll, "v1", v1)
      :ok = KV.put(coll, "v2", v2)
      
      tree = KV.snapshot(coll)
      tree_q = Tree.quantize(tree)
      
      assert tree_q.quantized != nil
      
      # Search using quantized index
      # Note: Quantization is lossy, scores might differ slightly
      results = Query.execute(tree_q, [:knn, v1, 1, 0.0])
      assert [{ "v1", _ }] = results
    end
  end

  describe "Sparse Vectors" do
    test "Hybrid search", %{collection: coll} do
      # Dense: v1 is close to q_dense
      v1_dense = vec([1.0, 0.0])
      v2_dense = vec([0.0, 1.0])
      
      :ok = KV.put(coll, "v1", v1_dense)
      :ok = KV.put(coll, "v2", v2_dense)

      # Add Sparse: v2 has high sparse overlap
      # Sparse dim 100
      tree = KV.snapshot(coll)
      tree = Tree.insert_sparse(tree, "v1", [{1, 0.1}], 100)
      tree = Tree.insert_sparse(tree, "v2", [{50, 1.0}], 100) # High match
      
      # Update KV
      :ok = KV.set_tree(coll, tree)
      
      # Query
      q_dense = vec([0.0, 1.0]) # Matches v2 dense
      q_sparse = {[{50, 1.0}], 100} # Matches v2 sparse
      
      # Hybrid search: should favor v2 strongly
      results = Query.execute(tree, [:hybrid, q_dense, q_sparse, 2, 0.0, [alpha: 0.5]])
      
      {top_id, _} = hd(results)
      assert top_id == "v2"
    end
  end
end
