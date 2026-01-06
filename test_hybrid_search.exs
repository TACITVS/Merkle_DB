alias MerkleDb.{KV, Query, Tree}

IO.puts "--- Hybrid Search (RRF) Test ---"

# Start App
{:ok, _} = Application.ensure_all_started(:merkle_db)
KV.reset()

# 1. Insert data
# Vector 1: Semantic "Apple", Keywords [1, 2]
v1 = for _ <- 1..64, into: <<>>, do: <<1.0::little-float-64>> # Dummy "Apple" vector
KV.put("apple", v1, %{"desc" => "red fruit"})
# Add sparse vector (keywords)
tree = KV.snapshot()
tree = Tree.insert_sparse(tree, "apple", [{1, 1.0}, {2, 1.0}], 1000)
KV.set_tree(tree)

# Vector 2: Semantic "Orange", Keywords [3, 4]
v2 = for _ <- 1..64, into: <<>>, do: <<0.1::little-float-64>> # Different semantic
KV.put("orange", v2, %{"desc" => "orange fruit"})
tree = KV.snapshot()
tree = Tree.insert_sparse(tree, "orange", [{3, 1.0}, {4, 1.0}], 1000)
KV.set_tree(tree)

# Vector 3: Semantic "Banana", Keywords [1, 3] (overlaps with both)
v3 = for _ <- 1..64, into: <<>>, do: <<0.5::little-float-64>>
KV.put("banana", v3, %{"desc" => "yellow fruit"})
tree = KV.snapshot()
tree = Tree.insert_sparse(tree, "banana", [{1, 1.0}, {3, 1.0}], 1000)
KV.set_tree(tree)

# 2. Perform Hybrid Query
# Semantic search for "Apple" (v1)
# Sparse search for Keyword 3 (matches Orange and Banana)
query_vec = v1
sparse_query = {[{3, 1.0}], 1000}

IO.puts "\nPerforming Hybrid Search (Semantic: Apple, Keyword: 3)..."
results = Query.execute(KV.snapshot(), [:hybrid, query_vec, sparse_query, 3, 0.0])

IO.puts "Results (Top 3):"
Enum.each(results, fn {key, score} ->
  IO.puts " - #{key}: RRF Score = #{Float.round(score, 6)}"
end)

# Verification
# "banana" should be high because it matches both (semantic similarity to apple + keyword 3)
# "apple" has high semantic, 0 keyword.
# "orange" has high keyword, low semantic.

if length(results) > 0 do
  IO.puts "✅ SUCCESS: Hybrid search returned results."
else
  IO.puts "❌ FAILURE: No results returned."
  exit({:error, :no_results})
end
