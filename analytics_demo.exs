# MerkleDb Analytics & IVF Demo
alias MerkleDb.{Tree, Analytics, Query}

IO.puts "--- MerkleDb Research & Analytics Demo ---"

# 1. Create a tree with 1000 random 64-dim vectors
dim = 64
count = 1000
IO.puts "Generating #{count} random vectors..."

tree = Enum.reduce(1..count, Tree.new(), fn i, acc ->
  vec = for _ <- 1..dim, into: <<>>, do: <<:rand.uniform()::little-float-64>>
  Tree.insert(acc, "Vec-#{i}", vec)
end)

# 2. Run Analytics: Column Stats
stats = Analytics.column_stats(tree, 0)
IO.puts "\n📊 Stats for Dimension 0:"
IO.inspect(stats)

# 3. Build IVF Index using K-Means
k_clusters = 10
IO.puts "\n🏗️  Building IVF Index (K-Means Clustering, k=#{k_clusters})..."
{time_ivf, tree_indexed} = :timer.tc(fn -> 
  Analytics.build_ivf_index(tree, k_clusters)
end)
IO.puts "Index built in #{time_ivf / 1000} ms"

# 4. Compare Search Speed
query_vec = for _ <- 1..dim, into: <<>>, do: <<0.5::little-float-64>>

IO.puts "\n🔍 Flat KNN Search:"
{time_flat, results_flat} = :timer.tc(fn -> 
  Query.execute(tree, [:knn, query_vec, 5, 0.0])
end)
IO.puts "Flat search took #{time_flat} us"
IO.inspect(results_flat)

IO.puts "\n🔍 IVF Accelerated Search:"
{time_ivf_search, results_ivf} = :timer.tc(fn -> 
  Query.execute(tree_indexed, [:knn, query_vec, 5, 0.0])
end)
IO.puts "IVF search took #{time_ivf_search} us"
IO.inspect(results_ivf)

speedup = time_flat / time_ivf_search
IO.puts "\n🚀 IVF Speedup: #{Float.round(speedup, 2)}x"
