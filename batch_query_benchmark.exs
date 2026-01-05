alias MerkleDb.{KV, Query, Tree}

defmodule BatchQueryBenchmark do
  @collection "bench_batch"
  @dim 64
  @total_vectors 10000
  @batch_size 20
  @k 10

  def run do
    try do
      IO.puts "=== MerkleDB Batch Query API Benchmark ==="
      setup()

      query_vectors = for _ <- 1..@batch_size, do: generate_random_vector_bin(@dim)
      
      IO.puts "Scenario: #{@total_vectors} vectors, Batch size: #{@batch_size}"

      # 1. Single Query Benchmark
      IO.puts "\n1. Running #{@batch_size} queries INDIVIDUALLY..."
      {time_single, _} = :timer.tc(fn ->
        tree = KV.snapshot(@collection)
        Enum.each(Enum.with_index(query_vectors), fn {q, _i} ->
          _ = Query.execute(tree, [:knn, q, @k, 0.0])
          IO.write "."
        end)
        IO.puts ""
      end)
      qps_single = Float.round(@batch_size / (time_single / 1_000_000.0), 2)
      IO.puts "   - Total time: #{Float.round(time_single/1000, 2)}ms"
      IO.puts "   - Throughput: #{qps_single} QPS"

      # 2. Batch Query Benchmark
      IO.puts "\n2. Running #{@batch_size} queries in a SINGLE BATCH..."
      {time_batch, _} = :timer.tc(fn ->
        tree = KV.snapshot(@collection)
        _ = Query.execute_batch(tree, query_vectors, @k, 0.0)
      end)
      qps_batch = Float.round(@batch_size / (time_batch / 1_000_000.0), 2)
      IO.puts "   - Total time: #{Float.round(time_batch/1000, 2)}ms"
      IO.puts "   - Throughput: #{qps_batch} QPS"

      # 3. Summary
      speedup = Float.round(time_single / time_batch, 2)
      IO.puts "\n=== RESULT ==="
      IO.puts "🚀 Batch Query is #{speedup}x faster than individual queries."
      IO.puts "Amortized NIF overhead: #{Float.round((time_single - time_batch)/@batch_size/1000, 4)}ms saved per query."
    rescue
      e -> IO.puts "❌ Benchmark Failed: #{inspect(e)}"
    end
  end

  defp setup do
    KV.create_collection(@collection)
    KV.reset(@collection)
    
    IO.write "Ingesting #{@total_vectors} vectors..."
    vectors = for i <- 1..@total_vectors, do: {"vec_#{i}", generate_random_vector_bin(@dim)}
    KV.put_batch(@collection, vectors)
    IO.puts " Done."
  end

  defp generate_random_vector_bin(dim) do
    for _ <- 1..dim, into: <<>>, do: <<:rand.uniform()::little-float-64>>
  end
end

BatchQueryBenchmark.run()
