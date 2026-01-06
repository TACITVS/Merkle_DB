alias MerkleDb.{KV, Query, Tree}

defmodule BatchQueryBenchmark do
  @collection "bench_batch"
  @dim 64
  @total_vectors 1000
  @batch_size 20
  @k 10

  def run do
    File.write!("bench.log", "Starting benchmark\n", [:append])
    try do
      IO.puts "=== MerkleDB Batch Query API Benchmark ==="
      setup()

      query_vectors = for _ <- 1..@batch_size, do: generate_random_vector_bin(@dim)
      
      IO.puts "Scenario: #{@total_vectors} vectors, Batch size: #{@batch_size}"

      # 1. Single Query Benchmark
      IO.puts "\n1. Running #{@batch_size} queries INDIVIDUALLY..."
      File.write!("bench.log", "Starting individual queries\n", [:append])
      {time_single, _} = :timer.tc(fn ->
        tree = try do
          File.write!("bench.log", "Calling snapshot\n", [:append])
          res = GenServer.call(KV, {:snapshot, @collection}, 30_000)
          File.write!("bench.log", "Snapshot returned\n", [:append])
          res
        catch
          kind, reason -> 
            IO.puts "ERROR during snapshot: #{inspect({kind, reason})}"
            File.write!("bench.log", "ERROR during snapshot: #{inspect({kind, reason})}\n", [:append])
            nil
        end
        
        if tree do
          Enum.each(Enum.with_index(query_vectors), fn {q, i} ->
            File.write!("bench.log", "Query #{i+1} starting\n", [:append])
            _ = Query.execute(tree, [:knn, q, @k, 0.0])
            File.write!("bench.log", "Query #{i+1} finished\n", [:append])
            IO.write "#{i+1} "
          end)
        end
        IO.puts ""
      end)
      qps_single = Float.round(@batch_size / (time_single / 1_000_000.0), 2)
      IO.puts "   - Total time: #{Float.round(time_single/1000, 2)}ms"
      IO.puts "   - Throughput: #{qps_single} QPS"

      # 2. Batch Query Benchmark
      IO.puts "\n2. Running #{@batch_size} queries in a SINGLE BATCH..."
      {time_batch, _} = :timer.tc(fn ->
        tree = GenServer.call(KV, {:snapshot, @collection}, 30_000)
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
    GenServer.call(KV, {:put_batch, @collection, vectors}, 60_000)
    IO.puts " Done."
  end

  defp generate_random_vector_bin(dim) do
    for _ <- 1..dim, into: <<>>, do: <<:rand.uniform()::little-float-64>>
  end
end

BatchQueryBenchmark.run()