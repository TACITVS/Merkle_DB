defmodule MerkleDb.SemanticDemo do
  alias MerkleDb.{KV, Query, TextEmbedding, Tree, Ingestor}

  @collection "crime_and_punishment"
  @filename "crime_and_punishment.txt"

  def run do
    # Set environment variable for test GloVe file
    System.put_env("GLOVE_FILE", "data/glove_test.txt")
    
    IO.puts "=== MerkleDB Semantic Retrieval Demo (Sliding Window) ==="
    IO.puts "Book: Crime and Punishment (Dummy)"
    
    # 1. Setup - 300 dimensions for GloVe
    setup_collection()
    
    # 2. Ingest with Sliding Window
    # Size 10 words, overlap 5 words
    chunks = Ingestor.chunk_file(@filename, 10, 5)
    IO.puts "Ingesting #{length(chunks)} overlapping text chunks..."
    
    ingest_chunks(chunks)
    IO.puts "Ingestion complete."
    
    # 3. Build Index
    IO.puts "Building HNSW index..."
    tree = KV.snapshot(@collection)
    updated_tree = Tree.build_hnsw(tree, m: 16, ef_construction: 64)
    KV.set_tree(@collection, updated_tree)
    IO.puts "Index built."
    
    # 4. Search Demo using the new :semantic query type
    IO.puts "\n=== Semantic Search Results (AVX2 Accelerated) ==="
    
    queries = [
      "philosophy",
      "logic",
      "philosophy logic"
    ]
    
    Enum.each(queries, fn q ->
      search(q)
    end)
  end

  defp setup_collection do
    # Create with 300 dims
    case KV.create_collection(@collection, dim: 300) do
      :ok -> :ok
      {:error, :already_exists} -> 
        KV.drop_collection(@collection)
        KV.create_collection(@collection, dim: 300)
    end
  end

  defp load_and_chunk do
    File.read!(@filename)
    |> String.split("\n", trim: true)
    |> Enum.map(&String.trim/1)
    |> Enum.with_index()
  end

  defp ingest_chunks(chunks) do
    kv_pairs = 
      chunks
      |> Enum.with_index()
      |> Enum.map(fn {text, idx} ->
        id = "chunk_" <> Integer.to_string(idx)
        # embed/1 returns f32 binary
        vec = TextEmbedding.embed(text)
        meta = %{"text" => text}
        {id, vec, meta}
      end)
    
    KV.put_batch(@collection, kv_pairs)
  end

  defp search(query_text) do
    IO.puts "\n🔎 Semantic Query: \"#{query_text}\""
    
    {time, results} = :timer.tc(fn ->
      # Use the new :semantic query type implemented in Query.execute
      Query.execute(KV.snapshot(@collection), [:semantic, query_text, 3, 0.0])
    end)
    
    IO.puts "   (Found #{length(results)} results in #{Float.round(time/1000, 2)}ms)"
    
    Enum.each(results, fn {key, score} ->
      tree = KV.snapshot(@collection)
      idx = Map.get(tree.key_index, key)
      meta = Map.get(tree.metadata, idx, %{})
      text = Map.get(meta, "text", "No preview")
      
      IO.puts "   - [#{Float.round(score, 4)}] #{key}: \"#{text}\""
    end)
  end
end

MerkleDb.SemanticDemo.run()
